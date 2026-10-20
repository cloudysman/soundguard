"""SoundGuard — audio -> frozen ResNet-18 features (serving side).

Mirrors the training notebook's preprocessing EXACTLY (driven by feature_config.json)
so there is no train/serve skew:
    resample -> mono -> pad/crop to sr*max_length_s -> log-mel -> 1->3 channels
    -> per-sample min-max[0,1] -> ImageNet normalisation -> frozen ResNet-18 -> [1, 512]

Imported lazily by predictor.py (needs torch + torchaudio + torchvision).
"""
import json
import os

DEFAULT_CONFIG = {
    "sample_rate": 16000, "max_length_s": 4, "n_mels": 64,
    "n_fft": 1024, "hop_length": 512,
    "normalize": "minmax01_then_imagenet",
    "in_features": 512, "num_classes": 10,
}


class FeatureExtractor:
    def __init__(self, config_path="checkpoints/feature_config.json", device=None):
        import torch
        import torchaudio
        import torchvision

        self.cfg = self._load_config(config_path)
        self.sr = self.cfg["sample_rate"]
        self.num_samples = self.sr * self.cfg["max_length_s"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._torchaudio = torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=self.cfg["n_fft"],
            hop_length=self.cfg["hop_length"], n_mels=self.cfg["n_mels"],
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        backbone.fc = torch.nn.Identity()
        self.backbone = backbone.to(self.device).eval()      # eval(): frozen BatchNorm

    def _load_config(self, path):
        if path and os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
            return {**DEFAULT_CONFIG, **cfg}
        return dict(DEFAULT_CONFIG)

    def _pad_or_crop(self, wav):
        import torch
        n = wav.shape[1]
        if n > self.num_samples:
            return wav[:, :self.num_samples]
        if n < self.num_samples:
            return torch.nn.functional.pad(wav, (0, self.num_samples - n))
        return wav

    def wav_to_log_mel(self, wav, sr):
        """wav: 1-D (or [C,T]) numpy/torch in [-1,1]; returns torch [1, n_mels, time] in decibels.

        Steps 1 to 5 of the pipeline. Split out from wav_to_features so the serving layer can show
        the spectrogram the classifier is actually given, without computing the pipeline twice."""
        import numpy as np
        import torch

        if not torch.is_tensor(wav):
            wav = torch.tensor(np.asarray(wav, dtype="float32"))
        wav = wav.float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)                            # [1, T]
        elif wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)               # downmix to mono

        # Discard everything past the window we keep BEFORE resampling. Those samples are cropped away
        # by _pad_or_crop regardless, so resampling them is pure waste: a 60 s upload cost 2.28x a 4 s
        # one for byte-identical features. The one-second margin leaves the resampling filter full
        # context for the samples that are kept, so the features are unchanged.
        keep = int(self.cfg["max_length_s"] * sr) + sr
        if wav.shape[1] > keep:
            wav = wav[:, :keep]

        if sr != self.sr:
            wav = self._torchaudio.transforms.Resample(sr, self.sr)(wav)
        wav = self._pad_or_crop(wav)

        return self.amp_to_db(self.mel(wav))                  # [1, n_mels, time]

    def log_mel_to_features(self, log_mel):
        """log_mel: [1, n_mels, time]; returns torch [1, 512]. Steps 6 to 9 of the pipeline."""
        import torch

        feats = log_mel.expand(3, -1, -1).clone()             # 1 -> 3 channels
        fmin, fmax = feats.min(), feats.max()
        feats = (feats - fmin) / (fmax - fmin + 1e-6)
        feats = (feats - self.mean) / self.std
        feats = feats.unsqueeze(0).to(self.device)            # [1, 3, n_mels, time]

        with torch.no_grad():
            return self.backbone(feats)                       # [1, 512]

    def wav_to_features(self, wav, sr):
        """wav: 1-D (or [C,T]) numpy/torch in [-1,1]; returns torch [1, 512].

        The whole pipeline in one call. Callers that also need the spectrogram should invoke the
        two halves themselves rather than calling this and recomputing."""
        return self.log_mel_to_features(self.wav_to_log_mel(wav, sr))
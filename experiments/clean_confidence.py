"""Per-recording confidence of the frozen-backbone classifier on clean UrbanSound8K.

The published numbers average over five seeds and the ten official folds. This script
reproduces that protocol and keeps the per-recording confidence, which the stored results do
not: each recording is scored by a head trained on the nine folds that exclude it, and the
confidence is averaged over the seeds.

Writes rerun_results/clean_conf.npz with
    conf     [N]  mean top-category confidence over the seeds
    correct  [N]  whether the mean prediction is right
    label    [N]
    fold     [N]

    python clean_confidence.py
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sg import backbone_resnet18 as BB
from sg import data

CACHE, OUT = "cache", "rerun_results"
EPOCHS, BS, LR, WD = 50, 128, 1e-3, 1e-4
SEEDS = 5
DEVICE = BB.DEVICE


def embed_clean(ds):
    dst = os.path.join(CACHE, f"{ds.name}_resnet18_emb_clean.pt")
    wp, y, folds = data.cache(ds, CACHE)
    if os.path.exists(dst):
        return torch.load(dst), y, folds
    waves = np.load(wp, mmap_mode="r")
    handle = BB.load()
    emb = BB.embed(handle, np.asarray(waves), batch=32)
    torch.save(emb, dst)
    print(f"  embedded clean: {tuple(emb.shape)}", flush=True)
    return emb, y, folds


def train_head(x, t, dim, n_cls, seed):
    torch.manual_seed(seed)
    head = nn.Linear(dim, n_cls).to(DEVICE)
    opt = optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    n = len(t)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            crit(head(x[idx]), t[idx]).backward()
            opt.step()
        sch.step()
    return head.eval()


def main():
    ds = data.REGISTRY["us8k"]
    emb, y, folds = embed_clean(ds)
    x = emb.to(DEVICE).float()
    t = torch.as_tensor(y, device=DEVICE).long()
    n, dim = x.shape
    probs = np.zeros((n, ds.n_classes), dtype=np.float64)

    for seed in range(SEEDS):
        for f in sorted(set(folds)):
            te = folds == f
            tr = ~te
            head = train_head(x[tr], t[tr], dim, ds.n_classes, seed)
            with torch.no_grad():
                p = torch.softmax(head(x[te]), dim=1).cpu().numpy()
            probs[te] += p
        print(f"  seed {seed} done", flush=True)

    probs /= SEEDS
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "clean_conf.npz"),
             conf=conf.astype(np.float32), correct=correct,
             label=np.asarray(y), fold=np.asarray(folds))
    print(f"accuracy {correct.mean():.4f} | mean confidence {conf.mean():.4f}", flush=True)
    print(f"wrote {OUT}/clean_conf.npz", flush=True)


if __name__ == "__main__":
    main()

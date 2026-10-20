"""SoundGuard — aggregate per-fold logits into results/dashboard.json.

Reads the dumps produced by the training notebook:
    results/{head}_fold{N}_val.pt            (clean    — required)
    results/{head}_fold{N}_val_corrupted.pt  (corrupted — optional, needs Hien's corruption + a re-run)
Each dump is a dict {"logits": [n,10], "labels": [n]}.

Writes results/dashboard.json in the exact schema the backend /api/dashboard serves,
so the dashboard switches from mock to real numbers with no code change.

Usage:
    python eval.py --head mlp        # or --head linear
"""
import os
import glob
import json
import argparse

import metrics as M

CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]


def load_dumps(head, tag, results_dir):
    import torch  # lazy: only needed to read the .pt dumps
    files = sorted(glob.glob(os.path.join(results_dir, f"{head}_fold*_{tag}.pt")))
    if not files:
        return None
    logits, labels = [], []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        logits.append(d["logits"])
        labels.append(d["labels"])
    return torch.cat(logits), torch.cat(labels)


def _block(logits, labels):
    rel = M.reliability_curve(logits, labels, n_bins=15)
    rc = M.risk_coverage_curve(logits, labels)
    return {
        "ece": M.ece(logits, labels, n_bins=15),
        "nll": M.nll(logits, labels),
        "brier": M.brier(logits, labels),
        "accuracy": M.accuracy(logits, labels),
        "rel_bins": rel["centers"],
        "rel_acc": rel["accuracy"],
        "coverage": rc["coverage"],
        "risk": rc["risk"],
        "per_class": M.per_class_accuracy(logits, labels, num_classes=len(CLASSES)),
    }


def build_dashboard(head="mlp", results_dir="./results", out_path=None):
    out_path = out_path or os.path.join(results_dir, "dashboard.json")

    clean_dump = load_dumps(head, "val", results_dir)
    if clean_dump is None:
        raise SystemExit(
            f"No clean dumps at {results_dir}/{head}_fold*_val.pt — "
            f"run the training notebook first and copy results/ out of Colab."
        )
    clean = _block(*clean_dump)

    corr_dump = load_dumps(head, "val_corrupted", results_dir)
    corrupted = _block(*corr_dump) if corr_dump is not None else None
    corr_pc = {p["class"]: p["accuracy"] for p in corrupted["per_class"]} if corrupted else {}

    def r3(xs):
        return [round(float(x), 3) for x in xs]

    dashboard = {
        "reliability": {
            "bins": r3(clean["rel_bins"]),
            "clean": r3(clean["rel_acc"]),
            "corrupted": r3(corrupted["rel_acc"]) if corrupted else None,
        },
        "riskCoverage": {
            "coverage": r3(clean["coverage"]),
            "clean": r3(clean["risk"]),
            "corrupted": r3(corrupted["risk"]) if corrupted else None,
        },
        "classAcc": [
            {
                "k": CLASSES[p["class"]],
                "clean": round(p["accuracy"] * 100, 1),
                "corrupt": (round(corr_pc[p["class"]] * 100, 1) if corrupted else None),
            }
            for p in sorted(clean["per_class"], key=lambda x: -x["accuracy"])
        ],
        "metrics": [
            {"k": "ECE", "desc": "Expected Calibration Error", "clean": round(clean["ece"], 4),
             "corrupt": (round(corrupted["ece"], 4) if corrupted else None), "lowerBetter": True},
            {"k": "NLL", "desc": "Negative Log-Likelihood", "clean": round(clean["nll"], 3),
             "corrupt": (round(corrupted["nll"], 3) if corrupted else None), "lowerBetter": True},
            {"k": "Brier", "desc": "Brier score", "clean": round(clean["brier"], 3),
             "corrupt": (round(corrupted["brier"], 3) if corrupted else None), "lowerBetter": True},
            {"k": "Accuracy", "desc": "Top-1 accuracy", "clean": round(clean["accuracy"], 3),
             "corrupt": (round(corrupted["accuracy"], 3) if corrupted else None),
             "lowerBetter": False, "pct": True},
        ],
        "_head": head,
        "_corrupted_pending": corrupted is None,
    }

    with open(out_path, "w") as f:
        json.dump(dashboard, f, indent=2)
    return dashboard, out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--results", default="./results")
    args = ap.parse_args()

    dash, path = build_dashboard(args.head, args.results)
    m = {x["k"]: x for x in dash["metrics"]}
    print(f"Wrote {path}  (head={args.head})")
    print(f"  clean   | ECE {m['ECE']['clean']}  NLL {m['NLL']['clean']}  "
          f"Brier {m['Brier']['clean']}  Acc {m['Accuracy']['clean']}")
    if dash["_corrupted_pending"]:
        print("  corrupted | PENDING — needs corrupted-audio logits:")
        print("            apply Hien's corruption to the val audio, re-run the model,")
        print("            dump results/{head}_fold{N}_val_corrupted.pt, then re-run this.")
    else:
        print(f"  corrupt | ECE {m['ECE']['corrupt']}  NLL {m['NLL']['corrupt']}  "
              f"Brier {m['Brier']['corrupt']}  Acc {m['Accuracy']['corrupt']}")
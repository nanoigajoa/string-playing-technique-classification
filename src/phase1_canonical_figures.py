"""Render manuscript figures from replay-verified canonical CSV artifacts."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/phase1-manuscript-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "phase1" / "canonical"
OUT = BASE / "manuscript" / "figures"
CLASSES = ["ordinario", "pizzicato_secco", "tremolo", "sul_ponticello", "col_legno_battuto"]
SHORT = ["ord.", "pizz.", "trem.", "sul pont.", "col legno"]
RATIOS = [0.05, 0.10, 0.20, 0.50]
INPUTS = [
    BASE / "mert_svm" / "canonical_summary.csv",
    BASE / "self_training" / "canonical_summary.csv",
    BASE / "verification" / "bootstrap_sensitivity.csv",
    BASE / "verification" / "confusion_counts.csv",
    BASE / "self_training" / "confusion_counts.csv",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, name):
    for extension in ("png", "pdf"):
        metadata = {"Creator": "Canonical figure renderer"} if extension == "pdf" else None
        fig.savefig(OUT / f"{name}.{extension}", dpi=300, metadata=metadata)
    plt.close(fig)


def confusion_matrix(frame):
    """Keep the common class order; counts pool three predictions per clip."""
    assert len(frame) == 25 and not frame.duplicated(["true_class", "predicted_class"]).any()
    counts = frame.pivot(index="true_class", columns="predicted_class", values="count")
    counts = counts.loc[CLASSES, CLASSES].to_numpy(dtype=int)
    assert np.all(counts >= 0) and counts.sum() == 3402 * 3
    assert np.all(counts.sum(axis=1) > 0)
    return counts / counts.sum(axis=1, keepdims=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-root", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    inputs = [args.tables_root.resolve() / path.relative_to(BASE) for path in INPUTS]
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    model, self_training, intervals, model_counts, self_counts = [pd.read_csv(path) for path in inputs]
    model = model.set_index("seed_ratio").loc[RATIOS]
    budgets = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for name, label, color in [("mert", "MERT head", "#0072B2"), ("svm", "Descriptor SVM", "#D55E00")]:
        axes[0].plot(budgets, model[f"{name}_macro_f1_fold_mean"], "o-", label=label, color=color)
    for name, label, color, style in [("supervised", "Matched supervised", "#0072B2", "o-"),
                                    ("pseudo", "Actual pseudo-targets", "#D55E00", "s-"),
                                    ("oracle", "Privileged oracle diagnostic", "#666666", "^--")]:
        rows = self_training[self_training.condition == name].set_index("ratio").loc[RATIOS]
        axes[1].plot(budgets, rows.macro_f1_fold_mean, style, label=label, color=color)
    for ax, title in zip(axes, ["(a) Validation-tuned models", "(b) Fixed self-training configuration"]):
        ax.set(title=title, xlabel="Labeled seed budget", ylabel="Mean macro F1", ylim=(0.60, 0.90))
        ax.set_xticks(budgets, ["5%", "10%", "20%", "50%"])
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, "figure1_budget_curves")

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for offset, unit, label, color in [(-0.18, "clip", "Clip", "#0072B2"),
                                     (0, "recording_proxy", "Filename proxy", "#009E73"),
                                     (0.18, "instrument_type", "Instrument type", "#D55E00")]:
        rows = intervals[(intervals.unit == unit) & (intervals.measure == "mert_minus_svm")]
        rows = rows.set_index("seed_ratio").loc[RATIOS]
        assert (rows.draws == 1000).all()
        assert np.allclose(rows.point, model.mert_minus_svm)
        errors = np.vstack([rows.point - rows.ci_low, rows.ci_high - rows.point])
        ax.errorbar(budgets + offset, rows.point, yerr=errors, fmt="o", capsize=4, color=color, label=label)
    ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
    ax.set_xticks(budgets, ["5%", "10%", "20%", "50%"])
    ax.set(xlabel="Labeled seed budget", ylabel="MERT minus SVM macro F1",
           title="Paired percentile intervals: sensitivity to resampling unit")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "figure2_resampling_units")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    panels = [
        (model_counts[(model_counts.condition == "ratio_0.5") & (model_counts.model == "mert")], "(a) MERT, 50%"),
        (model_counts[(model_counts.condition == "ratio_0.5") & (model_counts.model == "svm")], "(b) SVM, 50%"),
        (self_counts[(self_counts.ratio == 0.05) & (self_counts.condition == "supervised")], "(c) Matched supervised, 5%"),
        (self_counts[(self_counts.ratio == 0.05) & (self_counts.condition == "pseudo")], "(d) Actual pseudo-targets, 5%"),
    ]
    for ax, (frame, title) in zip(axes.flat, panels):
        values = confusion_matrix(frame)
        heatmap = ax.imshow(values, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(5), SHORT, rotation=35, ha="right")
        ax.set_yticks(range(5), SHORT)
        ax.set(title=title, xlabel="Predicted class", ylabel="True class")
        for row in range(5):
            for col in range(5):
                ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if values[row, col] > 0.55 else "black")
    fig.colorbar(heatmap, ax=axes, shrink=0.75, label="Row proportion of pooled predictions")
    save_figure(fig, "figure3_confusions")
    manifest = {
        "source_script_sha256": sha256(Path(__file__)),
        "inputs": {str(path.relative_to(args.tables_root.resolve())): sha256(path) for path in inputs},
        "outputs": {path.name: sha256(path) for path in sorted(OUT.glob("figure*.*"))},
        "dpi": 300,
        "point_estimate": "Mean trials within type, then mean four instrument types",
        "intervals": "1000 paired percentile draws conditional on fixed predictions; no population inference",
        "confusions": "Row-normalized pooled counts, three repeated predictions per clip",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Saved three PNG/PDF figures and input/output hashes.")


if __name__ == "__main__":
    main()

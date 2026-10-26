"""Leakage-safe rerun of the Phase 1 duration-confound baseline.

Unlike the historical script, residualization, scaling, SVC class weighting,
and the majority class are learned independently inside every outer LOSO fold.
Historical outputs are never overwritten; repaired outputs live under
``results/phase1/audit_repair``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phase1_train_only import (  # noqa: E402
    DurationResidualizer,
    array_hash,
    residualizer_hash,
    scaler_hash,
)


META = ROOT / "data" / "mert_meta.csv"
FEATS = ROOT / "data" / "features_full.csv"
DURATIONS = ROOT / "data" / "clip_durations.csv"
OUT_DIR = ROOT / "results" / "phase1" / "audit_repair"
SEED = 42


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def loso_predictions(X, y, groups, duration=None, name=""):
    """Return out-of-fold predictions and a fit provenance record per fold."""
    splitter = GroupKFold(n_splits=len(np.unique(groups)))
    predictions = np.full(y.shape, -1, dtype=np.int64)
    records = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        X_train = np.asarray(X[train_idx], dtype=np.float64)
        X_test = np.asarray(X[test_idx], dtype=np.float64)
        residual_hash = None
        if duration is not None:
            residualizer = DurationResidualizer.fit(X_train, duration[train_idx])
            X_train = residualizer.transform(X_train, duration[train_idx])
            X_test = residualizer.transform(X_test, duration[test_idx])
            residual_hash = residualizer_hash(residualizer)

        pipeline = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=SEED,
            ),
        )
        pipeline.fit(X_train, y[train_idx])
        predictions[test_idx] = pipeline.predict(X_test)
        scaler = pipeline.named_steps["standardscaler"]
        svc = pipeline.named_steps["svc"]
        records.append({
            "condition": name,
            "fold": fold,
            "held_out_group": str(np.unique(groups[test_idx]).tolist()),
            "n_train": int(train_idx.size),
            "n_test": int(test_idx.size),
            "train_index_hash": array_hash(train_idx),
            "test_index_hash": array_hash(test_idx),
            "train_class_counts": np.bincount(y[train_idx], minlength=len(np.unique(y))).tolist(),
            "svc_class_weight": svc.class_weight_.tolist(),
            "svc_class_weight_hash": array_hash(svc.class_weight_),
            "scaler_hash": scaler_hash(scaler),
            "residualizer_hash": residual_hash,
            "fit_scope": "train_idx_only",
        })
    if np.any(predictions < 0):
        raise AssertionError(f"{name}: missing out-of-fold predictions")
    return predictions, records


def fold_majority_predictions(y, groups):
    splitter = GroupKFold(n_splits=len(np.unique(groups)))
    predictions = np.full(y.shape, -1, dtype=np.int64)
    records = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros((len(y), 1)), y, groups)):
        counts = np.bincount(y[train_idx])
        majority = int(counts.argmax())
        predictions[test_idx] = majority
        records.append({
            "condition": "majority",
            "fold": fold,
            "held_out_group": str(np.unique(groups[test_idx]).tolist()),
            "train_class_counts": counts.tolist(),
            "majority_class": majority,
            "fit_scope": "train_idx_only",
        })
    return predictions, records


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    meta = pd.read_csv(META)
    features = pd.read_csv(FEATS)
    duration_frame = pd.read_csv(DURATIONS)
    if len(meta) != len(features) or len(meta) != len(duration_frame):
        raise ValueError("meta, features, and duration cache row counts differ")
    if "filename" in duration_frame and not np.array_equal(
        duration_frame["filename"].to_numpy(), meta["filename"].to_numpy()
    ):
        raise ValueError("duration cache filename order differs from metadata")

    y = meta["label_id"].to_numpy(dtype=np.int64)
    groups = meta["instrument"].to_numpy()
    duration = duration_frame["duration"].to_numpy(dtype=np.float64)
    feature_columns = [c for c in features if c not in ("label_id", "instrument")]
    onset_columns = [c for c in feature_columns if c.startswith("flux") or c == "log_first_onset_time"]
    X_full = features[feature_columns].to_numpy(dtype=np.float64)
    X_onset = features[onset_columns].to_numpy(dtype=np.float64)

    predictions = {}
    provenance = []
    predictions["majority"], records = fold_majority_predictions(y, groups)
    provenance.extend(records)
    conditions = [
        ("duration_only", duration.reshape(-1, 1), None),
        ("full_245", X_full, None),
        ("full_245_residualized", X_full, duration),
        ("onset_only", X_onset, None),
        ("onset_residualized", X_onset, duration),
    ]
    for name, matrix, residual_duration in conditions:
        predictions[name], records = loso_predictions(
            matrix, y, groups, duration=residual_duration, name=name
        )
        provenance.extend(records)
        print(f"{name}: {f1_score(y, predictions[name], average='macro'):.4f}", flush=True)

    metrics = {
        name: float(f1_score(y, pred, average="macro", zero_division=0))
        for name, pred in predictions.items()
    }
    row = {
        "f1_duration_only": metrics["duration_only"],
        "f1_full_245": metrics["full_245"],
        "f1_majority": metrics["majority"],
        "f1_full_residualized": metrics["full_245_residualized"],
        "f1_onset_only": metrics["onset_only"],
        "f1_onset_residualized": metrics["onset_residualized"],
    }
    csv_path = OUT_DIR / "duration_baseline_repaired.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    pred_path = OUT_DIR / "duration_baseline_repaired_predictions.npz"
    np.savez_compressed(pred_path, y_true=y, **predictions)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "outer instrument-type LOSO; every learned transform fit on train_idx only",
        "seed": SEED,
        "inputs": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (META, FEATS, DURATIONS)
        },
        "n_samples": int(len(y)),
        "groups": np.unique(groups).tolist(),
        "n_features": len(feature_columns),
        "n_onset_features": len(onset_columns),
        "metrics": row,
        "prediction_file": str(pred_path.relative_to(ROOT)),
        "prediction_sha256": file_sha256(pred_path),
        "fold_fits": provenance,
        "limitations": [
            "The four groups are instrument types, not independent players, sessions, or physical instruments.",
            "Whole-dataset duration distributions and correlations are descriptive and are not used to fit a model.",
            "This repair does not retroactively validate the historical duration_baseline.csv.",
        ],
    }
    manifest_path = OUT_DIR / "duration_baseline_repaired_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Duration baseline — train-only repaired rerun",
        "",
        "All reported scores are out-of-fold predictions from instrument-type LOSO. "
        "Residualization, scaling, SVC class weights, and the majority class were fit separately on each training fold.",
        "",
        "| condition | macro F1 |",
        "|---|---:|",
    ]
    report.extend(f"| {name} | {value:.4f} |" for name, value in row.items())
    report += [
        "",
        "These results replace neither the historical file nor any headline result automatically. "
        "Use the JSON manifest to trace every fitted artifact and the NPZ as the canonical prediction source.",
        "",
        f"Runtime: {time.time() - started:.1f} s.",
    ]
    (OUT_DIR / "duration_baseline_repaired.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"saved: {csv_path}")
    print(f"saved: {manifest_path}")


if __name__ == "__main__":
    main()

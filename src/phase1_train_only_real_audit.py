"""Build train-only preprocessing manifests for every configured Phase 1 split.

This is a no-training audit: it does not fit an AI classifier or evaluate test
labels.  It reconstructs the declared train/early-stop/selection partitions,
fits each learned preprocessing artifact from train indices only, and verifies
that mutations confined to the held-out fold cannot change those artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phase1_train_only import (  # noqa: E402
    DurationResidualizer,
    array_hash,
    balanced_class_weights,
    fit_train_scaler,
    residualizer_hash,
    scaler_hash,
)


DEFAULT_OUT = ROOT / "results" / "phase1" / "audit_repair" / "train_only_real_folds.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_val(indices, labels, n_classes, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices, dtype=np.int64)
    local_labels = labels[indices]
    train, validation = [], []
    for class_id in range(n_classes):
        positions = indices[local_labels == class_id]
        if len(positions) == 0:
            continue
        n_validation = max(1, round(len(positions) * val_ratio))
        selected = rng.choice(len(positions), n_validation, replace=False)
        validation.extend(positions[selected].tolist())
        train.extend(np.delete(positions, selected).tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def partition(split, labels, n_classes, remap=None):
    labeled = np.asarray(split["labeled_idx"], dtype=np.int64)
    test = np.asarray(split["test_idx"], dtype=np.int64)
    if remap is not None:
        keep, old_to_new = remap
        labeled = old_to_new[labeled[keep[labeled]]]
        test = old_to_new[test[keep[test]]]
    seed_core, val_select = split_val(
        labeled, labels, n_classes, seed=split["trial"] + 1000
    )
    train, early_stop = split_val(
        seed_core, labels, n_classes, seed=split["trial"]
    )
    return train, early_stop, val_select, test


def audit_partition(split, labels, features, duration, n_classes, task, remap=None):
    train, early_stop, val_select, test = partition(split, labels, n_classes, remap=remap)
    named = {
        "train": train,
        "early_stop": early_stop,
        "val_select": val_select,
        "test": test,
    }
    pairs_disjoint = {
        f"{left}_{right}": len(np.intersect1d(named[left], named[right])) == 0
        for i, left in enumerate(named)
        for right in list(named)[i + 1:]
    }

    weights = balanced_class_weights(labels[train], n_classes)
    scaler = fit_train_scaler(features[train])
    residualizer = DurationResidualizer.fit(features[train], duration[train])
    before = {
        "class_weight": array_hash(weights),
        "scaler": scaler_hash(scaler),
        "residualizer": residualizer_hash(residualizer),
    }

    # Mutations are deliberately applied only to held-out copies.  Re-fitting
    # from the declared train indices must reproduce byte-identical artifacts.
    labels_mutated = labels.copy()
    labels_mutated[test] = 0
    features_mutated = features.copy()
    features_mutated[test] = features_mutated[test] * 1000.0 + 777.0
    duration_mutated = duration.copy()
    duration_mutated[test] = duration_mutated[test] * 50.0 + 100.0
    after = {
        "class_weight": array_hash(balanced_class_weights(labels_mutated[train], n_classes)),
        "scaler": scaler_hash(fit_train_scaler(features_mutated[train])),
        "residualizer": residualizer_hash(
            DurationResidualizer.fit(features_mutated[train], duration_mutated[train])
        ),
    }

    return {
        "task": task,
        "fold": int(split["fold"]),
        "seed_ratio": float(split["seed_ratio"]),
        "trial": int(split["trial"]),
        "n_train": int(len(train)),
        "n_early_stop": int(len(early_stop)),
        "n_val_select": int(len(val_select)),
        "n_test": int(len(test)),
        "index_hashes": {name: array_hash(value) for name, value in named.items()},
        "pairwise_disjoint": pairs_disjoint,
        "train_class_counts": np.bincount(labels[train], minlength=n_classes).tolist(),
        "train_class_weights": weights.tolist(),
        "artifacts_before_test_mutation": before,
        "artifacts_after_test_mutation": after,
        "test_mutation_invariant": before == after,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    label_path = ROOT / "data" / "mert_labels.npy"
    feature_path = ROOT / "data" / "features_full.csv"
    duration_path = ROOT / "data" / "clip_durations.csv"
    split_path = ROOT / "data" / "split_configs.json"
    labels5 = np.load(label_path).astype(np.int64)
    feature_frame = pd.read_csv(feature_path)
    feature_columns = [c for c in feature_frame if c not in ("label_id", "instrument")]
    features5 = feature_frame[feature_columns].to_numpy(dtype=np.float64)
    duration5 = pd.read_csv(duration_path)["duration"].to_numpy(dtype=np.float64)
    splits = json.loads(split_path.read_text(encoding="utf-8"))["splits"]

    records = [
        audit_partition(split, labels5, features5, duration5, 5, "five_class")
        for split in splits
    ]

    keep = labels5 != 0
    old_to_new = np.full(len(labels5), -1, dtype=np.int64)
    old_to_new[keep] = np.arange(keep.sum())
    labels4 = labels5[keep] - 1
    features4 = features5[keep]
    duration4 = duration5[keep]
    remap = (keep, old_to_new)
    records.extend(
        audit_partition(
            split, labels4, features4, duration4, 4, "four_class_without_ordinario", remap=remap
        )
        for split in splits
    )

    all_disjoint = all(all(record["pairwise_disjoint"].values()) for record in records)
    all_invariant = all(record["test_mutation_invariant"] for record in records)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "train-only artifact provenance without classifier training or test evaluation",
        "inputs": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (label_path, feature_path, duration_path, split_path)
        },
        "covered_consumers": {
            "five_class": [
                "MERT tuning head class weight",
                "layer probe class weight",
                "LoRA class weight",
                "SVM scaler and balanced class weight",
                "duration residualizer",
            ],
            "four_class_without_ordinario": [
                "supplementary MERT head class weight",
                "supplementary SVM scaler and balanced class weight",
            ],
        },
        "n_configured_splits": len(splits),
        "n_audited_task_splits": len(records),
        "all_partitions_pairwise_disjoint": all_disjoint,
        "all_artifacts_invariant_to_test_mutation": all_invariant,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_configured_splits": len(splits),
        "n_audited_task_splits": len(records),
        "all_partitions_pairwise_disjoint": all_disjoint,
        "all_artifacts_invariant_to_test_mutation": all_invariant,
    }, indent=2))
    print(f"saved: {args.out}")
    if not all_disjoint or not all_invariant:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

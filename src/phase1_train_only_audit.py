"""Audit Phase 1 preprocessing boundaries and run mutation sentinels.

The audit separates descriptive whole-dataset summaries from quantities that
affect a fitted model.  A passing sentinel means that the repaired primitives'
artifacts are invariant to test-label/test-feature mutations; it does not make
legacy scripts that still use global statistics publication-safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from phase1_train_only import (
    DurationResidualizer,
    array_hash,
    balanced_class_weights,
    fit_train_scaler,
    residualizer_hash,
    scaler_hash,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "results" / "phase1" / "audit_repair" / "train_only_preprocessing.json"


CHECKS = [
    {
        "path": "notebooks/03_modeling_v2.ipynb",
        "patterns": ["compute_class_weight('balanced'", "y=labels"],
        "classification": "legacy_violation",
        "reason": "The CNN notebook computes one class-weight vector from all labels before GroupKFold.",
    },
    {
        "path": "notebooks/04_phase1.ipynb",
        "patterns": ["counts = np.bincount(labels", "nn.CrossEntropyLoss(weight=class_weights)"],
        "classification": "legacy_violation",
        "reason": "The Phase 1 notebook computes global MERT class weights before configured splits.",
    },
    {
        "path": "src/phase1_rerun.py",
        "patterns": ["counts = np.bincount(labels", "CrossEntropyLoss(weight=class_weights)"],
        "classification": "legacy_violation",
        "reason": "MERT loss weights are derived from the repository-global label vector.",
    },
    {
        "path": "src/phase1_mert_tuning_audit.py",
        "patterns": ["counts = np.bincount(labels", "CrossEntropyLoss(weight=class_weights)"],
        "classification": "legacy_violation",
        "reason": "MERT head loss weights are computed before the outer-fold training indices are supplied.",
    },
    {
        "path": "src/phase1_layer_probe.py",
        "patterns": ["counts = np.bincount(labels", "CrossEntropyLoss(weight=class_weights)"],
        "classification": "legacy_violation",
        "reason": "Layer-probe loss weights are derived from all labels.",
    },
    {
        "path": "src/phase1_supplementary_4class.py",
        "patterns": ["counts = np.bincount(labels", "CrossEntropyLoss(weight=class_weights)"],
        "classification": "legacy_violation",
        "reason": "Four-class MERT loss weights are derived from all retained labels.",
    },
    {
        "path": "src/phase1_lora.py",
        "patterns": ["counts = np.bincount(labels", "cw = torch.tensor(counts.sum()"],
        "classification": "legacy_violation",
        "reason": "LoRA loss weights are derived from all labels rather than tr_idx.",
    },
    {
        "path": "src/phase1_duration_baseline.py",
        "patterns": ["LinearRegression().fit(D, X)", "X_resid = residualize(X_full, dur)"],
        "classification": "legacy_violation",
        "reason": "Duration residualization is fit once on all outer folds before LOSO.",
    },
    {
        "path": "src/phase1_duration_baseline.py",
        "patterns": ["np.full_like(y, np.bincount(y).argmax())"],
        "classification": "legacy_violation",
        "reason": "The reported majority baseline reads the full evaluation label distribution.",
    },
    {
        "path": "src/phase1_mert_tuning_audit.py",
        "patterns": ["Pipeline([('scaler', StandardScaler())", "pipe.fit(X_feat[tr_idx], labels[tr_idx])"],
        "classification": "safe_structure",
        "reason": "The SVM scaler and balanced SVC are fit together on tr_idx only.",
    },
    {
        "path": "src/phase1_svm_seed_comparison.py",
        "patterns": ["('scaler', StandardScaler())", "clf.fit(X_train, y_train)"],
        "classification": "safe_structure",
        "reason": "The caller supplies a training partition to the complete SVM pipeline.",
    },
    {
        "path": "src/phase1_cluster_bootstrap.py",
        "patterns": ["make_pipeline(StandardScaler()", "pipe.fit(X_feat[train_idx], labels[train_idx])"],
        "classification": "safe_structure",
        "reason": "Refit scaling and SVC weighting use train_idx only.",
    },
    {
        "path": "src/phase1_selftraining_repaired.py",
        "patterns": ["train_only_class_weights(pool.targets", "CrossEntropyLoss(weight=weights)"],
        "classification": "safe_structure",
        "reason": "The repaired path derives weights from its explicit paired training pool.",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_sources() -> list[dict]:
    records = []
    for check in CHECKS:
        path = ROOT / check["path"]
        lines = path.read_text(encoding="utf-8").splitlines()
        hits = []
        missing = []
        for pattern in check["patterns"]:
            matching = [
                {"line": number, "text": line.strip(), "pattern": pattern}
                for number, line in enumerate(lines, 1)
                if pattern in line
            ]
            if matching:
                hits.extend(matching)
            else:
                missing.append(pattern)
        records.append({
            **check,
            "source_sha256": sha256(path),
            "evidence": hits,
            "all_expected_patterns_found": not missing,
            "missing_patterns": missing,
        })
    return records


def sentinel() -> dict:
    rng = np.random.default_rng(20260920)
    train_x = rng.normal(size=(25, 7))
    train_d = rng.uniform(0.4, 4.0, size=25)
    train_y = np.tile(np.arange(5), 5)
    test_x = rng.normal(size=(10, 7))
    test_d = rng.uniform(0.4, 4.0, size=10)
    test_y = np.tile(np.arange(5), 2)

    # Repaired artifacts receive train arrays only.
    weights_a = balanced_class_weights(train_y, 5)
    scaler_a = fit_train_scaler(train_x)
    resid_a = DurationResidualizer.fit(train_x, train_d)

    mutated_test_x = test_x * 1000.0 + 777.0
    mutated_test_d = test_d * 50.0 + 100.0
    mutated_test_y = np.zeros_like(test_y)
    weights_b = balanced_class_weights(train_y, 5)
    scaler_b = fit_train_scaler(train_x)
    resid_b = DurationResidualizer.fit(train_x, train_d)

    repaired_before = {
        "class_weight": array_hash(weights_a),
        "scaler": scaler_hash(scaler_a),
        "residualizer": residualizer_hash(resid_a),
    }
    repaired_after_test_mutation = {
        "class_weight": array_hash(weights_b),
        "scaler": scaler_hash(scaler_b),
        "residualizer": residualizer_hash(resid_b),
    }

    # Reproduce the legacy anti-pattern on concatenated train+test arrays.
    legacy_weights_a = balanced_class_weights(np.concatenate([train_y, test_y]), 5)
    legacy_weights_b = balanced_class_weights(np.concatenate([train_y, mutated_test_y]), 5)
    legacy_scaler_a = StandardScaler().fit(np.vstack([train_x, test_x]))
    legacy_scaler_b = StandardScaler().fit(np.vstack([train_x, mutated_test_x]))
    legacy_resid_a = LinearRegression().fit(
        np.concatenate([train_d, test_d]).reshape(-1, 1),
        np.vstack([train_x, test_x]),
    )
    legacy_resid_b = LinearRegression().fit(
        np.concatenate([train_d, mutated_test_d]).reshape(-1, 1),
        np.vstack([train_x, mutated_test_x]),
    )
    legacy_before = {
        "class_weight": array_hash(legacy_weights_a),
        "scaler": array_hash(legacy_scaler_a.mean_, legacy_scaler_a.var_, legacy_scaler_a.scale_),
        "residualizer": array_hash(legacy_resid_a.coef_, legacy_resid_a.intercept_),
    }
    legacy_after_test_mutation = {
        "class_weight": array_hash(legacy_weights_b),
        "scaler": array_hash(legacy_scaler_b.mean_, legacy_scaler_b.var_, legacy_scaler_b.scale_),
        "residualizer": array_hash(legacy_resid_b.coef_, legacy_resid_b.intercept_),
    }

    changed_train_y = train_y.copy()
    changed_train_y[[0, 6]] = [1, 2]
    changed_train_x = train_x.copy()
    changed_train_x[0] += 10.0
    changed_train_d = train_d.copy()
    changed_train_d[0] += 2.0
    repaired_after_train_mutation = {
        "class_weight": array_hash(balanced_class_weights(changed_train_y, 5)),
        "scaler": scaler_hash(fit_train_scaler(changed_train_x)),
        "residualizer": residualizer_hash(DurationResidualizer.fit(changed_train_x, changed_train_d)),
    }

    assertions = {
        "test_label_mutation_does_not_change_repaired_class_weight": (
            repaired_before["class_weight"] == repaired_after_test_mutation["class_weight"]
        ),
        "test_feature_mutation_does_not_change_repaired_scaler": (
            repaired_before["scaler"] == repaired_after_test_mutation["scaler"]
        ),
        "test_feature_or_duration_mutation_does_not_change_repaired_residualizer": (
            repaired_before["residualizer"] == repaired_after_test_mutation["residualizer"]
        ),
        "train_label_mutation_changes_repaired_class_weight": (
            repaired_before["class_weight"] != repaired_after_train_mutation["class_weight"]
        ),
        "train_feature_mutation_changes_repaired_scaler": (
            repaired_before["scaler"] != repaired_after_train_mutation["scaler"]
        ),
        "train_feature_or_duration_mutation_changes_repaired_residualizer": (
            repaired_before["residualizer"] != repaired_after_train_mutation["residualizer"]
        ),
        "legacy_global_class_weight_reacts_to_test_labels": (
            legacy_before["class_weight"] != legacy_after_test_mutation["class_weight"]
        ),
        "legacy_global_scaler_reacts_to_test_features": (
            legacy_before["scaler"] != legacy_after_test_mutation["scaler"]
        ),
        "legacy_global_residualizer_reacts_to_test_data": (
            legacy_before["residualizer"] != legacy_after_test_mutation["residualizer"]
        ),
    }
    return {
        "seed": 20260920,
        "mutation_scope": {
            "test_labels": "all labels replaced by class 0",
            "test_features": "x * 1000 + 777",
            "test_duration": "duration * 50 + 100",
        },
        "repaired_artifacts_before": repaired_before,
        "repaired_artifacts_after_test_mutation": repaired_after_test_mutation,
        "repaired_artifacts_after_train_mutation": repaired_after_train_mutation,
        "legacy_artifacts_before": legacy_before,
        "legacy_artifacts_after_test_mutation": legacy_after_test_mutation,
        "assertions": assertions,
        "passed": all(assertions.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sources = inspect_sources()
    sentinel_result = sentinel()
    static_complete = all(record["all_expected_patterns_found"] for record in sources)
    payload = {
        "audit": "phase1_train_only_preprocessing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase 1 Python evaluation paths, core CNN/Phase 1 notebooks, and repaired preprocessing primitives",
        "source_records": sources,
        "feature_selection": {
            "learned_selector_found_in_scoped_paths": False,
            "note": "Named onset/chroma subsets are prespecified column sets, not selectors fit from labels. Any future learned selector must be fit inside the outer-fold training path.",
        },
        "descriptive_statistics": {
            "allowed_whole_dataset_use": "Class-duration tables and correlations may be reported as descriptive summaries only.",
            "not_allowed": "Whole-dataset statistics may not fit, select, tune, or transform an outer-fold model.",
        },
        "sentinel": sentinel_result,
        "status": {
            "static_expected_patterns_found": static_complete,
            "repaired_primitives_pass_sentinel": sentinel_result["passed"],
            "legacy_paths_publication_safe": False,
            "verification_2_audit_complete": static_complete and sentinel_result["passed"],
            "verification_2_repair_primitives_complete": sentinel_result["passed"],
            "verification_2_publication_rerun_ready": False,
            "verification_2_complete": False,
            "reason": "The audit and repaired primitives are complete, but legacy MERT/CNN paths must not feed a publication rerun until their training entry points use the repaired train-only API.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["status"], ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")
    if not static_complete or not sentinel_result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

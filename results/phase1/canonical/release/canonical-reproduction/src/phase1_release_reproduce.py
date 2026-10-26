"""Recompute canonical statistics using released predictions, without models.

This audit verifies saved predictions and explicit partitions, not checkpoint
inference or model training. NumPy and pandas are the only required packages.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["ordinario", "pizzicato_secco", "tremolo", "sul_ponticello", "col_legno_battuto"]
CONDITIONS = ("supervised", "pseudo", "oracle")
RATIOS = (0.05, 0.10, 0.20, 0.50)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def metrics(y, prediction):
    require(y.shape == prediction.shape and y.ndim == 1, "Invalid prediction shape")
    require(np.isin(y, range(5)).all() and np.isin(prediction, range(5)).all(), "Invalid class ID")
    counts = np.bincount(y * 5 + prediction, minlength=25).reshape(5, 5)
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    per_class = np.divide(2.0 * np.diag(counts), denominator, out=np.zeros(5), where=denominator != 0)
    return {"macro_f1": float(per_class.mean()), "per_class_f1": per_class.tolist(),
            "confusion_matrix": counts.tolist()}


def check_metrics(actual, expected):
    require(np.isclose(actual["macro_f1"], expected["macro_f1"], rtol=0, atol=1e-12), "Macro F1 mismatch")
    require(np.allclose(actual["per_class_f1"], expected["per_class_f1"], rtol=0, atol=1e-12), "Class F1 mismatch")
    require(actual["confusion_matrix"] == expected["confusion_matrix"], "Confusion mismatch")


def integers(values, size):
    array = np.asarray(values)
    require(array.ndim == 1 and array.dtype.kind in "iu", "Indices must be integers")
    array = array.astype(np.int64)
    require(np.isin(array, np.arange(size)).all(), "Index out of bounds")
    require(len(array) == len(np.unique(array)), "Duplicate indices")
    return array


def check_partitions(parts, labeled, unlabeled, test, meta, instrument):
    arrays = {name: integers(parts[name], len(meta)) for name in ("train", "early_stop", "val_select", "test")}
    for left, right in itertools.combinations(arrays.values(), 2):
        require(not np.intersect1d(left, right).size, "Partition overlap")
    require(np.array_equal(arrays["test"], test), "Test order mismatch")
    observed = np.concatenate([arrays[name] for name in ("train", "early_stop", "val_select")])
    require(np.array_equal(np.sort(observed), np.sort(labeled)), "Observed seed not partitioned completely")
    require(not np.intersect1d(observed, unlabeled).size, "Unlabeled/observed overlap")
    expected_test = np.flatnonzero(meta.instrument.to_numpy() == instrument)
    require(np.array_equal(np.sort(test), expected_test), "Outer holdout does not cover instrument type")
    require(not (meta.instrument.to_numpy()[observed] == instrument).any(), "Held-out type enters training")
    return arrays


def load_records(root):
    base = root / "results/phase1/canonical"
    meta = pd.read_csv(root / "data/release_meta.csv")
    require(len(meta) == 3402 and np.array_equal(meta.sample_index, np.arange(3402)), "Metadata order changed")
    require([CLASSES[i] for i in meta.label_id] == meta.label.tolist(), "Label mapping changed")
    labels = meta.label_id.to_numpy(dtype=np.int64)
    splits = json.loads((root / "data/split_configs.json").read_text())["splits"]
    splits = {f"r{round(s['seed_ratio'] * 100):02d}_f{s['fold']}_t{s['trial']}": s for s in splits}
    parts = json.loads((root / "data/release_partitions.json").read_text())
    records = {name: [] for name in ("mert_svm", "cnn", "self_training")}
    for family, expected_count in (("mert_svm", 48), ("cnn", 4), ("self_training", 48)):
        paths = sorted((base / family / "manifests").glob("*.json"))
        require(len(paths) == expected_count, f"Wrong manifest count: {family}")
        for path in paths:
            manifest = json.loads(path.read_text())
            key = manifest.get("split_key", manifest.get("fold_key", manifest.get("key")))
            artifact = manifest if family == "self_training" else manifest["artifacts"]
            prediction_path = root / artifact["predictions"]
            require(prediction_path.resolve().is_relative_to(root.resolve()), "Unsafe artifact path")
            require(file_hash(prediction_path) == artifact["predictions_sha256"], f"Prediction hash mismatch: {key}")
            with np.load(prediction_path, allow_pickle=False) as stored:
                index = integers(stored["sample_index"], len(meta))
                y = stored["y_true"].astype(np.int64)
                fields = {"mert": "mert_prediction", "svm": "svm_prediction"} if family == "mert_svm" else (
                    {"cnn": "prediction"} if family == "cnn" else {name: name for name in CONDITIONS})
                predictions = {name: stored[field].astype(np.int64) for name, field in fields.items()}
            require(np.array_equal(y, labels[index]), f"Saved test labels mismatch: {key}")
            instrument = manifest["held_out_instrument"]
            require(np.array_equal(np.sort(index), np.flatnonzero(meta.instrument.to_numpy() == instrument)), "Invalid type holdout")
            row = {"key": key, "fold": int(manifest["fold"]), "instrument": instrument, "index": index,
                   "y": y, "manifest": manifest, **predictions}
            if family != "cnn":
                row.update(ratio=float(manifest["seed_ratio"]), trial=int(manifest["trial"]))
                split = splits[key]
                labeled, unlabeled, test = [integers(split[name], len(meta)) for name in ("labeled_idx", "unlabeled_idx", "test_idx")]
                for left, right in itertools.combinations((labeled, unlabeled, test), 2):
                    require(not np.intersect1d(left, right).size, "Outer split overlap")
                require(np.array_equal(np.sort(np.concatenate([labeled, unlabeled, test])), np.arange(len(meta))), "Incomplete split")
                arrays = check_partitions(parts[family][key], labeled, unlabeled, test, meta, instrument)
                require(np.array_equal(index, test), "Saved test positions differ from configured split")
                if family == "mert_svm":
                    require({name: array_hash(value) for name, value in arrays.items()} == manifest["partition_hashes"], "Partition hashes mismatch")
                    counts = np.bincount(labels[arrays["train"]], minlength=5).astype(float)
                    require((counts > 0).all(), "Missing training class")
                    weights = counts.sum() / (5 * counts)
                    require(array_hash(weights) == manifest["mert_class_weight_hash"], "Training weight hash mismatch")
                    require(array_hash(index, y, row["mert"], row["svm"]) == manifest["prediction_content_hash"], "Prediction content mismatch")
                    require(manifest["test_prediction_calls_per_model"] == 1, "Unexpected test access declaration")
                else:
                    for name, values in (("observed", labeled), ("unlabeled", unlabeled), ("test", test)):
                        require(array_hash(values) == manifest[f"{name}_index_sha256"], "Seed boundary hash mismatch")
                    check_self_training(manifest, arrays, labels, unlabeled, root)
            else:
                train = integers(parts[family][key]["train"], len(meta))
                require(np.array_equal(np.sort(train), np.setdiff1d(np.arange(len(meta)), index)), "CNN training complement mismatch")
                require(array_hash(train) == manifest["train_index_hash"] and array_hash(index) == manifest["test_index_hash"], "CNN partition hash mismatch")
                require(array_hash(index, y, row["cnn"]) == manifest["prediction_content_hash"], "CNN prediction content mismatch")
                require(manifest["test_prediction_calls"] == 1, "Unexpected CNN test access declaration")
            for model, prediction in predictions.items():
                expected = manifest["conditions"][model]["test_metrics"] if family == "self_training" else manifest[
                    f"{model}_metrics" if family == "mert_svm" else "metrics"]
                check_metrics(metrics(y, prediction), expected)
                if family == "self_training":
                    require(array_hash(prediction) == manifest["conditions"][model]["prediction_sha256"], "Condition prediction hash mismatch")
            records[family].append(row)
        require(len({r["key"] for r in records[family]}) == expected_count, "Duplicate manifest key")
        if family != "cnn":
            expected_keys = {(ratio, fold, trial) for ratio in RATIOS for fold in range(4) for trial in range(3)}
            require({(r["ratio"], r["fold"], r["trial"]) for r in records[family]} == expected_keys, "Incomplete trial grid")
    return records, meta


def check_self_training(manifest, parts, labels, unlabeled, root):
    require(manifest["training_code_sha256"] == file_hash(root / "src/phase1_canonical_selftraining.py"), "Training source hash mismatch")
    require(manifest["repaired_core_sha256"] == file_hash(root / "src/phase1_selftraining_repaired.py"), "Core source hash mismatch")
    require((manifest["epochs"], manifest["patience"], manifest["max_iter"], manifest["tau"]) == (150, 15, 5, 0.8), "Training spec mismatch")
    for condition in CONDITIONS:
        item = manifest["conditions"][condition]
        report = item["training"]
        require(item["test_evaluation_count"] == 1, "Unexpected self-training test declaration")
        require(report["selected_iteration"] == int(np.argmax(report["selector_macro_f1"])), "Validation-selected iteration mismatch")
        if condition != "oracle":
            for name, field in (("train", "train_observed_indices_sha256"), ("early_stop", "early_stop_indices_sha256"), ("val_select", "selector_indices_sha256")):
                require(array_hash(parts[name]) == report["fixed_partitions"][field], "Observed partition provenance mismatch")
    report = manifest["conditions"]["pseudo"]["training"]
    index = parts["train"].copy()
    targets = labels[index].copy()
    observed_n = len(index)
    require(len(report["acceptance"]) == len(report["training_pool_by_iteration"]), "Pseudo trace length mismatch")
    for step, recorded_pool in zip(report["acceptance"], report["training_pool_by_iteration"]):
        source_counts = {"observed": observed_n}
        if len(index) > observed_n:
            source_counts["pseudo"] = len(index) - observed_n
        actual = {"n_examples": len(index), "source_counts": source_counts,
                  "indices_sha256": array_hash(index), "targets_sha256": array_hash(targets)}
        require(actual == recorded_pool, "Explicit pseudo-target pool mismatch")
        if step["status"] == "pseudo_targets_added_for_next_iteration":
            accepted = integers(step["accepted_indices"], len(labels)) if step["accepted_indices"] else np.empty(0, dtype=np.int64)
            predicted = np.asarray(step["accepted_targets"], dtype=np.int64)
            require(len(accepted) == len(predicted) == step["accepted"], "Accepted pair count mismatch")
            require(np.isin(accepted, unlabeled).all() and not np.intersect1d(index, accepted).size, "Invalid accepted indices")
            require(np.isin(predicted, range(5)).all(), "Invalid pseudo-target IDs")
            require(array_hash(accepted) == step["accepted_indices_sha256"] and array_hash(predicted) == step["accepted_targets_sha256"], "Accepted pair hash mismatch")
            index = np.concatenate([index, accepted])
            targets = np.concatenate([targets, predicted])
    require(report["selected_training_pool"] == report["training_pool_by_iteration"][report["selected_iteration"]], "Selected pool mismatch")
    oracle = manifest["conditions"]["oracle"]["training"]
    require(oracle["privileged_oracle_access"] is True, "Oracle diagnostic declaration missing")
    for step in oracle["accepted_counts"]:
        indices = integers(step["indices"], len(labels)) if step["indices"] else np.empty(0, dtype=np.int64)
        require(np.isin(indices, unlabeled).all() and np.array_equal(labels[indices], step["oracle_targets"]), "Oracle diagnostic trace mismatch")


def fold_values(records, model, positions=None, per_class=False):
    result = []
    for fold in sorted({r["fold"] for r in records}):
        values = []
        for record in records:
            if record["fold"] == fold:
                selected = positions[fold] if positions is not None else np.arange(len(record["y"]))
                score = metrics(record["y"][selected], record[model][selected])
                values.append(score["per_class_f1" if per_class else "macro_f1"])
        result.append(np.mean(values, axis=0))
    return np.asarray(result)


def bootstrap(records, meta, unit, draws, seed):
    # Same positions for both models and all trials in a type. Training is fixed.
    rng = np.random.default_rng(seed)
    positions = {r["fold"]: np.arange(len(r["index"])) for r in records}
    for fold in positions:
        indices = [r["index"] for r in records if r["fold"] == fold]
        require(all(np.array_equal(indices[0], index) for index in indices), "Trials have different test order")
    mert, svm = [fold_values(records, model, positions) for model in ("mert", "svm")]
    base_mert, base_svm = mert.copy(), svm.copy()
    point = np.asarray([mert.mean(), svm.mean(), (mert - svm).mean()])
    samples = []
    for _ in range(draws):
        if unit != "instrument_type":
            for fold in sorted(positions):
                representative = next(r for r in records if r["fold"] == fold)
                n = len(representative["index"])
                if unit == "clip":
                    positions[fold] = rng.integers(n, size=n)
                else:
                    proxy = meta.recording_id.to_numpy()[representative["index"]]
                    unique, inverse = np.unique(proxy, return_inverse=True)
                    groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
                    positions[fold] = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            mert, svm = [fold_values(records, model, positions) for model in ("mert", "svm")]
        else:
            chosen = rng.integers(len(base_mert), size=len(base_mert))
            mert, svm = base_mert[chosen], base_svm[chosen]
        samples.append([mert.mean(), svm.mean(), (mert - svm).mean()])
    return point, np.quantile(samples, [0.025, 0.975], axis=0)


def make_tables(records, meta, draws, seed):
    model_rows, model_summary, self_rows, self_summary = [], [], [], []
    model_class, model_confusion, self_class, self_confusion, intervals = [], [], [], [], []
    for record in records["mert_svm"]:
        m = record["manifest"]
        mert, svm = [metrics(record["y"], record[model])["macro_f1"] for model in ("mert", "svm")]
        model_rows.append(dict(split_key=record["key"], fold=record["fold"], held_out_instrument=record["instrument"],
                               seed_ratio=record["ratio"], trial=record["trial"], mert_macro_f1=mert,
                               svm_macro_f1=svm, mert_minus_svm=mert-svm,
                               **{k: m["artifacts"][k] for k in ("mert_checkpoint_sha256", "svm_checkpoint_sha256", "predictions_sha256")}))
    for ratio in RATIOS:
        subset = [r for r in records["mert_svm"] if r["ratio"] == ratio]
        mert, svm = [fold_values(subset, model) for model in ("mert", "svm")]
        model_summary.append(dict(seed_ratio=ratio, n_splits=12, n_outer_folds=4,
                                  mert_macro_f1_fold_mean=mert.mean(), mert_macro_f1_fold_sd=mert.std(ddof=1),
                                  svm_macro_f1_fold_mean=svm.mean(), svm_macro_f1_fold_sd=svm.std(ddof=1),
                                  mert_minus_svm=(mert-svm).mean()))
        for unit in ("clip", "recording_proxy", "instrument_type"):
            point, ci = bootstrap(subset, meta, unit, draws, seed)
            for i, measure in enumerate(("mert", "svm", "mert_minus_svm")):
                intervals.append(dict(seed_ratio=ratio, unit=unit, measure=measure,
                                      point=point[i], ci_low=ci[0, i], ci_high=ci[1, i], draws=draws))
        for condition in CONDITIONS:
            group = [r for r in records["self_training"] if r["ratio"] == ratio]
            values = fold_values(group, condition)
            self_summary.append(dict(ratio=ratio, condition=condition, macro_f1_fold_mean=values.mean(), macro_f1_fold_sd=values.std(ddof=1)))
    for record in records["self_training"]:
        for condition in CONDITIONS:
            self_rows.append(dict(key=record["key"], fold=record["fold"], ratio=record["ratio"], trial=record["trial"],
                                  condition=condition, macro_f1=metrics(record["y"], record[condition])["macro_f1"],
                                  selected_iteration=record["manifest"]["conditions"][condition]["training"]["selected_iteration"]))
    cnn_rows = [dict(fold_key=r["key"], fold=r["fold"], held_out_instrument=r["instrument"],
                     macro_f1=metrics(r["y"], r["cnn"])["macro_f1"],
                     checkpoint_sha256=r["manifest"]["artifacts"]["checkpoint_sha256"],
                     predictions_sha256=r["manifest"]["artifacts"]["predictions_sha256"]) for r in records["cnn"]]
    groups = [(f"ratio_{ratio:g}", model, [r for r in records["mert_svm"] if r["ratio"] == ratio]) for ratio in RATIOS for model in ("mert", "svm")]
    groups += [("full_labels", "cnn", records["cnn"])]
    for condition, model, group in groups:
        per_class = fold_values(group, model, per_class=True).mean(axis=0)
        counts = sum(np.asarray(metrics(r["y"], r[model])["confusion_matrix"]) for r in group)
        for i, name in enumerate(CLASSES):
            model_class.append(dict(condition=condition, model=model, **{"class": name}, fold_mean_f1=per_class[i]))
            for j, prediction in enumerate(CLASSES):
                model_confusion.append(dict(condition=condition, model=model, true_class=name, predicted_class=prediction, count=int(counts[i,j])))
    for ratio in RATIOS:
        group = [r for r in records["self_training"] if r["ratio"] == ratio]
        for condition in CONDITIONS:
            per_class = fold_values(group, condition, per_class=True).mean(axis=0)
            counts = sum(np.asarray(metrics(r["y"], r[condition])["confusion_matrix"]) for r in group)
            for i, name in enumerate(CLASSES):
                self_class.append(dict(ratio=ratio, condition=condition, **{"class": name}, f1=per_class[i]))
                for j, prediction in enumerate(CLASSES):
                    self_confusion.append(dict(ratio=ratio, condition=condition, true_class=name, predicted_class=prediction, count=int(counts[i,j])))
    return {"mert_svm/canonical_results.csv": pd.DataFrame(model_rows),
            "mert_svm/canonical_summary.csv": pd.DataFrame(model_summary),
            "cnn/canonical_results.csv": pd.DataFrame(cnn_rows),
            "self_training/canonical_results.csv": pd.DataFrame(self_rows),
            "self_training/canonical_summary.csv": pd.DataFrame(self_summary),
            "self_training/per_class_f1.csv": pd.DataFrame(self_class),
            "self_training/confusion_counts.csv": pd.DataFrame(self_confusion),
            "verification/per_class_f1.csv": pd.DataFrame(model_class),
            "verification/confusion_counts.csv": pd.DataFrame(model_confusion),
            "verification/bootstrap_sensitivity.csv": pd.DataFrame(intervals)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    out = (args.output or root / "reproduced").resolve()
    require(not out.is_relative_to(root / "results") and out != root, "Output must not replace reference results")
    require(args.draws == 1000 and args.seed == 20260928, "Reference comparison requires canonical draws/seed")
    release = json.loads((root / "RELEASE_MANIFEST.json").read_text())
    for relative, digest in release["files"].items():
        path = root / relative
        require(path.resolve().is_relative_to(root), "Unsafe release manifest path")
        require(file_hash(path) == digest, f"Release file changed: {relative}")
    records, meta = load_records(root)
    print("Validated 100 prediction files, instrument holdouts, observed partitions and explicit pseudo-target traces.", flush=True)
    tables = make_tables(records, meta, args.draws, args.seed)
    for relative, actual in tables.items():
        expected = pd.read_csv(root / "results/phase1/canonical" / relative)
        require(set(expected.columns) == set(actual.columns), f"Columns changed: {relative}")
        columns = expected.columns.tolist()
        keys = [c for c in columns if c in ("key", "split_key", "fold_key", "seed_ratio", "ratio", "condition", "model", "class", "true_class", "predicted_class", "unit", "measure")]
        actual, expected = [frame[columns].sort_values(keys).reset_index(drop=True) for frame in (actual, expected)]
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        actual.to_csv(target, index=False)
        print(f"PASS: {relative} ({len(actual)} rows)", flush=True)
    report = {"schema_version": 1, "mode": "saved_predictions_only", "passed": True,
              "checkpoint_inference_performed": False, "training_performed": False,
              "prediction_files_checked": 100, "model_prediction_vectors_checked": 244,
              "tables_checked": len(tables), "absolute_metric_tolerance": 1e-12,
              "draws": args.draws, "seed": args.seed,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
              "release_manifest_sha256": file_hash(root / "RELEASE_MANIFEST.json"),
              "output_sha256": {relative: file_hash(out / relative) for relative in tables}}
    (out / "REPRODUCTION.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: all canonical statistics reproduced without audio, embeddings, checkpoints or torch.")


if __name__ == "__main__":
    main()

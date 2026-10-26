"""Replay canonical checkpoints and report uncertainty from their saved predictions.

The test labels are read only during this post-training audit.  All estimates
refer to held-out instrument *types*, not independent players or sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import SmallCNN  # noqa: E402
from phase1_canonical_mert_svm import Head, partition, split_key  # noqa: E402
from phase1_train_only import array_hash, balanced_class_weights  # noqa: E402

BASE = ROOT / "results" / "phase1" / "canonical"
OUT = BASE / "verification"
CLASSES = ["ordinario", "pizzicato_secco", "tremolo", "sul_ponticello", "col_legno_battuto"]
N_CLASSES = len(CLASSES)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_artifact(manifest, name):
    path = ROOT / manifest["artifacts"][name]
    assert path.is_file(), path
    assert sha256(path) == manifest["artifacts"][name + "_sha256"], path
    return path


def scores(y, prediction):
    labels = np.arange(N_CLASSES)
    return {
        "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y, prediction, labels=labels, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist(),
    }


def fast_macro_f1(y, prediction):
    counts = np.bincount(y * N_CLASSES + prediction, minlength=N_CLASSES ** 2)
    matrix = counts.reshape(N_CLASSES, N_CLASSES)
    true_positive = np.diag(matrix).astype(float)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.divide(2 * true_positive, denominator,
                           out=np.zeros(N_CLASSES), where=denominator != 0).mean())


def assert_scores(actual, recorded):
    assert np.isclose(actual["macro_f1"], recorded["macro_f1"], atol=1e-12)
    assert np.allclose(actual["per_class_f1"], recorded["per_class_f1"], atol=1e-12)
    assert actual["confusion_matrix"] == recorded["confusion_matrix"]


def read_mert_svm(meta, replay):
    embeddings = np.load(ROOT / "data" / "mert_embeddings.npy", mmap_mode="r") if replay else None
    frame = pd.read_csv(ROOT / "data" / "features_full.csv") if replay else None
    features = frame[[c for c in frame if c not in ("label_id", "instrument")]].to_numpy(np.float32) if replay else None
    manifests = sorted((BASE / "mert_svm" / "manifests").glob("*.json"))
    assert len(manifests) == 48, f"expected 48 MERT/SVM manifests, got {len(manifests)}"
    labels = meta["label_id"].to_numpy(dtype=np.int64)
    split_config = json.loads((ROOT / "data" / "split_configs.json").read_text())
    split_map = {split_key(split): split for split in split_config["splits"]}
    records = []
    for path in manifests:
        manifest = json.loads(path.read_text())
        split = split_map[manifest["split_key"]]
        parts = partition(split, labels)
        assert {name: array_hash(indices) for name, indices in parts.items()} == manifest["partition_hashes"]
        weights = balanced_class_weights(labels[parts["train"]], N_CLASSES)
        assert array_hash(weights) == manifest["mert_class_weight_hash"]
        assert np.allclose(weights, manifest["mert_class_weights"])
        checkpoint = checked_artifact(manifest, "mert_checkpoint")
        svm_checkpoint = checked_artifact(manifest, "svm_checkpoint")
        prediction_path = checked_artifact(manifest, "predictions")
        with np.load(prediction_path) as saved:
            index = saved["sample_index"].astype(np.int64)
            y = saved["y_true"].astype(np.int64)
            mert = saved["mert_prediction"].astype(np.int64)
            svm = saved["svm_prediction"].astype(np.int64)
        assert len(index) == len(np.unique(index))
        assert np.array_equal(y, meta["label_id"].to_numpy()[index])
        assert (meta["instrument"].to_numpy()[index] == manifest["held_out_instrument"]).all()
        assert array_hash(index) == manifest["partition_hashes"]["test"]
        assert array_hash(index, y, mert, svm) == manifest["prediction_content_hash"]
        assert manifest["partitions_pairwise_disjoint"] and manifest["test_prediction_calls_per_model"] == 1
        assert_scores(scores(y, mert), manifest["mert_metrics"])
        assert_scores(scores(y, svm), manifest["svm_metrics"])
        if replay:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = Head(state["config"]["arch"], state["config"]["dropout"], state["input_dim"])
            model.load_state_dict(state["state_dict"])
            model.eval()
            with torch.no_grad():
                replay_mert = model(torch.from_numpy(np.asarray(embeddings[index], dtype=np.float32))).argmax(1).numpy()
            replay_svm = joblib.load(svm_checkpoint).predict(features[index])
            assert np.array_equal(replay_mert, mert), manifest["split_key"]
            assert np.array_equal(replay_svm, svm), manifest["split_key"]
        records.append({
            "key": manifest["split_key"], "fold": int(manifest["fold"]),
            "ratio": float(manifest["seed_ratio"]), "trial": int(manifest["trial"]),
            "instrument": manifest["held_out_instrument"], "index": index,
            "y": y, "mert": mert, "svm": svm,
            "prediction_sha256": manifest["artifacts"]["predictions_sha256"],
        })
    original = pd.read_csv(BASE / "mert_svm" / "canonical_results.csv")
    assert set(original["split_key"]) == {r["key"] for r in records}
    for record in records:
        row = original.set_index("split_key").loc[record["key"]]
        assert row["predictions_sha256"] == record["prediction_sha256"]
        assert np.isclose(row["mert_macro_f1"], scores(record["y"], record["mert"])["macro_f1"])
        assert np.isclose(row["svm_macro_f1"], scores(record["y"], record["svm"])["macro_f1"])
    return records


def read_cnn(meta, replay):
    mels = np.load(BASE / "cnn" / "cache" / "mels_128.npy", mmap_mode="r") if replay else None
    manifests = sorted((BASE / "cnn" / "manifests").glob("*.json"))
    assert len(manifests) == 4, f"expected 4 CNN manifests, got {len(manifests)}"
    labels = meta["label_id"].to_numpy(dtype=np.int64)
    groups = meta["instrument"].to_numpy()
    folds = list(GroupKFold(n_splits=len(np.unique(groups))).split(np.zeros((len(labels), 1)), labels, groups))
    records = []
    for path in manifests:
        manifest = json.loads(path.read_text())
        train_index, test_index = folds[manifest["fold"]]
        assert array_hash(train_index) == manifest["train_index_hash"]
        assert array_hash(test_index) == manifest["test_index_hash"]
        weights = balanced_class_weights(labels[train_index], N_CLASSES)
        assert array_hash(weights) == manifest["class_weight_hash"]
        assert np.allclose(weights, manifest["class_weights"])
        checkpoint = checked_artifact(manifest, "checkpoint")
        prediction_path = checked_artifact(manifest, "predictions")
        with np.load(prediction_path) as saved:
            index = saved["sample_index"].astype(np.int64)
            y = saved["y_true"].astype(np.int64)
            prediction = saved["prediction"].astype(np.int64)
        assert np.array_equal(y, meta["label_id"].to_numpy()[index])
        assert (meta["instrument"].to_numpy()[index] == manifest["held_out_instrument"]).all()
        assert array_hash(index) == manifest["test_index_hash"]
        assert array_hash(index, y, prediction) == manifest["prediction_content_hash"]
        assert manifest["partitions_disjoint"] and manifest["test_prediction_calls"] == 1
        assert_scores(scores(y, prediction), manifest["metrics"])
        if replay:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = SmallCNN(n_classes=N_CLASSES)
            model.load_state_dict(state["state_dict"])
            model.eval()
            pieces = []
            with torch.no_grad():
                for batch in np.array_split(index, max(1, (len(index) + 31) // 32)):
                    tensor = torch.from_numpy(np.asarray(mels[batch], dtype=np.float32))[:, None]
                    pieces.append(model(tensor).argmax(1).numpy())
            assert np.array_equal(np.concatenate(pieces), prediction), manifest["fold_key"]
        records.append({"key": manifest["fold_key"], "fold": int(manifest["fold"]),
                        "instrument": manifest["held_out_instrument"], "index": index,
                        "y": y, "cnn": prediction,
                        "prediction_sha256": manifest["artifacts"]["predictions_sha256"]})
    original = pd.read_csv(BASE / "cnn" / "canonical_results.csv")
    assert set(original["fold_key"]) == {r["key"] for r in records}
    for record in records:
        row = original.set_index("fold_key").loc[record["key"]]
        assert row["predictions_sha256"] == record["prediction_sha256"]
        assert np.isclose(row["macro_f1"], scores(record["y"], record["cnn"])["macro_f1"])
    return records


def statistic(records, model, positions_by_fold):
    # Identical sampled test positions are reused across trials of one fold.
    fold_values = []
    for fold in sorted(positions_by_fold):
        trial_values = []
        for record in records:
            if record["fold"] == fold:
                positions = positions_by_fold[fold]
                trial_values.append(fast_macro_f1(record["y"][positions], record[model][positions]))
        fold_values.append(np.mean(trial_values))
    return np.asarray(fold_values)


def sample_positions(records, meta, unit, rng):
    positions = {}
    for fold in sorted({r["fold"] for r in records}):
        representative = next(r for r in records if r["fold"] == fold)
        n = len(representative["index"])
        if unit == "clip":
            positions[fold] = rng.integers(n, size=n)
        else:
            proxy = meta["recording_id"].to_numpy()[representative["index"]]
            unique, inverse = np.unique(proxy, return_inverse=True)
            groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
            positions[fold] = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
    return positions


def bootstrap(records, meta, unit, draws, seed):
    rng = np.random.default_rng(seed)
    base_positions = {r["fold"]: np.arange(len(r["index"])) for r in records}
    models = ["mert", "svm"]
    base = {model: statistic(records, model, base_positions) for model in models}
    point = np.array([base["mert"].mean(), base["svm"].mean(),
                      (base["mert"] - base["svm"]).mean()])
    samples = np.empty((draws, 3))
    for draw in range(draws):
        positions = base_positions if unit == "instrument_type" else sample_positions(records, meta, unit, rng)
        mert = statistic(records, "mert", positions)
        svm = statistic(records, "svm", positions)
        if unit == "instrument_type":
            chosen = rng.integers(len(mert), size=len(mert))
            mert, svm = mert[chosen], svm[chosen]
        samples[draw] = [mert.mean(), svm.mean(), (mert - svm).mean()]
    return point, np.quantile(samples, [0.025, 0.975], axis=0)


def detail_tables(mert_svm, cnn):
    class_rows = []
    confusion_rows = []
    conditions = []
    for ratio in sorted({r["ratio"] for r in mert_svm}):
        subset = [r for r in mert_svm if r["ratio"] == ratio]
        conditions.extend([(f"ratio_{ratio:g}", model, subset) for model in ("mert", "svm")])
    conditions.append(("full_labels", "cnn", cnn))
    for condition, model, records in conditions:
        fold_class_scores = []
        confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        for fold in sorted({r["fold"] for r in records}):
            trials = [r for r in records if r["fold"] == fold]
            fold_class_scores.append(np.mean([scores(r["y"], r[model])["per_class_f1"] for r in trials], axis=0))
            for record in trials:
                confusion += np.asarray(scores(record["y"], record[model])["confusion_matrix"])
        for class_id, class_name in enumerate(CLASSES):
            class_rows.append({"condition": condition, "model": model, "class": class_name,
                               "fold_mean_f1": float(np.mean(np.asarray(fold_class_scores)[:, class_id]))})
            for predicted_id, count in enumerate(confusion[class_id]):
                confusion_rows.append({"condition": condition, "model": model,
                                       "true_class": class_name, "predicted_class": CLASSES[predicted_id],
                                       "count": int(count)})
    return pd.DataFrame(class_rows), pd.DataFrame(confusion_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("--draws must be at least 100")
    meta = pd.read_csv(ROOT / "data" / "mert_meta.csv")
    mert_svm = read_mert_svm(meta, replay=not args.skip_replay)
    cnn = read_cnn(meta, replay=not args.skip_replay)
    rows = []
    for ratio in sorted({r["ratio"] for r in mert_svm}):
        subset = [r for r in mert_svm if r["ratio"] == ratio]
        for unit in ("clip", "recording_proxy", "instrument_type"):
            point, interval = bootstrap(subset, meta, unit, args.draws, args.seed)
            for i, name in enumerate(("mert", "svm", "mert_minus_svm")):
                rows.append({"seed_ratio": ratio, "unit": unit, "measure": name,
                             "point": point[i], "ci_low": interval[0, i],
                             "ci_high": interval[1, i], "draws": args.draws})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "bootstrap_sensitivity.csv", index=False)
    class_table, confusion_table = detail_tables(mert_svm, cnn)
    class_table.to_csv(OUT / "per_class_f1.csv", index=False)
    confusion_table.to_csv(OUT / "confusion_counts.csv", index=False)
    summary = {
        "schema_version": 1, "replayed_checkpoints": not args.skip_replay,
        "mert_svm_splits": len(mert_svm), "cnn_folds": len(cnn),
        "mert_svm_prediction_set_sha256": hashlib.sha256("".join(r["prediction_sha256"] for r in mert_svm).encode()).hexdigest(),
        "cnn_prediction_set_sha256": hashlib.sha256("".join(r["prediction_sha256"] for r in cnn).encode()).hexdigest(),
        "cnn_macro_f1_fold_mean": float(np.mean([scores(r["y"], r["cnn"])["macro_f1"] for r in cnn])),
        "uncertainty_method": "paired percentile bootstrap; trial mean then instrument-type fold mean",
        "scope": "Four held-out instrument types only; recording_id is a filename-derived proxy, not a verified session or performer.",
    }
    (OUT / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

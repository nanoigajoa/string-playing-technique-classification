"""Replay and summarize matched canonical self-training conditions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from phase1_selftraining_repaired import (  # noqa: E402
    CLASSES, ExplicitTargetPool, LinearHead, predict_probabilities,
    sha256_array, stratified_holdout,
)

BASE = ROOT / "results" / "phase1" / "canonical" / "self_training"
CONDITIONS = ("supervised", "pseudo", "oracle")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y, prediction):
    ids = np.arange(len(CLASSES))
    return {
        "macro_f1": float(f1_score(y, prediction, labels=ids, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y, prediction, labels=ids, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y, prediction, labels=ids).tolist(),
    }


def main():
    paths = sorted((BASE / "manifests").glob("*.json"))
    assert len(paths) == 48, f"expected 48 manifests, found {len(paths)}"
    embeddings = np.load(ROOT / "data" / "mert_embeddings.npy", mmap_mode="r")
    labels = np.load(ROOT / "data" / "mert_labels.npy", mmap_mode="r")
    meta = pd.read_csv(ROOT / "data" / "mert_meta.csv")
    splits = json.loads((ROOT / "data" / "split_configs.json").read_text())["splits"]
    split_by_key = {f"r{round(s['seed_ratio'] * 100):02d}_f{s['fold']}_t{s['trial']}": s for s in splits}
    rows, class_rows, confusion_rows = [], [], []
    prediction_hashes = []
    for path in paths:
        manifest = json.loads(path.read_text())
        assert manifest["training_code_sha256"] == file_hash(ROOT / "src" / "phase1_canonical_selftraining.py")
        assert manifest["repaired_core_sha256"] == file_hash(ROOT / "src" / "phase1_selftraining_repaired.py")
        split = split_by_key[manifest["key"]]
        assert (manifest["epochs"], manifest["patience"], manifest["max_iter"], manifest["tau"]) == (150, 15, 5, 0.8)
        for field, array_name in (("observed_index_sha256", "labeled_idx"),
                                  ("unlabeled_index_sha256", "unlabeled_idx"),
                                  ("test_index_sha256", "test_idx")):
            assert manifest[field] == sha256_array(np.asarray(split[array_name], dtype=np.int64))
        prediction_path = ROOT / manifest["predictions"]
        assert file_hash(prediction_path) == manifest["predictions_sha256"]
        prediction_hashes.append(manifest["predictions_sha256"])
        with np.load(prediction_path) as stored:
            test = stored["sample_index"].astype(np.int64)
            y = stored["y_true"].astype(np.int64)
            predictions = {name: stored[name].astype(np.int64) for name in CONDITIONS}
        assert np.array_equal(y, labels[test])
        assert (meta["instrument"].to_numpy()[test] == manifest["held_out_instrument"]).all()
        assert set(manifest["conditions"]) == set(CONDITIONS)
        pseudo_report = manifest["conditions"]["pseudo"]["training"]
        observed = np.asarray(split["labeled_idx"], dtype=np.int64)
        unlabeled = np.asarray(split["unlabeled_idx"], dtype=np.int64)
        seed_core, seed_y, _, _ = stratified_holdout(
            observed, labels[observed], ratio=0.2, seed=manifest["seed"] + 1000)
        train, train_y, _, _ = stratified_holdout(
            seed_core, seed_y, ratio=0.2, seed=manifest["seed"] + 2000)
        pool = ExplicitTargetPool.from_labeled(train, train_y)
        assert len(pseudo_report["training_pool_by_iteration"]) == len(pseudo_report["acceptance"])
        for iteration, step in enumerate(pseudo_report["acceptance"]):
            assert pool.provenance() == pseudo_report["training_pool_by_iteration"][iteration]
            if step["status"] == "pseudo_targets_added_for_next_iteration":
                accepted_index = np.asarray(step["accepted_indices"], dtype=np.int64)
                accepted_target = np.asarray(step["accepted_targets"], dtype=np.int64)
                assert len(accepted_index) == step["accepted"]
                assert np.isin(accepted_index, unlabeled).all()
                assert not np.intersect1d(accepted_index, test).size
                assert sha256_array(accepted_index) == step["accepted_indices_sha256"]
                assert sha256_array(accepted_target) == step["accepted_targets_sha256"]
                pool.append_pseudo(accepted_index, accepted_target)
        assert pseudo_report["selected_training_pool"] == pseudo_report["training_pool_by_iteration"][pseudo_report["selected_iteration"]]
        oracle_report = manifest["conditions"]["oracle"]["training"]
        assert oracle_report["privileged_oracle_access"] is True
        for batch in oracle_report["accepted_counts"]:
            oracle_index = np.asarray(batch["indices"], dtype=np.int64)
            assert np.isin(oracle_index, unlabeled).all()
            assert np.array_equal(labels[oracle_index], batch["oracle_targets"])
        for name in CONDITIONS:
            item = manifest["conditions"][name]
            checkpoint = ROOT / item["checkpoint"]
            assert file_hash(checkpoint) == item["checkpoint_sha256"]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["condition"] == name
            assert saved["selected_iteration"] == item["training"]["selected_iteration"]
            model = LinearHead(embeddings.shape[1], len(CLASSES))
            model.load_state_dict(saved["state_dict"])
            replay = predict_probabilities(model, embeddings, test, "cpu").argmax(axis=1)
            assert np.array_equal(replay, predictions[name]), (manifest["key"], name)
            assert sha256_array(replay) == item["prediction_sha256"]
            actual = metrics(y, replay)
            assert np.isclose(actual["macro_f1"], item["test_metrics"]["macro_f1"])
            assert np.allclose(actual["per_class_f1"], item["test_metrics"]["per_class_f1"])
            assert actual["confusion_matrix"] == item["test_metrics"]["confusion_matrix"]
            assert item["test_evaluation_count"] == 1
            rows.append({"key": manifest["key"], "fold": manifest["fold"],
                         "ratio": manifest["seed_ratio"], "trial": manifest["trial"],
                         "condition": name, "macro_f1": actual["macro_f1"],
                         "selected_iteration": item["training"]["selected_iteration"]})
            for class_id, class_name in enumerate(CLASSES):
                class_rows.append({"key": manifest["key"], "fold": manifest["fold"],
                                   "ratio": manifest["seed_ratio"], "condition": name,
                                   "class": class_name, "f1": actual["per_class_f1"][class_id]})
                for predicted_id, count in enumerate(actual["confusion_matrix"][class_id]):
                    confusion_rows.append({"key": manifest["key"], "ratio": manifest["seed_ratio"],
                                           "condition": name, "true_class": class_name,
                                           "predicted_class": CLASSES[predicted_id], "count": count})
    frame = pd.DataFrame(rows)
    assert len(frame) == 48 * 3
    assert frame.groupby(["ratio", "condition"])["fold"].nunique().eq(4).all()
    summary = frame.groupby(["ratio", "condition", "fold"], as_index=False)["macro_f1"].mean()
    summary = summary.groupby(["ratio", "condition"], as_index=False).agg(
        macro_f1_fold_mean=("macro_f1", "mean"), macro_f1_fold_sd=("macro_f1", "std"))
    summary.to_csv(BASE / "canonical_summary.csv", index=False)
    frame.to_csv(BASE / "canonical_results.csv", index=False)
    class_frame = pd.DataFrame(class_rows)
    class_frame.groupby(["ratio", "condition", "class", "fold"], as_index=False)["f1"].mean().groupby(
        ["ratio", "condition", "class"], as_index=False)["f1"].mean().to_csv(BASE / "per_class_f1.csv", index=False)
    pd.DataFrame(confusion_rows).groupby(
        ["ratio", "condition", "true_class", "predicted_class"], as_index=False)["count"].sum().to_csv(
            BASE / "confusion_counts.csv", index=False)
    report = {"schema_version": 1, "verified_manifests": len(paths),
              "verified_checkpoint_replays": len(rows),
              "prediction_set_sha256": hashlib.sha256("".join(prediction_hashes).encode()).hexdigest(),
              "training_code_sha256": file_hash(ROOT / "src" / "phase1_canonical_selftraining.py"),
              "repaired_core_sha256": file_hash(ROOT / "src" / "phase1_selftraining_repaired.py"),
              "input_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in (
                  ROOT / "data" / "mert_embeddings.npy", ROOT / "data" / "mert_labels.npy",
                  ROOT / "data" / "split_configs.json")},
              "selection": "fixed tau=0.8; iteration selected by observed-label validation only",
              "oracle_scope": "explicit privileged diagnostic; not an attainable unsupervised result or guaranteed upper bound",
              "inference_scope": "four held-out instrument types only"}
    (BASE / "VERIFICATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

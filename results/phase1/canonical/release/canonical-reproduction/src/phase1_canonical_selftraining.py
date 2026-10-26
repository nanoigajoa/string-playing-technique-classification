"""Matched supervised, pseudo-label and explicit oracle control on Phase 1 splits.

The oracle condition is a privileged diagnostic; it is never presented as a
feasible self-training method. Threshold and iteration are selected without
test labels. Historical self-training outputs are left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score

from phase1_selftraining_repaired import (
    CLASSES, ExplicitTargetPool, LinearHead, predict_probabilities,
    run_repaired_self_training, score_observed, select_iteration,
    sha256_array, stratified_holdout, train_head,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase1" / "canonical" / "self_training"
MODELS = ROOT / "models" / "phase1" / "canonical" / "self_training"
TAU = 0.8  # Fixed before test evaluation; no threshold search on the test set.
BASE_SEED = 620_000


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key_for(split):
    return f"r{round(split['seed_ratio'] * 100):02d}_f{split['fold']}_t{split['trial']}"


def oracle_control(embeddings, observed_idx, observed_y, unlabeled_idx, oracle_y, *,
                   seed, device, epochs, patience, max_iter):
    """Upper-bound diagnostic with overt privileged access to unlabeled truth."""
    seed_core, seed_y, selector_idx, selector_y = stratified_holdout(
        observed_idx, observed_y, ratio=0.2, seed=seed + 1000)
    train_idx, train_y, early_idx, early_y = stratified_holdout(
        seed_core, seed_y, ratio=0.2, seed=seed + 2000)
    pool = ExplicitTargetPool.from_labeled(train_idx, train_y)
    remaining = np.asarray(unlabeled_idx, dtype=np.int64)
    oracle_lookup = dict(zip(remaining.tolist(), np.asarray(oracle_y, dtype=np.int64).tolist()))
    scores = []
    states = []
    accepted = []
    for iteration in range(max_iter):
        model, _ = train_head(
            embeddings, pool, early_idx, early_y, n_classes=len(CLASSES),
            device=device, seed=seed + iteration, epochs=epochs, patience=patience)
        selector = score_observed(model, embeddings, selector_idx, selector_y,
                                  n_classes=len(CLASSES), device=device)["macro_f1"]
        scores.append(selector)
        states.append({name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()})
        if iteration == max_iter - 1 or not len(remaining):
            break
        probabilities = predict_probabilities(model, embeddings, remaining, device)
        chosen = remaining[probabilities.max(axis=1) >= TAU]
        if not len(chosen):
            break
        # This explicit read is confined to the oracle diagnostic.
        chosen_targets = [oracle_lookup[int(index)] for index in chosen]
        pool.append_pseudo(chosen, chosen_targets)
        accepted.append({"indices": chosen.tolist(), "oracle_targets": chosen_targets})
        remaining = remaining[probabilities.max(axis=1) < TAU]
    selected = select_iteration(scores)
    result = LinearHead(embeddings.shape[1], len(CLASSES)).to(device)
    result.load_state_dict(states[selected])
    return result, {"selection_contract": "argmax observed selector macro F1",
                    "privileged_oracle_access": True, "selector_macro_f1": scores,
                    "selected_iteration": selected, "accepted_counts": accepted,
                    "oracle_target_source": "labels[unlabeled_idx] only in this control"}


def run_split(split, embeddings, labels, device, epochs, patience, max_iter):
    key = key_for(split)
    observed = np.asarray(split["labeled_idx"], dtype=np.int64)
    unlabeled = np.asarray(split["unlabeled_idx"], dtype=np.int64)
    test = np.asarray(split["test_idx"], dtype=np.int64)
    assert not np.intersect1d(observed, unlabeled).size
    assert not np.intersect1d(observed, test).size
    assert not np.intersect1d(unlabeled, test).size
    observed_y = labels[observed].astype(np.int64)
    seed = BASE_SEED + round(split["seed_ratio"] * 100) * 1000 + split["fold"] * 100 + split["trial"]
    conditions = {}
    for name, iterations in (("supervised", 1), ("pseudo", max_iter)):
        model, report = run_repaired_self_training(
            embeddings, observed, observed_y, unlabeled, tau=TAU,
            n_classes=len(CLASSES), device=device, seed=seed,
            max_iter=iterations, epochs=epochs, patience=patience)
        conditions[name] = (model, report)
    conditions["oracle"] = oracle_control(
        embeddings, observed, observed_y, unlabeled, labels[unlabeled],
        seed=seed, device=device, epochs=epochs, patience=patience,
        max_iter=max_iter)

    # Freeze and reload every selected checkpoint before one test evaluation.
    checkpoint_dir = MODELS / key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    predictions = {"sample_index": test, "y_true": labels[test].astype(np.int64)}
    condition_manifest = {}
    for name, (model, report) in conditions.items():
        checkpoint = checkpoint_dir / f"{name}.pt"
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "condition": name, "seed": seed, "tau": TAU,
                    "selected_iteration": report["selected_iteration"]}, checkpoint)
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        loaded = LinearHead(embeddings.shape[1], len(CLASSES)).to(device)
        loaded.load_state_dict(saved["state_dict"])
        probabilities = predict_probabilities(loaded, embeddings, test, device)
        prediction = probabilities.argmax(axis=1).astype(np.int64)
        predictions[name] = prediction
        labels_all = np.arange(len(CLASSES))
        truth = predictions["y_true"]
        evaluation = {
            "macro_f1": float(f1_score(truth, prediction, labels=labels_all,
                                        average="macro", zero_division=0)),
            "per_class_f1": f1_score(truth, prediction, labels=labels_all,
                                      average=None, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(truth, prediction,
                                                   labels=labels_all).tolist(),
        }
        condition_manifest[name] = {
            "training": report, "test_metrics": evaluation,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": file_hash(checkpoint),
            "prediction_sha256": sha256_array(prediction),
            "test_evaluation_count": 1,
        }
    prediction_path = RESULTS / "predictions" / f"{key}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, **predictions)
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_sha256": file_hash(Path(__file__).resolve()),
        "repaired_core_sha256": file_hash(ROOT / "src" / "phase1_selftraining_repaired.py"),
        "key": key, "fold": int(split["fold"]),
        "held_out_instrument": split["test_instrument"],
        "seed_ratio": float(split["seed_ratio"]), "trial": int(split["trial"]),
        "seed": int(seed), "tau": TAU, "epochs": epochs,
        "patience": patience, "max_iter": max_iter,
        "observed_index_sha256": sha256_array(observed),
        "unlabeled_index_sha256": sha256_array(unlabeled),
        "test_index_sha256": sha256_array(test),
        "conditions": condition_manifest,
        "predictions": str(prediction_path.relative_to(ROOT)),
        "predictions_sha256": file_hash(prediction_path),
    }
    manifest_path = RESULTS / "manifests" / f"{key}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def valid(manifest_path, epochs, patience, max_iter):
    try:
        manifest = json.loads(manifest_path.read_text())
        if (manifest["epochs"], manifest["patience"], manifest["max_iter"], manifest["tau"]) != (epochs, patience, max_iter, TAU):
            return False
        if manifest.get("training_code_sha256") != file_hash(Path(__file__).resolve()):
            return False
        if manifest.get("repaired_core_sha256") != file_hash(ROOT / "src" / "phase1_selftraining_repaired.py"):
            return False
        if file_hash(ROOT / manifest["predictions"]) != manifest["predictions_sha256"]:
            return False
        return all(file_hash(ROOT / item["checkpoint"]) == item["checkpoint_sha256"]
                   for item in manifest["conditions"].values())
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    embeddings = np.load(ROOT / "data" / "mert_embeddings.npy").astype(np.float32)
    labels = np.load(ROOT / "data" / "mert_labels.npy").astype(np.int64)
    splits = json.loads((ROOT / "data" / "split_configs.json").read_text())["splits"]
    selected = splits[:args.limit] if args.limit is not None else splits
    RESULTS.mkdir(parents=True, exist_ok=True)
    for number, split in enumerate(selected, 1):
        key = key_for(split)
        manifest_path = RESULTS / "manifests" / f"{key}.json"
        if valid(manifest_path, args.epochs, args.patience, args.max_iter):
            print(f"[{number}/{len(selected)}] {key} valid -> skip", flush=True)
            continue
        manifest = run_split(split, embeddings, labels, args.device,
                             args.epochs, args.patience, args.max_iter)
        values = {name: round(item["test_metrics"]["macro_f1"], 4)
                  for name, item in manifest["conditions"].items()}
        print(f"[{number}/{len(selected)}] {key} {values}", flush=True)
    paths = sorted((RESULTS / "manifests").glob("*.json"))
    manifests = [json.loads(path.read_text()) for path in paths]
    rows = [{"key": m["key"], "fold": m["fold"], "ratio": m["seed_ratio"],
             "trial": m["trial"], "condition": name,
             "macro_f1": item["test_metrics"]["macro_f1"]}
            for m in manifests for name, item in m["conditions"].items()
            if valid(RESULTS / "manifests" / f"{m['key']}.json", args.epochs, args.patience, args.max_iter)]
    pd.DataFrame(rows).to_csv(RESULTS / "progress.csv", index=False)
    print(f"valid manifests: {len(rows) // 3}/48", flush=True)


if __name__ == "__main__":
    main()

"""Canonical train-only MERT-head versus SVM evaluation.

This entry point replaces the publication use of the historical tuning script.
It preserves the configured outer splits, keeps early-stop and model-selection
validation sets disjoint, computes every learned quantity from ``train_idx``,
selects configurations without reading the test fold, saves the selected
checkpoints, and evaluates each selected checkpoint on test exactly once.

Outputs are isolated under ``results/phase1/canonical`` and
``models/phase1/canonical``; historical files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phase1_train_only import array_hash, balanced_class_weights  # noqa: E402


CLASSES = [
    "ordinario",
    "pizzicato_secco",
    "tremolo",
    "sul_ponticello",
    "col_legno_battuto",
]
N_CLASSES = len(CLASSES)
BASE_SEED = 420_000

EMBEDDING_PATH = ROOT / "data" / "mert_embeddings.npy"
LABEL_PATH = ROOT / "data" / "mert_labels.npy"
FEATURE_PATH = ROOT / "data" / "features_full.csv"
META_PATH = ROOT / "data" / "mert_meta.csv"
SPLIT_PATH = ROOT / "data" / "split_configs.json"
RESULTS_DIR = ROOT / "results" / "phase1" / "canonical" / "mert_svm"
MODEL_DIR = ROOT / "models" / "phase1" / "canonical" / "mert_svm"

MERT_GRID = {
    "arch": ["linear", "mlp256"],
    "lr": [3e-4, 1e-3, 3e-3],
    "wd": [1e-4, 1e-3],
    "dropout": [0.1, 0.3],
}
SVM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.001],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Head(nn.Module):
    def __init__(self, arch: str, dropout: float, in_dim: int = 768):
        super().__init__()
        if arch == "linear":
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, N_CLASSES)
            )
        elif arch == "mlp256":
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Dropout(dropout),
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, N_CLASSES),
            )
        else:
            raise ValueError(arch)

    def forward(self, features):
        return self.net(features)


def split_val(indices, labels, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices, dtype=np.int64)
    local_labels = labels[indices]
    train, validation = [], []
    for class_id in range(N_CLASSES):
        positions = indices[local_labels == class_id]
        if not len(positions):
            raise ValueError(f"class {class_id} absent from partition")
        n_validation = max(1, round(len(positions) * val_ratio))
        chosen = rng.choice(len(positions), n_validation, replace=False)
        validation.extend(positions[chosen].tolist())
        train.extend(np.delete(positions, chosen).tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def partition(split, labels):
    seed_core, val_select = split_val(
        split["labeled_idx"], labels, seed=int(split["trial"]) + 1000
    )
    train, early_stop = split_val(seed_core, labels, seed=int(split["trial"]))
    test = np.asarray(split["test_idx"], dtype=np.int64)
    named = {
        "train": train,
        "early_stop": early_stop,
        "val_select": val_select,
        "test": test,
    }
    for left, right in itertools.combinations(named, 2):
        if np.intersect1d(named[left], named[right]).size:
            raise AssertionError(f"partition overlap: {left}/{right}")
    return named


@torch.no_grad()
def head_predictions(model, embeddings, indices, device):
    model.eval()
    matrix = torch.from_numpy(embeddings[indices]).to(device)
    return model(matrix).argmax(1).cpu().numpy()


def train_head(
    embeddings,
    labels,
    train_idx,
    early_stop_idx,
    config,
    class_weights,
    device,
    seed,
    epochs,
    patience,
):
    set_seed(seed)
    model = Head(config["arch"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    X_train = torch.from_numpy(embeddings[train_idx]).to(device)
    y_train = torch.from_numpy(labels[train_idx]).to(device)
    X_early = torch.from_numpy(embeddings[early_stop_idx]).to(device)
    y_early = labels[early_stop_idx]

    best_score = -1.0
    best_epoch = -1
    best_state = None
    wait = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        criterion(model(X_train), y_train).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(X_early).argmax(1).cpu().numpy()
        score = f1_score(y_early, prediction, average="macro", zero_division=0)
        if score > best_score + 1e-12:
            best_score = float(score)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise AssertionError("early stopping failed to retain a checkpoint")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    return model, best_state, best_score, best_epoch


def metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(
            y_true, y_pred, labels=np.arange(N_CLASSES), average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(N_CLASSES)
        ).tolist(),
    }


def split_key(split) -> str:
    return f"r{int(round(float(split['seed_ratio']) * 100)):02d}_f{int(split['fold'])}_t{int(split['trial'])}"


def run_split(split, arrays, device, epochs, patience):
    embeddings, labels, features, meta = arrays
    key = split_key(split)
    parts = partition(split, labels)
    train_idx = parts["train"]
    class_weights = balanced_class_weights(labels[train_idx], N_CLASSES)
    split_seed = BASE_SEED + int(round(split["seed_ratio"] * 100)) * 1000 + split["fold"] * 100 + split["trial"]

    # MERT selection: test arrays are not passed into this block.
    best_mert = None
    mert_trials = []
    for config_id, values in enumerate(itertools.product(*MERT_GRID.values())):
        config = dict(zip(MERT_GRID.keys(), values))
        model, state, early_score, best_epoch = train_head(
            embeddings,
            labels,
            train_idx,
            parts["early_stop"],
            config,
            class_weights,
            device,
            split_seed + config_id,
            epochs,
            patience,
        )
        val_prediction = head_predictions(model, embeddings, parts["val_select"], device)
        val_score = float(
            f1_score(labels[parts["val_select"]], val_prediction, average="macro", zero_division=0)
        )
        trial = {
            "config_id": config_id,
            "config": config,
            "seed": split_seed + config_id,
            "early_stop_f1": early_score,
            "best_epoch_zero_based": best_epoch,
            "val_select_f1": val_score,
        }
        mert_trials.append(trial)
        if best_mert is None or val_score > best_mert["val_select_f1"] + 1e-12:
            best_mert = {**trial, "state": state}

    # SVM selection: the complete scaler+SVC pipeline is fit on train only.
    best_svm = None
    svm_trials = []
    for config_id, values in enumerate(itertools.product(*SVM_GRID.values())):
        config = dict(zip(SVM_GRID.keys(), values))
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=config["C"],
                gamma=config["gamma"],
                class_weight="balanced",
                random_state=split_seed + config_id,
            )),
        ])
        pipeline.fit(features[train_idx], labels[train_idx])
        val_prediction = pipeline.predict(features[parts["val_select"]])
        val_score = float(
            f1_score(labels[parts["val_select"]], val_prediction, average="macro", zero_division=0)
        )
        trial = {"config_id": config_id, "config": config, "val_select_f1": val_score}
        svm_trials.append(trial)
        if best_svm is None or val_score > best_svm["val_select_f1"] + 1e-12:
            best_svm = {**trial, "pipeline": pipeline}

    checkpoint_dir = MODEL_DIR / key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mert_checkpoint = checkpoint_dir / "mert_head.pt"
    svm_checkpoint = checkpoint_dir / "svm.joblib"
    torch.save({
        "state_dict": best_mert["state"],
        "config": best_mert["config"],
        "classes": CLASSES,
        "input_dim": int(embeddings.shape[1]),
        "split_key": key,
        "train_index_hash": array_hash(train_idx),
        "class_weights": class_weights.tolist(),
    }, mert_checkpoint)
    joblib.dump(best_svm["pipeline"], svm_checkpoint)

    # Load the persisted selected checkpoints, then touch test exactly once.
    saved_mert = torch.load(mert_checkpoint, map_location=device, weights_only=True)
    selected_head = Head(saved_mert["config"]["arch"], saved_mert["config"]["dropout"]).to(device)
    selected_head.load_state_dict(saved_mert["state_dict"])
    loaded_svm = joblib.load(svm_checkpoint)
    test_idx = parts["test"]
    mert_prediction = head_predictions(selected_head, embeddings, test_idx, device)
    svm_prediction = loaded_svm.predict(features[test_idx])
    y_true = labels[test_idx]

    prediction_path = RESULTS_DIR / "predictions" / f"{key}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        sample_index=test_idx,
        y_true=y_true,
        mert_prediction=mert_prediction,
        svm_prediction=svm_prediction,
    )

    partition_hashes = {name: array_hash(indices) for name, indices in parts.items()}
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_key": key,
        "fold": int(split["fold"]),
        "held_out_instrument": str(split["test_instrument"]),
        "seed_ratio": float(split["seed_ratio"]),
        "trial": int(split["trial"]),
        "split_seed": int(split_seed),
        "training_spec": {"epochs": int(epochs), "patience": int(patience)},
        "selection_rule": "maximum val_select macro F1; fixed grid order breaks ties",
        "test_role": "selected persisted checkpoints evaluated once after all selection",
        "partition_sizes": {name: int(len(indices)) for name, indices in parts.items()},
        "partition_hashes": partition_hashes,
        "partitions_pairwise_disjoint": True,
        "class_weight_source": "labels[train_idx] only",
        "train_class_counts": np.bincount(labels[train_idx], minlength=N_CLASSES).tolist(),
        "mert_class_weights": class_weights.tolist(),
        "mert_class_weight_hash": array_hash(class_weights),
        "mert_grid": MERT_GRID,
        "svm_grid": SVM_GRID,
        "mert_search": mert_trials,
        "svm_search": svm_trials,
        "selected_mert": {key_: value for key_, value in best_mert.items() if key_ != "state"},
        "selected_svm": {key_: value for key_, value in best_svm.items() if key_ != "pipeline"},
        "svm_fitted_class_weights": loaded_svm.named_steps["svm"].class_weight_.tolist(),
        "mert_metrics": metrics(y_true, mert_prediction),
        "svm_metrics": metrics(y_true, svm_prediction),
        "artifacts": {
            "mert_checkpoint": str(mert_checkpoint.relative_to(ROOT)),
            "mert_checkpoint_sha256": file_sha256(mert_checkpoint),
            "svm_checkpoint": str(svm_checkpoint.relative_to(ROOT)),
            "svm_checkpoint_sha256": file_sha256(svm_checkpoint),
            "predictions": str(prediction_path.relative_to(ROOT)),
            "predictions_sha256": file_sha256(prediction_path),
        },
        "prediction_content_hash": array_hash(test_idx, y_true, mert_prediction, svm_prediction),
        "test_prediction_calls_per_model": 1,
    }
    manifest_path = RESULTS_DIR / "manifests" / f"{key}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def artifact_valid(manifest_path: Path) -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for path_key, hash_key in (
            ("mert_checkpoint", "mert_checkpoint_sha256"),
            ("svm_checkpoint", "svm_checkpoint_sha256"),
            ("predictions", "predictions_sha256"),
        ):
            path = ROOT / manifest["artifacts"][path_key]
            if not path.exists() or file_sha256(path) != manifest["artifacts"][hash_key]:
                return False
        return True
    except Exception:
        return False


def build_index(input_hashes):
    manifest_paths = sorted((RESULTS_DIR / "manifests").glob("*.json"))
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    rows = []
    for manifest in manifests:
        rows.append({
            "split_key": manifest["split_key"],
            "fold": manifest["fold"],
            "held_out_instrument": manifest["held_out_instrument"],
            "seed_ratio": manifest["seed_ratio"],
            "trial": manifest["trial"],
            "mert_macro_f1": manifest["mert_metrics"]["macro_f1"],
            "svm_macro_f1": manifest["svm_metrics"]["macro_f1"],
            "mert_minus_svm": manifest["mert_metrics"]["macro_f1"] - manifest["svm_metrics"]["macro_f1"],
            "mert_checkpoint_sha256": manifest["artifacts"]["mert_checkpoint_sha256"],
            "svm_checkpoint_sha256": manifest["artifacts"]["svm_checkpoint_sha256"],
            "predictions_sha256": manifest["artifacts"]["predictions_sha256"],
        })
    frame = pd.DataFrame(rows).sort_values(["seed_ratio", "fold", "trial"])
    frame.to_csv(RESULTS_DIR / "canonical_results.csv", index=False)

    summary_rows = []
    for ratio, group in frame.groupby("seed_ratio"):
        fold_means = group.groupby("fold")[["mert_macro_f1", "svm_macro_f1"]].mean()
        summary_rows.append({
            "seed_ratio": ratio,
            "n_splits": len(group),
            "n_outer_folds": fold_means.shape[0],
            "mert_macro_f1_fold_mean": fold_means["mert_macro_f1"].mean(),
            "mert_macro_f1_fold_sd": fold_means["mert_macro_f1"].std(ddof=1),
            "svm_macro_f1_fold_mean": fold_means["svm_macro_f1"].mean(),
            "svm_macro_f1_fold_sd": fold_means["svm_macro_f1"].std(ddof=1),
            "mert_minus_svm": (
                fold_means["mert_macro_f1"] - fold_means["svm_macro_f1"]
            ).mean(),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "canonical_summary.csv", index=False)
    index = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": input_hashes,
        "code": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "n_manifests": len(manifests),
        "manifest_set_hash": canonical_json_hash([
            {"split_key": m["split_key"], "prediction_content_hash": m["prediction_content_hash"]}
            for m in manifests
        ]),
        "all_test_prediction_calls_equal_one": all(
            m["test_prediction_calls_per_model"] == 1 for m in manifests
        ),
        "all_partitions_disjoint": all(m["partitions_pairwise_disjoint"] for m in manifests),
        "result_csv": "results/phase1/canonical/mert_svm/canonical_results.csv",
        "summary_csv": "results/phase1/canonical/mert_svm/canonical_summary.csv",
    }
    (RESULTS_DIR / "RUN_INDEX.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return frame, summary, index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", type=float, nargs="*", default=[0.05, 0.10, 0.20, 0.50])
    parser.add_argument("--trials", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    device = device_name(args.device)
    set_seed(BASE_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(EMBEDDING_PATH).astype(np.float32)
    labels = np.load(LABEL_PATH).astype(np.int64)
    feature_frame = pd.read_csv(FEATURE_PATH)
    feature_columns = [c for c in feature_frame if c not in ("label_id", "instrument")]
    features = feature_frame[feature_columns].to_numpy(dtype=np.float32)
    meta = pd.read_csv(META_PATH)
    if not (len(embeddings) == len(labels) == len(features) == len(meta)):
        raise ValueError("canonical inputs have unequal rows")
    arrays = (embeddings, labels, features, meta)

    splits = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["splits"]
    selected = [
        split for split in splits
        if any(np.isclose(split["seed_ratio"], ratio) for ratio in args.ratios)
        and split["trial"] in args.trials
    ]
    if args.limit is not None:
        selected = selected[:args.limit]
    input_hashes = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in (EMBEDDING_PATH, LABEL_PATH, FEATURE_PATH, META_PATH, SPLIT_PATH)
    }
    print(
        f"device={device} splits={len(selected)} MERT_configs=24 SVM_configs=12 "
        f"epochs<={args.epochs} patience={args.patience}",
        flush=True,
    )
    started = time.time()
    for number, split in enumerate(selected, 1):
        key = split_key(split)
        manifest_path = RESULTS_DIR / "manifests" / f"{key}.json"
        if not args.no_resume and artifact_valid(manifest_path):
            print(f"[{number}/{len(selected)}] {key} valid artifacts -> skip", flush=True)
            continue
        split_started = time.time()
        manifest = run_split(split, arrays, device, args.epochs, args.patience)
        print(
            f"[{number}/{len(selected)}] {key} "
            f"MERT={manifest['mert_metrics']['macro_f1']:.4f} "
            f"SVM={manifest['svm_metrics']['macro_f1']:.4f} "
            f"selected={manifest['selected_mert']['config']} "
            f"{time.time() - split_started:.1f}s",
            flush=True,
        )
    frame, summary, index = build_index(input_hashes)
    print(summary.to_string(index=False), flush=True)
    print(f"n_manifests={index['n_manifests']} runtime={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

"""Canonical train-only mel-CNN LOSO baseline with artifact provenance.

The checkpoint is preregistered as the final epoch of a fixed training budget;
there is no validation or test-based model selection.  Class weights are fit
inside each outer training fold.  Historical notebook outputs are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import (  # noqa: E402
    FIXED_FRAMES,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SR,
    MelDataset,
    audio_to_mel,
)
from model import SmallCNN  # noqa: E402
from phase1_train_only import array_hash, balanced_class_weights  # noqa: E402


CLASSES = [
    "ordinario",
    "pizzicato_secco",
    "tremolo",
    "sul_ponticello",
    "col_legno_battuto",
]
N_CLASSES = len(CLASSES)
BASE_SEED = 520_000
META_PATH = ROOT / "data" / "mert_meta.csv"
RESULTS_DIR = ROOT / "results" / "phase1" / "canonical" / "cnn"
MODEL_DIR = ROOT / "models" / "phase1" / "canonical" / "cnn"
CACHE_DIR = RESULTS_DIR / "cache"
MEL_CACHE = CACHE_DIR / "mels_128.npy"
MEL_CACHE_META = CACHE_DIR / "mels_128_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def preprocessing_spec():
    return {
        "sample_rate": SR,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "fixed_frames": FIXED_FRAMES,
        "frame_policy": "first 128 frames or right zero padding",
        "normalization": "per-image min-max to [0,1] after power_to_db(ref=max)",
        "implementation": "src/dataset.py:audio_to_mel",
        "implementation_sha256": file_sha256(ROOT / "src" / "dataset.py"),
    }


def cache_valid(n_samples: int, meta_hash: str) -> bool:
    if not MEL_CACHE.exists() or not MEL_CACHE_META.exists():
        return False
    try:
        manifest = json.loads(MEL_CACHE_META.read_text(encoding="utf-8"))
        if manifest["n_samples"] != n_samples or manifest["metadata_sha256"] != meta_hash:
            return False
        if manifest["preprocessing"] != preprocessing_spec():
            return False
        cached = np.load(MEL_CACHE, mmap_mode="r")
        return cached.shape == (n_samples, N_MELS, FIXED_FRAMES) and cached.dtype == np.float32
    except Exception:
        return False


def build_or_load_cache(meta):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_hash = file_sha256(META_PATH)
    if cache_valid(len(meta), meta_hash):
        print(f"mel cache valid -> {MEL_CACHE}", flush=True)
        return np.load(MEL_CACHE, mmap_mode="r")

    print(f"building mel cache for {len(meta)} files", flush=True)
    cache = np.lib.format.open_memmap(
        MEL_CACHE,
        mode="w+",
        dtype=np.float32,
        shape=(len(meta), N_MELS, FIXED_FRAMES),
    )
    started = time.time()
    for index, filepath in enumerate(meta["filepath"]):
        cache[index] = audio_to_mel(filepath)
        if (index + 1) % 250 == 0 or index + 1 == len(meta):
            cache.flush()
            print(
                f"mel {index + 1}/{len(meta)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    del cache
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(meta),
        "metadata_path": str(META_PATH.relative_to(ROOT)),
        "metadata_sha256": meta_hash,
        "preprocessing": preprocessing_spec(),
        "cache_path": str(MEL_CACHE.relative_to(ROOT)),
        "cache_sha256": file_sha256(MEL_CACHE),
    }
    MEL_CACHE_META.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return np.load(MEL_CACHE, mmap_mode="r")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outputs = []
    for features, _ in loader:
        outputs.append(model(features.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(outputs)


def fold_metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": f1_score(
            y_true, y_pred, labels=np.arange(N_CLASSES), average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(N_CLASSES)
        ).tolist(),
    }


def run_fold(fold, train_idx, test_idx, mels, labels, groups, device, epochs, batch_size, lr):
    seed = BASE_SEED + fold
    set_seed(seed)
    class_weights = balanced_class_weights(labels[train_idx], N_CLASSES)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        MelDataset(mels[train_idx], labels[train_idx], augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = DataLoader(
        MelDataset(mels[test_idx], labels[test_idx], augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = SmallCNN(n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        scheduler.step()
        epoch_losses.append(total_loss / n_batches)
        print(
            f"fold={fold} epoch={epoch + 1}/{epochs} loss={epoch_losses[-1]:.6f}",
            flush=True,
        )

    fold_key = f"fold{fold}_{np.unique(groups[test_idx])[0]}"
    checkpoint = MODEL_DIR / fold_key / "final_epoch.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "classes": CLASSES,
        "fold": fold,
        "seed": seed,
        "epoch": epochs,
        "selection_rule": "preregistered final epoch; no validation or test selection",
        "train_index_hash": array_hash(train_idx),
        "class_weights": class_weights.tolist(),
    }, checkpoint)
    del model

    # Persist first, then load the fixed checkpoint and touch test once.
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    loaded = SmallCNN(n_classes=N_CLASSES).to(device)
    loaded.load_state_dict(saved["state_dict"])
    prediction = predict(loaded, test_loader, device)
    y_true = labels[test_idx]
    prediction_path = RESULTS_DIR / "predictions" / f"{fold_key}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        sample_index=test_idx,
        y_true=y_true,
        prediction=prediction,
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold_key": fold_key,
        "fold": fold,
        "held_out_instrument": str(np.unique(groups[test_idx])[0]),
        "seed": seed,
        "training_spec": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR(T_max=epochs)",
            "augmentation": "one random frequency mask and one random time mask per training access",
        },
        "selection_rule": "preregistered final epoch; no validation or test selection",
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_index_hash": array_hash(train_idx),
        "test_index_hash": array_hash(test_idx),
        "partitions_disjoint": np.intersect1d(train_idx, test_idx).size == 0,
        "class_weight_source": "labels[train_idx] only",
        "train_class_counts": np.bincount(labels[train_idx], minlength=N_CLASSES).tolist(),
        "class_weights": class_weights.tolist(),
        "class_weight_hash": array_hash(class_weights),
        "epoch_train_losses": epoch_losses,
        "metrics": fold_metrics(y_true, prediction),
        "artifacts": {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": file_sha256(checkpoint),
            "predictions": str(prediction_path.relative_to(ROOT)),
            "predictions_sha256": file_sha256(prediction_path),
        },
        "prediction_content_hash": array_hash(test_idx, y_true, prediction),
        "test_prediction_calls": 1,
    }
    manifest_path = RESULTS_DIR / "manifests" / f"{fold_key}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def artifact_valid(path: Path, epochs, batch_size, lr) -> bool:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = {"epochs": epochs, "batch_size": batch_size, "learning_rate": lr}
        if any(manifest["training_spec"][key] != value for key, value in expected.items()):
            return False
        for path_key, hash_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("predictions", "predictions_sha256"),
        ):
            artifact = ROOT / manifest["artifacts"][path_key]
            if not artifact.exists() or file_sha256(artifact) != manifest["artifacts"][hash_key]:
                return False
        return True
    except Exception:
        return False


def build_index(meta_hash):
    paths = sorted((RESULTS_DIR / "manifests").glob("*.json"))
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows = [{
        "fold_key": item["fold_key"],
        "fold": item["fold"],
        "held_out_instrument": item["held_out_instrument"],
        "macro_f1": item["metrics"]["macro_f1"],
        "checkpoint_sha256": item["artifacts"]["checkpoint_sha256"],
        "predictions_sha256": item["artifacts"]["predictions_sha256"],
    } for item in manifests]
    frame = pd.DataFrame(rows).sort_values("fold")
    frame.to_csv(RESULTS_DIR / "canonical_results.csv", index=False)
    index = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_sha256": meta_hash,
        "mel_cache_sha256": file_sha256(MEL_CACHE),
        "code": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "n_manifests": len(manifests),
        "macro_f1_fold_mean": float(frame["macro_f1"].mean()),
        "macro_f1_fold_sd": float(frame["macro_f1"].std(ddof=1)),
        "all_partitions_disjoint": all(item["partitions_disjoint"] for item in manifests),
        "all_test_prediction_calls_equal_one": all(item["test_prediction_calls"] == 1 for item in manifests),
        "prediction_content_hashes": {
            item["fold_key"]: item["prediction_content_hash"] for item in manifests
        },
    }
    (RESULTS_DIR / "RUN_INDEX.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return frame, index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    device = device_name(args.device)
    set_seed(BASE_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META_PATH)
    labels = meta["label_id"].to_numpy(dtype=np.int64)
    groups = meta["instrument"].to_numpy()
    mels = build_or_load_cache(meta)
    if args.cache_only:
        return
    splitter = GroupKFold(n_splits=len(np.unique(groups)))
    folds = list(splitter.split(np.zeros((len(labels), 1)), labels, groups))
    print(
        f"device={device} folds={args.folds} epochs={args.epochs} batch={args.batch_size}",
        flush=True,
    )
    started = time.time()
    for fold in args.folds:
        train_idx, test_idx = folds[fold]
        fold_key = f"fold{fold}_{np.unique(groups[test_idx])[0]}"
        manifest_path = RESULTS_DIR / "manifests" / f"{fold_key}.json"
        if not args.no_resume and artifact_valid(manifest_path, args.epochs, args.batch_size, args.lr):
            print(f"{fold_key}: valid artifacts -> skip", flush=True)
            continue
        fold_started = time.time()
        manifest = run_fold(
            fold,
            train_idx,
            test_idx,
            mels,
            labels,
            groups,
            device,
            args.epochs,
            args.batch_size,
            args.lr,
        )
        print(
            f"{fold_key}: macro_f1={manifest['metrics']['macro_f1']:.4f} "
            f"runtime={time.time() - fold_started:.1f}s",
            flush=True,
        )
    frame, index = build_index(file_sha256(META_PATH))
    print(frame.to_string(index=False), flush=True)
    print(
        f"mean={index['macro_f1_fold_mean']:.4f} sd={index['macro_f1_fold_sd']:.4f} "
        f"runtime={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

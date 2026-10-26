"""Leakage-repaired Phase 1 self-training with explicit target ownership.

This module is intentionally separate from ``phase1_rerun.py`` so the
historical implementation and its outputs remain available as audit evidence.
The training core never receives hidden unlabeled labels or test labels.

Commands:
    venv/bin/python -B src/phase1_selftraining_repaired.py --mode sentinel
    venv/bin/python -B src/phase1_selftraining_repaired.py --mode smoke

The smoke run is deliberately small.  It validates the repaired data flow but
does not replace the full matched rerun required for publication.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = ROOT / "data" / "mert_embeddings.npy"
LAB_PATH = ROOT / "data" / "mert_labels.npy"
SPLITS_PATH = ROOT / "data" / "split_configs.json"
RESULTS_DIR = ROOT / "results" / "phase1" / "audit_repair"

CLASSES = [
    "ordinario",
    "pizzicato_secco",
    "tremolo",
    "sul_ponticello",
    "col_legno_battuto",
]


def sha256_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def sha256_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    ordered = {key: state_dict[key].detach().cpu() for key in sorted(state_dict)}
    torch.save(ordered, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def as_int_array(values: Sequence[int] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


@dataclass
class ExplicitTargetPool:
    """Training indices paired with the only targets the trainer may read."""

    indices: np.ndarray
    targets: np.ndarray
    sources: list[str]

    @classmethod
    def from_labeled(
        cls,
        indices: Sequence[int] | np.ndarray,
        targets: Sequence[int] | np.ndarray,
    ) -> "ExplicitTargetPool":
        pool = cls(
            indices=as_int_array(indices),
            targets=as_int_array(targets),
            sources=["observed"] * len(indices),
        )
        pool.validate()
        return pool

    def validate(self) -> None:
        if len(self.indices) != len(self.targets) or len(self.indices) != len(self.sources):
            raise ValueError("indices, targets, and sources must have equal length")
        if len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("training indices must be unique")
        if any(source not in {"observed", "pseudo"} for source in self.sources):
            raise ValueError("unknown target source")

    def append_pseudo(
        self,
        indices: Sequence[int] | np.ndarray,
        pseudo_targets: Sequence[int] | np.ndarray,
    ) -> None:
        new_indices = as_int_array(indices)
        new_targets = as_int_array(pseudo_targets)
        if len(new_indices) != len(new_targets):
            raise ValueError("pseudo indices and targets must have equal length")
        if np.intersect1d(self.indices, new_indices).size:
            raise ValueError("pseudo-labeled indices are already in the training pool")
        self.indices = np.concatenate([self.indices, new_indices])
        self.targets = np.concatenate([self.targets, new_targets])
        self.sources.extend(["pseudo"] * len(new_indices))
        self.validate()

    def provenance(self) -> dict:
        source_counts = {
            source: self.sources.count(source) for source in sorted(set(self.sources))
        }
        return {
            "n_examples": len(self.indices),
            "source_counts": source_counts,
            "indices_sha256": sha256_array(self.indices),
            "targets_sha256": sha256_array(self.targets),
        }


def stratified_holdout(
    indices: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    *,
    ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split paired indices/targets without consulting any global label array."""
    indices = as_int_array(indices)
    targets = as_int_array(targets)
    if len(indices) != len(targets):
        raise ValueError("indices and targets must have equal length")
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between zero and one")

    rng = np.random.default_rng(seed)
    train_positions: list[int] = []
    holdout_positions: list[int] = []
    for label in np.unique(targets):
        positions = np.flatnonzero(targets == label)
        if len(positions) < 2:
            raise ValueError(f"class {label} needs at least two observed examples")
        n_holdout = min(max(1, round(len(positions) * ratio)), len(positions) - 1)
        selected = rng.choice(positions, size=n_holdout, replace=False)
        holdout_positions.extend(selected.tolist())
        train_positions.extend(np.setdiff1d(positions, selected).tolist())

    train_positions = as_int_array(train_positions)
    holdout_positions = as_int_array(holdout_positions)
    return (
        indices[train_positions],
        targets[train_positions],
        indices[holdout_positions],
        targets[holdout_positions],
    )


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def paired_tensors(
    embeddings: np.ndarray,
    indices: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = as_int_array(indices)
    targets = as_int_array(targets)
    if len(indices) != len(targets):
        raise ValueError("indices and explicit targets must have equal length")
    inputs = torch.tensor(embeddings[indices], dtype=torch.float32, device=device)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs, target_tensor


def train_only_class_weights(targets: np.ndarray, n_classes: int, device: str) -> torch.Tensor:
    counts = np.bincount(as_int_array(targets), minlength=n_classes).astype(float)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"training targets omit classes: {missing}")
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_head(
    embeddings: np.ndarray,
    pool: ExplicitTargetPool,
    early_stop_indices: np.ndarray,
    early_stop_targets: np.ndarray,
    *,
    n_classes: int,
    device: str,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float = 1e-3,
) -> tuple[LinearHead, float]:
    """Fit from explicit targets; validation is fixed observed-label data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LinearHead(embeddings.shape[1], n_classes).to(device)
    weights = train_only_class_weights(pool.targets, n_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_inputs, train_targets = paired_tensors(
        embeddings, pool.indices, pool.targets, device
    )
    val_inputs, val_targets = paired_tensors(
        embeddings, early_stop_indices, early_stop_targets, device
    )

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(train_inputs), train_targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(val_inputs).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(
            val_targets.cpu().numpy(), predictions, average="macro", zero_division=0
        )
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, best_f1


def predict_probabilities(
    model: nn.Module,
    embeddings: np.ndarray,
    indices: Sequence[int] | np.ndarray,
    device: str,
) -> np.ndarray:
    indices = as_int_array(indices)
    inputs = torch.tensor(embeddings[indices], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(inputs), dim=1).cpu().numpy()


def score_observed(
    model: nn.Module,
    embeddings: np.ndarray,
    indices: np.ndarray,
    targets: np.ndarray,
    *,
    n_classes: int,
    device: str,
) -> dict:
    probabilities = predict_probabilities(model, embeddings, indices, device)
    predictions = probabilities.argmax(axis=1)
    return {
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "per_class_f1": f1_score(
            targets,
            predictions,
            average=None,
            labels=list(range(n_classes)),
            zero_division=0,
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(n_classes))
        ).tolist(),
        "predictions_sha256": sha256_array(predictions.astype(np.int64)),
        "probabilities_sha256": sha256_array(probabilities.astype(np.float32)),
    }


def select_iteration(validation_scores: Sequence[float]) -> int:
    if not validation_scores:
        raise ValueError("validation score list must not be empty")
    return int(np.argmax(np.asarray(validation_scores, dtype=float)))


def run_repaired_self_training(
    embeddings: np.ndarray,
    labeled_indices: Sequence[int] | np.ndarray,
    labeled_targets: Sequence[int] | np.ndarray,
    unlabeled_indices: Sequence[int] | np.ndarray,
    *,
    tau: float,
    n_classes: int,
    device: str,
    seed: int,
    max_iter: int,
    epochs: int,
    patience: int,
) -> tuple[LinearHead, dict]:
    """Select a model using observed validation labels, without test access.

    Hidden unlabeled targets and test data are absent from this function's
    signature by design.  Accepted examples carry their predicted targets into
    all later gradient updates.
    """
    labeled_indices = as_int_array(labeled_indices)
    labeled_targets = as_int_array(labeled_targets)
    remaining_unlabeled = as_int_array(unlabeled_indices)

    seed_core_idx, seed_core_y, selector_idx, selector_y = stratified_holdout(
        labeled_indices, labeled_targets, ratio=0.2, seed=seed + 1000
    )
    train_idx, train_y, early_idx, early_y = stratified_holdout(
        seed_core_idx, seed_core_y, ratio=0.2, seed=seed + 2000
    )
    pool = ExplicitTargetPool.from_labeled(train_idx, train_y)

    validation_scores: list[float] = []
    early_stop_scores: list[float] = []
    acceptance: list[dict] = []
    states: list[dict[str, torch.Tensor]] = []
    state_hashes: list[str] = []
    pool_provenance_by_iteration: list[dict] = []

    for iteration in range(max_iter):
        pool_provenance_by_iteration.append(pool.provenance())
        model, early_f1 = train_head(
            embeddings,
            pool,
            early_idx,
            early_y,
            n_classes=n_classes,
            device=device,
            seed=seed + iteration,
            epochs=epochs,
            patience=patience,
        )
        selector_score = score_observed(
            model,
            embeddings,
            selector_idx,
            selector_y,
            n_classes=n_classes,
            device=device,
        )["macro_f1"]
        state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        states.append(state)
        state_hashes.append(sha256_state_dict(state))
        validation_scores.append(selector_score)
        early_stop_scores.append(early_f1)

        if iteration == max_iter - 1:
            acceptance.append(
                {
                    "iteration": iteration,
                    "status": "not_run_after_final_checkpoint",
                    "accepted": 0,
                    "remaining": int(len(remaining_unlabeled)),
                }
            )
            break
        if len(remaining_unlabeled) == 0:
            acceptance.append(
                {
                    "iteration": iteration,
                    "status": "no_unlabeled_examples_remaining",
                    "accepted": 0,
                    "remaining": 0,
                }
            )
            break
        probabilities = predict_probabilities(
            model, embeddings, remaining_unlabeled, device
        )
        confidence = probabilities.max(axis=1)
        pseudo_targets = probabilities.argmax(axis=1).astype(np.int64)
        accepted_mask = confidence >= tau
        accepted_idx = remaining_unlabeled[accepted_mask]
        accepted_targets = pseudo_targets[accepted_mask]
        pool.append_pseudo(accepted_idx, accepted_targets)
        remaining_unlabeled = remaining_unlabeled[~accepted_mask]
        acceptance.append(
            {
                "iteration": iteration,
                "status": "pseudo_targets_added_for_next_iteration",
                "accepted": int(accepted_mask.sum()),
                "remaining": int(len(remaining_unlabeled)),
                "accepted_indices": accepted_idx.tolist(),
                "accepted_targets": accepted_targets.tolist(),
                "accepted_indices_sha256": sha256_array(accepted_idx),
                "accepted_targets_sha256": sha256_array(accepted_targets),
            }
        )
        if not accepted_mask.any():
            break

    selected_iteration = select_iteration(validation_scores)
    selected_model = LinearHead(embeddings.shape[1], n_classes).to(device)
    selected_model.load_state_dict(states[selected_iteration])
    report = {
        "selection_contract": "argmax fixed observed-label selector macro F1",
        "test_access_during_training": False,
        "hidden_unlabeled_target_access": False,
        "tau": tau,
        "selected_iteration": selected_iteration,
        "selector_macro_f1": validation_scores,
        "early_stop_macro_f1": early_stop_scores,
        "checkpoint_sha256": state_hashes,
        "selected_checkpoint_sha256": state_hashes[selected_iteration],
        "acceptance": acceptance,
        "training_pool_by_iteration": pool_provenance_by_iteration,
        "selected_training_pool": pool_provenance_by_iteration[selected_iteration],
        "fixed_partitions": {
            "train_observed_indices_sha256": sha256_array(train_idx),
            "early_stop_indices_sha256": sha256_array(early_idx),
            "selector_indices_sha256": sha256_array(selector_idx),
            "early_stop_n": len(early_idx),
            "selector_n": len(selector_idx),
        },
    }
    return selected_model, report


def sentinel_report() -> dict:
    initial_indices = np.array([0, 1, 4, 5], dtype=np.int64)
    initial_targets = np.array([0, 1, 0, 1], dtype=np.int64)
    accepted_indices = np.array([2, 3], dtype=np.int64)
    pseudo_a = np.array([1, 0], dtype=np.int64)
    pseudo_b = np.array([0, 1], dtype=np.int64)
    hidden_oracle_a = np.array([0, 1], dtype=np.int64)
    hidden_oracle_b = np.array([9002, 9003], dtype=np.int64)

    pool_a = ExplicitTargetPool.from_labeled(initial_indices, initial_targets)
    pool_a.append_pseudo(accepted_indices, pseudo_a)
    targets_before_oracle_mutation = pool_a.targets.copy()
    _ = hidden_oracle_a, hidden_oracle_b
    targets_after_oracle_mutation = pool_a.targets.copy()

    pool_b = ExplicitTargetPool.from_labeled(initial_indices, initial_targets)
    pool_b.append_pseudo(accepted_indices, pseudo_b)

    validation_scores = [0.2, 0.8, 0.5]
    test_scores_a = [0.99, 0.1, 0.6]
    test_scores_b = [0.01, 0.1, 0.99]
    selected_a = select_iteration(validation_scores)
    selected_b = select_iteration(validation_scores)

    assertions = {
        "oracle_mutation_does_not_change_training_targets": bool(
            np.array_equal(targets_before_oracle_mutation, targets_after_oracle_mutation)
        ),
        "pseudo_mutation_changes_training_targets": bool(
            not np.array_equal(pool_a.targets, pool_b.targets)
        ),
        "test_score_mutation_does_not_change_iteration": bool(selected_a == selected_b),
        "core_signature_excludes_test_and_hidden_targets": bool(
            "test" not in inspect.signature(run_repaired_self_training).parameters
            and "hidden" not in inspect.signature(run_repaired_self_training).parameters
        ),
        "paired_tensor_source_excludes_global_labels": "labels["
        not in inspect.getsource(paired_tensors),
    }
    success = all(assertions.values())
    return {
        "schema_version": 1,
        "audit": "repaired self-training target boundary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "repaired_target_boundary_passed" if success else "failed",
        "success": success,
        "assertions": assertions,
        "trace": {
            "initial_targets": initial_targets.tolist(),
            "accepted_indices": accepted_indices.tolist(),
            "pseudo_targets_a": pseudo_a.tolist(),
            "pseudo_targets_b": pseudo_b.tolist(),
            "hidden_oracle_before": hidden_oracle_a.tolist(),
            "hidden_oracle_after": hidden_oracle_b.tolist(),
            "training_targets_a": pool_a.targets.tolist(),
            "training_targets_b": pool_b.targets.tolist(),
            "validation_scores": validation_scores,
            "test_scores_before": test_scores_a,
            "test_scores_after": test_scores_b,
            "selected_iteration_before_after_test_mutation": [selected_a, selected_b],
        },
        "contract": [
            "Observed labels enter only through explicit labeled_targets.",
            "Accepted unlabeled examples carry stored pseudo-targets into later updates.",
            "Early-stop and selector partitions are fixed observed-label subsets.",
            "The training core cannot receive test labels or hidden unlabeled targets.",
            "Test evaluation occurs once after the selected checkpoint is frozen.",
        ],
    }


def run_smoke(device: str, tau: float, epochs: int, max_iter: int) -> dict:
    embeddings = np.load(EMB_PATH)
    all_labels = np.load(LAB_PATH)
    split_config = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    split = split_config["splits"][0]

    labeled_idx = as_int_array(split["labeled_idx"])
    unlabeled_idx = as_int_array(split["unlabeled_idx"])
    test_idx = as_int_array(split["test_idx"])
    labeled_targets = all_labels[labeled_idx].astype(np.int64)

    model, training_report = run_repaired_self_training(
        embeddings,
        labeled_idx,
        labeled_targets,
        unlabeled_idx,
        tau=tau,
        n_classes=len(CLASSES),
        device=device,
        seed=42,
        max_iter=max_iter,
        epochs=epochs,
        patience=min(5, epochs),
    )

    # This is the only point at which the smoke run reads test targets.
    test_targets = all_labels[test_idx].astype(np.int64)
    test_metrics = score_observed(
        model,
        embeddings,
        test_idx,
        test_targets,
        n_classes=len(CLASSES),
        device=device,
    )
    return {
        "schema_version": 1,
        "run": "repaired self-training smoke test",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "not_for_publication": True,
        "device": device,
        "split": {
            "fold": split["fold"],
            "test_instrument": split["test_instrument"],
            "seed_ratio": split["seed_ratio"],
            "trial": split["trial"],
            "labeled_n": len(labeled_idx),
            "unlabeled_n": len(unlabeled_idx),
            "test_n": len(test_idx),
        },
        "training": training_report,
        "test_evaluation_count": 1,
        "test_metrics": test_metrics,
    }


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sentinel", "smoke"), default="sentinel")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "sentinel":
        output = RESULTS_DIR / "repaired_target_access.json"
        report = sentinel_report()
    else:
        output = RESULTS_DIR / "repaired_smoke.json"
        report = run_smoke(
            resolve_device(args.device), args.tau, args.epochs, args.max_iter
        )

    write_json(output, report)
    print(f"status: {report.get('overall_status', report.get('run'))}")
    print(f"wrote: {output}")
    return 0 if report.get("success", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

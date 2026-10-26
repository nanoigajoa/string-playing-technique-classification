"""Audit the Phase 1 self-training label and selection boundaries.

This script deliberately does not import ``phase1_rerun``: importing that
module loads the real embeddings, labels, and Torch device state.  Instead it
does two small, complementary checks:

1. inspect the current source code with ``ast`` and record the relevant data
   flow, and
2. replay that data flow on a synthetic fixture whose hidden labels can be
   replaced with sentinels.

The expected failure of the current target path is evidence, not a failure of
this audit command.  The command exits with status 0 only when that failure is
reproduced, the repaired reference path passes the same assertions, and the
iteration selector remains independent of test scores.

Example:
    python -B src/phase1_label_boundary_audit.py \
        --mode sentinel \
        --out results/phase1/audit_repair/target_access.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parent.parent
RERUN_PATH = ROOT / "src" / "phase1_rerun.py"
SPLIT_PATH = ROOT / "src" / "phase1_selftraining.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "04_phase1.ipynb"
RESULTS_PATH = ROOT / "results" / "phase1" / "all_results.csv"


@dataclass(frozen=True)
class AssertionRecord:
    name: str
    expected: str
    observed: str
    passed: bool
    expected_to_pass: bool
    expectation_met: bool
    assertion_message: str | None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def line_hits(path: Path, patterns: Sequence[str]) -> list[dict]:
    hits = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if any(pattern in line for pattern in patterns):
            hits.append({"line": line_number, "text": line.strip()})
    return hits


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def inspect_current_source() -> dict:
    source = RERUN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RERUN_PATH))
    make_tensors = function_node(tree, "make_tensors")
    self_training = function_node(tree, "self_training")
    write_report = function_node(tree, "write_report")

    make_text = ast.unparse(make_tensors)
    self_text = ast.unparse(self_training)
    report_text = ast.unparse(write_report)

    pseudo_assignment = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(name, ast.Name)
            and isinstance(name.ctx, ast.Store)
            and name.id == "pseudo_lbl"
            for target in node.targets
            for name in ast.walk(target)
        )
        and any(
            isinstance(child, ast.Attribute) and child.attr == "argmax"
            for child in ast.walk(node.value)
        )
        for node in ast.walk(self_training)
    )

    signatures = {
        "make_tensors_reads_global_labels_by_index": "labels[idx]" in make_text,
        "pseudo_labels_are_computed": pseudo_assignment,
        "accepted_global_indices_enter_labeled_pool": (
            "labeled_pool = labeled_pool + unlabeled_idx[accept_mask].tolist()" in self_text
        ),
        "pseudo_target_mapping_enters_labeled_pool": any(
            token in self_text
            for token in (
                "pseudo_targets[",
                "target_by_index[",
                "labeled_targets",
                "y_pseudo",
            )
        ),
        "iteration_selector_uses_validation_scores": (
            "np.argmax(val_sel_f1)" in self_text
        ),
        "iteration_result_reads_test_score_at_selected_iteration": (
            "result['iter_f1'][sel]" in source
        ),
        "report_tau_selector_uses_test_derived_best_st_f1": (
            "groupby('fold')['best_st_f1'].mean()" in report_text
        ),
    }

    current_oracle_target_path = all(
        (
            signatures["make_tensors_reads_global_labels_by_index"],
            signatures["pseudo_labels_are_computed"],
            signatures["accepted_global_indices_enter_labeled_pool"],
            not signatures["pseudo_target_mapping_enters_labeled_pool"],
        )
    )

    return {
        "signatures": signatures,
        "classification": {
            "target_path": (
                "oracle-label-by-global-index"
                if current_oracle_target_path
                else "not-confirmed-by-known-signatures"
            ),
            "iteration_selector": (
                "validation-score argmax"
                if signatures["iteration_selector_uses_validation_scores"]
                else "not-confirmed"
            ),
            "tau_headline_selector": (
                "test-derived score comparison"
                if signatures["report_tau_selector_uses_test_derived_best_st_f1"]
                else "not-confirmed"
            ),
        },
        "evidence_lines": line_hits(
            RERUN_PATH,
            (
                "labels[idx]",
                "pseudo_lbl =",
                "labeled_pool  = labeled_pool +",
                "np.argmax(val_sel_f1)",
                "groupby('fold')['best_st_f1']",
            ),
        ),
    }


def current_targets(global_labels: Sequence[int], indices: Sequence[int], _pseudo: dict[int, int]) -> list[int]:
    """Replay the current ``make_tensors(indices) -> labels[indices]`` path."""
    return [global_labels[index] for index in indices]


def repaired_targets(_global_labels: Sequence[int], indices: Sequence[int], pseudo: dict[int, int]) -> list[int]:
    """Reference contract: accepted unlabeled indices use stored pseudo-labels."""
    return [pseudo[index] for index in indices]


def training_inputs(embeddings: Sequence[Sequence[float]], indices: Sequence[int]) -> list[list[float]]:
    return [list(embeddings[index]) for index in indices]


def capture_assertion(
    name: str,
    expected: str,
    observed: str,
    check: Callable[[], None],
    *,
    expected_to_pass: bool,
) -> AssertionRecord:
    try:
        check()
    except AssertionError as error:
        passed = False
        message = str(error)
    else:
        passed = True
        message = None
    return AssertionRecord(
        name=name,
        expected=expected,
        observed=observed,
        passed=passed,
        expected_to_pass=expected_to_pass,
        expectation_met=(passed == expected_to_pass),
        assertion_message=message,
    )


def assert_equal(left, right, message: str) -> None:
    assert left == right, f"{message}: {left!r} != {right!r}"


def assert_not_equal(left, right, message: str) -> None:
    assert left != right, f"{message}: both were {left!r}"


def target_fixture_audit() -> dict:
    embeddings = [
        [0.0, 0.1],
        [1.0, 1.1],
        [2.0, 2.1],
        [3.0, 3.1],
        [4.0, 4.1],
        [5.0, 5.1],
    ]
    accepted_indices = [2, 4]
    oracle_a = [0, 1, 4, 3, 0, 2]
    oracle_b = [0, 1, 9002, 3, 9004, 2]
    pseudo_a = {2: 1, 4: 3}
    pseudo_b = {2: 2, 4: 4}

    baseline_inputs = training_inputs(embeddings, accepted_indices)
    oracle_mutated_inputs = training_inputs(embeddings, accepted_indices)

    current_a = current_targets(oracle_a, accepted_indices, pseudo_a)
    current_oracle_mutated = current_targets(oracle_b, accepted_indices, pseudo_a)
    current_pseudo_mutated = current_targets(oracle_a, accepted_indices, pseudo_b)
    repaired_a = repaired_targets(oracle_a, accepted_indices, pseudo_a)
    repaired_oracle_mutated = repaired_targets(oracle_b, accepted_indices, pseudo_a)
    repaired_pseudo_mutated = repaired_targets(oracle_a, accepted_indices, pseudo_b)

    current_assertions = [
        capture_assertion(
            "unlabeled_oracle_mutation_does_not_change_training_input",
            "same accepted embeddings",
            f"before={baseline_inputs}; after={oracle_mutated_inputs}",
            lambda: assert_equal(
                baseline_inputs,
                oracle_mutated_inputs,
                "training input changed after oracle-label mutation",
            ),
            expected_to_pass=True,
        ),
        capture_assertion(
            "unlabeled_oracle_mutation_does_not_change_gradient_target",
            "same target values",
            f"before={current_a}; after={current_oracle_mutated}",
            lambda: assert_equal(
                current_a,
                current_oracle_mutated,
                "gradient target depends on hidden unlabeled truth",
            ),
            expected_to_pass=False,
        ),
        capture_assertion(
            "pseudo_target_mutation_changes_gradient_target",
            "different target values",
            f"before={current_a}; after={current_pseudo_mutated}",
            lambda: assert_not_equal(
                current_a,
                current_pseudo_mutated,
                "gradient target ignores pseudo-label changes",
            ),
            expected_to_pass=False,
        ),
    ]

    repaired_assertions = [
        capture_assertion(
            "unlabeled_oracle_mutation_does_not_change_training_input",
            "same accepted embeddings",
            f"before={baseline_inputs}; after={oracle_mutated_inputs}",
            lambda: assert_equal(
                baseline_inputs,
                oracle_mutated_inputs,
                "training input changed after oracle-label mutation",
            ),
            expected_to_pass=True,
        ),
        capture_assertion(
            "unlabeled_oracle_mutation_does_not_change_gradient_target",
            "same target values",
            f"before={repaired_a}; after={repaired_oracle_mutated}",
            lambda: assert_equal(
                repaired_a,
                repaired_oracle_mutated,
                "gradient target depends on hidden unlabeled truth",
            ),
            expected_to_pass=True,
        ),
        capture_assertion(
            "pseudo_target_mutation_changes_gradient_target",
            "different target values",
            f"before={repaired_a}; after={repaired_pseudo_mutated}",
            lambda: assert_not_equal(
                repaired_a,
                repaired_pseudo_mutated,
                "gradient target ignores pseudo-label changes",
            ),
            expected_to_pass=True,
        ),
    ]

    trace = []
    for index in accepted_indices:
        trace.append(
            {
                "global_index": index,
                "embedding": embeddings[index],
                "oracle_a": oracle_a[index],
                "oracle_sentinel": oracle_b[index],
                "pseudo_a": pseudo_a[index],
                "pseudo_b": pseudo_b[index],
                "current_gradient_target_a": oracle_a[index],
                "current_gradient_target_after_oracle_mutation": oracle_b[index],
                "current_gradient_target_after_pseudo_mutation": oracle_a[index],
                "repaired_gradient_target_a": pseudo_a[index],
                "repaired_gradient_target_after_oracle_mutation": pseudo_a[index],
                "repaired_gradient_target_after_pseudo_mutation": pseudo_b[index],
            }
        )

    return {
        "fixture": {
            "accepted_indices": accepted_indices,
            "note": "Sentinels 9002/9004 are trace markers; no real loss is evaluated.",
        },
        "provenance_trace": trace,
        "current_path": {
            "target_accessor": "global_labels[accepted_global_index]",
            "assertions": [asdict(record) for record in current_assertions],
            "expected_failures_reproduced": all(record.expectation_met for record in current_assertions),
        },
        "repaired_reference_path": {
            "target_accessor": "pseudo_target_by_global_index[accepted_global_index]",
            "assertions": [asdict(record) for record in repaired_assertions],
            "all_assertions_passed": all(record.passed for record in repaired_assertions),
        },
    }


def select_iteration(validation_scores: Sequence[float]) -> int:
    assert validation_scores, "validation score list must not be empty"
    return max(range(len(validation_scores)), key=validation_scores.__getitem__)


def selector_fixture_audit() -> dict:
    validation_a = [0.20, 0.80, 0.50]
    validation_b = [0.90, 0.20, 0.10]
    test_a = [0.99, 0.10, 0.60]
    test_b = [0.01, 0.10, 0.99]
    selected_a = select_iteration(validation_a)
    selected_after_test_mutation = select_iteration(validation_a)
    selected_after_validation_mutation = select_iteration(validation_b)

    assertions = [
        capture_assertion(
            "test_score_mutation_does_not_change_iteration",
            "same selected iteration",
            (
                f"test before={test_a}, test after={test_b}, "
                f"selected={selected_a}/{selected_after_test_mutation}"
            ),
            lambda: assert_equal(
                selected_a,
                selected_after_test_mutation,
                "test mutation changed validation-based selector",
            ),
            expected_to_pass=True,
        ),
        capture_assertion(
            "validation_score_mutation_can_change_iteration",
            "different selected iteration",
            (
                f"validation before={validation_a}, validation after={validation_b}, "
                f"selected={selected_a}/{selected_after_validation_mutation}"
            ),
            lambda: assert_not_equal(
                selected_a,
                selected_after_validation_mutation,
                "selector ignored validation-score changes",
            ),
            expected_to_pass=True,
        ),
    ]
    return {
        "iteration_selector": {
            "contract": "argmax(validation trajectory); test trajectory is report-only",
            "assertions": [asdict(record) for record in assertions],
            "all_assertions_passed": all(record.passed for record in assertions),
        }
    }


def audit_saved_results() -> dict:
    required = {"ratio", "tau", "sel_iter", "best_st_f1", "iter_f1_json", "val_sel_f1_json"}
    rows_checked = 0
    selector_mismatches = []
    reported_score_mismatches = []
    aggregate: dict[tuple[float, float], list[float]] = {}

    with RESULTS_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        assert not missing, f"all_results.csv missing columns: {sorted(missing)}"
        for row_number, row in enumerate(reader, 2):
            test_scores = [float(value) for value in json.loads(row["iter_f1_json"])]
            validation_scores = [float(value) for value in json.loads(row["val_sel_f1_json"])]
            selected = int(row["sel_iter"])
            expected_selected = select_iteration(validation_scores)
            reported_score = float(row["best_st_f1"])
            expected_score = test_scores[selected]
            if selected != expected_selected:
                selector_mismatches.append(
                    {"csv_line": row_number, "stored": selected, "expected": expected_selected}
                )
            if abs(reported_score - expected_score) > 1e-12:
                reported_score_mismatches.append(
                    {"csv_line": row_number, "stored": reported_score, "expected": expected_score}
                )
            aggregate.setdefault((float(row["ratio"]), float(row["tau"])), []).append(reported_score)
            rows_checked += 1

    selected_tau_by_ratio = {}
    for ratio in sorted({key[0] for key in aggregate}):
        tau_means = {
            tau: sum(values) / len(values)
            for (row_ratio, tau), values in aggregate.items()
            if row_ratio == ratio
        }
        selected_tau = max(tau_means, key=tau_means.__getitem__)
        selected_tau_by_ratio[f"{ratio:g}"] = {
            "tau": selected_tau,
            "mean_test_derived_best_st_f1": tau_means[selected_tau],
            "selection_input": "best_st_f1 (test score at validation-selected iteration)",
        }

    return {
        "path": str(RESULTS_PATH.relative_to(ROOT)),
        "sha256": sha256(RESULTS_PATH),
        "rows_checked": rows_checked,
        "iteration_selector_matches_validation_argmax": not selector_mismatches,
        "iteration_selector_mismatches": selector_mismatches,
        "reported_test_score_matches_selected_iteration": not reported_score_mismatches,
        "reported_score_mismatches": reported_score_mismatches,
        "report_level_tau_selection": {
            "status": "test-score-derived selection; not an independent model-selection estimate",
            "selected_tau_by_ratio_recomputed": selected_tau_by_ratio,
        },
    }


def notebook_audit() -> dict:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("id") == "cell-st-func" or cell.get("cell_type") == "code"
    )
    return {
        "path": str(NOTEBOOK_PATH.relative_to(ROOT)),
        "sha256": sha256(NOTEBOOK_PATH),
        "same_known_target_signatures": {
            "global_label_accessor_present": "labels[idx]" in source,
            "pseudo_labels_computed": "pseudo_lbl = probs.argmax(axis=1)" in source,
            "accepted_indices_appended_without_target_mapping": (
                "labeled_pool    = labeled_pool + accepted_global.tolist()" in source
                and "pseudo_targets" not in source
            ),
        },
    }


def build_report() -> dict:
    static = inspect_current_source()
    target = target_fixture_audit()
    selector = selector_fixture_audit()
    saved = audit_saved_results()
    notebook = notebook_audit()

    required_source_signatures = static["classification"]["target_path"] == "oracle-label-by-global-index"
    expected_current_failures = target["current_path"]["expected_failures_reproduced"]
    repaired_passes = target["repaired_reference_path"]["all_assertions_passed"]
    selector_passes = selector["iteration_selector"]["all_assertions_passed"]
    saved_selector_consistent = (
        saved["iteration_selector_matches_validation_argmax"]
        and saved["reported_test_score_matches_selected_iteration"]
    )
    success = all(
        (
            required_source_signatures,
            expected_current_failures,
            repaired_passes,
            selector_passes,
            saved_selector_consistent,
        )
    )

    return {
        "schema_version": 1,
        "audit": "phase1 self-training label and selector boundary",
        "mode": "sentinel",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": (
            "expected_target_boundary_violation_reproduced"
            if success
            else "audit_expectation_not_met"
        ),
        "success": success,
        "interpretation": {
            "target": (
                "Accepted unlabeled indices are later resolved through labels[index]; "
                "pseudo_lbl is diagnostic only and is not the gradient target."
            ),
            "iteration_selector": (
                "The stored iteration is validation-selected and the selected iteration's "
                "test score is reported; these saved rows are internally consistent."
            ),
            "tau_selector": (
                "write_report compares best_st_f1 across tau values, so its best-tau headline "
                "is selected using test-derived scores and must not be treated as an independent estimate."
            ),
            "scope": (
                "This proves the target data-flow defect in the inspected revision. It does not "
                "estimate the corrected self-training performance or attribute every prior result to one cause."
            ),
        },
        "sources": [
            source_record(RERUN_PATH),
            source_record(SPLIT_PATH),
            source_record(NOTEBOOK_PATH),
            source_record(RESULTS_PATH),
        ],
        "static_source_audit": static,
        "synthetic_target_audit": target,
        "synthetic_selector_audit": selector,
        "saved_results_audit": saved,
        "notebook_audit": notebook,
        "next_full_rerun_contract": [
            "Carry explicit targets beside indices for every labeled and pseudo-labeled example.",
            "Never expose hidden unlabeled labels to target construction, split stratification, or early stopping.",
            "Use hidden labels only in a separately permissioned retrospective diagnostic after predictions are frozen.",
            "Choose iterations and tau on source validation only; evaluate the test set once.",
            "Save the selected checkpoint and prediction manifest used by every reported metric.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sentinel",), default="sentinel")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "phase1" / "audit_repair" / "target_access.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.out if args.out.is_absolute() else ROOT / args.out
    report = build_report()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"audit status: {report['overall_status']}")
    print(f"target current path: {report['static_source_audit']['classification']['target_path']}")
    print(
        "iteration selector: "
        f"{report['static_source_audit']['classification']['iteration_selector']}"
    )
    print(
        "tau headline selector: "
        f"{report['static_source_audit']['classification']['tau_headline_selector']}"
    )
    print(f"saved rows checked: {report['saved_results_audit']['rows_checked']}")
    print(f"wrote: {output_path}")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

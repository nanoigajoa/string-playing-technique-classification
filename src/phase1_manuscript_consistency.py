"""Check canonical manuscript tables against saved prediction summaries."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "phase1" / "canonical"
DOCUMENTS = [
    ROOT / "연구과정" / "주법분류_논문초안_국문_v1_canonical.md",
    ROOT / "연구과정" / "주법분류_논문초안_영문_v1_canonical.md",
    ROOT / "README.md",
]


def table_rows(document):
    rows = []
    for line in document.splitlines():
        if not line.startswith("|") or not re.match(r"^\| (5|10|20|50)% \|", line):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        rows.append(fields)
    return rows


def main():
    model = pd.read_csv(BASE / "mert_svm" / "canonical_summary.csv").set_index("seed_ratio")
    self_training = pd.read_csv(BASE / "self_training" / "canonical_summary.csv")
    self_training = self_training.pivot(index="ratio", columns="condition", values="macro_f1_fold_mean")
    intervals = pd.read_csv(BASE / "verification" / "bootstrap_sensitivity.csv")
    intervals = intervals[(intervals.unit == "instrument_type") & (intervals.measure == "mert_minus_svm")].set_index("seed_ratio")
    cnn = pd.read_csv(BASE / "cnn" / "canonical_results.csv")["macro_f1"].mean()
    expected_model = {}
    expected_self = {}
    for ratio in (0.05, 0.10, 0.20, 0.50):
        percent = round(ratio * 100)
        row = model.loc[ratio]
        ci = intervals.loc[ratio]
        expected_model[percent] = [
            f"{row.mert_macro_f1_fold_mean:.4f}",
            f"{row.svm_macro_f1_fold_mean:.4f}",
            f"{row.mert_minus_svm:+.4f}",
            f"[{ci.ci_low:+.4f}, {ci.ci_high:+.4f}]",
        ]
        expected_self[percent] = [f"{self_training.loc[ratio, condition]:.4f}"
                                  for condition in ("supervised", "pseudo", "oracle")]
    for path in DOCUMENTS:
        document = path.read_text(encoding="utf-8")
        rows = table_rows(document)
        assert len(rows) == (4 if path.name == "README.md" else 8), (path, len(rows))
        for fields in rows:
            percent = int(fields[0].rstrip("%"))
            if len(fields) == 5:
                assert [value.replace("−", "-") for value in fields[1:]] == expected_model[percent], (path, percent, fields[1:])
            elif len(fields) == 4:
                if path.name == "README.md":
                    assert [value.replace("−", "-") for value in fields[1:]] == expected_model[percent][:3], (path, percent, fields[1:])
                else:
                    assert fields[1:] == expected_self[percent], (path, percent, fields[1:])
            else:
                raise AssertionError((path, fields))
        assert f"{cnn:.4f}" in document, path
        if path.name == "README.md":
            assert expected_self[5][0] in document and expected_self[5][1] in document, path
        for unsupported in ("0.745 → 0.638 |", "0.900 | 0.775", "statistically indistinguishable"):
            assert unsupported not in document, (path, unsupported)
        print(f"PASS: {path.relative_to(ROOT)}: {len(rows)} canonical rows and CNN {cnn:.4f}")
    assert np.isclose(model.loc[0.5, "mert_minus_svm"], model.loc[0.5, "mert_macro_f1_fold_mean"] - model.loc[0.5, "svm_macro_f1_fold_mean"])


if __name__ == "__main__":
    main()

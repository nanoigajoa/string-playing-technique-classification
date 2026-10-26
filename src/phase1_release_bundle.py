"""Prepare an allowlisted local reproduction bundle; never upload artifacts.

Building explicit partition records requires the project's training environment.
Using the finished bundle requires only its small analysis dependency list.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_canonical_mert_svm import partition
from phase1_selftraining_repaired import stratified_holdout

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/phase1/canonical"
SOURCES = [
    "phase1_release_reproduce.py", "phase1_canonical_figures.py",
    "phase1_canonical_mert_svm.py", "phase1_canonical_cnn.py", "phase1_canonical_selftraining.py",
    "phase1_canonical_verify.py", "phase1_canonical_selftraining_verify.py",
    "phase1_selftraining_repaired.py", "phase1_train_only.py", "features.py", "dataset.py", "model.py", "build_dataset.py",
]
TABLES = {
    "mert_svm": ["canonical_results.csv", "canonical_summary.csv", "RUN_INDEX.json"],
    "cnn": ["canonical_results.csv", "RUN_INDEX.json"],
    "self_training": ["canonical_results.csv", "canonical_summary.csv", "per_class_f1.csv", "confusion_counts.csv", "VERIFICATION.json"],
    "verification": ["per_class_f1.csv", "confusion_counts.csv", "bootstrap_sensitivity.csv", "verification.json", "report.md", "excluded_models.md"],
}

README = """# Canonical string-technique evaluation: reproduction bundle

Local release candidate. No public deposit or new code license declaration has
been made. This bundle covers five classes, 3,402 clips, four held-out instrument
types, three trials and four observed-label budgets.

## Recompute reported statistics

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python -B src/phase1_release_reproduce.py
.venv/bin/python -B src/phase1_canonical_figures.py --tables-root reproduced --output-dir reproduced/figures
```

The first script verifies the release file hashes, 100 saved prediction files
(244 model prediction vectors), all type holdouts and explicit observed-label
partitions, validation-selected iterations, and stored pseudo-target pool traces.
It recomputes ten canonical CSV tables, including macro F1, per-class F1,
confusion counts and the three resampling analyses, then compares every row to
the supplied reference CSVs (absolute numeric tolerance 1e-12). The reference
CSVs are used for comparison after recomputation, not as metric inputs.

Outputs go into `reproduced/`; original reference results are preserved.
`REPRODUCTION.json` distinguishes this calculation from checkpoint inference.
The second command draws three figures from the recomputed CSVs. Font/layout
and PDF metadata can vary across systems; numerical reproduction does not
require byte-identical figure files.

## Reproduction scope

| Level | Included / validated |
|---|---|
| Statistics from fixed saved predictions | Included; separate environment validation is recorded outside the archive alongside its SHA-256 |
| Explicit observed partitions and stored pseudo-target ownership | Included and checked without a label-reading training routine |
| Checkpoint inference | Previously verified locally; checkpoints and embeddings are absent from this bundle |
| Full feature extraction and retraining | Audited source code supplied; requires licensed source audio and additional inputs, and was not rerun from this bundle |

No audio, mel cache, embedding cache, model checkpoint, credential file, or
local absolute audio path is included. Metadata preserves stable sample indices,
filenames, class labels, instrument types and filename-derived grouping proxies.
The labels are evaluation metadata and must not be passed into the pseudo-target
training pool. Oracle labels are privileged diagnostic information.

Original training entry points still expect `data/mert_meta.csv` with locally
restored `filepath` values, `mert_labels.npy`, `features_full.csv`, and
`mert_embeddings.npy`. Restore rows using `data/release_meta.csv` sample indices;
do not reorder them. The CNN builds its own mel cache from local paths. Obtain
OrchideaSOL separately; hashes of omitted inputs/checkpoints in the original
manifests are historical provenance, not evidence that this bundle contains
or replays them. The original checkpoint replay scripts require those omitted
files even when their metric calculations are otherwise available here.

## Environment

`requirements-analysis.txt` pins the packages used for this release check.
`environment-observed.json` records the packaging environment.
`requirements-training-observed.txt` records relevant currently installed
training packages. Earlier training runs did not preserve a complete environment
lock, so the observed environment must not be described as a recovered original
training environment. The repository's historical requirements are retained as
`requirements-project-historical.txt` for comparison.

## Interpretation and access

Read `CONTRIBUTION.md` for supported claims and remaining limitations.
There are four instrument TYPES, without verified performer/session/physical
instrument identifiers. All bootstrap results are conditional sensitivity
analyses of fixed predictions. Intervals containing zero do not prove model
equivalence. The custom four-fold holdout is not OrchideaSOL's recommended
five-fold protocol. Dataset exploration preceded the canonical runs; the
historical CNN manifest wording “preregistered” does not establish an independent
preregistered confirmation.

See `SOURCE_NOTICE.md` for external data access and pending release declarations.
"""

NOTICE = """# Sources and release declarations

OrchideaSOL source: https://doi.org/10.5281/zenodo.3686252
Cella, Ghisi, Lostanlen, Lévy, Fineberg and Maresz (2020), version 1.0.
The source record identifies the audio's Ircam Forum license and metadata's
CC BY 4.0 license. This bundle contains derived subset metadata and predictions,
not source audio. Changes to metadata: removed local filepath, retained row order,
and added explicit sample_index. recording_id is a filename-derived proxy.

MERT reference: Li et al. (2024), ICLR.
Model record: https://huggingface.co/m-a-p/MERT-v1-95M
No pretrained model weights or embeddings are distributed in this packet.

All original run manifests are copied byte-for-byte. A separate release manifest
describes the transformed metadata and the explicit partition exports. No new
code license, author declarations, archival DOI or public access URL has been
invented. Those release declarations remain to be supplied before publication.
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(root, relative):
    source = ROOT / relative
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def export_partitions(meta):
    labels = meta.label_id.to_numpy(dtype=np.int64)
    splits = json.loads((ROOT / "data/split_configs.json").read_text())["splits"]
    exported = {name: {} for name in ("mert_svm", "self_training", "cnn")}
    for split in splits:
        key = f"r{round(split['seed_ratio'] * 100):02d}_f{split['fold']}_t{split['trial']}"
        exported["mert_svm"][key] = {name: value.tolist() for name, value in partition(split, labels).items()}
        manifest = json.loads((BASE / "self_training/manifests" / f"{key}.json").read_text())
        observed = np.asarray(split["labeled_idx"], dtype=np.int64)
        core, core_y, selector, _ = stratified_holdout(observed, labels[observed], ratio=0.2, seed=manifest["seed"] + 1000)
        train, _, early, _ = stratified_holdout(core, core_y, ratio=0.2, seed=manifest["seed"] + 2000)
        exported["self_training"][key] = {"train": train.tolist(), "early_stop": early.tolist(),
                                           "val_select": selector.tolist(), "test": split["test_idx"]}
    for path in sorted((BASE / "cnn/manifests").glob("*.json")):
        m = json.loads(path.read_text())
        test = np.flatnonzero(meta.instrument.to_numpy() == m["held_out_instrument"])
        exported["cnn"][m["fold_key"]] = {"train": np.setdiff1d(np.arange(len(meta)), test).tolist(), "test": test.tolist()}
    return exported


def archive_bundle(root, archive):
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                # Stable timestamps and permissions; no local host attributes.
                info = zipfile.ZipInfo(str(Path("canonical-reproduction") / path.relative_to(root)), (2026, 10, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                handle.writestr(info, path.read_bytes())
    archive.with_suffix(".zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "release/canonical-reproduction")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("Choose a new output directory; release candidates are not overwritten")
    out.mkdir(parents=True)
    for name in SOURCES:
        copy(out, Path("src") / name)
    copy(out, Path("data/split_configs.json"))
    meta = pd.read_csv(ROOT / "data/mert_meta.csv")
    release_meta = meta[["filename", "label", "label_id", "instrument", "recording_id"]].copy()
    release_meta.insert(0, "sample_index", np.arange(len(meta)))
    release_meta.to_csv(out / "data/release_meta.csv", index=False)
    (out / "data/release_partitions.json").write_text(json.dumps(export_partitions(meta), separators=(",", ":")) + "\n")
    for family in ("mert_svm", "cnn", "self_training"):
        for folder, suffix in (("manifests", "json"), ("predictions", "npz")):
            for path in sorted((BASE / family / folder).glob(f"*.{suffix}")):
                copy(out, path.relative_to(ROOT))
    for family, names in TABLES.items():
        for name in names:
            copy(out, (BASE / family / name).relative_to(ROOT))
    for path in sorted((BASE / "manuscript/figures").glob("*")):
        if path.suffix in (".png", ".pdf", ".json"):
            copy(out, path.relative_to(ROOT))
    copy(out, (BASE / "manuscript/review_preview.pdf").relative_to(ROOT))
    for path in sorted((ROOT / "연구과정").glob("*v1_canonical.*")):
        if path.suffix in (".md", ".docx"):
            copy(out, path.relative_to(ROOT))
    copy(out, Path("docs/phase1_contribution_scope.md"))
    (out / "CONTRIBUTION.md").write_text((ROOT / "docs/phase1_contribution_scope.md").read_text())
    (out / "README.md").write_text(README)
    (out / "SOURCE_NOTICE.md").write_text(NOTICE)
    training_packages = ["numpy", "pandas", "matplotlib", "torch", "scikit-learn", "librosa", "transformers", "scipy", "joblib", "soundfile", "tqdm"]
    observed = {name: importlib.metadata.version(name) for name in training_packages}
    shutil.copyfile(ROOT / "requirements-phase1-analysis.txt", out / "requirements-analysis.txt")
    (out / "requirements-training-observed.txt").write_text("".join(f"{name}=={version}\n" for name, version in observed.items()))
    shutil.copyfile(ROOT / "requirements.txt", out / "requirements-project-historical.txt")
    (out / "environment-observed.json").write_text(json.dumps({"observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
        "packages": observed, "scope": "packaging environment; original complete training environment was not recorded"}, indent=2) + "\n")
    files = sorted(path for path in out.rglob("*") if path.is_file())
    forbidden = re.compile(r"/Users/|/home/|file://|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY")
    for path in files:
        if path.suffix in (".py", ".md", ".json", ".csv", ".txt") and forbidden.search(path.read_text()):
            raise ValueError(f"Local path or sensitive marker in {path.relative_to(out)}")
        if path.suffix in (".wav", ".flac", ".npy", ".pt", ".pkl"):
            raise ValueError("Unexpected raw audio/cache/checkpoint in release")
    manifest = {"schema_version": 1, "status": "local_release_candidate", "published": False,
                "files": {str(path.relative_to(out)): sha256(path) for path in files},
                "metadata_transformation": "removed local filepath; stable row order; sample_index added",
                "original_metadata_sha256": sha256(ROOT / "data/mert_meta.csv"),
                "builder_code_sha256": sha256(Path(__file__)),
                "scope": "fixed saved-prediction statistics; checkpoint inference and retraining not included",
                "checks": {"prediction_files": 100, "local_absolute_paths_in_text": 0,
                           "source_audio_caches_checkpoints_included": False},
                "external_publication_pending": ["code license and author declarations", "persistent anonymous/public archive", "final file review"]}
    (out / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive_bundle(out, out.parent / "canonical-reproduction.zip")
    print(f"Prepared {len(files) + 1} allowlisted files, archive and SHA-256 sidecar.")


if __name__ == "__main__":
    main()

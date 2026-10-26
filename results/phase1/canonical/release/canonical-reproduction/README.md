# Canonical string-technique evaluation: reproduction bundle

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

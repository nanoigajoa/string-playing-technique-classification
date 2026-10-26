# Phase 1 repair verification status

Date: 2026-09-20

## Verification 1 — self-training target boundary

Status: **structural and smoke-test pass; publication rerun not started**.

The historical implementation and outputs were left unchanged. The repaired
path is implemented in `src/phase1_selftraining_repaired.py` and uses paired
`(index, target)` training records. Initially observed examples retain observed
targets; accepted unlabeled examples retain the pseudo-target predicted at the
time of acceptance. Fixed observed-label subsets are reserved for early
stopping and iteration selection. The training core does not accept hidden
unlabeled targets or test targets, and the smoke test evaluates the selected
checkpoint on the test fold once.

The sentinel audit passed all five assertions in
`repaired_target_access.json`:

- mutating hidden oracle labels did not change training targets;
- mutating pseudo-targets did change training targets;
- mutating test scores did not change the selected iteration;
- the training-core signature excludes test and hidden targets;
- tensor construction does not read a global label array.

The real-data smoke test in `repaired_smoke.json` used the first configured
split, `tau=0.7`, two iterations, and 30 maximum epochs on CPU. Iteration zero
accepted 11 pseudo-labeled examples. The selected iteration-one checkpoint was
therefore trained on 78 observed-label and 11 pseudo-label examples. The test
fold was evaluated once after checkpoint selection. A repeated run produced
the same selected-checkpoint hash
`6b96a50d1c89e0c011e7bef7c7435c7a81d351be520f95602caccc308c66a864`
and prediction hash
`05eef74794c3cd1163b8edefd039f79121610efd52c05287d7f9f9ac8e028121`.

The smoke-test performance is explicitly not a publication result. It covers
one split, one threshold, shortened training, and two iterations. No headline
metric or old/new performance difference may be derived from it.

## Commands used

```bash
venv/bin/python -m py_compile src/phase1_selftraining_repaired.py
venv/bin/python -B src/phase1_selftraining_repaired.py --mode sentinel
venv/bin/python -B src/phase1_selftraining_repaired.py \
  --mode smoke --device cpu --tau 0.7 --epochs 30 --max-iter 2
```

## Next gate

Verification 2 was audited next; see the section below. A publication-scale
rerun remains blocked until the unsafe legacy training entry points are
migrated or replaced.

## Verification 2 — train-only learned preprocessing

Status: **audit and repaired primitives pass; one repaired real-data analysis
passes; legacy MERT/CNN reruns remain blocked**.

The static audit found nine legacy boundary violations. Global class weights
occur in `notebooks/03_modeling_v2.ipynb`, `notebooks/04_phase1.ipynb`,
`src/phase1_rerun.py`, `src/phase1_mert_tuning_audit.py`,
`src/phase1_layer_probe.py`, `src/phase1_supplementary_4class.py`, and
`src/phase1_lora.py`. The historical duration script also fits its linear
residualizer on the full dataset before LOSO and chooses its majority class
from the full evaluation label distribution. These files and their existing
outputs are preserved as historical evidence, not approved inputs to the
canonical publication rerun.

The audit also confirmed four safe structures: the SVM scaler and balanced
class weighting are inside pipelines fit on training indices in the tuning,
seed-comparison, and cluster-bootstrap paths; the repaired self-training path
derives weights from its explicit training pool. No learned feature selector
was found in the scoped Phase 1 paths. Named onset/chroma column subsets are
prespecified subsets rather than label-fit selectors.

`src/phase1_train_only.py` now supplies explicit train-only class weighting,
scaling, and duration residualization. The synthetic mutation sentinel in
`train_only_preprocessing.json` passed all nine assertions: held-out label,
feature, and duration mutations left repaired artifact hashes unchanged;
training-data mutations changed them; and the same held-out mutations changed
the reproduced global-fit artifacts.

The real split audit in `train_only_real_folds.json` reconstructed all 48
configured splits for both the five-class and ordinario-excluded four-class
tasks (96 task-splits). Every train, early-stop, selection, and test partition
was pairwise disjoint, and every class-weight, scaler, and residualizer hash
was unchanged after held-out-fold mutation. This audit fits preprocessing only;
it does not train a classifier or evaluate test labels.

The duration analysis was rerun through
`src/phase1_duration_baseline_repaired.py`. All 24 fitted fold artifacts (six
conditions including the majority baseline across four outer folds) are logged
as train-only in `duration_baseline_repaired_manifest.json`. The repaired macro
F1 values are: duration only 0.4004, full 245 features 0.9065, fold-local
duration-residualized full features 0.8891, onset only 0.6733, and fold-local
duration-residualized onset features 0.6936. The historical global-fit values
for the two residualized conditions were 0.8934 and 0.6937. Thus the full
residualized score changed by -0.0043 and the onset residualized score by
-0.0001. The qualitative observation that performance does not collapse after
linear duration adjustment survives this repair, but the repaired prediction
file—not the historical CSV—must be the source for any later claim.

Unsafe legacy MERT/CNN entry points remain quarantined. Verification 2 is
therefore complete as an audit and repair-contract test, but it is **not yet a
publication-rerun pass**. Their model-training entry points must consume
`balanced_class_weights(train_targets, n_classes)` (or an equivalent explicit
train-only implementation), and then be rerun before their metrics can enter a
canonical table.

## Verification 2 commands

```bash
venv/bin/python -m py_compile \
  src/phase1_train_only.py \
  src/phase1_train_only_audit.py \
  src/phase1_train_only_real_audit.py \
  src/phase1_duration_baseline_repaired.py
venv/bin/python -B src/phase1_train_only_audit.py
venv/bin/python -B src/phase1_train_only_real_audit.py
venv/bin/python -B src/phase1_duration_baseline_repaired.py
```

## Next gate

Before verification 3, create canonical repaired MERT/CNN training entry
points, run the required folds without overwriting historical outputs, and
bind each selected checkpoint to its prediction file and hash. Existing
`mert_tuning_audit.csv`, `layer_probe.csv`, `lora_results.csv`, CNN notebook
outputs, and supplementary four-class outputs are not eligible for the final
table until that migration and rerun are complete.

## 2026-09-28 continuation

The above "next gate" paragraph records the 2026-09-20 state. Since then,
`src/phase1_canonical_mert_svm.py` and `src/phase1_canonical_cnn.py` completed
the canonical MERT/SVM (48 splits) and CNN (four folds) runs. The repaired
self-training entry point completed matched supervised/pseudo/oracle diagnostics
for all 48 splits. `src/phase1_canonical_verify.py` replayed 100 MERT/SVM/CNN
checkpoints and `src/phase1_canonical_selftraining_verify.py` replayed 144
self-training checkpoints. Stored predictions, metrics, confusion matrices and
manifest hashes matched. The corresponding canonical report is
`results/phase1/canonical/verification/report.md`.

Verification 3 is complete for these canonical models. Verification 4 has a
paired clip/recording-proxy/instrument-type resampling sensitivity analysis for
MERT versus SVM. The old v0 manuscript is marked historical. New Korean and
English v1 canonical drafts and the README use verified rows only, with a
numeric consistency script. Layer-probe, LoRA, and external-transfer claims
remain outside v1 until separately repaired and rerun. The v1 drafts still
need literature review and journal formatting before submission.

## 2026-09-30 manuscript continuation

Both v1 manuscripts now include six primary-source references, exact feature
and training configurations, and three figures generated from the canonical
CSVs. The distinction between OrchideaSOL's recommended five-fold protocol
and this study's four instrument-type holdouts is explicit. Checkpoint replay
counts above are corrected to 100 (48 paired models plus four CNN folds).

`src/phase1_canonical_figures.py` records source/input/output hashes and
exports 300 dpi PNG and vector PDF figures. `src/phase1_manuscript_export.py`
exports both DOCX drafts and an anonymous English HTML preview. The printed
eight-page PDF was visually reviewed; both DOCX files contain two editable
tables and three drawings. Canonical table consistency passed for both
manuscripts and README. Remaining submission work is recorded in
`results/phase1/canonical/manuscript/READINESS.md`; no submission or public
artifact release has been made.

## 2026-10-01 release preparation

An allowlisted local release candidate now includes sanitized row-aligned
metadata, explicit observed partitions, original manifests and predictions,
audited code, tables, figures and the revised manuscripts. Source audio,
embedding/mel caches and checkpoints are omitted. The original run manifests
retain their bytes; release transformations have their own manifest.

`src/phase1_release_reproduce.py` recomputed ten canonical tables from 100
prediction files (244 model prediction vectors), checked observed-label
partitions and stored pseudo-target pool provenance, and compared all table
rows to the reference outputs. The archive was extracted and verified in a
separate environment without torch. Figures were rendered from those
recomputed tables. Archive SHA-256 and validation environment are recorded in
`results/phase1/canonical/release/VALIDATION.json`.

The original project requirements differ from packages currently installed.
The release records both, uses a separately verified analysis dependency lock,
and does not claim that the complete original training environment was
recovered. `docs/phase1_contribution_scope.md` and both manuscripts now state
the contribution as an auditable case study conditional on four instrument
types. Code licensing, author declarations and persistent public deposit
remain pending; the candidate has not been uploaded or submitted.

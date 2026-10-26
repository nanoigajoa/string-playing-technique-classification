# Canonical Phase 1 verification (2026-09-28)

Run: `venv/bin/python -B src/phase1_canonical_verify.py --draws 1000`.
The verification reloaded **all 48 MERT/SVM pairs and all four CNN checkpoints** on CPU. Every replayed prediction matched its saved prediction array. All checkpoint, prediction, split-index, class-weight and metric checks passed. The original training scripts and historical result files were not overwritten.

## Canonical results

Macro F1 is the mean of the four held-out **instrument-type** fold means. For MERT/SVM each fold first averages its three trials. CNN uses all available training labels and is therefore a different label-budget condition.

| Observed-label budget | MERT head | SVM | MERT − SVM |
|---:|---:|---:|---:|
| 5% | 0.6655 | 0.6613 | +0.0042 |
| 10% | 0.7634 | 0.7542 | +0.0092 |
| 20% | 0.8242 | 0.8316 | −0.0074 |
| 50% | 0.8599 | 0.8681 | −0.0082 |

Full-label CNN: **0.8729** fold-mean macro F1. The lowest 50%-budget MERT class F1 values are `col_legno_battuto` 0.7423 and `sul_ponticello` 0.7678; for SVM they are `sul_ponticello` 0.7386 and `col_legno_battuto` 0.8248. Exact class scores and confusion counts are in `per_class_f1.csv` and `confusion_counts.csv`. Confusion counts sum repeated trial predictions and must not be described as distinct audio clips.

## Paired resampling sensitivity

`bootstrap_sensitivity.csv` contains 1,000 paired percentile resamples at each label budget. The same sampled test positions are used for MERT and SVM and across trials within an instrument fold. The point statistic remains trial mean, then mean of four instrument-type folds. Intervals are descriptive sensitivity calculations conditional on these four types; they do not establish uncertainty for new players, physical instruments, or recording sessions.

| Budget | Clip 95% interval, MERT − SVM | Recording-proxy interval | Instrument-type interval |
|---:|---:|---:|---:|
| 5% | [−0.0093, +0.0199] | [−0.0135, +0.0198] | [−0.0544, +0.0451] |
| 10% | [−0.0073, +0.0249] | [−0.0058, +0.0250] | [−0.0514, +0.0698] |
| 20% | [−0.0216, +0.0076] | [−0.0235, +0.0080] | [−0.0515, +0.0403] |
| 50% | [−0.0212, +0.0054] | [−0.0224, +0.0059] | [−0.0345, +0.0179] |

All four instrument-type intervals include zero. This does **not** prove model equivalence. Equivalence would require a prespecified meaningful margin and an appropriate design. `recording_id` is derived from filenames and is only a grouping proxy, not a verified session or performer identifier. Resampling four instrument types cannot manufacture evidence about a wider population of performers.

## Repaired self-training (matched rerun)

`src/phase1_canonical_selftraining.py` ran all 48 configured splits with fixed `tau=0.8`, at most five iterations, up to 150 epochs and patience 15. The supervised, actual pseudo-target, and explicitly privileged oracle conditions share the observed-label budget, split, seed, and validation selection rule. Each selected checkpoint was persisted and reloaded before test prediction. `src/phase1_canonical_selftraining_verify.py` replayed all **144 checkpoints** and matched their saved predictions, manifest metrics, per-class scores, and confusion matrices. It also reconstructed every pseudo-label training pool from the stored `(sample index, predicted target)` pairs and checked each pool hash. The second complete rerun with this trace produced the same prediction-set hash as the first (`268c7a7f1b4af7d8dbc026e615c6e1cfb7d24a3b1751ccac0d6d5ff62a3dc3dc`).

| Observed-label budget | Supervised | True pseudo-label | Privileged oracle diagnostic |
|---:|---:|---:|---:|
| 5% | 0.6686 | 0.6364 | 0.6489 |
| 10% | 0.7295 | 0.7357 | 0.7356 |
| 20% | 0.7639 | 0.7737 | 0.7919 |
| 50% | 0.8361 | 0.8409 | 0.8350 |

These are four-instrument-type fold means after averaging three trials per fold. The oracle condition uses hidden unlabeled targets overtly as a diagnostic and is neither deployable nor mathematically guaranteed to exceed the other conditions after training and validation selection. Its result must not be called a performance ceiling. The historical 5% `0.745 → 0.638` comparison cannot isolate test-selection leakage: the previous pipeline also used hidden training targets and differed in training seed/configuration. No causal before/after effect size is claimed from that comparison.

At 5%, pseudo-label self-training's weakest class is `sul_ponticello` (fold-mean F1 0.3324), followed by `col_legno_battuto` (0.4237); at 50% they reach 0.6912 and 0.7327. Full class scores and confusion counts are in `../self_training/per_class_f1.csv` and `../self_training/confusion_counts.csv`.

## Four verification gates

1. **Self-training target boundary:** the repaired implementation passed sentinel and one-split repeatability checks in `results/phase1/audit_repair/VERIFICATION_STATUS.md`. The full matched supervised/pseudo/oracle rerun and 144-checkpoint replay now pass. Historical self-training performance and its claimed causal before/after drop remain **withdrawn from publication use**.
2. **Train-only preprocessing:** the real-fold mutation audit passed, and the canonical MERT/SVM/CNN runs derive class weights or scalers from training indices. This verification reconstructed every split and class-weight hash. Other historical layer-probe, LoRA and four-class results remain unqualified until separately migrated and rerun.
3. **Checkpoint → prediction → result provenance:** passed for the canonical MERT/SVM/CNN runs. `verification.json` records the prediction-set hashes. Checkpoint replay, saved predictions, per-class scores, confusion counts and published canonical CSV rows match.
4. **Inference and manuscript consistency:** clip, filename-proxy and instrument-type resampling are now compared from the **same saved MERT/SVM predictions**. The README and new Korean/English v1 drafts use the verified canonical table; `src/phase1_manuscript_consistency.py` checks their principal numerical rows. The v0 Korean/English drafts remain historical. Layer-probe/LoRA and external-transfer claims are excluded from v1 until those models receive separate canonical reruns.

The present canonical table supports only the tested OrchideaSOL setting with four held-out instrument **types**. It is not an independent-player, independent-session, or cross-dataset validation.

The dataset and its test folds were inspected in earlier exploratory work. The canonical reruns enforce a clean **within-run** training/selection/test boundary, but do not retroactively make the overall research process prospectively preregistered. Independent confirmation data would be needed for that stronger claim.

The specific stale claims in the Korean and English v0 drafts are listed in `manuscript_claim_audit.md`. The v1 drafts are concise canonical replacements, with DOCX copies generated from the v1 Markdown. Historical v0 DOCX copies have not been updated and must not be submitted.

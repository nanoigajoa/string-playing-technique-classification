# Duration baseline — train-only repaired rerun

All reported scores are out-of-fold predictions from instrument-type LOSO. Residualization, scaling, SVC class weights, and the majority class were fit separately on each training fold.

| condition | macro F1 |
|---|---:|
| f1_duration_only | 0.4004 |
| f1_full_245 | 0.9065 |
| f1_majority | 0.1039 |
| f1_full_residualized | 0.8891 |
| f1_onset_only | 0.6733 |
| f1_onset_residualized | 0.6936 |

These results replace neither the historical file nor any headline result automatically. Use the JSON manifest to trace every fitted artifact and the NPZ as the canonical prediction source.

Runtime: 2.3 s.

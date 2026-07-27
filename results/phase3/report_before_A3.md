# Phase 3 Results — SHAP Interpretability

**Model**: XGBoost v1 (F1=0.891)  |  **Features**: 245-dim acoustic descriptors

---

## 1. Feature Group Importance (dimensionality-corrected)

> Raw sum inflates groups with more features (MFCC has 80 dims vs Flux 4 dims).
> Right column shows per-feature mean — the dimensionality-corrected ranking.

| Rank | Group | n features | Raw sum | Per-feature mean |
|---|---|---|---|---|
| 1 | Spectral Flux | 4 | 1.9242 | **0.4811** |
| 2 | RMS | 4 | 0.1857 | **0.0464** |
| 3 | Spectral (sc) | 28 | 0.7988 | **0.0285** |
| 4 | MFCC+Δ+ΔΔ | 156 | 3.5593 | **0.0228** |
| 5 | ZCR | 4 | 0.0446 | **0.0112** |
| 6 | Chroma | 48 | 0.1780 | **0.0037** |
| 7 | Log Attack Time | 1 | 0.0017 | **0.0017** |

Chroma rank (raw): 5/7  →  rank (normalized): 6/7

---

## 2. Per-class Top Feature & Directional SHAP

> `flux_std_0` is the top feature shared by ordinario and col_legno_battuto,
> but each class occupies a **different range** of the feature (see `shap_flux_direction.png`).
> Measured medians: col_legno 0.18 (lowest) < sul_pont 0.48 < ordinario 0.51 < tremolo 0.93 (highest).

| Class | Top SHAP Feature | Direction | ALM Open Miss |
|---|---|---|---|
| ordinario | `flux_std_0` | **medium** flux_std (median 0.51) → class | 100% |
| pizzicato_secco | `mfcc_d_mean_1` | — | 0% |
| tremolo | `flux_q75_0` | **high** flux variance → class | 100% |
| sul_ponticello | `flux_q75_0` | intermediate | 100% |
| col_legno_battuto | `flux_std_0` | **low** flux_std (median 0.18) → class | 100% |

---

## 3. Surrogate Fidelity (XGBoost → MERT)

Overall prediction agreement: **0.998**

| Class | Agreement | SHAP interpretation validity |
|---|---|---|
| ordinario | 0.998 | high |
| pizzicato_secco | 1.000 | high |
| tremolo | 1.000 | high |
| sul_ponticello | 0.997 | high |
| col_legno_battuto | 0.982 | high |

> Classes with fidelity < 0.70: SHAP values explain XGBoost decisions only,
> not necessarily how MERT reaches its predictions.

---

## 4. Phase 2 / Phase 3 Synthesis

Qwen2-Audio: yes/no accuracy 50.4%  |  open-ended accuracy 20.0%

**Observation** (correlation, not confirmed causation):

- **pizzicato_secco**: ALM 100% correct. SHAP shows high mfcc_delta + low RMS — acoustically distinct, consistent with ALM success.
- **ordinario**: ALM 0% open correct. SHAP top feature is `flux_std` at a medium level (median 0.51, close to sul_ponticello 0.48). *Hypothesis*: ordinarios sustained, stable bow sits right next to sul_ponticello on flux_std, so ALM struggles to separate them without fine-grained spectral cues. Causal link unconfirmed — could reflect ALM training data bias toward percussive/extreme techniques.
- **tremolo**: ALM 0% open correct. SHAP top feature is high `flux_q75`. *Hypothesis*: rapid bow repetition produces high flux variance, which XGBoost captures well. Whether ALM fails due to this specific cue is unverified; intervention experiment needed to confirm.

> **Limitation**: The Phase 2→3 connection is observational.
> Proving causation requires an intervention experiment (e.g., artificially varying
> flux_std on ordinario clips and re-querying the ALM).

---

## Output Files

| File | Description |
|---|---|
| `shap_group_bar.png` | Group importance: raw vs normalized side-by-side |
| `shap_beeswarm_top20.png` | Top-20 features beeswarm |
| `shap_per_class_bar.png` | Top-8 features per class |
| `shap_flux_direction.png` | Directional SHAP — flux_std for ordinario vs col_legno |
| `shap_surrogate_fidelity.png` | XGBoost vs MERT-LR prediction agreement |
| `shap_case_ordinario.png` | Waterfall case studies — ordinario |
| `shap_case_tremolo.png` | Waterfall case studies — tremolo |
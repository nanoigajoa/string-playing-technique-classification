# 18 · 원고 초안 — Data & Method (§4, v0)

> 아웃라인 `13_논문초안_아웃라인.md` §4의 산문화. 근거: `CLAUDE.md`(설계), `src/features.py`(245d),
> `src/dataset.py`·`model.py`(v2), `phase1_extract_13layer.py`(MERT), `mert_tuning_audit.md`(3분할·예산),
> `cluster_bootstrap.md`(CI), `duration_baseline.md`(클래스 수). 본문=영어 초안, `▸`=한글 주석.

---

## 4. Data and Method

### 4.1 Data

We use the string subset of **OrchideaSOL** [cite], a library of isolated instrumental notes.
We retain five bowed-string articulation classes — *ordinario, pizzicato secco, tremolo,
sul ponticello, col legno battuto* — across four instruments (violin, viola, violoncello,
contrabass), yielding **3,402 clips** with the class distribution in Table 4.1. The set is
imbalanced (ordinario 1,193 to col legno battuto 227), which motivates macro-F1 as the primary
metric.

| Class | n | mean dur (s) |
|---|---|---|
| ordinario | 1,193 | 7.16 |
| tremolo | 1,041 | 7.25 |
| pizzicato secco | 639 | 1.36 |
| sul ponticello | 302 | 7.61 |
| col legno battuto | 227 | 2.71 |

We deliberately do **not** use TinySOL (ordinario only) and note that clip duration itself
carries class information (§5.4/§6); durations are reported for transparency, not used as a
feature except in the confound-control baseline.

▸ 데이터=OrchideaSOL 현악 5클래스 3,402. 불균형→macro-F1. 길이 confound 투명 공개.
  수치출처=`duration_baseline.md`§2.

### 4.2 Evaluation protocol: leave-one-instrument-out

All models are evaluated under **instrument-grouped GroupKFold (leave-one-instrument-out,
LOSO)**: four folds, each holding out one of the four instruments entirely for testing. This
prevents notes from the same instrument/recording appearing in both train and test — the
optimistic bias that random splits incur on this data. Grouping keys are the recording
identifiers parsed by `features.py::parse_recording_id`. Because the generalization claim is
across *instruments*, the instrument/fold is the correct unit for both training splits and
uncertainty (§4.5).

▸ 평가=악기단위 GroupKFold(LOSO) 4fold. 랜덤분할 낙관편향 차단. fold=일반화 주장의 단위.

To study limited-label regimes we sample labeled **seed budgets of 5/10/20/50%** (per fold,
3 random trials, averaged). To avoid selection-time leakage (§5.1), each seed is split
**three ways**:

```
labeled seed ─┬─ val_select 20%   → hyperparameter/config selection ONLY (disjoint, fixed)
              └─ seed_core 80% ─┬─ tr     80%  → training
                                └─ es_val 20% → early-stopping ONLY
```

`val_select ∩ es_val = ∅`, and both are disjoint from the LOSO test fold. Model selection
(config, iteration, threshold) touches **only** `val_select`; early stopping touches **only**
`es_val`; nothing touches test. For the SVM (no early stopping) we report the equal-training-
data condition (`tr`, 64% of labeled) as the primary comparison and the `seed_core` condition
(80%) as a sensitivity variant.

▸ 3분할(선택-누수 방지 핵심). val_select=선택전용, es_val=조기종료전용, 둘 다 test와 분리.
  SVM 주수치=tr(동일학습데이터), 민감도=seed_core. 출처=`mert_tuning_audit.md`.

### 4.3 Features and models

**Handcrafted (v1).** We extract a **245-dimensional** descriptor vector per clip
(`features.py`). Frame-level descriptors are aggregated over time by four statistics —
**mean, std, Q25, Q75** — to preserve temporal variation that a mean alone discards. The
vector comprises: MFCC+Δ+ΔΔ (156), Chroma (48), spectral contrast (28), onset-strength
envelope (4), RMS energy (4), zero-crossing rate (4), and log-first-onset-time (1). Chroma
(pitch-class) is retained deliberately as a **negative control**, not because it is expected
to help articulation classification. Classifiers: **SVM (RBF)** — the primary model — with
XGBoost and random forest as alternates; XGBoost additionally serves as the SHAP surrogate
(§4.6).

▸ v1=245d(4통계 집계). 내역 명시. Chroma=음성 대조군(제거 안 함). 주모델 SVM, SHAP대상 XGB.

**Spectrogram CNN (v2).** As a deep baseline on the same LOSO protocol, we train a small CNN
(`model.py`) on log-mel spectrograms (`dataset.py`), with SpecAugment and an **asymmetric
(2,1) pooling** that halves only the frequency axis to preserve temporal resolution.

▸ v2=log-mel + SmallCNN(비대칭 풀링). 같은 LOSO.

**Self-supervised backbone (MERT).** We use **MERT-v1-95M** [cite] via HuggingFace
`transformers` (`AutoModel`, `trust_remote_code`), resampling audio to 24 kHz. Embeddings are
the mean-pooled hidden states over time (batch=1, no padding, to match the cached-embedding
pipeline exactly); all 13 layers are retained for the probe and the layer-weighted head. On
top we train a classification **head** — a single linear layer, an MLP-256, or a learned
convex combination over layers ("layer-weighted"). A **CLAP zero-shot** classifier
(`laion/clap-htsat-unfused`) provides a no-label reference point. The heaviest tuning point,
**LoRA** (r=8, α=16, on q/v projections), adapts the backbone itself (§5.4). All models run on
Apple-Silicon MPS with CPU fallback; no CUDA, no bitsandbytes.

▸ 백본=MERT-v1-95M, 24kHz, time-mean-pool, 13층. head=linear/mlp256/layer-weighted. CLAP=제로샷
  기준선. LoRA=최대예산점. MPS(구현제약). 출처=`phase1_extract_13layer.py`·`phase1_lora.py`.

### 4.4 Tuning budgets

To make the SVM-vs-MERT comparison fair, both are tuned on `val_select` (never on test).
The SVM searches **12 configurations** (C ∈ {0.1,1,10,100} × γ ∈ {scale,0.01,0.001}); the
MERT head searches **24** (arch ∈ {linear, mlp256} × lr × weight-decay × dropout) — a *larger*
budget for the backbone, guarding against undertuning it (§5.2).

▸ 공정튜닝: SVM 12 / MERT 24 config, 전부 val_select에서만. MERT에 더 큰 예산(언더튜닝 방지).

### 4.5 Uncertainty quantification

We report a **BCa bootstrap** (B=2,000, 95%). Because clips from one recording are strongly
correlated, we resample at the **recording-cluster** level rather than the clip level.
Critically, since the claim generalizes across instruments, we also report a **two-level**
bootstrap that resamples the **instrument/fold** unit — the dominant variance component — and
treat *its* zero-coverage as the significance test (§5.3). With four instruments these
intervals are necessarily wide; that width is the honest answer, not a defect to hide.

▸ CI=BCa B2000, 녹음클러스터 재표집 + 2수준(fold) 재표집을 유의성 기준으로. n=4라 넓음=정직.
  출처=`cluster_bootstrap.md`.

### 4.6 Interpretability: descriptor probe and SHAP

Two analyses explain *why* (§6). **(a) Layer-wise descriptor probe:** for each MERT layer we
ridge-regress each handcrafted descriptor from the 768-d embedding (LOSO, out-of-fold R²,
`RidgeCV` per target on train only), reading each layer as "how linearly available is this
descriptor here." Chroma serves as a control, following the control-task rationale of Hewitt
& Liang [cite]. **(b) SHAP:** we compute SHAP values on the XGBoost surrogate, report
**dimensionality-corrected** (per-feature) group importances, and validate the surrogate
against the MERT head **out-of-sample** (Cohen's κ), reporting fidelity honestly rather than
the misleading in-sample agreement.

▸ 해석 두 축: (a) 층별 ridge probe(OOF R², chroma 대조군), (b) SHAP(차원보정 그룹중요도 +
  OOS κ 정직 보고). 출처=`descriptor_probe.md`·`report.md`.

---

## 손볼 곳 (TODO)
- [ ] `[cite]` 확정 (OrchideaSOL, MERT, CLAP, Hewitt&Liang)
- [ ] Table 4.1 캡션·번호, §4.1↔§5.4(길이 confound) 상호참조
- [ ] v2 CNN 하이퍼파라미터(에폭·lr·batch) 부록에 넣을지 결정
- [ ] MERT transformers 버전 명시(5.12.1로 실행됨, 원 프로젝트 타깃 4.40과 상이 — 재현성 각주)
- [ ] 3분할 그림을 정식 figure로 승격할지 결정

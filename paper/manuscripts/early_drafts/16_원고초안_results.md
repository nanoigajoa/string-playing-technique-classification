# 16 · 원고 초안 — Results (§5, v0)

> 아웃라인 `13_논문초안_아웃라인.md` §5의 산문화. 수치는 결과 파일 직접 인용:
> `results/누수제거_전후비교.md`(①), `results/phase1/mert_tuning_audit.md`(②),
> `results/phase1/cluster_bootstrap.md`(③), `results/phase1/svm_seed_comparison.md`·
> `duration_baseline.md`(착지). 본문=영어 초안, `▸`=한글 주석(무엇을/근거·수치출처).
> 원칙(B6-doc §0): ①②는 **실제 반전**, ③은 **미수(near-miss)** — 과장 금지.

---

## 5. Results

We report each pitfall as a before/after pair on the same data, in the order we
encountered it. Pitfalls (i) and (ii) each *reversed* our working conclusion; pitfall
(iii) would have reversed it a third time but was caught before it entered any reported
result. §5.4 states the corrected, stable conclusion.

▸ 섹션 전체 규칙: 마주친 순서대로 전후쌍. ①②=반전, ③=미수. 마지막 5.4=교정 후 착지.

Throughout, macro-F1 is averaged over four instrument-grouped folds (leave-one-instrument-
out; violin / viola / cello / contrabass), across label budgets of 5/10/20/50%. Unless
noted, MERT and SVM are trained on **identical data** (the 64% `tr` partition) for an
equal-budget comparison; a sensitivity variant (`seed_core`, 80%) is reported where the
two differ.

▸ 공통 프로토콜 한 문단. 주 비교=동일 학습데이터(tr 64%), 민감도=seed_core 80%.

---

### 5.1 Pitfall (i): selection-time test-set leakage — *first reversal*

Our initial self-training pipeline selected the iteration (and confidence threshold τ) by
test-set macro-F1. Removing this — selecting on a disjoint held-out split instead — lowers
self-training macro-F1 across every budget, most at low label counts (Table 5.1):

| Label budget | Leaked (selected on test) | De-leaked | Δ |
|---|---|---|---|
| 5%  | 0.745 | 0.638 | **−0.107** |
| 10% | 0.817 | 0.739 | −0.078 |
| 20% | 0.837 | 0.798 | −0.039 |
| 50% | 0.857 | 0.829 | −0.028 |
| cross-domain (Phase 4) | 0.461 | 0.449 | −0.012 |

The drop grows as labels shrink (−0.107 at 5%), showing the "iteration-selection gain" was
largely leaked test signal, worst where the selection split is tiny (~25 samples). Crucially,
the leak had produced an apparent **crossover**: in the leaked pipeline MERT+self-training
exceeded the SVM at the 5–10% budgets, supporting a "foundation models win under limited
labels" reading. After de-leaking, the SVM matches or exceeds MERT+self-training at *every*
budget, and self-training no longer beats even its own MERT baseline at 5% (0.638 vs 0.726).
**The crossover — and the conclusion resting on it — disappears.**

▸ 함정①=반전1. 핵심표=`누수제거_전후비교.md`§1. 교차점 소멸=§2. "foundation model 유리" 인상이
  누수 산물. 5%에서 self-training이 자기 baseline보다도 낮음(0.638<0.726).

---

### 5.2 Pitfall (ii): backbone undertuning — *second reversal*

The de-leaked result above used a single linear head on MERT's last hidden state — below
common probing convention (e.g., MARBLE's MLP-512). Matching convention with a proper head
search (arch ∈ {linear, mlp256} × lr × weight-decay × dropout = 24 configs, all selected on
a **disjoint** `val_select` split, giving MERT a *larger* search budget than the SVM's 12)
raises tuned MERT over its default by **+0.025 to +0.046**. Under equal training data, tuned
MERT and tuned SVM become statistically indistinguishable — every gap falls within ±0.7 fold-
std (Table 5.2):

| Label budget | SVM (tuned, tr) | MERT (tuned, tr) | MERT − SVM | in fold-std |
|---|---|---|---|---|
| 5%  | 0.6613 ± 0.053 | 0.6714 ± 0.040 | **+0.0101** | +0.21σ |
| 10% | 0.7542 ± 0.078 | 0.7477 ± 0.037 | −0.0065 | −0.11σ |
| 20% | 0.8316 ± 0.039 | 0.8074 ± 0.030 | −0.0242 | −0.69σ |
| 50% | 0.8681 ± 0.040 | 0.8593 ± 0.025 | −0.0088 | −0.26σ |

This **reverses the §5.1 verdict a second time**: the "SVM wins everywhere" conclusion was
itself an artifact — of undertuning the backbone, not of the data. Two further points guard
against over-claiming. First, the correct head is *budget-dependent*: at low labels the
search prefers a plain linear head (5%: 9/12 folds linear; 10%: 8/12), splits evenly at 20%,
and prefers mlp256 only at 50%. The lesson is not "convention undertunes SSL" but "probe
design can flip the conclusion, and we sat below convention and fooled ourselves once."
Second, if the SVM is allowed its structural advantage — it needs no early-stopping split
and can train on the full `seed_core` 80% — it edges ahead everywhere (up to −1.04σ at 10%),
but still within ~1σ at n=4, a tilt rather than a reversal.

▸ 함정②=반전2. 주표=`mert_tuning_audit.md`§성능비교+종합분석1. default→tuned +0.025~0.046.
  head는 라벨량 의존(저=linear, 고=mlp256) — "우리가 관행 이하에서 속았다" 프레이밍. seed_core
  민감도=SVM 소폭 우세하나 ~1σ, 뒤집힘 아님.

---

### 5.3 Pitfall (iii): resampling unit in uncertainty quantification — *a near-miss*

Having landed on parity, we quantified it with a BCa bootstrap (B=2000). Resampling at the
**recording-cluster** level while holding the four instrument folds fixed, the paired
MERT−SVM interval **excludes zero at the 5% and 20% budgets** (−0.027 [−0.044, −0.012] and
−0.026 [−0.043, −0.011]), reading as "SVM significantly ahead" — in tension with the parity
of §5.2. But this interval models only test-clip sampling variance; the folds — the very
axis our leave-one-instrument-out claim generalizes over — are fixed. A generalization claim
across instruments must resample at the instrument/fold unit. Doing so (a two-level
bootstrap) widens the intervals **2.6–5.0×** and every paired interval now **covers zero**
(Table 5.3):

| Label budget | MERT − SVM | 1-level CI (clip) | 0? | 2-level CI (fold) | 0? |
|---|---|---|---|---|---|
| 5%  | −0.027 | [−0.044, −0.012] | no  | [−0.107, +0.050] | **yes** |
| 10% | −0.000 | [−0.016, +0.015] | yes | [−0.071, +0.069] | **yes** |
| 20% | −0.026 | [−0.043, −0.011] | no  | [−0.074, +0.035] | **yes** |
| 50% | −0.007 | [−0.021, +0.006] | yes | [−0.046, +0.025] | **yes** |

Had we reported the clip-level interval, we would have flipped the conclusion a **third**
time — back to "SVM significantly wins." Unlike (i) and (ii), this pitfall never entered a
reported result: we caught it before publication. We flag it precisely because it is *harder*
to catch — it arises from adopting a more rigorous method (a cluster bootstrap) and stopping
one level too shallow. The wide fold-level intervals (n=4 instruments) are not a weakness to
hide but the honest answer to "would this hold on other instruments?"

▸ 함정③=미수. 주표=`cluster_bootstrap.md`§2·§3. 1수준은 5%·20%에서 0 배제(SVM 유의)→2수준은
  전 구간 0 포함, 2.6~5.0× 확장. **정직 강조**: ①②와 달리 보고에 들어간 적 없음. 더 엄밀한
  방법 도입에서 생겨 포착이 더 어려움. n=4 넓은 구간=숨길 약점 아니라 정직한 답.

---

### 5.4 Landing: conditional parity

Removing all three pitfalls, the corrected picture is **conditional parity**: on this low-
dimensional, strong-signal, small-data fine-grained task, a tuned 245-dimensional SVM and a
tuned MERT head are statistically indistinguishable across 5–50% label budgets (all gaps
within ±0.7 fold-std, §5.2; all fold-level intervals cover zero, §5.3). We rule out a
duration confound — that the SVM merely exploits clip-length differences between techniques
— with a length-only baseline (single duration feature), which reaches only macro-F1 0.400 under
LOSO versus 0.906 for the full 245-dimensional feature set — a 0.506 gap. Moreover,
regressing the duration component out of the features leaves performance essentially intact
(0.906 → 0.893), and the onset-family features actually *rise* after residualization
(0.673 → 0.694); a mere length proxy would have collapsed. The discriminative signal is
correlated with, but not reducible to, clip length.

The parity is *conditional*, not universal: it is a claim about tasks whose discriminative
signal is concentrated in a few low-level acoustic descriptors, not about MIR at large. §6
explains, mechanistically, *why* depth adds nothing here.

**Does a larger tuning budget rescue the backbone?** A natural objection to §5.2 is that we
only tuned a *frozen* head — perhaps adapting the backbone itself would open a gap for MERT.
It does not; it does the opposite. We place five points on a single tuning-budget spectrum
(Table 5.4): SVM → frozen-linear head → frozen-tuned head → learned layer-weighting → LoRA
adaptation of the backbone (r=8, q/v projections).

| seed | SVM (tuned) | frozen-linear | frozen-tuned | layer-weighted | LoRA |
|---|---|---|---|---|---|
| 20% | 0.832 | 0.776 | 0.807 | **0.855** | 0.804 |
| 50% | 0.868 | 0.814 | 0.859 | **0.900** | 0.775 |

Two readings follow. First, the *fair ceiling* for MERT is the tuned frozen probe
(layer-weighted): it matches or slightly exceeds the SVM (0.855 vs 0.832 at 20%; 0.900 vs
0.868 at 50%) — the same "convention-matched probing reaches parity" result as §5.2, now at
the top of the spectrum. Second, adapting the backbone with LoRA does **not** continue the
climb; it *regresses* — below even the frozen-tuned head, and worse with more data (0.804 at
20% → 0.775 at 50%), while its early-stop-val F1 stays high (0.90–1.0). This is the signature
of fine-tuning **distorting** pretrained features under distribution shift: our LOSO protocol
tests on an unseen instrument, exactly the out-of-distribution regime where full fine-tuning
underperforms linear probing (the LP-FT effect, Kumar et al. [cite, 2022 ICLR]). So the
"you simply did not fine-tune hard enough" objection is answered in the strongest form: fine-
tuning harder makes MERT *worse* here, not better. (We report a single LoRA configuration,
not a grid — a limitation noted in §7 — but the inverted data-scaling and the train–test gap
are the expected direction of the effect, not an artifact of one setting.)

▸ B3 착지. 주표=`lora_spectrum.md`. 핵심: (1) MERT 공정 상한=layer-weighted, SVM과 동급+.
  (2) LoRA는 상승 안 하고 *퇴행*(frozen보다 낮고, 데이터 늘수록 악화) → LP-FT(Kumar 2022,
  OOD서 FT<LP) 서명. "덜 튜닝했다" 반박을 가장 강하게 격파. 단일 config 한계는 §7에 명시.

▸ 착지=조건부 동급. 주표=`svm_seed_comparison.md`+§5.2·§5.3 종합. duration confound 배제=
  `duration_baseline.md`([DUR] 수치 채울 것). "conditional"을 못박아 과일반화 차단. §5.4 마지막
  칸(B3 LoRA)은 도는 중 — 완료 시 한 문장.

---

## 손볼 곳 (TODO)
- [x] ~~`[DUR]`~~ length-only 0.400 / 잔차화 논거까지 반영 완료 (`duration_baseline.md`)
- [x] ~~B3 LoRA~~ 완료·반영 (LP-FT 프레임, 5.7h run). §7에 "단일 config" 한계 추가 필요
- [ ] Table 5.4 캡션·번호, §5.4↔§6.1·§3.2(Kumar LP-FT) 상호참조
- [ ] Table 번호·캡션 정식화, 본문↔표 상호참조
- [ ] §5.1 cross-domain 행은 Phase4로 미루고 여기선 각주 처리할지 결정
- [ ] fold-std(σ) 정의를 §4 Method에서 1회 정의 후 여기선 기호만 사용
- [ ] significance.md의 Wilcoxon p=0.125(n=4 하한) 언급을 §5.2 각주로 넣을지 결정

# 17 · 원고 초안 — Discussion (§6, v0)

> 아웃라인 `13_논문초안_아웃라인.md` §6의 산문화. 근거:
> `results/phase3/descriptor_probe.md`(층별 R²), `results/phase2/layer_probe.md`(probe효과>계열효과),
> `results/phase3/report.md`(SHAP 그룹중요도·방향성·surrogate κ). 본문=영어 초안, `▸`=한글 주석.
> 원칙: 유의하지 않은 방향성을 결론으로 승격시키지 않는다. 음성결과는 음성결과로 싣는다.

---

## 6. Discussion

§5 established *that* a tuned MERT head and a tuned SVM are indistinguishable on this task.
This section asks *why*, and reports two honest negative results that bound the claim.

▸ §5=사실(동급), §6=원인(메커니즘)+정직한 음성결과 2개.

---

### 6.1 Why depth does not help: a layer-wise descriptor probe

To locate what the backbone contributes, we regress handcrafted acoustic descriptors from
each MERT layer's embedding (ridge, LOSO, out-of-fold R²) — reading each layer as "how
linearly available is this descriptor here." The pattern explains the parity (Table 6.1):

| Descriptor (role) | R² at L0 (conv front-end) | peak layer | R² at peak | high−low band |
|---|---|---|---|---|
| Onset-strength envelope (core cue) | **0.767** | L8 | 0.850 | +0.008 |
| Spectral contrast (mid) | 0.709 | L5 | 0.751 | −0.009 |
| MFCC (mid) | 0.585 | L2 | 0.638 | **−0.086** |
| Chroma (negative control) | 0.506 | L9 | 0.604 | **+0.023** |

Three things stand out. **(1)** The task's dominant cue — the onset-strength envelope — is
already almost fully linearly present at the *convolutional front-end* (R²=0.767 at L0,
before any transformer layer), and the twelve transformer layers add essentially nothing to
the band average (+0.008); its peak (0.850 at L8) exceeds L0 by only +0.083. **(2)** MFCC —
timbral detail — actually *degrades* with depth (−0.086). **(3)** The only descriptor that
consistently *gains* with depth is Chroma (+0.023), i.e. pitch-class / harmonic abstraction
— and Chroma is our negative control, near-zero in SHAP importance (§6.2) because articulation
is pitch-invariant. So the representational work the transformer stack performs is largely
orthogonal to what this task needs: the discriminative signal sits at the front-end that any
mel-style feature extractor (including our SVM's inputs) also sees, while depth buys harmonic
abstraction the task discards. This is the audio-articulation counterpart to Pasad et al.'s
[cite] layer-wise finding in speech, and it is the mechanistic reason a heavy backbone offers
no practical gain here.

▸ 핵심 메커니즘. 주표=`descriptor_probe.md`§3. onset은 L0=0.767에서 이미 포화(깊이 기여 +0.008,
  정점도 +0.083), MFCC는 깊이서 악화(−0.086), 깊이서 오르는 건 chroma(음성 대조군)뿐 →
  "깊이가 하는 일이 이 과제와 직교". Pasad(speech)의 오디오-주법 판.

A companion probe-configuration analysis (Table 6.1b) reinforces this from the modeling side:
matching probing convention (best-layer / layer-weighted head vs a last-layer linear head)
raises MERT by a consistent **+0.039–0.047**, whereas the model-family effect at equal budget
is only **−0.002 to +0.035** (sign-inconsistent). *Probe design moves the number more than the
choice of model family does* — the MIR instance of Musgrave et al.'s [cite] "tuning-budget
effects exceed method effects," and a direct quantification of why pitfall (ii) was able to
flip the conclusion.

▸ 보조. `layer_probe.md`착지: probe설정효과(+0.042평균) > 계열효과(+0.018평균, 부호불일치).
  함정②가 왜 결론을 뒤집을 힘이 있었는지의 정량 근거.

The same mechanism explains why *deeper* adaptation backfires (§5.4, Table 5.4). If the
task-relevant signal is already exposed at the front-end and the depth encodes abstraction
the task discards, then fine-tuning the backbone on a small in-distribution training set has
little useful signal to sharpen and ample opportunity to overfit — distorting the very front-
end features the frozen probe relied on. Under our cross-instrument (out-of-distribution)
LOSO test this manifests as LoRA falling below the frozen-tuned probe and degrading with more
data, the LP-FT effect of Kumar et al. [cite]. The probe result is thus not just descriptive:
it *predicts* that parameter-efficient fine-tuning cannot help — and can hurt — on this task.

▸ probe가 LoRA 퇴행까지 예측. 신호가 front-end에 있고 깊이가 버리는 추상만 더하면, 소량
  in-dist 학습은 날카롭게 할 신호가 적고 과적합 여지만 커 front-end 특징을 왜곡 → OOD(LOSO)서
  LP-FT(Kumar). probe가 서술을 넘어 "PEFT가 못 돕고 해칠 수 있음"을 예측한다는 연결.

---

### 6.2 What the cheap features encode: SHAP against musician knowledge

If depth is unnecessary, what *are* the 245 handcrafted features using? A dimensionality-
corrected SHAP analysis (per-feature mean over an XGBoost surrogate) ranks the **onset-
strength envelope first by a wide margin** (0.481 per feature vs 0.046 for the next group,
RMS), with Chroma second-to-last (0.004) — exactly the ordering the probe predicts, and
exactly what a string player would expect: articulation is defined by *how the note begins
and evolves*, not by *which pitch* it is. The direction of the top flux feature aligns with
physical intuition per class: `col legno battuto` occupies **low** flux variance (a single
dry wood-on-string strike, median flux_std 0.18), `tremolo` occupies **high** flux variance
(rapid re-bowing, high flux_q75), and `ordinario` sits at a **medium** level near
`sul ponticello` — the very adjacency that makes those two the hardest to separate.

▸ SHAP↔음악가 지식. 주표=`report.md`§1(그룹중요도)·§2(방향성). onset-strength 1위(per-feature
  0.481), chroma 뒤에서 2번째 → probe예측·연주자직관과 일치. flux 방향: col legno=낮은 분산(마른
  타격 1회), tremolo=높은 분산(빠른 재보잉), ordinario≈sul pont 인접(→혼동 최대).

We report surrogate fidelity honestly. Measured **out-of-sample** (instrument-grouped), the
XGBoost surrogate agrees with the MERT head on 91.4% of held-out clips (Cohen's κ=0.882) —
substantial, but not the near-perfect in-sample 0.998 an earlier version mistakenly reported.
Fidelity is weakest precisely on the rare classes, `sul ponticello` (κ=0.74) and
`col legno battuto` (κ=0.71); SHAP explanations for those two transfer to MERT only with
caution. And when either model errs, they agree on only 16% of the error cases — the two
families fail on *different* clips. SHAP thus explains the XGBoost decision faithfully and the
MERT decision only approximately; we do not overstate it as a single mechanism for both.

▸ surrogate 정직 보고. `report.md`§3: OOS κ=0.882(과거 0.998은 in-sample 착오), 희귀클래스서 약함
  (κ 0.74/0.71), 오류일치 16%(서로 다른 클립서 실패). "SHAP=XGB 설명, MERT는 근사"로 못박음.

---

### 6.3 Two honest negative results

**Descriptor–layer alignment is *not* supported.** A tempting story is that MERT's learned
layer weights concentrate where the core descriptor is most available. They do not: the
correlation between the learned layer-weight profile and per-layer descriptor availability is
higher for the **Chroma control (ρ=0.73)** than for the onset-strength core cue (ρ=0.42).
With only 13 layers this comparison is underpowered, and the control out-aligning the target
is the opposite of what an alignment claim needs. We therefore *drop* the alignment narrative
and report the correlation table as a null result rather than dress it as confirmation.

▸ 음성결과①. `descriptor_probe.md`§1 "B2가중과 ρ" 열: chroma(대조군) 0.73 > onset(핵심) 0.42.
  대조군이 더 정렬 → alignment 주장 폐기, ρ표를 null로 게재. 13층이라 검정력도 부족.

**The ALM→SHAP causal link is observational only.** Qwen2-Audio's failures correlate
suggestively with SHAP structure — it is 100% correct on the acoustically isolated
`pizzicato secco` yet 0% (open-ended) on `ordinario` and `tremolo`, whose top cues place them
near neighbors in flux space. But this is correlation; confirming it would require an
intervention (e.g., synthetically varying flux_std on ordinario clips and re-querying the ALM),
which we did not run. We present it as a hypothesis-generating observation, not a mechanism.

▸ 음성결과②. `report.md`§4: ALM 실패↔SHAP 구조 상관은 관찰일 뿐. pizzicato 100% / ordinario·tremolo
  0%. 인과 확정은 개입실험 필요(미수행) → 가설로만 제시.

---

### 6.4 Robustness and scope

The parity and its mechanism are consistent under an ordinario-excluded 4-class variant and
across the cross-domain Philharmonia transfer, where we additionally localize the fragile
classes (`col legno`, `sul ponticello` collapse to F1≈0 off-domain) — a robustness boundary
rather than a counterexample. [▸ Phase4 `results/phase4/report.md` 수치로 확정]. The scope
condition is explicit: our claim concerns tasks whose discriminative signal is low-dimensional
and front-end-accessible. Where the signal instead requires the harmonic abstraction that only
depth supplies, we would expect the gap to reopen — a prediction our probe makes testable.

▸ 강건성·범위. 4클래스·cross-domain서 유지, 취약클래스 규명. "저차원·front-end 접근가능 신호"라는
  범위조건 명시 + probe가 그 경계를 검증가능하게 만든다는 예측으로 마무리.

---

## 손볼 곳 (TODO)
- [ ] §7 Limitations에 추가: **LoRA는 단일 config(그리드 미탐색)** — 방향성은 LP-FT와 일치하나
      "LoRA 최적 튜닝"은 주장하지 않음. (우리가 비판하는 언더튜닝 죄 자기적용 방지 문장)
- [ ] `14_` Intro ¶4의 "transformer layers add only +0.03" 를 정확히 정정 —
      band차 +0.008 / 정점 대비 +0.083 중 어느 표현 쓸지 통일 (현재 +0.03은 부정확)
- [ ] Phase4 4클래스·cross-domain 수치를 §6.4에 확정 인용 (`results/phase4/report.md`)
- [ ] Table 6.1 캡션·번호 정식화, §6.1↔§3.2(Pasad)·§5.2 상호참조
- [ ] surrogate κ 표(class별)를 부록으로 뺄지 본문 유지할지 결정
- [ ] ρ null-result를 표로 실을지 문장으로만 둘지 (음성결과 게재 형식)

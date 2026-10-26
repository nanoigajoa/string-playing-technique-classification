# 19 · 원고 초안 — Limitations & Future Work (§7, v0)

> 아웃라인 `13_논문초안_아웃라인.md` §7의 산문화. 흩어진 한계 취합:
> §5.2(n=4 검정력), §5.3(③ 1건·2수준 BCa), §5.4(단일 LoRA config), §6.3(음성결과), §6.4(범위).
> 본문=영어 초안, `▸`=한글 주석. 원칙: 한계를 방어적으로 숨기지 말고 기여의 경계로 명시.

---

## 7. Limitations and Future Work

We state the boundaries of the claim plainly; several are direct consequences of the honest
protocol this paper argues for.

▸ 원칙: 한계=기여의 경계. 정직한 프로토콜이 오히려 만든 한계도 있음(예: n=4 넓은 CI).

**Statistical power (n=4).** Our generalization claim is across four instruments, so the
fold-level unit has n=4 — the honest but low-powered unit (§4.5). No paired test can reach
p<0.05 at this size (the two-sided Wilcoxon floor is p=0.125), and the two-level bootstrap
intervals are correspondingly wide. We therefore frame the central result as *statistical
indistinguishability*, not as a proven equality, and we do not promote any within-std
directional tilt to a conclusion. More instruments (or instrument families) would tighten this.

▸ n=4 검정력. Wilcoxon 하한 p=0.125. "동급"은 구분불가지 등가증명 아님. 방향성 승격 안 함.
  악기·악기군 확장이 해법.

**Single-dataset scope.** All results are on OrchideaSOL. The parity is *conditional* — it
concerns tasks whose discriminative signal is low-dimensional and front-end-accessible
(§6.4), and we do not claim it for MIR at large. Cross-domain transfer to Philharmonia probes
robustness but is not an independent replication of the pitfalls. A second, independent task
is the highest-value extension (below).

▸ 단일 데이터셋. 조건부 동급(저차원·front-end 접근 신호). MIR 일반 주장 아님. 2차 태스크가 최고가치.

**A single LoRA configuration.** §5.4 reports one LoRA setting (r=8, fixed lr), not a grid —
in contrast to the 12/24-config searches we gave the SVM and frozen head. We therefore do
**not** claim LoRA was optimally tuned, and we are careful not to commit the very undertuning
we critique. The finding we do stand behind is *directional and mechanistic*: LoRA regresses
below the frozen-tuned probe and worsens with more data, matching the LP-FT feature-distortion
effect under our out-of-distribution (LOSO) evaluation. A LoRA hyperparameter sweep would
strengthen this from a consistent signal to a tuned lower bound.

▸ 단일 LoRA config(그리드 X) — 우리가 비판하는 언더튜닝 자기적용 방지 위해 "LoRA 최적" 주장
  안 함. 방향성·메커니즘(LP-FT)만 주장. 스윕하면 튜닝된 하한으로 격상.

**Pitfall (iii) observed once, one-sided fix.** The resampling-unit pitfall (§5.3) is a
near-miss caught in this study; we did not replicate it across datasets, and we apply the
two-level bootstrap as a correction rather than deriving a full BCa two-level interval with
acceleration at both levels. The variance-decomposition framing (Bouthillier et al. [cite])
is the natural formalization we leave to future work.

▸ ③은 1건·미수. 데이터셋 간 재현 없음. 2수준 BCa(양수준 가속)까지는 안 감. 분산분해 형식화=향후.

**Interpretability transfers only approximately.** SHAP explains the XGBoost surrogate, which
agrees with the MERT head out-of-sample at κ=0.88 and less on the two rare classes (§6.2), so
mechanistic claims for `sul ponticello` / `col legno battuto` carry more uncertainty. The
descriptor–layer alignment and the ALM→SHAP causal link are reported as null / observational
respectively (§6.3), not as findings.

▸ 해석 근사성. surrogate κ=0.88(희귀클래스 약함). alignment=null, ALM인과=관찰. 승격 안 함.

**Reproducibility note.** The MERT remote code ran under a newer `transformers` than the
project originally targeted; we pin versions in the release and cache all embeddings and splits
for exact reproduction.

▸ transformers 버전 상이(5.12.1 실행). 릴리스에 버전 고정 + 임베딩·split 캐시로 재현.

### Future work

- **Independent second task (highest priority).** Replicating the three pitfalls on a separate
  articulation dataset — e.g., Erhu playing techniques — would turn "a pitfall we hit" into "a
  pattern," and directly address the single-dataset limit. It requires a performer/session
  grouping key to avoid re-introducing leakage.
- **Reverse and broader cross-domain.** Train-on-Philharmonia / test-on-OrchideaSOL and other
  instrument families, to map where front-end-accessible parity holds and where depth begins
  to pay.
- **From checklist to formal variance model.** Promote the practitioner checklist (§ appendix)
  into a variance-decomposition procedure that names the correct resampling unit for a given
  generalization claim.
- **Practitioner-facing tool (application thread).** The mechanistic account (SHAP ↔ bowing
  physics) points toward a lightweight, interpretable feedback tool for string pedagogy — a
  separate applied paper, not a claim of this one.

▸ Future: ①Erhu 2차 태스크(최우선, 그룹키 선결) ②역방향·광역 cross-domain ③체크리스트→분산모델
  형식화 ④연주자 피드백 도구(2번 논문). 12a 문서의 perception 편입은 메커니즘 한정으로만.

---

## 손볼 곳 (TODO)
- [ ] `[cite]` (Bouthillier) 확정, 체크리스트를 appendix로 뺄지 §4/§7 어디 둘지 결정
- [ ] Erhu 데이터 그룹키 확보 여부에 따라 Future 1번 문장 톤 조정
- [ ] §7 길이: 투고지 확정 후 rare-class·재현성 문단을 1문장씩 압축 가능
- [ ] 12a(perception 재고) 처리 방침을 Future에 한 줄로 명시할지 결정 ([[leakage-audit-trajectory]] 참조)

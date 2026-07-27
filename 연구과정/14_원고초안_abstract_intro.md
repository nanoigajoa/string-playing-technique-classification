# 14 · 원고 초안 — Abstract & Introduction (v0)

> 아웃라인 `13_논문초안_아웃라인.md` §1·§2의 산문화. 수치는 감사 최종본
> (`results/phase1/mert_tuning_audit.md`, `results/누수제거_전후비교.md`, `results/phase3/descriptor_probe.md`) 기준.
> 본문 = 영어(투고용 초안), `▸` = 한글 주석(무엇을 하는 문단인가 / 근거 파일). 확정 아님, 손볼 것.

---

## Working title

**When Foundation Models Only *Look* Better: Three Evaluation Pitfalls in Fine-Grained String Articulation Classification**

▸ 대안: "Understanding When Heavy Audio Models Don't Help." 제목은 마지막에 확정.

---

## Abstract (draft)

Self-supervised audio foundation models are increasingly assumed to outperform
handcrafted acoustic features on music understanding tasks. We test this assumption
on fine-grained string articulation classification (five techniques —
*ordinario, pizzicato secco, tremolo, sul ponticello, col legno battuto* — on
OrchideaSOL, evaluated leave-one-instrument-out). An initial pipeline suggested a
foundation-model (MERT) advantage under limited labels. Auditing our own protocol,
we find this advantage is an artifact of three evaluation pitfalls: (i) selection-time
test-set leakage in self-training, (ii) backbone undertuning, and (iii) a mismatched
resampling unit in uncertainty quantification. Removing the first two **flips the
conclusion twice**; the third, left uncorrected, would flip it a third time. After
correction, a tuned MERT head and a tuned SVM over 245-dimensional descriptors are
**statistically indistinguishable** across 5–50% label budgets (all gaps within
±0.7 fold-std, n=4). A layer-wise descriptor probe explains why: the discriminative
low-level cue (onset-strength envelope) is already linearly recoverable at MERT's
convolutional front-end (R²=0.77, before any transformer layer), and depth adds mainly
harmonic abstraction this task does not require. Our contribution is not a new
state-of-the-art model but a reproducible account of three pitfalls, a practical
checklist, and a scoped negative result: for low-dimensional strong-signal, small-data
fine-grained tasks, heavy models offer no practical gain.

▸ 문장별 근거: 반전2회=`누수제거_전후비교.md`+`mert_tuning_audit.md` / 미수③=`cluster_bootstrap.md` /
  동급±0.7σ=`mert_tuning_audit.md` §종합분석1 / R²0.77=`descriptor_probe.md`.

---

## 1. Introduction (draft)

**¶1 — 배경/통념.**
Automatic recognition of instrumental playing techniques is a core problem in music
information retrieval, with applications in performance analysis, music education, and
expressive-content indexing. String articulations in particular carry fine acoustic
distinctions — the bow striking the string (*col legno*), bowing near the bridge
(*sul ponticello*), rapid re-bowing (*tremolo*). As self-supervised audio foundation
models (e.g., MERT) achieve strong results across MIR benchmarks, a common expectation
is that such models also dominate fine-grained technique classification, rendering
handcrafted acoustic features obsolete.

▸ 이 과제가 왜 세밀분류인지 + "foundation model이 당연히 낫다"는 통념 설정. Related은 §3에서 32편으로.

**¶2 — 함정은 쉽게 만들어진다.**
We show this expectation is easy to *manufacture* and hard to *verify*. Small,
individually reasonable evaluation choices can inflate an apparent foundation-model
advantage without any real modeling gain. We isolate three such choices that arose in
our own pipeline and quantify each with a before/after comparison.

▸ 논문의 태도: 남을 저격하는 게 아니라 "우리 파이프라인에서 실제로 발동한" 함정을 정직하게 해부.

**¶3 — 세 함정과 두 번의 반전.**
*(i) Selection-time test-set leakage.* Choosing the self-training iteration (or a
decision threshold) by test-set F1 leaks the target into model selection; removing it
drops self-training macro-F1 by up to 0.107 at the 5% label budget and erases an
apparent SVM–MERT crossover. *(ii) Backbone undertuning.* A single linear head on the
last hidden state understates the backbone; a convention-matched search (MLP /
layer-weighted head, disjoint selection val) raises tuned MERT by 0.025–0.046. *(iii)
Resampling unit in uncertainty quantification.* A BCa bootstrap that resamples only
test clips — holding folds fixed — reports SVM as significantly ahead at 5% and 20%;
resampling at the correct unit (instruments/folds, the dominant variance component for
a cross-instrument claim) returns confidence intervals that cover zero everywhere.
Pitfalls (i) and (ii) each **flipped our conclusion** — first toward, then away from
the foundation model; (iii) was caught before it flipped it a third time.

▸ 핵심 서사 문단. ①②는 실제 반전, ③은 미수(과장 금지 — B6-doc §0 원칙 그대로).

**¶4 — 교정 후 결론 + 메커니즘.**
With all three controlled, and under **equal training data**, tuned MERT and a tuned
245-dimensional SVM are indistinguishable across label budgets (5–50%; all gaps within
±0.7 fold-std). A layer-wise descriptor probe locates the reason: the onset-strength
envelope that separates these techniques is already linearly present at MERT's
convolutional front-end (R²=0.77, pre-transformer), and the twelve transformer layers add
almost nothing to its band average (+0.008, peaking only +0.083 above the front-end); the
only descriptor that *gains* with depth is chroma — harmonic abstraction this articulation
task does not need. The heavy model has nothing task-
relevant to add that the front-end has not already exposed.

▸ 착지 = 조건부 동급 + "왜"(descriptor probe). alignment 주장은 안 함(검정력 부족, §6.1).

**¶5 — 기여.**
Our contributions are: (1) a reproducible account of three evaluation pitfalls in
fine-grained audio classification, each with a before/after quantification on the same
data; (2) a practitioner checklist for avoiding them; and (3) a scoped negative result
— on low-dimensional, strong-signal, small-data fine-grained tasks, self-supervised
foundation models yield no practical advantage over well-tuned handcrafted features —
supported by a mechanistic layer-wise analysis rather than accuracy alone.

▸ 기여 3개. "새 SOTA 아님"을 명시해 리뷰어의 "그래서?"를 선제 차단.

---

## 손볼 곳 (TODO)
- [ ] 제목 확정
- [ ] Abstract 길이 다듬기(학회 규정 자수)
- [ ] ¶3 수치 인용을 표 참조(§5.1~5.3)로 연결
- [ ] "handcrafted features obsolete" 통념에 인용 붙이기(Related 32편에서)
- [ ] §5.4 B3(LoRA) 결과 나오면 ¶4에 "even at the highest tuning budget (LoRA)" 한 구절 추가

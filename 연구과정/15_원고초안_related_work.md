# 15 · 원고 초안 — Related Work (§3, v0)

> 아웃라인 `13_논문초안_아웃라인.md` §3의 산문화 (Track C).
> 근거: `docs/기여정리_선행연구대비.md`, `논문/05_PlayingTechnique_Classification`(20편 tracker),
> `논문/06_SSL_Evaluation_Methodology`(SSL·평가방법론). 본문=영어 초안, `▸`=한글 주석(무엇을/근거).
> 확정 아님 — 인용키(`\cite{}`)는 최종 bib 확정 후 채움. `[cite]` = 자리표시.

---

## 3. Related Work

이 섹션은 두 갈래로 구성한다. §3.1은 **무엇을 분류하는가**(현악 주법분류의 특징·모델 계보),
§3.2는 **어떻게 평가하는가**(SSL·준지도 평가의 함정 문헌). §3.3의 코딩표가 둘을 잇는다 —
선행 주법분류 연구를 *평가 관행* 축으로 코딩해, 본 논문의 세 함정이 사는 빈틈을 드러낸다.

▸ 구조 의도: 05폴더(도메인)와 06폴더(방법론)를 병치하고, 코딩표로 "도메인 논문들이 방법론
  논문들의 교훈을 아직 흡수하지 않았다"를 보여 기여를 위치시킴.

---

### 3.1 Playing-technique classification

**Handcrafted acoustic features on bowed strings.**
The task closest to ours is instrument playing-technique (IPT) classification on bowed
strings from audio. Su et al. [cite, 2014 TASLP] classify nine techniques on exactly the
four instruments we use (violin, viola, cello, contrabass), showing that phase-derived
features (group-delay, instantaneous-frequency) with sparse coding lift macro-F1 from
0.835 (magnitude-only) to 0.915 — but also that cross-corpus transfer (RWC → real
recordings) collapses to 0.547. Kruger & Jacobs [cite, 2020 JNMR] classify up to eleven
techniques with an SVM over handcrafted features on RWC under a random 70/30 split.
Barbancho et al. [cite, 2009 ICASSP] detect six violin techniques with a hand-built
decision flowchart (no learning). These establish that (a) fine articulation distinctions
are carried by low-level spectral/temporal cues, and (b) generalization *across recording
sources* is the hard part — the same tension our leave-one-instrument-out protocol targets.

▸ 우리와 가장 가까운 좌표. Su 2014 = "정확히 같은 4악기". 공통 발견: 판별 신호는 저수준,
  진짜 어려움은 소스 간 일반화(→ 우리 LOSO 동기). `[Highest/High relevance in tracker]`

**Scattering-transform and multi-scale lineage.**
A second line argues MFCCs miss the amplitude/frequency modulation that defines many
techniques (tremolo, vibrato), and replaces them with (joint time-frequency) scattering
transforms. Lostanlen et al. [cite, 2018 DLfM] introduce this on SOL — the dataset
underlying our OrchideaSOL — and frame IPT as harder than instrument recognition. Wang et
al. extend adaptive scattering to Chinese bamboo flute across two papers [cite, 2020
ICASSP; 2022 TASLP], reporting macro-F1 79.9% vs 35.9% for MFCCs. Ducher & Esling [cite,
2019 ISMIR] decompose cello IPT into orthogonal axes and cross-validate over five
soundbanks (their *minus-1db* protocol), showing a flat 18-way classifier fails (41.5%)
where a taxonomy-aware CQT-RCNN reaches 57.6%. This lineage motivates our *Chroma-as-
control* ablation and our reading of `col legno battuto` as a technique that moves several
acoustic axes at once.

▸ 산란변환 계보. Lostanlen 2018 = OrchideaSOL 전신. Ducher-Esling의 소스분리 교차검증(minus-1db)
  = 우리 LOSO 철학의 도메인 선례(코딩표에서 "source-disjoint ✓"로 코딩). `[High]`

**Spectrogram-CNN and onset-fusion.**
On the deep-learning side, frame-level detectors on Guzheng and Erhu [cite, 2019 arXiv;
2022 ISMIR; 2023 ICASSP] show that fusing note-onset information roughly doubles note-level
F1 (39.5% → 80.8%), directly evidencing that *attack/onset* cues dominate IPT — consistent
with our SHAP finding that the onset-strength envelope is the single most important feature.
CNN-on-spectrogram studies on violin bowing [cite, 2021 MLwithApps] and electric guitar
[cite, 2025 SciRep] report >90% accuracy but under evaluation designs (mixed-source random
5-fold; same-data testing) that their own authors flag as generalization-weak, with
accuracy dropping to ~70% on held-out real recordings.

▸ CNN/온셋 계보. 온셋이 IPT의 핵심이라는 ablation 근거(→ 우리 SHAP 1위 onset-strength와 일치).
  동시에 이 군의 평가설계 약점(랜덤분할·동일데이터 테스트)이 코딩표의 근거. `[Medium~High]`

**Self-supervised pretraining on IPT.**
Closest to the "foundation model" question, MERTech [cite, 2024 ICASSP] fine-tunes
MERT-v1-95M — the same backbone we use — for multi-task IPT/pitch/onset, reaching SOTA on
Guzheng (MI-F1 90.0%) yet only 53.36% on violin RWC technique classification; VioPTT [cite,
2025 arXiv] surpasses it (77.22%) via synthetic augmentation but covers only four
techniques (excluding col legno, tremolo, sul ponticello). Both report accuracy on a single
task and do **not** compare the SSL model against a tuning-matched handcrafted baseline —
precisely the comparison our audit isolates.

▸ 정확히 "foundation model" 질문. MERTech = 우리와 동일 백본. 핵심 관찰: SSL 논문들도 동등
  튜닝예산의 수동특징 baseline과 비교하지 않음 → 우리 함정②가 사는 자리. `[Medium-High]`

**Motion / multimodal (context).**
A distinct branch classifies bowing gestures from IMU/MOCAP/EMG rather than audio
[cite, 2018 iWOAR; 2019/2021 FrontPsychol; 2024 JAIT], often exceeding 99% but typically
with a single performer (N=1) and fixed 80/20 splits, and with motion dominating audio.
We note these only to bound scope: our claim is about *audio-only* classification, where
the technique boundary is acoustically — not kinematically — defined.

▸ 모션/멀티모달은 범위 밖으로 명시 격리(N=1·모션우세). 우리는 오디오 전용 주장임을 못박음. `[Low]`

---

### 3.2 Evaluation pitfalls, semi-supervised learning, and SSL probing

**Tuned baselines and realistic SSL evaluation.**
Our central methodological ancestor is Oliver et al. [cite, 2018 NeurIPS], who show in the
image domain that once baselines are properly tuned, the apparent benefit of semi-supervised
learning shrinks dramatically. We port this warning to audio/MIR in the foundation-model era
and add a failure mode they do not treat: **selection-time leakage** from choosing the
self-training iteration / confidence threshold by test performance. Musgrave et al. [cite,
2020 ECCV] make a parallel "reality check" for metric learning, attributing inflated gains
to weak baselines and flawed protocols.

▸ 방법론적 조상 = Oliver 2018. 우리 추가분 = self-training 회차/τ의 선택-시점 누수(함정①).
  Musgrave = 인접 "reality check" 선례.

**Leakage as a general failure mode.**
Kapoor & Narayanan [cite, 2023 Patterns] document leakage across 294 papers in 17 fields
and propose a reporting taxonomy; our pitfall (i) is an instance of their "test-set
information used in model selection" category, localized to self-training and quantified
with a before/after table. We adopt their *model info sheet* as the template for our §4
checklist.

▸ 누수 일반론 배경 + 체크리스트 형식 모델. 우리 함정①을 이들 taxonomy 한 칸에 매핑.

**Resampling unit and variance decomposition.**
Bouthillier et al. [cite, 2021 MLSys] argue that a valid benchmark must model *all* sources
of variance; fixing some (e.g., data split) while resampling others fabricates significance.
This is the general theory behind our pitfall (iii): a bootstrap that resamples only test
clips while holding folds fixed narrows the interval 2.6–5.0× and reports a difference that
vanishes once the dominant cross-instrument variance component is resampled at the fold unit.

▸ 함정③(재표집 단위)의 일반 이론 = Bouthillier. 우리 2수준 CI 발견의 근거.

**Layer-wise probing and control tasks.**
Pasad et al. [cite, 2021 ASRU] show, layer-by-layer, that a wav2vec2-style speech model's
early layers mirror low-level acoustics (mel-filterbank-like) while upper layers encode
global/semantic structure. As MERT is architecturally the same family, this is the direct
precedent for our descriptor probe, where the onset-strength envelope saturates at the
convolutional front-end (R²=0.77) before any transformer layer. Hewitt & Liang [cite, 2019
EMNLP] justify pairing probes with control tasks so a probe reads representations rather
than learning the task itself — the rationale for keeping our probes low-capacity and our
Chroma control in place.

▸ descriptor probe(메커니즘)의 두 근거. Pasad = 층별 저수준→의미 (우리 L0 포화와 판박이),
  Hewitt = probe 저용량·대조군 설계 정당화.

**Probing conventions in music SSL benchmarks.**
Standardized SSL benchmarks — MARBLE [cite, 2023 arXiv] for music, USB [cite, 2022 NeurIPS]
for SSL broadly — matter to our pitfall (ii) in a specific way: MARBLE's probing head is an
MLP (hidden 512), not a single linear layer. Thus the "convention" does not undertune the
backbone; *we* did, by starting from a last-layer single-linear head. Pitfall (ii) is
therefore not "the field undertunes SSL" but "probe design (head capacity, layer selection)
can flip the conclusion, and we fooled ourselves once by sitting below convention." MERT
itself [cite, 2024 ICLR] is the backbone under test.

▸ 중요한 정직 포인트: 관행(MARBLE=MLP512)은 SSL을 과소평가하지 않음. 과소평가한 건 *우리*
  출발설정(마지막층+단일linear). 함정②를 "우리 자신의 실수"로 정확히 프레이밍.

---

### 3.3 Positioning: an evaluation-practice coding of prior IPT work  (C1.5)

우리는 §3.1의 주법분류 논문들을 세 가지 *평가 관행* 축으로 코딩한다: **(A) 소스/악기-분리
평가인가**(랜덤 분할이면 낙관 편향), **(B) 무거운 모델을 동등한 튜닝·라벨 예산의 값싼
baseline과 비교하는가**, **(C) 일반화 주장의 단위에 맞춘 불확실성을 보고하는가**. 표가
보여주듯, 일부 연구는 (A)를 잘 지키지만(소스분리 교차검증), **(B)와 (C)를 함께 만족하는
연구는 없다** — 본 논문의 세 함정이 정확히 그 빈칸에 있다.

▸ 코딩표 = C1.5. 핵심 메시지: (A)는 일부 지킴, (B)·(C)는 아무도 안 함 → 우리 기여의 자리.

| Prior work | Domain | (A) Source/instrument-disjoint eval | (B) Heavy vs *tuned* cheap baseline, equal budget | (C) Uncertainty at claim's unit |
|---|---|---|---|---|
| Su et al. 2014 [cite] | violin/viola/cello/CB | ~ cross-corpus (RWC→real) | ✗ (no deep/SSL) | ✗ |
| Barbancho 2009 [cite] | violin | ✗ (pooled) | ✗ (rule-based) | ✗ |
| Kruger & Jacobs 2020 [cite] | bowed string | ✗ random 70/30 | ✗ (SVM only) | ✗ |
| Lostanlen 2018 [cite] | SOL (all fam.) | ~ held-out | ✗ (scattering vs MFCC only) | ✗ |
| Ducher & Esling 2019 [cite] | cello | ✓ 5 soundbanks (minus-1db) | ✗ (no cheap baseline audit) | ✗ |
| Wang 2020/2022 [cite] | CBF | ~ held-out | ✗ (scattering+SVM) | ✗ |
| Guzheng/Erhu 2019–2023 [cite] | Chinese strings | ✓ VSB→RW (synth→real) | ✗ | ✗ |
| Alar 2021 [cite] | violin bowing | ✗ mixed-source random 5-fold | ✗ | ✗ |
| MERTech 2024 [cite] | multi (incl. violin) | ~ dataset-standard | ✗ (SSL only, no tuned SVM) | ✗ |
| VioPTT 2025 [cite] | violin | ~ synth→RWC | ✗ (SSL/CRNN only) | ✗ |
| SpectroFusionNet 2025 [cite] | e-guitar | ✗ same-data test | ✗ | ✗ |
| **This work** | **violin/viola/cello/CB** | **✓ LOSO (instrument-grouped)** | **✓ tuned MERT vs tuned 245-d SVM, matched budget** | **✓ 2-level BCa (instrument-resampled)** |

> ✗ = 미충족, ~ = 부분/암묵, ✓ = 명시적으로 충족. (A)열의 `~`는 소스분리를 하되 우리 LOSO만큼
> 엄격하지 않거나 합성→실제 방향의 특수 형태.

▸ 표 코딩 근거는 각 tracker Summary. `~`/`✗` 판정은 최종 정독 후 재확인(특히 Guzheng 계열의
  분할 정의, MERTech의 데이터 표준 분할). 열 (B)·(C)의 전열 `✗`가 이 논문의 논거.

---

## 손볼 곳 (TODO)
- [ ] 인용키 확정 후 `[cite]` → `\cite{}` 치환, bib 작성
- [ ] 06폴더 papers_tracker.csv 완성(현재 4행뿐, 폴더엔 ~15편) — 코딩·인용 대조용
- [ ] Oliver/Musgrave/Bouthillier/Pasad/Hewitt md 정독 후 §3.2 문장 근거 재확인
- [ ] 코딩표 (A)열 `~` 판정을 원문 분할 정의로 확정 (Guzheng VSB→RW, MERTech, VioPTT)
- [ ] §3.1↔§5/§6 상호참조 연결 (Su 일반화붕괴↔우리 LOSO, 온셋fusion↔SHAP)
- [ ] 분량 조절: 투고지 확정 후 §3.1 모션/멀티모달 문단을 1~2문장으로 압축 가능

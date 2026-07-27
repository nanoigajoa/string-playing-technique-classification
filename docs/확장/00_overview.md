# 확장 프로젝트 개요

> 최종 갱신: 2026-07-10

v1/v2가 완료된 시점에서 이 프로젝트가 어디로 확장되는지를 정리한 문서다.
개별 Phase 설계는 각 문서를 참고할 것.

---

## v1/v2에서 확장이 시작되는 지점

v1(SVM F1=0.916)과 v2(CNN F1=0.905)는 두 가지 핵심 한계를 드러냈다.

**첫째, 라벨 의존성.** OrchideaSOL 3,402개 샘플이 전부 전문가 라벨이다.
현실에서 새로운 기법이나 악기로 분류기를 확장하려면 다시 전문가 라벨링이 필요하다.
"이 의존성이 실제로 얼마나 병목인가"를 정량화한 연구는 없다.

**둘째, 해석 범위의 한계.** v1 SHAP는 "어떤 특징이 중요한가"를 보여줬지만,
"대형 모델도 같은 지점에서 실패하는가"라는 비교 질문은 다루지 않았다.
특히 최근 부상하는 Audio-Language Model이 주법처럼 미세한 영역에서 어떤 한계를 보이는지는
학계에서 아직 체계적으로 측정된 바 없다.

이 두 질문이 확장의 출발점이다.

---

## 네 Phase의 내러티브 연결

```
Phase 1 (Self-Training — 같은 도메인)
  "라벨이 X%일 때 분류기가 어디까지 되고 어디서 한계가 나는가"
  실험: OrchideaSOL 라벨을 인위적으로 가려 비라벨 풀 시뮬레이션
      ↓ 실패 클래스 식별: MERT 최악 → sul_ponticello, col_legno_battuto

Phase 2 (ALM 환각 평가)
  "대형 Audio-Language Model도 같은 클래스에서 실패하는가"
  비교 대상: Phase 1 MERT 실패 클래스
      → 실제 결과: 교집합 없음 (ALM 최악: ordinario, tremolo)
      ↓ 두 접근의 약점이 서로 다른 곳에 있다 — 보완적 관계 발견

Phase 3 (SHAP 해석가능성)
  "각 모델의 실패는 음향학적으로 왜 일어나는가"
      → Spectral Flux 방향성이 핵심 (낮음→col_legno 0.18, 중간→ordinario 0.51, 높음→tremolo 0.93)
      → Surrogate fidelity: out-of-sample κ≈0.88 (in-sample 0.998은 암기 착시) — XGBoost SHAP은 MERT의 근사 proxy, 소수 클래스에선 주의

Phase 4 (Cross-Domain 일반화 검증)
  "self-training 효과가 진짜 독립 데이터에서도 성립하는가"
  실험: Philharmonia(별도 녹음)를 Pool A(비라벨)/Pool B(holdout)로 분리
  추가 기여: pseudo-label 정확도 직접 측정 (보통 self-training 연구에서 불가)
      → zero-shot F1=0.423, self-training(τ=0.7, 소스 val 선택) F1=0.449 (+2.7%p)
      → col_legno/sul_ponticello: pseudo-label 채택 0% — 도메인 이동에 구조적 취약
      → Phase 3 연결: Spectral Flux 기반 경계가 도메인 간 유지되지 않는 클래스
```

> ⚠️ **2026-07 방향 전환**: STEP 1 감사에서 **두 평가 함정**(self-training test 누수 + MERT 백본 언더튜닝)이
> 발견되어, 논문의 주 주장이 아래로 재정의됨. 위 4-Phase는 이제 **보조 근거**로 재배치된다.
> 상세: [../method_leakage_note.md](../method_leakage_note.md), [../../연구과정/00_흐름정리.md](../../연구과정/00_흐름정리.md)

**최종 주 주장 (반증된 두 극단 사이의 정확한 지점)**:

**"저차원 강판별 신호(Spectral Flux)가 존재하는 소규모 세밀분류 태스크에서는, 공정하게 평가하면
값싼 수동특징 SVM과 대형 SSL(MERT)이 오차범위 내 동급이다. 이 결론에 이르기까지 self-training 평가의
두 함정(test 누수 → 착각①, 백본 언더튜닝 → 착각②)을 정량화했다."**

보조 근거로 재배치: 대형 ALM 한계(Phase 2), 음향학적 설명(Phase 3), 도메인 취약 클래스(Phase 4).

> **실제 완료 현황**: Phase 1/2/3/4 실행 완료 + 두 함정 감사·수정 완료.
> Phase 1 수치는 누수 제거본으로 갱신됨. Phase 4는 소스-val τ 선택본(τ=0.7).

---

## 연구 흐름과의 연결

이 구조는 2025~2026년 MIR 연구 흐름에서 네 가지 활성화 영역에 대응한다.

| Phase | 대응 연구 흐름 |
|---|---|
| Phase 1 | "라벨링 비용이 세부 음악 태스크 확장의 실질 병목" — self-training/pseudo-labeling 수요 증가 |
| Phase 2 | "ALM 환각 측정이 독립 연구주제로 부상" — 화성/구조가 아닌 미세 영역의 환각 미측정 |
| Phase 3 | "ISMIR 2025 XAI 전용 트랙 신설" — 맞추는 것에서 설명하는 것으로 평가 축 이동 |
| Phase 4 | "사운드뱅크 간 도메인 이동" — 합성 데이터로 실연주 일반화(VioPTT)와 대비되는 접근 |

---

## 파일 구조

```
src/
  phase1_selftraining.py    ← Phase 1 독립 실행 스크립트
  phase2_alm_eval.py        ← Phase 2 독립 실행 스크립트
  phase3_shap.py            ← Phase 3 독립 실행 스크립트
  phase4_crossdomain.py     ← Phase 4 독립 실행 스크립트 (예정)
data/
  split_configs.json        ← Phase 1: 분할 설정
  mert_embeddings.npy       ← Phase 1: OrchideaSOL MERT 임베딩 (3402 × 768)
  mert_labels.npy           ← Phase 1: 대응 라벨 벡터
  mert_meta.csv             ← Phase 1: 파일 경로·악기·클래스 메타데이터
  philharmonia_pool_a.json  ← Phase 4: self-training용 Pool A 파일 목록
  philharmonia_pool_b.json  ← Phase 4: holdout Pool B 파일 목록
  philharmonia_mert_embeddings.npy  ← Phase 4: Philharmonia MERT 임베딩
  eval_set.json             ← Phase 2: 평가 질문-답 세트 (125개)
  eval_results.json         ← Phase 2: 추론 결과 전체 (125개)
results/
  phase1/
    all_results.csv
    learning_curve.png
    tau_tradeoff.png
    per_class_f1.png
    clap_zeroshot.csv
    clap_confusion.png
    report.md
  phase2/
    misclassification_heatmap.png
    report.md
  phase3/
    shap_group_bar.png
    shap_beeswarm_top20.png
    shap_per_class_bar.png
    shap_flux_direction.png
    shap_surrogate_fidelity.png
    shap_case_ordinario.png
    shap_case_tremolo.png
    report.md
  phase4/
    poolB_zeroshot_confusion.png
    pseudolabel_accuracy_by_class.png
    poolB_selftraining_confusion.png
    kshot_curve.png              ← 선택
    report.md
```

---

## 각 Phase 문서

| 문서 | 내용 |
|---|---|
| [01_phase1_selftraining.md](01_phase1_selftraining.md) | MERT 임베딩 추출, 분할 설계, self-training 루프 |
| [02_phase2_alm_eval.md](02_phase2_alm_eval.md) | Qwen2-Audio 로드 전략, 평가 세트 설계, 지표 정의 |
| [03_phase3_shap.md](03_phase3_shap.md) | XGBoost SHAP 심화, 케이스 스터디, 삼각 연결 리포트 |
| [04_philharmonia_crossdomain.md](04_philharmonia_crossdomain.md) | Pool A/B 분할, cross-domain self-training, pseudo-label 검증 |

---

## 최종 패키징 목표

Phase 1+2+3 완료 상태로 ISMIR 2026 LBD(Late-Breaking Demo) 제출 가능.
Phase 4까지 붙이면 MIREX 2026 신규 task 제안 또는 workshop paper 수준의 완성도.

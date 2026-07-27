# 프로젝트: 보잉 주법 분류기 — Self-Training 파이프라인 + ALM 환각 평가 + SHAP 해석가능성

## 배경

- OrchideaSOL 데이터셋으로 5-클래스 보잉 주법(ordinario / pizzicato_secco / tremolo /
  sul_ponticello / col_legno_battuto) 분류기 v1(SVM F1=0.916) + v2(CNN F1=0.905) 완료.
- 이 프로젝트는 자료/연구로드맵_2026.md의 TOP 5 중 #2(SSL+Self-training) +
  #5(ALM 환각 평가) + #3(SHAP 해석가능성)을 통합한 단일 프로젝트.
- **핵심 내러티브**: "라벨 부족이 세부 음악 태스크 확장의 실질적 병목"임을 보이고(Phase 1),
  대형 ALM도 동일 지점에서 실패함을 실증하며(Phase 2), 그 음향적 원인을 SHAP으로 설명(Phase 3).
- 개발 환경: MacBook Air M5, 10코어 CPU, 24GB unified memory, GPU 없음, MPS 백엔드.
- 유료 API 금지. Audio Flamingo 3 / Music Flamingo는 NVIDIA OneWay Noncommercial License
  (학술/포트폴리오 목적은 허용). **Phase 2 ALM 대상은 Qwen2-Audio-7B-Instruct로 결정**
  (audio-flamingo-3는 CUDA 전용으로 MPS 사용 불가).

---

## 결정사항 요약 (계획 수립 단계에서 확정)

| 항목 | 결정 | 근거 |
|---|---|---|
| Phase 1 비라벨 데이터 출처 | OrchideaSOL 라벨 은닉 (a안) | oracle 보유로 pseudo-label 품질 직접 측정 가능, 학습 곡선으로 라벨 병목 실증 |
| Phase 2 ALM | Qwen2-Audio-7B-Instruct | audio-flamingo-3는 CUDA 전용 A100 전용 (8B BF16), MPS 불가 |
| Phase 2 양자화 | float16 우선, OOM 시 optimum-quanto int8 | bitsandbytes는 CUDA 전용, MPS에서 불가 |
| Phase 3 SHAP 방법 | B안: XGBoost SHAP + librosa 직접 연결 | MERT 임베딩→librosa 간접 매핑보다 설명력·명확성 높음 |

---

## Phase 1 — Self-Training 파이프라인 정식화

**목적**: 비라벨 데이터로 어디까지 확장 가능하고 어디서 한계가 나는지를 드러낸다.
"self-training이 잘 작동했다"보다 "라벨 X%에서 plateau가 온다"가 더 중요한 결과.

**백본**: m-a-p/MERT-v1-95M (HuggingFace transformers). MPS 우선, 실패 시 CPU fallback.

### 세부 단계

| 단계 | 내용 | 출력 |
|---|---|---|
| **1-0** | **CLAP 제로샷 기준선**: `laion/clap-htsat-unfused`로 텍스트 프롬프트 기반 제로샷 분류 (학습 없음). "라벨 0개" 성능 기준점 설정 | `results/phase1/clap_zeroshot.csv`, `clap_confusion.png` |
| **1-1** | 분할 설계: GroupKFold(악기 단위, parse_recording_id), seed 4종(5%/10%/20%/50%), 각 3 trial, stratified by class. 나머지 = unlabeled pool | `data/split_configs.json` |
| **1-2** | MERT 임베딩 사전 추출·캐싱 (전체 3,402개, time-axis mean pool) | `data/mert_embeddings.npy` |
| **1-3** | seed만으로 baseline head 학습 (지도학습 기준선, self-training 없음) | seed ratio별 baseline F1 |
| **1-4** | Self-training 루프: τ ∈ {0.7, 0.8, 0.9}, 최대 5 iter, 조기종료(채택률<1%). 매 iter마다 macro F1 / confusion matrix / 채택률 / 클래스별 pseudo-label 분포 기록 | `results/phase1/iter_*.png` |
| **1-5** | 학습 곡선 시각화 (x=labeled ratio, y=F1, CLAP 기준선 + baseline vs. best self-training) + τ별 채택률-정확도 트레이드오프 + markdown 리포트 | `results/phase1/report.md` |

**스크립트**: `src/phase1_selftraining.py` (독립 실행)

---

## Phase 2 — Audio-Language Model 환각(hallucination) 평가

**목적**: 대형 ALM이 화성/구조처럼 데이터가 많은 영역은 잘하지만, 주법처럼
전문가 라벨링이 필요한 미세 영역에서 한계를 보인다는 걸 실증한다.
모델 사양보다 "한계를 드러내는 평가 설계" 자체가 핵심.

**대상 모델**: Qwen2-Audio-7B-Instruct (Apache 2.0, HuggingFace).
로드: `torch_dtype=torch.float16, device_map="auto"`. OOM 시 optimum-quanto int8.

### 세부 단계

| 단계 | 내용 | 출력 |
|---|---|---|
| **2-1** | 모델 로드 + 30초 청크 메모리/속도 테스트, OOM 시 quanto int8 fallback | 로드 확인 |
| **2-2** | 평가 세트 코드 생성: yes/no 템플릿 + open-ended 템플릿, 클래스별 25개(총 125개), OrchideaSOL 정답 라벨 포함. 외부 API 불필요 | `data/eval_set.json` |
| **2-3** | 배치 추론 실행 + 응답 파싱 (yes/no 정규식, open-ended 키워드 매칭) | raw results |
| **2-4** | 주법별 오인식률 표 + 캘리브레이션, Phase 1 공통 실패 클래스 교집합 확인 | `results/phase2/report.md` + 히트맵 PNG |

**스크립트**: `src/phase2_alm_eval.py` (독립 실행)

---

## Phase 3 — SHAP 기반 해석가능성

**목적**: Phase 1 분류기가 왜 맞히고 왜 틀리는지를 음향학적으로 설명하고,
Phase 2에서 드러난 ALM의 한계와 대조한다. SHAP 구현 방법은 유연하게 선택.

**방법**: B안 — XGBoost v1 SHAP + librosa 직접 연결.
MERT 임베딩 SHAP은 librosa feature와 직접 연결 불가(간접 상관분석 필요, 설명력 낮음).

### 세부 단계

| 단계 | 내용 | 출력 |
|---|---|---|
| **3-1** | Phase 1/2 교집합 식별: 두 모델 공통 실패 클래스 → 케이스 스터디 대상 선정 | 대상 클래스 목록 |
| **3-2** | XGBoost SHAP 심화: 대상 클래스 SHAP waterfall/beeswarm, feature group별 기여도 (MFCC+Δ+ΔΔ / Spectral Flux / ZCR / LAT / Chroma 대조군) | SHAP PNG |
| **3-3** | 케이스 스터디 4개: ①둘 다 실패(음향 원인), ②Phase 1만 성공(어떤 특징), ③Chroma ablation 연결, ④pseudo-label 오류 샘플의 음향 특성 | 케이스별 분석 |
| **3-4** | 통합 리포트: "라벨 부족 병목(Phase 1) → 음향적 원인(Phase 3) → 대형 ALM도 동일 지점 실패(Phase 2)" 삼각 연결 | `results/phase3/report.md` |

**스크립트**: `src/phase3_shap.py` (독립 실행)

---

## 코딩 스타일

- 각 phase는 독립 실행 가능한 스크립트 (`src/phase1_selftraining.py` 등).
- requirements.txt 갱신 시 pip install은 venv 안에서 진행 (`--break-system-packages` 금지).
- 모든 결과는 `results/` 폴더에 markdown 리포트 + PNG로 저장.
- 매 phase 끝나면 macro F1 / 환각률 / SHAP 요약을 한 페이지 markdown으로 정리.

## 최종 패키징 목표

완성 후 → ISMIR 2026 LBD(Late-Breaking Demo) 또는 MIREX 2026 신규 task 제안으로 연결.
포트폴리오 서사: "분류기 제작 → 거대 ALM의 한계 실증 → 음향학적 원인 설명" 3단 구조.

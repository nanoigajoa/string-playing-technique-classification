# v2 확장 — CNN + v1 비교

> 작성: 2026-06-15

## Phase 5 — mel-spectrogram + CNN

### 입력 설계

- 오디오를 고정 길이 세그먼트로 자르거나 패딩
- mel-spectrogram을 이미지로 변환 후 CNN 입력

```python
import librosa
import numpy as np

def audio_to_melspec(y, sr, n_mels=128, fixed_length=128):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # 시간축 고정 (패딩 or 자르기)
    if mel_db.shape[1] < fixed_length:
        mel_db = np.pad(mel_db, ((0,0),(0, fixed_length - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :fixed_length]
    return mel_db  # shape: (n_mels, fixed_length)
```

### 모델 선택

| 옵션 | 조건 | 비고 |
|---|---|---|
| 소규모 CNN (직접 구성) | 데이터 충분 시 | 구조 이해에 유리 |
| ResNet 전이학습 | 데이터 부족 시 | ImageNet 사전학습 가중치 활용 |

> OrchideaSOL 샘플 수가 적으면 전이학습이 유리. Phase 1 샘플 카운트 후 결정.

---

## v1 vs v2 비교표

v1과 **동일한 평가 프로토콜**(같은 fold, 같은 클래스)로 비교할 것.

| 항목 | v1 (XGBoost) | v2 (CNN) |
|---|---|---|
| 정확도 (Macro F1) | 측정 후 기입 | 측정 후 기입 |
| 해석성 | Feature Importance / SHAP | 낮음 (GradCAM으로 부분 보완) |
| 데이터 요구량 | 적음 | 많음 |
| 핵심 내용 | 특징 엔지니어링, 도메인 지식 | 딥러닝 파이프라인 |

> 단일 모델의 성능 제시보다, **v1과 v2를 동일한 평가 프로토콜로 비교한 트레이드오프 분석**에 의미가 있다.

---

## Phase 6 — 문서화 및 결과 정리

### 산출물 체크리스트

- [ ] Jupyter Notebook — 재현 가능한 전체 파이프라인 (데이터 로드 → 특징 추출 → 학습 → 평가)
- [ ] Feature Importance 시각화 + 음향학적 해석 서술
- [ ] SHAP summary plot — 기법별 결정적 특징 도출
- [ ] v1 vs v2 트레이드오프 비교 표
- [ ] Confusion Matrix (v1, v2 각각)

### 핵심 서술

> "X 기법은 Y 특징이 결정적이었고, 그 이유는 음향학적으로 Z이며, 누수는 그룹 분할로 막았습니다."

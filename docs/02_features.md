# 특징 추출 파이프라인

## 핵심 설계

오디오는 길이가 제각각이고 XGBoost는 고정 길이 표 데이터를 입력으로 받는다.  
**프레임 단위 특징 → 통계 집계 → 고정 길이 벡터** 변환이 이 프로젝트의 핵심 엔지니어링이다.

```python
import librosa
import numpy as np

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    zcr  = librosa.feature.zero_crossing_rate(y)

    feats = []
    for f in [mfcc, np.atleast_2d(flux), zcr]:
        feats += [
            f.mean(axis=1),
            f.std(axis=1),
            np.percentile(f, 25, axis=1),
            np.percentile(f, 75, axis=1),
        ]
    return np.concatenate(feats)   # 길이 고정 벡터
```

> **평균만 쓰지 말 것.** std·Q25·Q75까지 포함해야 시간적 변화(어택의 날카로움, 감쇠 속도)가 살아남는다.

---

## 특징 우선순위

| 특징 | 변별력 | 이유 |
|---|---|---|
| Log-Attack-Time, Temporal Centroid | ★★★ | Pizz/Spiccato 짧은 어택 vs Arco 지속음 |
| Spectral Flux / ZCR | ★★★ | 과도부·잡음성 (활 소음, 트레몰로 진동) 포착 |
| MFCC (+Δ, ΔΔ) | ★★ | 전반적 음색 + 시간 변화 |
| Spectral Contrast | ★★ | 배음 구조 차이 |
| **Chroma** | ☆ | 음높이 정보 — **음성 대조군(negative control)으로만 포함** |

### Chroma를 왜 빼지 않는가

Chroma는 pitch class 정보라 Pizzicato/Arco처럼 같은 음으로 연주 가능한 기법은 구분하지 못한다.  
빼는 대신 **"왜 안 되는지 증명하는 대조군"으로 명시적으로 포함** → 면접에서 오히려 도메인 이해가 드러난다.

---

## librosa 주요 함수 참조

| 특징 | librosa 함수 |
|---|---|
| MFCC | `librosa.feature.mfcc` |
| Spectral Contrast | `librosa.feature.spectral_contrast` |
| Spectral Flux | `librosa.onset.onset_strength` |
| ZCR | `librosa.feature.zero_crossing_rate` |
| Chroma | `librosa.feature.chroma_stft` |
| Onset (어택 타이밍) | `librosa.onset.onset_detect` |

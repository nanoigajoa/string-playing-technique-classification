"""
특징 추출 모듈.
오디오 파일 하나를 받아 고정 길이 특징 벡터를 반환한다.
"""

from pathlib import Path
import numpy as np
import librosa


def parse_recording_id(filename: str) -> str:
    """파일명에서 녹음 ID 추출 (GroupKFold 그룹 키로 사용)
    Vn-ord-A#3-ff-4c-N.wav → Vn-A#3-ff  (악기 + 음 + 강약)
    """
    parts = Path(filename).stem.split('-')
    if len(parts) >= 4:
        return f'{parts[0]}-{parts[2]}-{parts[3]}'
    return parts[0]


def _aggregate(feat: np.ndarray) -> np.ndarray:
    """프레임 단위 특징 → [mean, std, Q25, Q75] 통계 집계"""
    if feat.ndim == 1:
        feat = feat[np.newaxis, :]
    return np.concatenate([
        feat.mean(axis=1),
        feat.std(axis=1),
        np.percentile(feat, 25, axis=1),
        np.percentile(feat, 75, axis=1),
    ])


def extract_features(filepath: str, sr: int = 22050) -> np.ndarray:
    """오디오 파일 → 245차원 고정 길이 특징 벡터

    특징 구성:
        MFCC (13) + Δ + ΔΔ  × 4통계 = 156
        Spectral Contrast (7) × 4통계 = 28
        Onset-strength envelope (1) × 4통계 = 4   (코드 prefix 'flux' — 상세 아래)
        ZCR (1)               × 4통계 = 4
        RMS Energy (1)        × 4통계 = 4
        Chroma (12, 음성대조군) × 4통계 = 48
        Log first-onset time (1 scalar)  = 1
        합계: 245

    용어 주의 (A5):
      · 'flux_*' 컬럼의 실체는 librosa.onset.onset_strength — mel 스펙트로그램의
        프레임간 차분을 half-wave rectify 후 mel축 합산한 **onset strength envelope**다.
        (수학적으로 mel 기반 spectral flux의 일종이지만) 논문 표기는 "onset strength
        envelope"로 통일한다. 이 특징이 per-feature SHAP 1위다.
      · 'log_first_onset_time'은 **첫 onset의 시각**에 log1p를 취한 값이다. MIR의
        log-attack-time(onset→peak 상승시간)이 **아니다** — 과거 오표기를 이름으로 정정했다.
        (SHAP 중요도 최하위라 결론에 영향 없음)
    """
    y, sr = librosa.load(filepath, sr=sr)
    parts = []

    # MFCC + delta + delta2
    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_d  = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    parts  += [_aggregate(mfcc), _aggregate(mfcc_d), _aggregate(mfcc_d2)]

    # Spectral Contrast
    sc = librosa.feature.spectral_contrast(y=y, sr=sr)
    parts.append(_aggregate(sc))

    # Onset-strength envelope (mel 기반 half-wave-rectified spectral flux)
    # 컬럼 prefix는 하위호환 위해 'flux' 유지 — 실체는 onset_strength envelope
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    parts.append(_aggregate(flux))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    parts.append(_aggregate(zcr))

    # RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    parts.append(_aggregate(rms))

    # Chroma — 음성 대조군 (pitch class 정보, 기법 분류에 무효)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    parts.append(_aggregate(chroma))

    # Log first-onset time — 첫 onset의 시각에 log1p (attack time이 아님, A5 참조)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    log_first_onset = np.log1p(onsets[0]) if len(onsets) > 0 else 0.0
    parts.append(np.array([log_first_onset]))

    return np.concatenate(parts)


def make_feature_names() -> list[str]:
    """특징 벡터의 각 차원에 대응하는 이름 목록 반환"""
    names = []
    stats = ['mean', 'std', 'q25', 'q75']
    for prefix, n in [('mfcc', 13), ('mfcc_d', 13), ('mfcc_d2', 13)]:
        for s in stats:
            names += [f'{prefix}_{s}_{i}' for i in range(n)]
    for prefix, n in [('sc', 7), ('flux', 1), ('zcr', 1), ('rms', 1), ('chroma', 12)]:
        for s in stats:
            names += [f'{prefix}_{s}_{i}' for i in range(n)]
    names.append('log_first_onset_time')
    return names


FEATURE_NAMES = make_feature_names()

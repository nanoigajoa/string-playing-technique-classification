"""
v2 CNN용 PyTorch Dataset.
미리 계산된 mel-spectrogram numpy 배열을 받아 텐서로 변환한다.
학습 시 SpecAugment(시간/주파수 마스킹)를 적용한다.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

# mel-spectrogram 공통 파라미터
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FIXED_FRAMES = 128  # ~3초 @ hop=512, sr=22050


def audio_to_mel(filepath: str, sr: int = SR) -> np.ndarray:
    """오디오 파일 → 정규화된 mel-spectrogram (n_mels, FIXED_FRAMES)"""
    y, sr = librosa.load(filepath, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 시간축 고정
    if mel_db.shape[1] < FIXED_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, FIXED_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :FIXED_FRAMES]

    # [0, 1] 정규화
    lo, hi = mel_db.min(), mel_db.max()
    mel_db = (mel_db - lo) / (hi - lo + 1e-8)
    return mel_db.astype(np.float32)


def spec_augment(mel: np.ndarray,
                 freq_mask_max: int = 15,
                 time_mask_max: int = 20) -> np.ndarray:
    """SpecAugment: 주파수/시간축 무작위 마스킹"""
    mel = mel.copy()
    n_mels, n_frames = mel.shape

    # 주파수 마스킹
    f = np.random.randint(0, freq_mask_max + 1)
    f0 = np.random.randint(0, n_mels - f + 1)
    mel[f0:f0 + f, :] = 0.0

    # 시간 마스킹
    t = np.random.randint(0, time_mask_max + 1)
    t0 = np.random.randint(0, n_frames - t + 1)
    mel[:, t0:t0 + t] = 0.0

    return mel


class MelDataset(Dataset):
    """
    Parameters
    ----------
    mels   : np.ndarray, shape (N, n_mels, frames)
    labels : np.ndarray, shape (N,)
    augment: bool — 학습 시 True, 평가 시 False
    """
    def __init__(self, mels: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.mels    = mels
        self.labels  = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mels[idx]
        if self.augment:
            mel = spec_augment(mel)
        return torch.FloatTensor(mel).unsqueeze(0), self.labels[idx]

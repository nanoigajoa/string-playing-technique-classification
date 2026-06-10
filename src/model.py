"""
v2 CNN 모델 정의.
Input: (batch, 1, 128, 128) — mono mel-spectrogram
Output: (batch, n_classes)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: tuple = (2, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class SmallCNN(nn.Module):
    """
    4 ConvBlock → GlobalAvgPool → Dropout → FC

    비대칭 풀링 (2, 1): 주파수 축만 절반씩 줄이고 시간 축은 보존.
    멜 스펙트로그램에서 시간(가로)과 주파수(세로)는 물리적으로 다른 축이므로
    동일한 2×2 풀링은 배음 해상도와 시간 정보를 함께 손실시킨다.

    Feature map 크기 변화 (입력 1×128×128 기준):
        ConvBlock1 (1→32):    freq 64 × time 128
        ConvBlock2 (32→64):   freq 32 × time 128
        ConvBlock3 (64→128):  freq 16 × time 128
        ConvBlock4 (128→128): freq  8 × time 128
        GlobalAvgPool:         1×1 → 128-dim
    """

    def __init__(self, n_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32,  pool=(2, 1)),
            ConvBlock(32,  64,  pool=(2, 1)),
            ConvBlock(64,  128, pool=(2, 1)),
            ConvBlock(128, 128, pool=(2, 1)),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

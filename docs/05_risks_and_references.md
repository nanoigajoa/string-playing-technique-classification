# 리스크 및 참고 자료

## 리스크 대응표

| 리스크 | 대응 |
|---|---|
| 기법별 샘플 부족 (특히 Spiccato) | Phase 1에서 사전 카운트, 부족 클래스 통합/제외 |
| 데이터 누수 → 정확도 과대평가 | 공식 fold / GroupKFold 강제 사용 |
| Chroma에 시간 낭비 | 대조군으로 명시적 위치 부여, 빼지 않고 왜 안 되는지 증명 |
| v1 클래스가 너무 쉬워 학습 효과 없음 | 어려운 보잉 아티큘레이션 클래스 반드시 포함 |
| v2 데이터 부족으로 CNN 과적합 | ResNet 전이학습으로 전환, Data Augmentation (pitch shift, time stretch) |

---

## 참고 자료

### 데이터셋

- **OrchideaSOL** (사용 데이터셋) — 52종 연주기법 라벨
  - 논문: https://arxiv.org/abs/2007.00763
  - 다운로드: https://www.orch-idea.org/datasets/
- **TinySOL** (사용 불가 — ordinario only)
  - https://zenodo.org/records/3685367

### 라이브러리

- **mirdata** — 재현 가능한 MIR 데이터 로더: https://mirdata.readthedocs.io/
- **librosa** — 오디오 특징 추출: https://librosa.org/
- **SHAP** — 모델 해석: https://shap.readthedocs.io/

# 프로젝트 개요 및 로드맵

## 목표

현악기 연주기법(Arco / Pizzicato / Tremolo / Spiccato 등)을 자동 분류하는 ML 파이프라인을 두 단계로 구축한다.

- **v1:** 오디오 특징 추출 → 통계 집계 → XGBoost / SVM 비교 + Feature Importance 해석
- **v2:** mel-spectrogram → CNN, v1과 트레이드오프 비교

---

## 단계 로드맵

| 단계 | 내용 | 문서 |
|---|---|---|
| Phase 1 | 데이터 확보 및 클래스 설계 | [01_data.md](01_data.md) |
| Phase 2 | 특징 추출 파이프라인 | [02_features.md](02_features.md) |
| Phase 3 | 데이터 분할 및 누수 방지 | [01_data.md](01_data.md) |
| Phase 4 | v1 모델링 (RF / XGBoost / SVM) | [03_v1_modeling.md](03_v1_modeling.md) |
| Phase 5 | v2 확장 (CNN) 및 비교 | [04_v2_cnn.md](04_v2_cnn.md) |
| Phase 6 | 문서화 및 포트폴리오 | [04_v2_cnn.md](04_v2_cnn.md) |

리스크 및 참고 자료 → [05_risks_and_references.md](05_risks_and_references.md)

---

## 모델 역할 요약

| 모델 | 역할 |
|---|---|
| Random Forest | 베이스라인 |
| XGBoost | 주력 모델 (Feature Importance / SHAP 해석 대상) |
| SVM | 비교 모델 |
| CNN (v2) | mel-spectrogram 기반, v1과 트레이드오프 비교 |

[2025_SciRep] SpectroFusionNet a CNN approach utilizing spectrogram fusion for electric guitar play recognition

저자: Ganesh Kumar Chellamani, Aishwarya N, Chandhana C, Kanwaljeet Kaur, Rakesh Thoppaen Suresh Babu
출판: Scientific Reports, Vol. 15, Article 16842, 2025
링크/DOI: 10.1038/s41598-025-00287-w
읽은 날: 2026-07-05
상태: Done


CORE

1. 한줄 요약
MFCC·CWT·Gammatone 스펙트로그램을 ResNet50으로 추출한 후 late fusion(max voting)으로 결합해 전기 기타 9기법 분류 99.12% 달성.

2. 핵심 내용 & 주요 기여
3종 스펙트로그램을 병렬 추출한다: MFCC(청각 지각적), CWT Morlet(시변 해상도), Gammatone(청각 필터뱅크). ResNet50(26M), MobileNetV2(3.4M), InceptionV3(24M) 경량 모델로 각 스펙트로그램의 특징을 추출한다. 융합 전략을 early fusion(채널 방향 결합 후 단일 모델)과 late fusion(독립 처리 후 max voting/weighted avg/concatenation 결합) 두 가지로 체계적으로 비교한다. 핵심 발견은 late fusion이 early fusion보다 일관되게 우수하며, max voting이 가장 효과적이라는 점이다. 10종 ML 분류기(SVM, RF, KNN, LMT, MLP, Linear SVM 등)를 비교해 LMT와 Linear SVM이 우수함을 확인한다. Mitsou(2023) 데이터셋을 사용한다.

3. 한계 / 비판점
Mitsou 2023 단일 연주자 데이터셋 사용으로 99.12%는 과적합 의심이 크다. 실시간 테스트(YouTube 클립)에서 70.9%로 급락해 실세계 적용성이 제한적이다. ResNet50이 50M+ 파라미터로 "경량" 주장과 모순된다. AdaBoost와 Naive Bayes는 fused feature에서 24~46%로 매우 낮아 고차원 퓨전 피처에 취약함이 드러난다.

4. 추가로 읽을 references
Mitsou et al. (2023) Data in Brief — 사용 데이터셋 원논문
He et al. (2016) — ResNet 원논문
Sandler et al. (2018) — MobileNetV2 원논문


OPTIONAL

A. 방법론
입력: WAV 오디오(48kHz → 8kHz 리샘플). 3종 스펙트로그램 추출: S_M(MFCC), S_C(CWT Morlet, w0=6), S_G(Gammatone).

Early fusion: 두 스펙트로그램을 채널 방향으로 결합 후 단일 CNN 통과. 3가지 페어(SM+SC, SC+SG, SM+SG).

Late fusion: 각 스펙트로그램 → ResNet50 → flatten → feature vector. 그 후 세 가지 전략:
- Max voting: F_late,max = max(F_M, F_C, F_G) (element-wise)
- Weighted avg: F_late,wavg = Σ w_i·F_i
- Concatenation: [F_M; F_C; F_G]

분류기: 10종 ML, 하이퍼파라미터 그리드서치. 학습: 80% train / 20% test, 5-fold CV 검증.

B. 데이터셋
Mitsou (2023)과 동일: 549개 오디오 샘플, 9기법, 단일 기타리스트, 3기타×3앰프. 클래스별 train/test 분리(alternate picking: train 64/test 17, 등).

C. 결과
개별 스펙트로그램 최고: ResNet50+MFCC+LMT 96.49%.
5-fold CV 최고: MFCC+Gammatone, max voting, Linear SVM 98.79%.
단일 테스트 최고: MFCC+Gammatone, max voting, MLP 100%(Table 6).
전체 보고 최고: 99.12% accuracy, 100% precision, 100% recall.
실시간(YouTube): SM+SG, max voting, Linear SVM 70.9%.
Late > Early fusion: 평균 +5~15%p 향상.

D. 내 프로젝트(string-playing-technique-classification)와의 비교

유사점: SVM 계열 분류기를 사용하는 점이 동일하다. 다중 스펙트로그램 비교(MFCC, CWT, Gammatone)는 내 프로젝트의 피처 선택 실험에 직접 참고 가능하다.

차이점:
1. 악기/기법 — 전기 기타 9기법 vs 내 활현악기 5기법. 기법 겹침 없음.
2. 데이터 신뢰성 — 단일 연주자 데이터로 99.12% vs 내 GroupKFold LOSO 91.6%. 내 수치가 오히려 더 신뢰할 수 있는 일반화 지표.
3. 모델 복잡도 — ResNet50 피처 추출 + ML 분류기 앙상블 vs 내 경량 SVM.

적용 가능한 인사이트: max voting이 concatenation보다 효과적이라는 발견은 내 프로젝트에서 피처 앙상블을 설계할 때 직접 참고할 수 있다. 99.12%(단일 연주자)→70.9%(실시간)의 급락은, 내 프로젝트 논문의 limitation 섹션에서 "이솔레이션 데이터로 학습한 모델의 실세계 성능 하락 위험"을 논의할 때 대조 사례로 인용하기 좋다.

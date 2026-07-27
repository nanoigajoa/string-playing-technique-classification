[2024_JAIT] Ensemble of Multimodal Deep Learning Models for Violin Bowing Techniques Classification

저자: Zain Muhammed, Nagamanoj Karunakaran, Pranamya P. Bhat, Arti Arya
출판: Journal of Advances in Information Technology, Vol.15, No.1, pp.40-48 (2024)
링크/DOI: doi.org/10.12720/jait.15.1.40-48
읽은 날: 2026-06-22
상태: Done

CORE

1. 한줄 요약
바이올린 보잉기법 5가지(détaché/legato/martelé/spiccato/staccato)를 오디오(Essentia)+모션(Myo EMG+IMU) 멀티모달 feature로 1D-CNN(오디오)과 2개의 2D-LSTM(EMG, IMU) 앙상블(softmax voting)로 분류해서, 다이내믹/템포 변화가 섞인 데이터에서도 99%대의 매우 높고 일관된 정확도를 달성함.

2. 핵심 내용 & 주요 기여
8명의 전문 바이올리니스트가 Kreutzer Study No.2(표준 교습 레퍼토리)를 연주한 공개 데이터셋(Sarasua, Caramiaux, Tanaka, Ortiz 2017)을 사용해서, primary(5기법×80개=400), dynamic(3기법, pp~ff 다이내믹 변주, 240), tempo(3기법, slow~fast 템포 변주, 240), full(880, 전체 통합) 4개 서브셋으로 평가함. 오디오 73개(Essentia spectral/temporal/tonal descriptor)와 모션 105개(EMG 8채널+IMU 13채널×5개 time-domain feature: MAV/RMS/WL/ZC/SSC) feature를 각각 PCA(40/30 component)로 축소한 뒤, 먼저 전통 ML(LR/KNN/SVM/NB/MLP)로 오디오 단독 vs 모션 단독을 비교함 — MLP가 가장 우수하고, "detaché"는 오디오 분류기가 더 잘 잡고 "martelé"는 모션 분류기가 더 잘 잡는다는 상호보완적 패턴을 발견함(8번 HHMM 논문과는 반대로, 이 논문에서는 전반적으로 오디오가 모션보다 약간 더 우수함).

이 상호보완성에 기반해 1D-CNN(오디오)+2D-LSTM(EMG)+2D-LSTM(IMU) 3개 모델의 예측을 softmax voting으로 앙상블하는 구조를 제안함. 결과는 primary 98.8%, dynamic 100%, tempo 99.2%, full 99.5%로 모든 서브셋에서 일관되게 높음 — 특히 비교 대상인 Alvaro et al.(2017, 같은 데이터셋 원저자, Particle Filter/HMM5 모델)은 dynamic/tempo 서브셋에서 IMU+EMG 결합 시 정확도가 37~41.2%까지 무너지는데, 이 논문의 앙상블은 그런 조건 변화에도 견고함을 유지함 — 다이내믹/템포 변주에 대한 강건성이 핵심 기여.

3. 한계 / 비판점
8명의 연주자 데이터를 사용했지만, 10-fold stratified CV가 연주자 단위 분리(LOSO 등)인지 단순 무작위 샘플 분리인지 논문에 명시되지 않음 — 같은 연주자의 다른 반복 연주가 train/test에 같이 들어갈 가능성이 있어, 이전에 읽은 단일연주자 논문들(8,9번)의 문제가 다인원 데이터에서도 형태를 바꿔 재발할 수 있음. 5개 기법 모두 단일 정해진 악보(Kreutzer 연습곡)를 반복 연주한 것이라 음악적 맥락의 다양성이 제한적임. 5개 기법이 전부 보우 운동학적 기법(articulation)이라 우리 프로젝트의 음색 변화 기법(col_legno_battuto, sul_ponticello)은 다루지 않음. 앙상블의 voting 방식(가중치를 어떻게 정했는지)에 대한 구체적 설명이 부족하고, 단일 모델(CNN/LSTM) 대비 앙상블의 개선폭이 일부 서브셋(primary: 97.5%→98.8%)에서는 크지 않아 비용 대비 효과에 대한 논의가 부족함.

4. 추가로 읽을 references
Sarasua, Caramiaux, Tanaka, Ortiz (2017), "Datasets for the analysis of expressive musical gestures" (Movement Computing) — 이 논문이 사용한 8명 다인원 멀티모달 바이올린 데이터셋의 원 출처, 직접 받아서 우리 프로젝트의 오디오 단독 모델과 비교 실험에 활용할 수 있을지 검토할 가치가 있음.
Sun, Liu, Chang (2020), "Deep learning-based violin bowing action" (Sensors) — IMU+깊이카메라로 80%대 정확도를 보인 연구, 카메라 기반 모션 캡처와의 비교 지점으로 읽어볼 만함.

OPTIONAL

A. 방법론
오디오는 Essentia로 73개 spectral/temporal/tonal feature(MFCC, GFCC, Flux, HFC, RMS, zero-crossing, pitch salience 등) 추출 후 PCA로 67개 선택(CNN 입력 차원과는 약간 다름, 본문 표기 혼재). 모션은 Myo의 EMG 8채널+IMU 13채널(쿼터니언, Euler angle, 가속도, 자이로) 총 21채널에서 각각 MAV/RMS/WL/ZC/SSC 5개 time-domain feature를 0.25초 윈도우(50% overlap)로 추출. 모델은 3개: 1D-CNN(오디오, conv1d 2층+batchnorm+relu+maxpool, 67차원 입력), 2D-LSTM(EMG, 42 timestep×40 feature), 2D-LSTM(IMU, 42 timestep×65 feature). 세 모델의 출력을 softmax voting으로 결합해 최종 예측.

B. 데이터셋
8명의 전문 바이올리니스트가 Kreutzer Study No.2(C장조)를 각 조건당 10회 연주. primary(5기법: détaché/legato/martelé/spiccato/staccato, 80개씩=400), dynamic(3기법만: détaché/legato/spiccato, pp~ff 다이내믹 변주, 240), tempo(3기법만, slow~fast 템포 변주, 240), full(전체 통합 880). 오디오 44.1kHz, EMG 8채널+IMU 13채널 동시 기록.

C. 결과
전통 ML 비교(오디오 단독 vs 모션 단독, full dataset): MLP가 오디오 99.7%, 모션 97.6%로 가장 우수, NB가 가장 낮음(82.7%/81.7%) — 오디오가 모션보다 약간 더 정확함(8번 HHMM 논문과 반대 경향). 앙상블 최종 결과: primary 98.8%(CNN-오디오 97.5%, LSTM-EMG 90.5%, LSTM-IMU 95% 대비 개선), dynamic 100%(개별 최고 99.2%보다도 개선), tempo 99.2%, full 99.5%(CNN 99%, LSTM-EMG 89.5%, LSTM-IMU 96.7% 대비). 비교 대상 Alvaro et al.(2017)은 primary에서는 90%대지만 dynamic/tempo에서 IMU+EMG 결합 시 37~41.2%로 붕괴 — 이 논문의 앙상블이 조건 변화에 훨씬 강건함을 보여주는 핵심 비교.

D. 내 프로젝트(string-playing-technique-classification)와의 비교

유사점: 같은 바이올린 단일 악기 대상의 다중 클래스 보잉기법 분류이고, 8,9번에서 읽은 Dalmazzo & Ramirez 논문들을 직접 인용/비교 대상으로 다뤄서 우리가 이미 읽은 두 논문과 연결되는 문헌 네트워크를 형성함(같은 데이터 계보: Alar et al.[14]=7번 논문도 비교표에 포함됨). 8명의 다인원 데이터를 쓴다는 점은 우리 LOSO 설계(여러 악기)와 "단일 표본으로의 과적합을 피하려는" 방향성이 비슷하지만, 검증 방식(연주자 단위 분리 여부 불명)은 우리보다 불확실함.

차이점:
1. 입력 모달리티 — 오디오+모션(EMG+IMU) 멀티모달 앙상블이고, 우리는 순수 오디오. 이 논문은 오디오가 모션보다 근소하게 우수하다고 보고하는데, 이는 8번 논문(모션이 오디오보다 훨씬 우수, 93% vs 39%)과 상반된 결과 — 모달리티 간 상대적 중요도는 기법 종류/feature 설계에 따라 달라질 수 있음을 시사함.
2. 기법의 성격 — 5개 모두 보우 운동학적 articulation이고, 우리의 음색 변화 기법(col_legno_battuto, sul_ponticello)은 다루지 않음.
3. 검증 — 8명의 연주자가 있지만 연주자 단위 train/test 분리가 명시되지 않은 10-fold CV — 우리 LOSO(악기 단위 명시적 분리)보다 검증의 엄격성이 불투명함.
4. 모델 구조 — CNN+LSTM 앙상블(소프트맥스 voting)이라는 비교적 복잡한 구조. 우리는 v1(SVM 등)/v2(단일 CNN)로 더 단순한 구조 비교.

적용 가능한 인사이트: 오디오와 모션 중 어느 쪽이 더 중요한지가 기법군에 따라 달라진다는 이 논문과 8번 논문의 상반된 결과는, discussion에서 "멀티모달 확장의 효과는 기법 종류에 따라 다를 수 있다"는 균형 잡힌 논의로 인용할 수 있음. 또한 다이내믹/템포 변주(pp~ff, slow~fast)가 섞인 데이터에서 단일 모달리티 모델의 성능이 크게 흔들릴 수 있다는 Alvaro et al. 비교 결과는, 우리 OrchideaSOL 데이터도 다이내믹/템포 변주를 충분히 포함하는지 점검해보고, 포함하지 않는다면 이를 일반화의 한계로 discussion에 명시할 근거가 됨.

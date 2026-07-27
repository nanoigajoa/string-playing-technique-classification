[2018_iWOAR] A Machine Learning Approach to Violin Bow Technique Classification: a Comparison Between IMU and MOCAP systems

저자: David Dalmazzo, Simone Tassani, Rafael Ramírez
출판: iWOAR 2018 (5th international Workshop on Sensor-based Activity Recognition and Interaction), Berlin
링크/DOI: doi.org/10.1145/3266157.3266216
읽은 날: 2026-06-22
상태: Done

CORE

1. 한줄 요약
바이올린 보우 스트로크 4가지(Détaché/Martelé/Spiccato/Ricochet)를 오디오가 아니라 모션 데이터(고가의 MOCAP vs 저가의 Myo armband)로만 분류해 비교한 연구로, 두 기기 모두 99% 이상의 정확도를 보여 저가 웨어러블로도 충분히 정밀한 동작 인식이 가능함을 보임.

2. 핵심 내용 & 주요 기여
TELMI 프로젝트(음악교육 기술, 연습 중 실시간 피드백을 목표) 산하 연구로, 전문 바이올리니스트 1명이 4개 보우 기법을 스크립트된 음계 연습곡으로 연주하는 동안 MOCAP(적외선 카메라 8대, marker 기반, 250fps)과 Myo armband(IMU+EMG, 200Hz)를 동시에 기록함. 오디오는 분류 입력이 아니라 두 모션 데이터의 시간축 동기화에만 쓰임(Essentia onset 검출) — 즉 순수 모션 기반 분류임. WEKA로 J48 decision tree와 SVM을 비교한 결과, MOCAP 99.46%(J48)/99.51%(SVM), Myo 99.85%(J48)/98.92%(SVM)로 저가 기기(Myo)가 고가 MOCAP과 거의 동등하거나 오히려 살짝 더 나은 정확도를 보임 — 음악교육용 저가 웨어러블 제품의 실용성을 시사함.

3. 한계 / 비판점
단일 연주자(N=1) 데이터만 사용해서 다른 연주자에 대한 일반화는 전혀 검증되지 않음 — 우리 LOSO나 다른 논문들의 minus-1db/cross-soundbank 같은 일반화 검증이 전혀 없음. 28,820개 "샘플"이 연속된 제스처 시퀀스를 윈도우로 나눈 것으로 보이는데, train/test 분리가 단순 stratified CV(WEKA 기본)로만 이루어져서 같은 제스처 인스턴스의 인접 프레임이 train/test에 동시에 들어가는 시간적 leakage 가능성이 있음 — 99%대의 비현실적으로 높은 정확도가 이 때문일 수 있는데 논문에서 전혀 논의되지 않음. 오디오를 전혀 분류에 사용하지 않아 우리 프로젝트(오디오 전용)와의 직접적인 방법론 비교는 어려움. 4개 기법 모두 보우 스트로크 계열로, col_legno_battuto 같은 타격 기법이나 sul_ponticello 같은 위치 기법은 다루지 않음. 통계적 유의성 검정이나 신뢰구간 보고가 없음.

4. 추가로 읽을 references
Peiper, Warden, Garnett (2003), "An Interface for Real-time Classification of Articulations Produced by Violin Bowing" (NIME) — 초기 violin bow articulation 분류 연구(decision tree, 전자기 모션 추적), 84%(2 gesture)~71%(4+ gesture) 정확도. 모션 기반 violin technique 분류의 원류격 연구.
Gibet et al. (2006), "Gesture Analysis of Violin Bow Strokes" — augmented violin, 가속도계+kNN으로 3개 기법(Détaché/Martelé/Spiccato) 분류, 다이내믹(pp/mf/ff)에 따라 정확도가 달라진다는 발견이 흥미로움.

OPTIONAL

A. 방법론
입력은 오디오가 아니라 모션 데이터 2종: MOCAP(적외선 카메라 8대, marker 기반, 250fps, 0.1mm 정밀도, 몸/바이올린/활에 부착된 다수의 3D 마커)과 Myo armband(팔에 착용하는 저가 웨어러블, 8채널 EMG + 3축 자이로/가속도/지자기, 200Hz). 오디오는 Essentia로 onset만 추출해서 두 모션 데이터를 동기화하는 데만 쓰임(분류 input 아님). MOCAP과 Myo의 샘플링레이트를 선형보간으로 통일하고 신호 피크 정렬로 시간축을 맞춤. 분류기는 WEKA의 J48(decision tree)과 SVM을 비교.

B. 데이터셋
전문 바이올리니스트 1명(Madeleine Mitchell, Royal College of Music)이 4개 보우 기법(Détaché/Martelé/Spiccato/Ricochet)을 각각 스크립트된 2-octave 음계 연습곡으로 연주(기법별 BPM/박자가 다름, 각 32개 제스처). MOCAP과 Myo 데이터를 정렬해서 총 28,820개 샘플(윈도우 단위로 보임)을 구성.

C. 결과
J48 decision tree: MOCAP 99.46%, Myo 99.85%(Myo가 약간 더 높음). SVM: MOCAP 99.51%, Myo 98.92%(MOCAP이 약간 더 높음). 두 기기 모두 99%대의 매우 높은 정확도를 기록함. Confusion matrix(MOCAP, J48)에서는 Spiccato-Détaché 간 약간의 혼동이 있는데, 저자들은 오디오 feature(바운싱 효과를 들을 수 있는 정보)가 없어서라고 추정함.

D. 내 프로젝트(string-playing-technique-classification)와의 비교

유사점: 같은 바이올린 단일 악기 대상이고 기법을 자동으로 구분한다는 목표가 같음. Confusion에서 Spiccato-Détaché(바운싱 계열)가 헷갈린다는 점은, 우리의 col_legno_battuto 어려움과 비슷하게 물리적으로 유사한 동작/소리를 내는 기법군 내부 혼동이라는 공통 패턴.

차이점:
1. 입력 모달리티 — 이 논문은 순수 모션 데이터(오디오 전혀 사용 안 함)로 분류하고, 우리는 순수 오디오로 분류함. 완전히 다른 센서 모달리티라 직접적인 방법론 비교는 어려움.
2. 검증 엄격성 — 이 논문은 단일 연주자(N=1) 데이터의 단순 stratified CV만 사용하고 다른 연주자/세션에 대한 일반화는 전혀 검증하지 않음 — 우리 LOSO보다 훨씬 약한 검증임. 시간적 leakage 가능성도 99%대 정확도의 원인일 수 있음.
3. 기법의 성격 — 4개 모두 보우 스트로크(운동학적) 기법이고, col_legno_battuto나 sul_ponticello처럼 음색을 바꾸는 위치/타격 기법은 다루지 않음.
4. 목적 — 이 논문은 음악 교육(실시간 피드백)을 목표로 하고, 우리는 오디오 기반 자동 인식/분석을 목표로 함 — 어플리케이션 맥락이 다름.

적용 가능한 인사이트: 이 논문은 우리 프로젝트에 직접적인 방법론을 주기보다, 모션 데이터를 추가하면 오디오만으로 구분하기 힘든 기법(Spiccato-Détaché처럼 바운싱 여부)을 더 잘 구분할 수 있다는 가능성을 보여주는 참고 사례로 인용할 수 있음 — 향후 멀티모달 확장을 고려한다면 이 논문이 그 방향의 선례가 됨. 다만 col_legno_battuto 같은 우리의 핵심 어려움에는 직접 답을 주지 않으므로, "다음 단계" 레퍼런스 정도로 의미를 한정하는 게 적절함.

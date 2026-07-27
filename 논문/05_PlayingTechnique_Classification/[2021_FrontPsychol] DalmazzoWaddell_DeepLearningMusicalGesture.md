[2021_FrontPsychol] Applying Deep Learning Techniques to Estimate Patterns of Musical Gesture

저자: David Dalmazzo, George Waddell, Rafael Ramírez
출판: Frontiers in Psychology, Vol. 11, Article 584042, January 2021
링크/DOI: 10.3389/fpsyg.2020.584042
읽은 날: 2026-07-05
상태: Done


CORE

1. 한줄 요약
Myo 팔찌 IMU 데이터만으로 바이올리니스트의 8가지 보우잉 기법을 99.2% 정확도로 분류 — col legno 포함.

2. 핵심 내용 & 주요 기여
탐지 대상 기법은 martelé, staccato, détaché, ricochet, legato, trémolo, collé, col legno (8클래스)이며 오디오를 사용하지 않고 IMU 센서(Myo 팔찌)만 사용한다. 입력은 acc(x,y,z), gyro(x,y,z), Euler angles(x,y,z) = 9채널, 200Hz이다. 피험자는 5명 전문가 + 3명 학생 바이올리니스트(총 8명)이다. 전통 ML(Gaussian Process 97.4%, Extra Trees 95.3%, RF 94.9%)과 DNN(CNN, 3DMultiHeaded_CNN, CNN_LSTM)을 비교하며, CNN_LSTM(filter=512)이 99.234%로 최고 성능을 달성했다. 같은 연구팀의 이전 논문(HHMM, iWOAR 2018)에서 7기법 처리했던 것을 8기법으로 확장하고 DNN을 도입한 확장 연구이다.

3. 한계 / 비판점
오디오 피처가 없어 실제 MIR 파이프라인과의 통합이 어렵고, 악기에 별도 센서 착용이 필요하다. 80/20 고정 분리로 LOSO(Leave-One-Subject-Out) 검증이 없어 연주자 독립성이 보장되지 않는다. col legno 샘플 수가 가장 적어(20개) 결과 과대해석 주의가 필요하다. tremolo F1이 상대적으로 낮으며(precision 0.89), Myo 팔찌 단종으로 재현성 문제가 있다.

4. 추가로 읽을 references
Dalmazzo & Ramírez (2019) Frontiers in Psychology — 동일 연구팀의 HHMM 이전 연구
Dalmazzo, Tassani, Ramírez (2018) iWOAR — IMU vs MOCAP 비교 선행 연구
Sarasua et al. (2017) — 다인원 멀티모달 바이올린 데이터셋


OPTIONAL

A. 방법론
입력은 Myo 팔찌 9채널 IMU 시계열(200Hz)이다. 전처리: 윈도우 0.375초(75 프레임), 50% 오버랩, 각 윈도우 [75×9] 텐서. 모델 구조:
- CNN: 2D conv (75×9 입력) → flatten → softmax
- 3DMultiHeaded_CNN: 3개 병렬 CNN 헤드(각기 다른 커널 크기) → concat
- CNN_LSTM: CNN 피처 추출 → LSTM 시퀀스 처리 → softmax (filter 크기 32/64/128/256/512 실험, 512에서 최고 성능)
학습 방식은 80% train / 20% test, 5회 반복 평균이다.

B. 데이터셋
바이올리니스트 8명(전문가 5 + 학생 3). 8기법 × 각 약 50~100 샘플(기법별 불균형, col legno 최소 20개). Myo 팔찌를 활을 쥔 팔의 전완부에 착용. EMG 데이터는 수집했으나 노이즈 과다로 미사용.

C. 결과
전통 ML 최고: Gaussian Process 97.447%. DNN 최고: CNN_LSTM(filter=512) 99.234%±1.106. col legno F1=0.97(20개 샘플임에도 높은 성능 — 타악기성 제스처가 IMU에서 뚜렷). tremolo가 상대적으로 낮음(precision 0.89) — 진동 패턴이 다른 기법과 혼동. 전문가 vs 학생 차이 미보고.

D. 내 프로젝트(string-playing-technique-classification)와의 비교

유사점: col legno, tremolo가 내 프로젝트 5기법(col_legno_battuto, tremolo)과 직접 겹친다. 같은 연구팀(Dalmazzo & Ramírez)의 이전 HHMM 논문(iWOAR 2018)과 연결되는 문헌 계보로, 내 프로젝트가 다루는 기법군과 가장 직접적으로 연관된 IMU 연구다.

차이점:
1. 입력 모달리티 — IMU 전용 vs 내 프로젝트 오디오 전용. 두 접근은 상호보완적이다.
2. 검증 엄격성 — 80/20 고정 분리(LOSO 없음) vs 내 GroupKFold LOSO. 내 프로젝트가 더 엄격한 일반화 검증.
3. 기법 구성 — 8가지 보우 운동학적 기법 vs 내 5기법(음색 변화 포함). sul_ponticello, ordinario는 이 논문에 없음.

적용 가능한 인사이트: col legno의 타악기성 IMU 패턴이 오디오에서도 독특한 스펙트럼 패턴(고주파 타격음)을 가진다는 점을 discussion에서 연결하여, 오디오 전용으로도 col_legno_battuto 분류가 가능한 근거를 보강할 수 있다. tremolo IMU precision 0.89 저하가 이 논문과 내 프로젝트에서 모두 나타난다면, 해당 기법의 분류 난이도에 관한 교차 증거로 인용할 수 있다.

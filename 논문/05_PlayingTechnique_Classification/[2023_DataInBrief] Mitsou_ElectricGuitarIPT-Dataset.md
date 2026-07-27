[2023_DataInBrief] A multimodal dataset for electric guitar playing technique recognition

저자: Alexandros Mitsou, Antonia Petrogianni, Eleni Amvrosia Vakalaki, Christos Nikou, Theodoros Psallidas, Theodoros Giannakopoulos
출판: Data in Brief, Vol. 52, Article 109842, 2024 (available online 22 November 2023)
링크/DOI: 10.1016/j.dib.2023.109842
읽은 날: 2026-07-05
상태: Done


CORE

1. 한줄 요약
전기 기타 9가지 기법의 멀티모달(오디오+비디오) 공개 데이터셋을 구축하고 SVM/CNN 베이스라인 실험으로 검증.

2. 핵심 내용 & 주요 기여
549개 오디오+비디오 샘플(WAV 48kHz 스테레오 + MP4)로 구성된 데이터셋을 공개한다. 9기법: alternate picking, legato, tapping, sweep picking, vibrato, hammer-on, pull-off, slide, bend. 오디오뿐 아니라 영상 데이터를 포함해 자세/손 위치 인식 연구를 가능하게 한다. 3가지 기타 × 3가지 앰프 시뮬레이션 조합으로 음색 다양성을 확보하며, MuseScore 악보 파일도 제공한다. Zenodo(DOI: 10.5281/zenodo.10075352)에 공개되어 있다.

3. 한계 / 비판점
단일 연주자만 사용해 연주자 다양성이 없고 개인 특성에 편향될 수 있다. 실제 곡/공연 맥락 없이 짧은 개별 기법 이솔레이션 녹음이다. 스마트폰(Xiaomi Redmi Note 8)을 가정환경에서 사용한 녹음이라 전문 스튜디오 대비 노이즈가 포함된다. alternate picking/legato/tapping이 81개인 반면 sweep picking은 36개로 클래스 불균형이 있다. hammer-on과 pull-off 간 혼동이 심하다(역방향 제스처 유사성).

4. 추가로 읽을 references
Lostanlen et al. (2018) DLfM — Extended Playing Techniques 데이터셋 (이 논문이 인용)
Su et al. (2014) TASLP — 스파스 세프스트럴 기타 기법 분류
Giannakopoulos (2015) — pyAudioAnalysis 라이브러리 (베이스라인에 사용)


OPTIONAL

A. 방법론
데이터 수집: Xiaomi Redmi Note 8 스마트폰으로 비디오 촬영, FFmpeg으로 WAV 추출. 기타: River West RWG, Carvin DC-400, Palm Bay Cyclone P7-XR(모두 험버커 픽업). 앰프 시뮬레이션: Avid Eleven Rack(Soldano SL100, Plexiglass, DC-Modern).

SVM: pyAudioAnalysis 136-dim 피처(20개 68-dim 단기 벡터 평균), 1초 세그먼트, RBF 커널, GridSearch(C∈[0.1,1,10,50,100,1000], gamma∈[0.0001,0.001,0.01,0.1,1,10]).

CNN: mel-spectrogram 입력(1초, STFT 50ms/no overlap → 20×128), 4개 conv layer(5×5, 32→64→128→256채널, 2×2 MaxPool), 3개 linear layer(2048→1048→256→9), LeakyReLU + 2D BatchNorm.

검증: (1) leave-one-exercise-out 5-fold, (2) leave-one-guitar-out 3-fold, (3) leave-one-amplifier-out 3-fold.

B. 데이터셋
총 549개 오디오+비디오 파일. 클래스별: alternate picking/legato/tapping 81개, vibrato/hammer-on/pull-off/slide/bend 54개, sweep picking 36개. 총 13.6GB(비디오 13GB + 오디오 1.2GB). MuseScore(.mscz) 악보 파일 18개 포함.

C. 결과
SVM: Exp1(exercise) 67.8%/F1=62.3%, Exp2(guitar) 84.2%/F1=81.2%, Exp3(amp) 83.4%/F1=81.9%.
CNN: Exp1 76.5%/F1=76.9%, Exp2 75.4%/F1=83.1%, Exp3 80.7%/F1=82.3%.
주요 혼동: hammer-on↔pull-off(역방향 제스처), tapping↔legato(지판 공통점), sweep picking↔tapping.

D. 내 프로젝트(string-playing-technique-classification)와의 비교

유사점: SVM과 CNN을 베이스라인으로 비교하는 점이 내 프로젝트와 동일하다. leave-one-guitar-out / leave-one-amplifier-out 검증 방식이 내 GroupKFold LOSO와 동일한 "unseen 조건에서의 일반화" 철학을 공유한다.

차이점:
1. 악기 — 전기 기타 vs 내 활현악기. 기법 겹침 없음.
2. 데이터 규모와 다양성 — 단일 연주자 549개 vs OrchideaSOL 다중 악기 전문 녹음.
3. 목적 — 데이터셋 기여 논문 vs 내 프로젝트의 방법론 연구.

적용 가능한 인사이트: leave-one-guitar-out(84.2%)이 leave-one-exercise-out(67.8%)보다 훨씬 높다는 결과는, 내 프로젝트의 GroupKFold LOSO 설계 결정을 정당화하는 외부 증거로 인용할 수 있다. 또한 SpectroFusionNet(2025)이 같은 데이터에서 99.12%를 달성한 것은 단일 연주자 데이터의 과적합 위험을 경고하는 대조 사례로 유용하다.

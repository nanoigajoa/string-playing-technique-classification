[2019_ISMIR] Folded CQT RCNN for Real-time Recognition of Instrument Playing Techniques

저자: Jean-François Ducher, Philippe Esling
출판: ISMIR 2019, Delft, Netherlands
링크/DOI: zenodo.org/records/3527908 (archives.ismir.net/ismir2019)
읽은 날: 2026-06-22
상태: Done

CORE

1. 한줄 요약
첼로 연주기법(IPT)을 4개의 독립적인 축(exciter/vibrator, left-hand, waveform, interaction position)으로 분해해서 멀티태스크로 분류하고, 5개의 독립된 사운드뱅크로 교차검증(minus-1db)하면, 단일 18-class 분류기는 일반화에 완전히 실패하지만(41.5%) taxonomy+CQT+LSTM 조합은 훨씬 견고하게 일반화됨(57.6%)을 보여준 실시간 IPT 분류 연구.

2. 핵심 내용 & 주요 기여
첼로 IPT를 4개 독립 축으로 나누는 taxonomy를 제안함: Axis1 exciter/vibrator(ordinario, con legno, pizzicato, pizz Bartok, pressured bowing, hand/knuckle hit), Axis2 left-hand(vibrato, non vibrato, glissando, trill), Axis3 waveform(sustained, staccato, tremolo), Axis4 interaction position(ordinario, sul tasto, sul ponticello, harmonics, spiccato 등). 5개의 독립적인 상업용 첼로 사운드뱅크(EWQL, VSL, ISI, VO, ConTimbre)로 합성 데이터를 만들고, Livshin(2007)이 제안한 minus-1db(사운드뱅크 하나씩 빼고 교차평가) 방식으로 일반화 성능을 검증함.

가장 중요한 발견은, IPT를 하나의 평평한(flat) 18-class 문제로 직접 분류하면(A18 모델) 같은 사운드뱅크 안에서는 90%대 정확도가 나오지만 다른 사운드뱅크로 교차평가하면 41.5%까지 무너진다는 것. 반면 4축 taxonomy로 분해하고, 새로 제안한 Folded CQT(F-CQT, pitch spiral의 일반화로 배음 구조를 작은 conv kernel에 모아줌) 입력과 LSTM front-end를 쓰면 minus-1db 18-class accuracy가 57.6%까지 개선됨. 축별로는 exciter/vibrator가 가장 쉽고(87%) interaction position(bow position)이 가장 어려움(63.5%). 실시간 반응성도 측정해서, exciter 축은 공격 후 70ms 안에 70% 정확도에 도달할 만큼 빠르다는 것도 보임.

3. 한계 / 비판점
여전히 실제 첼로 솔로 녹음이 아니라 상업용 샘플러를 Max/MSP로 구동해 합성한 데이터로 전체 실험을 수행함(저자도 실제 녹음 코퍼스를 future work로 언급). minus-1db로 일반화를 개선했다고는 해도 최종 57.6%는 동일 사운드뱅크 내 90%대에 비하면 여전히 큰 격차이고, 완전히 해결된 문제는 아님. Taxonomy 구성이 다소 임의적인 부분이 있음 — 저자도 pressured bowing을 axis1에, harmonics를 axis4에 억지로 배정했다고 인정하고, harmonic trill처럼 실제로는 연주 가능하지만 어떤 사운드뱅크에도 없는 조합도 있음. 4축 예측을 18-class로 합성하는 rule-based 시스템 자체의 오류 전파는 따로 분석하지 않음. SHAP 같은 해석가능성 분석이나 confusion 기반 정성 분석은 없음 — 순수하게 정량적 ablation(CQT vs LMS, FC vs LSTM 등)만 제시함.

4. 추가로 읽을 references
[15] Livshin, "Automatic Musical Instrument Recognition and Related Topics" (PhD 논문, 2007) — minus-1db 방법론의 원조, 우리 LOSO 설계와 같은 철학적 뿌리.
[18] Lostanlen & Cella, "Deep Convolutional Networks on the Pitch Spiral for Music Instrument Recognition" (ISMIR 2016) — 이 논문의 F-CQT가 일반화한 pitch spiral 아이디어의 원본.

OPTIONAL

A. 방법론
여러 spectral transform을 비교함: log-mel-spectrogram(128bin) vs 저해상도/고해상도 CQT vs 새로 제안한 Folded CQT(F-CQT). F-CQT는 CQT의 크로마 빈을 5도권(cycle of fifths) 순서로 재배열하고 옥타브를 접어(fold) 인접하게 배치해서, 한 음의 배음들이 작은 conv kernel(3 fifths x 4 octaves) 안에 모이도록 만듦. 모델은 3개 conv module로 된 CNN back-end(F-CQT 필터, 3x3, 6x2(vibrato/trill 같은 긴 패턴 포착용) 세 필터를 concat)에, 4개의 독립적인 front-end classifier(IPT 축마다 하나, FC 또는 LSTM)를 붙임. Configuration E에서는 note의 공격(attack) 위치를 함께 예측하는 보조 분류기를 추가해서 그 예측을 IPT classifier 입력에 concat하는 멀티태스크 구조. 60프레임(~1.4초) 고정 길이 시퀀스로 처리하고, 공격 이후 시간에 따른 정확도 변화(reactivity)까지 측정함.

B. 데이터셋
5개의 독립적인 상업용 첼로 사운드뱅크(EastWest Quantum Leap, Vienna String Library, IRCAM Solo Instruments, Virtual Orchestra, ConTimbre)를 Max/MSP로 구동해 무작위 음/화음 시퀀스를 합성하고, 피치 디튠(±20Hz)과 리버브로 augmentation해서 총 13.5시간 분량을 만듦. JAMS 표준으로 어노테이션하고 4축 taxonomy로 라벨링.

C. 결과
같은 사운드뱅크 내 평가(homogeneous)에서는 모든 설정이 90%대를 기록하지만, 단일 18-class 분류기(A18)를 minus-1db(다른 사운드뱅크로 교차평가)로 보면 41.5%까지 무너짐(최저 30.3%). 4축 taxonomy와 다양한 입력/모델 설정을 적용하면 minus-1db 18-class accuracy가 49.6%(LMS)에서 57.6%(LSTM+joint onset)까지 개선됨. CQT 계열이 LMS보다 대부분 축에서 나음. 축별로는 exciter/vibrator가 가장 쉽고(87%) interaction position이 가장 어려움(63.5%) — bow position은 사람도 구분하기 어려운 축이라고 저자들도 설명함. Reactivity 실험에서는 exciter 축이 공격 후 70ms 안에 70% 정확도에 도달할 만큼 빠르지만, waveform 축(staccato 등)은 note가 끝나야 정확히 판별되어 반응이 느림.

D. 내 프로젝트(string-playing-technique-classification)와의 비교

유사점: 이 논문은 우리 프로젝트와 거의 같은 악기 계열(첼로, 활현악기)을 다루고, taxonomy를 자세히 보면 우리 5개 클래스가 사실 여러 축에 걸쳐 분산되어 있음을 알 수 있음 — ordinario(Axis1 베이스라인), pizzicato_secco(Axis1 finger/string), col_legno_battuto(Axis1 con legno이면서 동시에 Axis3 waveform 성격도 가짐), tremolo(Axis3 waveform), sul_ponticello(Axis4 interaction position). 즉 우리가 "5-class 단일 분류"로 묶은 것이 이 논문 기준으로는 서로 다른 독립적인 축이 섞인 것. 핵심 발견("단일 축으로 IPT를 분류하면 일반화가 약하다")도 우리 프로젝트의 동기(LOSO로 다른 악기에 대한 일반화 검증)와 본질적으로 같은 문제의식이고, minus-1db(Livshin 2007이 원조)가 사실상 LOSO의 원형이라는 점도 확인됨.

차이점:
1. Taxonomy 구조 — 이 논문은 IPT를 4개의 독립적 축으로 분해해서 멀티태스크로 학습하고 rule-based 시스템으로 18-class를 합성함. 우리는 5개 클래스를 평평한 단일 분류 문제로 둠.
2. 실시간성 — 이 논문은 real-time(저지연) 분류가 핵심 목표라 짧은 윈도우와 공격 이후 반응시간까지 정량 평가함. 우리는 이미 분리된 전체 클립을 한번에 분류하는 오프라인 설정이라 지연시간을 고려하지 않음.
3. 일반화 검증 단위 — 이 논문은 minus-1db로 사운드뱅크(가상악기/샘플러) 단위 분리, 우리는 LOSO로 실제 악기(violin/viola/cello/contrabass) 단위 분리 — 비슷한 철학이지만 분리 기준의 단위가 다름.
4. 입력표현 — 이 논문은 Folded CQT라는 배음 구조를 명시적으로 인코딩하는 변환을 새로 제안함. 우리는 mel-spectrogram(v2)과 표준 spectral feature(v1)를 쓰고 harmonic-aware한 변환은 시도하지 않음.

적용 가능한 인사이트: col_legno_battuto가 가장 어렵다는 우리 발견은, 이 논문의 taxonomy 관점에서 보면 "원래 다른 두 축(exciter + waveform)이 동시에 바뀌는 복합 기법이라 단일 라벨로는 분리하기 어렵다"는 구조적 설명이 가능함 — discussion에서 이 논문의 taxonomy를 인용해 "우리 5개 클래스가 균일한 난이도의 분류 문제가 아니라 서로 다른 축의 기법이 섞여 있다"는 한계로 언급할 수 있음. 또한 Folded CQT처럼 배음 구조를 명시적으로 인코딩하는 입력표현을 v2 CNN의 추가 실험으로 시도해서 mel-spectrogram과 성능을 비교해볼 수 있음.

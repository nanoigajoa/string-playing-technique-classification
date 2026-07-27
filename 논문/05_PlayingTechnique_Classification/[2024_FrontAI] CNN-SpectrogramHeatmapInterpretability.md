[2024_FrontAI] Interpreting CNN Models for Musical Instrument Recognition Using Multi-Spectrogram Heatmap Analysis: A Preliminary Study

저자: Rujia Chen, Akbar Ghobakhlou, Ajit Narayanan
출판: Frontiers in Artificial Intelligence, Vol.7, Article 1499913 (2024)
링크/DOI: doi.org/10.3389/frai.2024.1499913
읽은 날: 2026-06-22
상태: Done

CORE

1. 한줄 요약
악기 분류(주법 분류 아님) 문제에서 6가지 스펙트로그램 표현(STFT/Log-Mel/MFCC/Chroma/Spectral Contrast/Tonnetz)이 CNN에 각각 어떤 정보를 주는지, Integrated Gradients 히트맵과 4개 통계 지표(Difference Mean/KL Divergence/JS Divergence/EMD)로 정량 비교한 해석가능성 중심의 예비 연구로, MFCC/Log-Mel이 대부분 악기에서 우수하지만 악기별로 최적 표현이 다르다는 것을 보임.

2. 핵심 내용 & 주요 기여
이 논문은 주법(playing technique) 분류가 아니라 NSynth 데이터셋의 10개 악기(Bass, Brass, Flute, Guitar, Keyboard, Mallet, Organ, Reed, String, Vocal) 분류 문제를 다루지만, "어떤 스펙트로그램 표현이 CNN의 의사결정에 어떻게 기여하는가"를 정량적으로 분석하는 해석가능성(interpretability) 방법론이 핵심 기여임. 각 스펙트로그램 타입마다 별도의 CNN(6 conv layer + pooling/dropout/dense)을 학습시키고, Integrated Gradients로 입력의 어떤 time-frequency bin이 예측에 기여했는지 히트맵을 추출함. 이후 같은 악기의 여러 샘플 간 히트맵을 비교해서 Difference Mean(평균 차이), KL Divergence, JS Divergence, EMD(분포 형태 차이) 4개 지표로 "이 악기에 대해 feature importance가 얼마나 일관적인가"를 측정함.

결과적으로 정확도는 Log-Mel 0.62, STFT 0.56, MFCC 0.55, Spectral Contrast 0.51, Chroma 0.36, Tonnetz 0.21 순(본문 텍스트와 표가 다소 불일치하게 서술되어 있어 주의가 필요)이고, 악기별로 최적 표현이 다름을 보임 — Bass/Flute/Mallet/Organ/Vocal은 MFCC가 강하고, Brass/Reed는 Chroma/Tonnetz(harmonic relationship 포착)가 강하고, String/Vocal은 모든 표현에서 변동성이 크고 분류가 어려움. 히트맵 일관성 지표(EMD 등)에서 Bass는 변동성이 크고(분산된 feature) Organ/Reed는 일관적(국소화된 feature)이라는 발견이, 단순 정확도 숫자만으로는 보이지 않는 "왜 어떤 악기가 더 어려운가"에 대한 추가 설명을 제공함.

3. 한계 / 비판점
저자들이 명시한 것처럼 NSynth의 서브셋만 사용한 예비연구(preliminary study)로 샘플 규모가 작고, 합성 음원(synthesized)이라 실제 녹음의 다양성/노이즈/중첩(overlapping instruments)을 반영하지 못함 — 우리 OrchideaSOL도 동일한 한계를 공유하는 지점. 본문 텍스트의 정확도 서술("MFCC가 가장 높은 0.62")과 Figure 4의 표(Log-Mel이 0.62, MFCC가 0.55)가 서로 일치하지 않는 등 보고의 정합성에 문제가 있어 보임 — 수치를 인용할 때 주의가 필요함. 이 연구는 분류 성능을 개선하려는 목적이 아니라 순수하게 해석가능성 분석이 목적이라고 명시하며, 실제로 기존 벤치마크(Leaf 69.2%, EfficientLeaf 71.7%)보다 정확도가 낮음(최고 0.62)을 스스로 인정함 — 우리 프로젝트처럼 성능 자체를 최적화하려는 연구와는 목적이 다름. 4개 통계 지표(Difference Mean/KL/JS/EMD)가 각각 무엇을 측정하는지는 명확하지만, 이 지표들이 실제로 "어떤 feature가 중요한지"를 사람이 이해할 수 있는 설명으로 연결되는 구체적 사례(예: 특정 주파수대가 왜 중요한지)는 충분히 제시되지 않음 — 정성적 해석이 다소 추상적임.

4. 추가로 읽을 references
Sundararajan, Taly, Yan (2017), "Axiomatic attribution for deep networks" (ICML) — 이 논문이 사용한 Integrated Gradients 기법의 원 논문, 우리가 향후 v2 CNN의 해석가능성을 SHAP 외에 다른 방법으로도 검증하고 싶다면 참고할 방법론.
Han, Chen, Yang (2016), "Deep convolutional neural networks for predominant instrument recognition in polyphonic music" (IEEE/ACM TASLP) — MFCC+CNN으로 악기 인식 성능을 개선한 연구, feature representation과 CNN 아키텍처의 상호작용을 다룬 선행연구로 인용됨.

OPTIONAL

A. 방법론
입력은 NSynth 오디오 샘플을 6가지 스펙트로그램(STFT, Log-Mel, MFCC, Chroma, Spectral Contrast, Tonnetz)으로 변환한 2D 이미지. 스펙트로그램 타입마다 별도의 CNN(conv2d 3블록, 각 블록 32/64/128 필터, 3x3 kernel, ReLU+same padding, 2x2 max pooling, dropout 0.25 후 dense 256+dropout 0.5+sigmoid 출력)을 200 epoch, Adam optimizer, binary cross-entropy로 학습(10개 악기 각각에 대해 binary classification 형태로 보임). Integrated Gradients(baseline=0 행렬에서 실제 입력까지 보간하며 gradient를 누적·평균)로 각 샘플의 히트맵을 추출하고, 동일 악기의 샘플 간 히트맵을 Difference Mean/KL Divergence/JS Divergence/EMD 4개 지표로 비교해서 feature importance의 일관성과 분산을 정량화.

B. 데이터셋
NSynth 데이터셋의 서브셋, 10개 악기(Bass, Brass, Flute, Guitar, Keyboard, Mallet, Organ, Reed, String, Vocal) — 모두 신디사이저로 합성된 음원이며, 우리 프로젝트의 주법 분류 클래스와는 무관한 악기 종류(family) 분류 문제임.

C. 결과
정확도: Log-Mel 0.62(표 기준 최고), STFT 0.56, MFCC 0.55, Spectral Contrast 0.51, Chroma 0.36, Tonnetz 0.21. 악기별 precision은 표현마다 크게 다름 — MFCC는 Bass/Mallet/Organ에서 높은 precision, Tonnetz/Chroma는 Brass/Reed에서 강함(harmonic relationship 포착), Vocal/String은 모든 표현에서 상대적으로 어려움. 히트맵 일관성 분석(EMD)에서는 Bass가 가장 변동성이 크고(분산된 feature importance) Organ/Reed가 가장 일관적(국소화된 feature importance)임 — 이는 정확도만으로는 설명되지 않는 "어떤 악기가 모델에게 더 안정적인 신호를 주는가"에 대한 추가 정보.

D. 내 프로젝트(string-playing-technique-classification)와의 비교

유사점: 직접적인 task 유사성(주법 분류)은 없지만, 방법론적으로는 매우 관련성이 높음 — 우리 v2 CNN이 mel-spectrogram을 입력으로 쓰고 v1이 SHAP으로 feature importance를 분석하는 것처럼, 이 논문은 "스펙트로그램 표현 종류가 CNN의 의사결정에 어떻게 다르게 기여하는가"를 정량화하는 해석가능성 프레임워크를 제시함 — 우리 프로젝트의 SHAP 분석(v1)을 v2 CNN으로 확장할 때 참고할 수 있는 대안적/보완적 방법(Integrated Gradients + 분포 비교 지표).

차이점:
1. 과제 정의 — 이 논문은 악기 종류(family) 분류이고, 우리는 동일 악기 내 주법 분류 — 완전히 다른 task. 직접적인 성능/방법론 비교는 어려움.
2. 분석 목적 — 이 논문은 성능 개선이 아니라 순수 해석가능성이 목적이고, 실제로 최신 벤치마크보다 정확도가 낮음을 인정함. 우리는 성능 최적화(SVM 0.916)와 분석(SHAP)을 함께 추구함.
3. 데이터 — NSynth(완전 합성, 단일음, note 단위)는 우리 OrchideaSOL과 유사하게 통제된 고립음 데이터셋이라는 점은 공통적 한계.
4. 해석가능성 방법 — Integrated Gradients(gradient 기반, CNN 전용)과 우리의 SHAP(모델 무관, 주로 SVM/XGBoost에 적용)은 서로 다른 해석가능성 도구 — 우리가 v2 CNN에도 해석가능성 분석을 추가하고 싶다면 SHAP 대신 또는 함께 Integrated Gradients를 고려할 수 있음.

적용 가능한 인사이트: 이 논문이 보여준 "스펙트로그램 표현마다 악기/기법별로 더 잘 맞는 것이 다르다"는 핵심 발견은, 우리 프로젝트에서도 mel-spectrogram(v2) 외에 MFCC나 CQT 같은 다른 표현을 col_legno_battuto 같은 어려운 클래스에 별도로 테스트해볼 동기가 됨. 또한 이 논문의 4개 분포비교 지표(Difference Mean/KL/JS/EMD)는, 우리 SHAP 분석에서 "col_legno_battuto의 feature importance가 클래스 내에서 얼마나 일관적인가"를 정량화하는 보조 지표로 응용해볼 수 있음 — 만약 col_legno_battuto 샘플들의 feature importance 분포가 다른 클래스보다 더 분산되어 있다면(이 논문의 Bass 사례처럼), 이는 col_legno_battuto가 왜 가장 어려운 클래스인지에 대한 정량적 근거가 될 수 있음.

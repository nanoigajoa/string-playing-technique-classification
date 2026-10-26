# 악기종 보류 평가에서 현악 주법 분류 모델 비교: 라벨 경계, 모델 선택, 추론 범위의 감사

> 내부 검토용 원고, 2026-10-01. 표와 그림은 canonical 저장 예측에 근거한다. 문헌과 검토 형식을 보강했으며, 공개 재현 자료와 최종 저자 선언은 아직 준비 중이다.

## 초록

현악 주법 분류에서 모델 순위는 라벨 접근 경계와 모델 선택 절차에 영향을 받는다. 우리는 OrchideaSOL 현악 부분집합의 네 악기종, 다섯 주법, 3,402개 클립으로 이 절차를 감사했다. 외부 fold에서 한 악기종 전체를 보류하고, 제한 라벨 조건에서는 각 예산마다 세 번 표집했다. MERT 분류 헤드와 245차원 음향 특징 SVM은 분리된 검증셋에서 설정을 고르고, 학습 fold에서만 class weight와 scaler를 계산했다. 5%, 10%, 20%, 50% 라벨 예산에서 MERT/SVM의 악기종별 평균 macro F1은 각각 0.6655/0.6613, 0.7634/0.7542, 0.8242/0.8316, 0.8599/0.8681이었다. 수정된 자기학습은 5%에서 감독학습 0.6686보다 낮은 0.6364를 보였으나, 10~50%에서는 차이가 작고 방향이 달랐다. 선택 checkpoint를 재로딩한 예측으로 클립, 녹음 파일명 기반 proxy, 악기종 단위 재표집을 비교했을 때, MERT−SVM의 악기종 단위 구간은 네 예산 모두 0을 포함했다. 이 결과는 모델 동등성이나 새 연주자에 대한 일반화를 입증하지 않는다. 본 연구의 재현 가능한 기여는 숨은 정답 라벨 접근, 학습 밖 데이터로 계산한 통계량, 모델 선택과 평가 파일의 불일치를 찾아 수정하고, 주장을 관찰한 그룹 수준에 맞춘 평가 절차다.

**주제어:** 연주 주법; 음악 표현; 준지도학습; 평가; 누수; 재현성.

## 1. 연구 질문과 범위

본 연구는 두 질문을 다룬다. 첫째, 악기종을 통째로 보류했을 때 제한 라벨로 학습한 MERT 헤드와 수동 특징 SVM의 성능 차이가 어떻게 변하는가? 둘째, 자기학습의 의사 라벨과 학습 target, 모델 선택 검증셋, 저장 checkpoint와 예측 파일을 추적할 때 기존 성능 해석이 유지되는가? 결론은 OrchideaSOL의 네 **악기종**에 조건부이다. 물리적 악기, 연주자, 세션 식별자가 확보되지 않아 이들에 대한 일반화를 시험하지 못했다.

이 원고는 평가 경계와 결과 출처가 확인된 MERT 헤드, SVM, 자기학습, CNN을 다룬다. 기존 원고의 layer-weighted probe, LoRA, SHAP 기전 설명, 외부 데이터 전이 성능은 동일한 canonical 검증을 통과하지 않았으므로 모델 순위나 원인 설명의 근거로 사용하지 않는다.

### 1.1 관련 연구

Kruger and Jacobs (2020)는 음향·Hartley 특징과 SVM을 이용한 현악 주법 분류를 연구했다. Li et al. (2024)은 음악 특화 자기지도 표현 MERT를 제안했다. 본 연구는 다섯 클래스 부분집합에서 고정 MERT 표현과 특징 SVM을 비교하며, 과제와 평가 절차가 달라 선행 논문의 수치와 직접 비교하지 않는다.

Oliver et al. (2018)은 준지도학습 평가에서 감독학습 대조군의 중요성을 보여준다. Kapoor and Narayanan (2023)은 기계학습 기반 과학의 누수를 검토했고, Rodríguez-Algarra et al. (2019)은 음악 분류의 교란을 통제된 개입으로 분석했다. 이를 바탕으로 라벨 접근과 평가 요인을 추적한다. 여러 결함을 동시에 수정한 이번 결과로 각 결함의 인과 기여를 분리하지는 않는다.

### 1.2 연구 기여

본 연구는 다섯 현악 주법의 악기종 보류 평가에서 관측 라벨과 의사 target의 전달 경로, 학습 분할에서 적합한 통계량, 검증으로 선택한 checkpoint와 저장 예측을 연결한 감사 사례를 제시한다. 제한 라벨 조건의 MERT/SVM 비교와 설정을 맞춘 자기학습 진단을 보고하고, 고정 예측에 대한 재표집 단위를 바꿔 추론 범위의 민감도를 확인한다. 결과 자료는 모델 파일 없이 재계산할 수 있으며, 결론은 관찰된 네 악기종에 조건부이다.

## 2. 데이터와 방법

### 2.1 데이터와 분할

원자료는 OrchideaSOL (Cella et al., 2020)이다. 공식 자료는 악기·음고·다이내믹스 균형을 고려한 5분할을 제공한다. 이번 악기종 보류 4분할은 다른 연구 질문에 맞춘 자체 분할이며 공식 벤치마크 분할을 사용한 결과가 아니다.

다섯 주법은 `ordinario` 1,193개, `pizzicato_secco` 639개, `tremolo` 1,041개, `sul_ponticello` 302개, `col_legno_battuto` 227개다. 네 악기종은 바이올린, 비올라, 첼로, 콘트라베이스다. 주요 지표는 클래스 불균형을 반영하는 macro F1이며, 클래스별 F1과 confusion matrix도 함께 저장했다.

외부 평가는 악기종별 leave-one-instrument-out 네 fold를 사용한다. 5%, 10%, 20%, 50% 제한 라벨 예산은 fold마다 세 번 표집되어 예산당 12개 실행을 이룬다. 표의 값은 먼저 같은 fold의 세 실행을 평균하고, 그다음 네 fold를 같은 비중으로 평균한 것이다. 이때 `recording_id`는 파일명에서 유도한 묶음 proxy이며 독립 연주자 또는 세션 ID로 취급하지 않는다.

### 2.2 학습과 모델 선택

제한 라벨 시드는 `val_select` 20%, 나머지 `seed_core` 80%로 나눈다. `seed_core`를 다시 학습 80%, 조기종료 검증 20%로 나눈다. 따라서 학습 모델은 시드 라벨의 약 64%를 직접 사용하며, `val_select`, 조기종료 검증, 외부 테스트는 서로 겹치지 않는다. MERT 헤드는 선형/MLP-256, 학습률, weight decay, dropout을 조합한 24개 설정을 탐색하고, SVM은 C와 γ의 12개 설정을 탐색한다. 두 모델 모두 `val_select`에서 설정을 고른 뒤, 저장한 checkpoint를 다시 불러 테스트 예측을 한 번 계산했다. SVM의 scaler와 class weight, MERT의 class weight는 해당 실행의 학습 index에서만 적합·계산했다.

음향 특징은 모노 22,050 Hz 오디오에서 계산했다. 245차원은 MFCC와 1·2차 delta 156차원, spectral contrast 28차원, onset strength·zero-crossing rate·RMS 각 4차원, chroma 48차원, 첫 검출 onset까지 시간의 log1p 1차원으로 구성된다. 프레임별 각 채널의 클립 전체 평균·표준편차·25/75 백분위수를 사용한다. Chroma를 유지하며 이번 결과로 chroma의 인과 효과를 추정하지 않는다. 고정 `m-a-p/MERT-v1-95M` 인코더는 24,000 Hz 입력의 마지막 hidden layer를 시간 평균해 768차원 표현을 만든다 (Li et al., 2024).

MERT는 전체 학습 배치를 사용하는 AdamW로 학습률 {0.0003, 0.001, 0.003}, weight decay {0.0001, 0.001}, dropout {0.1, 0.3}을 탐색하며 최대 200 epoch, patience 20을 사용한다. SVM은 RBF 커널, C ∈ {0.1, 1, 10, 100}, γ ∈ {scale, 0.01, 0.001}을 탐색한다. 탐색 크기와 계산 비용이 달라 검증 분리만으로 튜닝 자원이 동등해지지는 않는다.

자기학습에서는 초기 관측 라벨과 채택한 의사 라벨을 `(표본 index, target)` 쌍으로 유지한다. 임계값 τ=0.8은 테스트 전에 고정했고, 최대 다섯 반복 중 선택은 관측 라벨의 `val_select` F1로 수행했다. supervised, 실제 의사 라벨, 숨은 정답을 **명시적으로** 사용한 oracle 진단을 같은 분할·시드에서 실행했다. Oracle은 실제 준지도학습 방법이나 수학적 성능 상한이 아니다. 기존 코드에서는 의사 라벨로 채택한 표본의 index를 추가한 다음 전체 정답 배열을 다시 읽는 오류가 있었으므로, 이전 자기학습 수치로는 누수 제거의 인과 효과를 계산하지 않는다.

자기학습 헤드는 LayerNorm–dropout(0.3)–linear이며 전체 학습 배치 AdamW, 학습률 0.001, weight decay 0.0001, 적합당 최대 150 epoch, patience 15를 사용한다. Class weight는 명시적 target 풀에서 다시 계산한다. 세 자기학습 조건 안에서 분할·설정을 맞췄으며, 별도로 튜닝한 MERT/SVM 실험과는 분할 시드·설정이 다르다.

전체 라벨 CNN은 128×128 mel 입력의 소형 모델이며, 고정 30 epoch의 마지막 checkpoint를 쓴다. 라벨 예산이 다르므로 제한 라벨 MERT/SVM 표의 직접 대조군으로 취급하지 않는다.

CNN 전처리는 22,050 Hz, FFT 2,048, hop 512, mel 128개를 사용한다. 로그 파워 입력을 앞 128프레임으로 자르거나 오른쪽에 0을 채운 뒤 이미지별 min–max 정규화한다. 학습에서 주파수 최대 15개·시간 최대 20프레임을 마스킹하고 평가에서는 적용하지 않는다. Adam, 학습률 0.001, weight decay 0.0001, 배치 32, cosine 학습률 감소를 사용한다.

### 2.3 출처 검증과 불확실성

MERT/SVM 48쌍과 CNN 네 fold의 checkpoint 100개, 자기학습 세 조건의 checkpoint 144개를 재로딩해 저장 예측 및 manifest 지표와 대조했다. 자기학습은 반복마다 채택한 index와 당시 target으로 학습 풀을 다시 구성해 해시를 확인했다. 지표·클래스별 F1·confusion matrix와 아래 재표집은 같은 저장 예측을 사용한다.

MERT−SVM 차이는 대응하는 테스트 위치를 함께 뽑아 1,000회 percentile 재표집했다. 단위는 클립, 파일명 기반 녹음 proxy, 악기종으로 각각 바꾸었다. 악기종 재표집 구간은 **네 관찰된 종류에 대한 민감도 계산**이다. 네 종류를 재표집해 새 연주자 모집단의 신뢰구간을 얻었다고 해석하지 않는다.

## 3. 결과

### 3.1 제한 라벨 모델 비교

| 라벨 시드 | MERT macro F1 | SVM macro F1 | MERT − SVM | 차이의 악기종 재표집 95% 구간 |
|---:|---:|---:|---:|---:|
| 5% | 0.6655 | 0.6613 | +0.0042 | [−0.0544, +0.0451] |
| 10% | 0.7634 | 0.7542 | +0.0092 | [−0.0514, +0.0698] |
| 20% | 0.8242 | 0.8316 | −0.0074 | [−0.0515, +0.0403] |
| 50% | 0.8599 | 0.8681 | −0.0082 | [−0.0345, +0.0179] |

표 1. 제한 라벨 성능과 대응 악기종 재표집 구간. 네 악기종에 같은 비중을 준 평균이다.

점추정의 우열은 예산에 따라 달라지며, 네 악기종 단위 구간은 모두 0을 포함한다. 따라서 이 표는 명확한 우승 모델을 정하지 않는다. 동시에 0 포함을 동등성 입증으로 해석하지 않는다. 50% 조건에서 MERT의 낮은 클래스별 F1은 `col_legno_battuto` 0.7423과 `sul_ponticello` 0.7678이었고, SVM은 `sul_ponticello` 0.7386과 `col_legno_battuto` 0.8248이었다. 반복 테스트 예측을 합친 confusion counts에서는 `ordinario`와 `sul_ponticello`, `pizzicato_secco`와 `col_legno_battuto` 사이의 혼동이 두 모델 모두에서 두드러졌다. 이 합계는 서로 다른 클립 수가 아니다.

### 3.2 수정된 자기학습

| 라벨 시드 | Supervised | 실제 의사 라벨 | 명시적 oracle 진단 |
|---:|---:|---:|---:|
| 5% | 0.6686 | 0.6364 | 0.6489 |
| 10% | 0.7295 | 0.7357 | 0.7356 |
| 20% | 0.7639 | 0.7737 | 0.7919 |
| 50% | 0.8361 | 0.8409 | 0.8350 |

표 2. 설정을 맞춘 자기학습 비교. Oracle target은 진단에만 사용한 특권 정보다.

5%에서 실제 의사 라벨 조건의 가장 낮은 클래스 F1은 `sul_ponticello` 0.3324와 `col_legno_battuto` 0.4237이다. 50%에서는 각각 0.6912, 0.7327이다. Oracle 진단이 항상 더 높은 성능을 내지는 않았다. 같은 숨은 정답을 제공해도 채택 경로와 검증셋 선택이 바뀌기 때문이다. 과거 자기학습의 5% `0.745 → 0.638` 서술은 서로 다른 결함과 실행 설정이 섞인 비교이므로 삭제한다.

그림 1은 두 실험 절차의 곡선을 요약한다.

![Budget curves](../../figures/canonical/figure1_budget_curves.png){width=100%}

그림 1. 표 1·2의 라벨 예산별 곡선. 악기종 안의 반복을 평균한 뒤 네 종류를 평균했다. (b)는 설정을 맞춘 조건이며 oracle은 진단이다. 두 패널의 튜닝 절차는 다르다.

### 3.3 전체 라벨 CNN과 재표집 단위

CNN의 네 fold 평균 macro F1은 0.8729였다. 이는 제한 라벨 실험과 다른 학습 예산에서 얻은 참고값이다. 50% MERT−SVM 차이 −0.0082의 clip 구간은 [−0.0212, +0.0054], 녹음 proxy 구간은 [−0.0224, +0.0059], 악기종 구간은 [−0.0345, +0.0179]이었다. 단위를 바꾸면 구간 폭이 달라지므로 클립만 재표집한 좁은 구간을 악기종 일반화의 불확실성으로 제시하지 않는다.

그림 2는 모든 예산·재표집 단위를, 그림 3은 합산 클래스 혼동을 보여준다.

![Resampling sensitivity](../../figures/canonical/figure2_resampling_units.png){width=100%}

그림 2. 같은 MERT−SVM 점추정과 단위별 대응 재표집 1,000회의 95% percentile 구간. 가로 위치는 구분을 위해 옮겼다. 고정 예측에 조건부인 민감도 요약이다.


![Confusion matrices](../../figures/canonical/figure3_confusions.png){width=90%}

그림 3. 50% MERT/SVM과 5% 감독/의사 라벨 조건의 행 정규화 혼동 행렬. 패널마다 클립당 세 번의 테스트 예측 10,206개를 합쳤으며 독립 신규 클립 수가 아니다. 값은 합산 예측의 행 비율이며 표의 fold 평균 F1과 다르다. 약칭 순서는 2.1절의 클래스 순서다.

## 4. 논의와 한계

이번 감사로 확인된 사실은 절차적이다. 의사 라벨은 정답 배열 대신 명시적 target으로 전달해야 하고, class weight·scaler는 각 학습 분할 안에서 구해야 한다. 검증셋은 조기종료와 설정 선택에 역할을 나누며, 테스트 예측은 선택된 저장 checkpoint에서 생성한다. 표와 구간은 해당 예측 파일의 해시로 연결된다.

네 악기종 LOSO는 같은 악기종이 학습과 테스트에 동시에 놓이는 것을 막지만, 새로운 물리적 악기·연주자·녹음 세션에 대한 독립 검증은 아니다. `recording_id`도 파일명 proxy다. 재표집 구간이 0을 포함한다는 사실은 동등성을 뜻하지 않으며, 실질적 동등성을 말하려면 사전에 정한 허용 격차와 더 많은 독립 그룹이 필요하다. 자기학습의 τ=0.8은 이번 canonical 실행 전에 고정했지만, 데이터셋 자체는 이전 실험에서 이미 탐색됐다. 그러므로 이 평가를 독립적인 사전등록 확인 실험으로 제시하지 않는다. 고정 τ 결과도 다른 임계값의 최적성을 증명하지 않는다. Layer-weighted probe와 LoRA, 외부 도메인 검증은 이번 canonical 비교에서 제외했다.

## 5. 결론

관찰된 네 악기종에서 튜닝한 MERT 헤드와 특징 SVM의 우열은 라벨 예산에 따라 달랐다. 명시적 target 자기학습은 가장 적은 라벨 예산에서 평균 F1을 낮췄다. 이번 감사는 추적 가능한 평가 자료를 제공하며, 더 넓은 일반화와 인과 해석에는 추가 독립 그룹과 통제된 개입이 필요하다.

## 6. 재현성과 자료 접근

실행 코드와 입력·checkpoint·예측 해시는 `results/phase1/canonical/mert_svm/`, `cnn/`, `self_training/`의 manifest 및 검증 파일에 있다. `src/phase1_canonical_verify.py --draws 1000`과 `src/phase1_canonical_selftraining_verify.py`가 저장 checkpoint 재실행과 표 수치 대조를 수행한다. 상세 결과와 과거 원고의 철회 대상 주장은 각각 `results/phase1/canonical/verification/report.md`, `manuscript_claim_audit.md`에 기록했다.

그림은 `src/phase1_canonical_figures.py`로 생성하며 manifest에 입력 CSV와 출력 SHA-256을 기록했다. 원본 오디오는 OrchideaSOL 원자료 기록의 Ircam Forum 라이선스로 접근하며 이 저장소에서 재배포하지 않는다. 로컬 공개 후보 묶음에 음원 경로를 제거한 부분집합 메타데이터, 명시적 분할, 감사 코드, 예측, manifest와 대조 표를 준비했다. `src/phase1_release_reproduce.py`는 음원·임베딩·checkpoint 없이 보고 통계를 재계산한다. 이 경로는 저장 예측의 통계와 target 기록을 확인하며 checkpoint 추론이나 모델 재학습을 수행하지 않는다. 영구 공개 자료는 아직 등록하지 않았다. 따라서 이 내부 초안은 완전한 공개 재현성을 주장하지 않는다.

## 참고문헌

Cella, C.-E., Ghisi, D., Lostanlen, V., Lévy, F., Fineberg, J., & Maresz, Y. (2020). *OrchideaSOL: An audio dataset of isolated musical notes, including mutes and extended playing techniques* (Version 1.0) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.3686252](https://doi.org/10.5281/zenodo.3686252).

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), 100804. [https://doi.org/10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804).

Kruger, A. B., & Jacobs, J. P. (2020). Playing technique classification for bowed string instruments from raw audio. *Journal of New Music Research, 49*(4), 320–333. [https://doi.org/10.1080/09298215.2020.1784957](https://doi.org/10.1080/09298215.2020.1784957).

Li, Y., Yuan, R., Zhang, G., Ma, Y., Chen, X., Yin, H., Xiao, C., Lin, C., Ragni, A., Benetos, E., Gyenge, N., Dannenberg, R., Liu, R., Chen, W., Xia, G., Shi, Y., Huang, W., Wang, Z., Guo, Y., & Fu, J. (2024). MERT: Acoustic music understanding model with large-scale self-supervised training. *International Conference on Learning Representations*. [Conference paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html).

Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., & Goodfellow, I. (2018). Realistic evaluation of deep semi-supervised learning algorithms. *Advances in Neural Information Processing Systems, 31*. [Conference paper](https://proceedings.neurips.cc/paper/2018/hash/c1fea270c48e8079d8ddf7d06d26ab52-Abstract.html).

Rodríguez-Algarra, F., Sturm, B. L., & Dixon, S. (2019). Characterising confounding effects in music classification experiments through interventions. *Transactions of the International Society for Music Information Retrieval, 2*(1), 52–66. [https://doi.org/10.5334/tismir.24](https://doi.org/10.5334/tismir.24).

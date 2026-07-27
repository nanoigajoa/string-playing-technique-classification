[2020_JNMR] Playing technique classification for bowed string instruments from raw audio

저자: A. B. Kruger, J. P. Jacobs
출판: Journal of New Music Research, 2020, Vol. 49, No. 4, pp. 320–333
링크/DOI: https://doi.org/10.1080/09298215.2020.1784957
읽은 날:
상태: Done


CORE — 모든 논문에 공통, 항상 작성

1. 한줄 요약
바이올린·비올라·첼로·콘트라베이스 각각에 대해 raw audio에서 추출한 전통 오디오 feature(MIRtoolbox)와 Discrete Hartley Transform(DHT) 기반 위상 특징을 SVM에 입력하여 최대 11가지 연주 기법을 분류하고, 36/44-class 악기+기법 공동 분류까지 구현한다.

2. 핵심 내용 & 주요 기여
- 기존 연구(Su, Lin et al. 2014)가 바이올린 9-class에 집중한 것을 바이올린/비올라/첼로/콘트라베이스 4악기로 확장.
- 9-class(단일 피치 기법)와 11-class(글리산도·포르타멘토 추가) 분류기를 각 악기별 구축.
- Feature: (1) MIRT set — MIRtoolbox 기반 시간/스펙트럼/켑스트럼 특징 213차원; (2) HT set — DHT magnitude 8통계값×4pooling=32차원; (3) HPD set — DHT phase 1차 차분 기반 32차원.
- Feature selection 후 multiclass SVM(RBF 커널) 분류.
- 주요 기법: normal(vibrato 포함), non-vibrato, sul ponticello, sul tasto, sordino, pizzicato, spiccato, tremolo, flageolet, glissando, portamento.
- 36-class(4악기×9기법), 44-class(4악기×11기법) 계층적 공동 분류기 구성 → MAFM 0.884.
- DHT 기반 위상 feature(HPD)가 기존 sparse coding 방식보다 우수.

3. 한계 / 비판점
- RWC 데이터베이스의 고립 음(isolated tones)만 사용: 실제 연주 구절(phrase)에서의 일반화 미검증.
- 악기당 연주자 3명뿐 → 연주자 독립성(speaker independence와 유사) 미검증.
- 수동 설계 feature(hand-crafted) 의존; deep learning / SSL 기반 표현과 미비교.
- 클래스 불균형 여부 명시 없음; 다이나믹(세기) 변화 처리가 제한적.
- 2020년 기준으로 end-to-end 딥러닝 방법 비교 없음.

4. 추가로 읽을 references
- Su, Lin et al. (2014): 바이올린 9-class 분류 선행 연구 (비교 기준선)
- Barbancho et al. (2009): 바이올린 기법 계층적 분류 선행 연구
- Lartillot et al. (2008): MIRtoolbox — 이 논문의 feature 추출 핵심 도구
- Paraskevas et al. (2006, 2015): DHT 기반 특징 원래 제안


OPTIONAL

A. 방법론
- 입력: 44.1kHz WAV 고립 음 샘플 (RWC 데이터베이스)
- Feature 추출:
  - MIRT set: 시간(ZCR, RMS, low-energy rate), 스펙트럼(centroid, spread, flux, brightness 등), 켑스트럼(MFCC 1~13 + Δ + ΔΔ) → 213차원 (aggregation: mean, std, slope, period entropy)
  - HT set: DHT magnitude의 8가지 통계값(mean, variance, IQR, MAD, range, skewness, kurtosis, entropy) × 4 pooling = 32차원
  - HPD set: DHT phase의 1차 차분(ΔΨ)에서 동일 8통계값 × 4 pooling = 32차원
- Feature selection: wrapper method (SVM 기반)
- 모델: multiclass SVM (RBF 커널), 계층적 구성
- 평가 지표: macro-average F-measure (MAFM)
- 프레임: 50ms, hop 10%, Hann window, zero-padding to 4096 samples

B. 데이터셋
- RWC Music Instrument Sound Set (Goto et al., 2004)
- 바이올린, 비올라, 첼로, 콘트라베이스 각 3명의 연주자
- 반음 간격 음정, 최대 3개 다이나믹 레벨
- 샘플 지속 시간: 0.33~7.81초 (기법별 상이)
- 9-class(단일 피치 기법) + 11-class(글리산도/포르타멘토 추가)

C. 결과 (원문 Table 1, p.327 대조 확정 — 이전 판 오기 수정)
- 바이올린 9-class MAFM: **0.941**(MIRT+HPD RAW, 최고) / 0.938({MIRT,HT}+ReliefF, §5). RAW·ReliefF 상위 설정은 0.938~0.941.
  ⚠️ 이전에 적힌 0.915는 **Su, Lin et al. 2014의 ALL+SC 점수**(원문 §5에서 Kruger가 Su의 값으로 명시 인용)이며 Kruger의 것이 아님.
- 바이올린 11-class MAFM: 0.934(MIRT+HPD RAW). 4악기 개별 11-class: violin 0.934 / viola 0.917 / cello 0.924 / contrabass 0.845.
- 36-class 공동 분류 MAFM: 0.885 / 44-class: 0.884.
- DHT 기반 위상 feature(HPD)가 Su et al.의 sparse coding보다 우수
- Feature selection 후 213차원 → 53차원으로 줄여도 성능 유지

D. 내 프로젝트(string-playing-technique-classification)와의 비교
유사점: 완전히 동일한 문제 설정 — 현악기 연주 기법 분류, raw audio 입력, 동일 악기군(violin family), 동일 기법 목록(sul ponticello, pizzicato, spiccato 등) 포함. 가장 직접적인 비교 기준선 논문.
차이점: 이 논문은 수동 feature + SVM; 딥러닝/SSL backbone 미사용. RWC 고립 음 한정으로 실제 연주 구절 적용 미검증. 연주자 수 적음(3명/악기).
적용 인사이트:
- MIRT feature + DHT phase feature를 딥러닝 모델의 입력으로 결합하는 hybrid 접근 가능성.
- 9/11-class 설정과 36/44-class 공동 분류 설정을 실험 baseline으로 재현해 SSL 기반 방법과 비교.
- Sul ponticello, sordino, flageolet 등 음색 변화 기법은 스펙트럼 위상 정보(DHT phase)에 민감 → MERT의 CQT 재구성 손실이 이를 잘 포착할 수 있는지 ablation 설계 가능.
- 장애물: RWC 데이터셋 접근성; 자체 데이터 수집 시 동일 프로토콜(3연주자, 반음 단위, 3다이나믹) 참조.

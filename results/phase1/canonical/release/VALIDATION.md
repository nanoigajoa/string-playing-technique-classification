# 공개 후보 자료 묶음 검증 — 2026-10-01

## 결과

**통과.** `canonical-reproduction.zip`을 원래 저장소와 다른 폴더에 풀고,
별도 가상환경에서 묶음 안의 코드만 실행했다. 음원·임베딩·checkpoint는
없으며 PyTorch도 설치하지 않았다. 시스템 site-packages를 사용하지 않았다.

- ZIP 무결성과 SHA-256 sidecar 대조 통과; 허용 목록의 253개 파일.
- 저장 예측 100파일, 모델 예측 벡터 244개, 3,402개 행의 클래스·악기종 정렬 확인.
- 관측 라벨의 학습·조기종료·설정 선택·테스트 분할, 반복 grid, 학습 class weight,
  검증으로 고른 반복, 저장 의사 target 쌍으로 구성한 풀의 해시 확인.
- 예측에서 결과표 10개·878행을 다시 계산해 원본 CSV와 대조: 숫자의 절대 허용 오차
  1e-12, 나머지 값과 행·열 구성도 일치.
- 재계산 표에서 그림 3개를 PNG/PDF로 생성; 입력·출력 해시 확인.
- `pip check` 통과. 기존 학습 환경의 패키지를 변경하지 않았다.

묶음 SHA-256, 실제 분석 패키지와 재계산 출력 해시는 [VALIDATION.json](VALIDATION.json),
통계 재생성 보고서는 [REPRODUCTION.json](REPRODUCTION.json)에 있다. 후보 ZIP의
manifest 자체는 외부 검증 보고서를 포함하지 않으므로 ZIP 내부의 자체 해시 순환이 없다.

## 실행한 명령

아래 명령은 별도 환경을 만든 뒤 ZIP을 풀어 묶음의 루트에서 실행했다.

```bash
python -B src/phase1_release_reproduce.py
python -B src/phase1_canonical_figures.py --tables-root reproduced --output-dir reproduced/figures
python -m pip check
```

Python 3.14.5, macOS arm64; NumPy 2.4.6, pandas 3.0.6,
Matplotlib 3.11.0 및 전이 의존성은 `requirements-analysis.txt`에 고정했다.
기존 환경은 pandas 3.0.4를 사용했지만, 해당 버전의 [PyPI 배포 철회 기록](https://pypi.org/project/pandas/3.0.4/)을 확인해 분석 환경에는
[3.0.6](https://pypi.org/project/pandas/3.0.6/)을 설치하고 결과 일치를 검증했다.
별도 환경의 설치 패키지 전체는 JSON에 기록했다.

## 검증 범위

이번 통과는 **고정 저장 예측의 통계적 재현과 저장 target 경로 검증**이다.
새 코드가 숨은 정답에 접근하지 않음을 학습으로 다시 증명한 것은 아니며,
checkpoint를 실행하거나 모델을 재학습하지 않았다. 원래 로컬 checkpoint
재실행 기록은 별도로 보존했고 원래 manifest를 묶음에 바이트 그대로 넣었다.
해시만 남은 생략 입력·모델 파일은 공개 후보에서 다시 검증할 수 없다.

과거 학습 실행의 완전한 환경 잠금 기록이 없으므로 현재 설치 버전을
그 실행 당시 환경이라고 주장하지 않는다. 패키징 시점의 환경 기록과
별도 분석 환경을 구분했다. 다른 운영체제·Python 버전의 실행은 아직 확인하지 않았다.

코드 라이선스·최종 저자 선언·영구 보관 주소는 아직 확정하지 않았다.
이 묶음은 로컬 공개 후보이며 외부 업로드나 학술지 제출은 수행하지 않았다.

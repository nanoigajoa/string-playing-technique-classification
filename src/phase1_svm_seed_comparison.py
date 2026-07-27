"""
Phase 1 보완 — SVM seed별 성능 비교

연구과정/03_피드백_전문가.md 1순위 작업:
    SVM을 Phase 1과 동일한 seed 비율 + LOSO 분할로 평가해
    "Foundation model이 언제 SVM보다 유리한가" 교차점 테이블을 생성한다.

실행:
    python src/phase1_svm_seed_comparison.py

의존 파일 (사전 생성 필요):
    data/features_full.csv       — v1 특징 행렬 (3402 × 247)
    data/split_configs.json      — Phase 1 동일 분할 (phase1_selftraining.py --step splits)
    results/phase1/all_results.csv — Phase 1 MERT 결과

출력:
    results/phase1/svm_seed_comparison.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

# ── 상수 (phase1_selftraining.py와 통일) ──────────────────────────────────────

CLASSES     = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
LABEL2STR   = dict(enumerate(CLASSES))
STR2LABEL   = {v: k for k, v in LABEL2STR.items()}
SEED_RATIOS = [0.05, 0.10, 0.20, 0.50]

DATA_PATH    = ROOT / 'data' / 'features_full.csv'
SPLITS_PATH  = ROOT / 'data' / 'split_configs.json'
MERT_RESULTS = ROOT / 'results' / 'phase1' / 'all_results.csv'
OUT_PATH     = ROOT / 'results' / 'phase1' / 'svm_seed_comparison.md'

CLAP_F1 = 0.363  # phase1/report.md의 CLAP zero-shot 고정값


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_features() -> tuple[np.ndarray, np.ndarray]:
    """features_full.csv → (X: float32 배열, y: int 배열)"""
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ('label_id', 'instrument')]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df['label_id'].to_numpy(dtype=np.int32)
    print(f"  특징 행렬: {X.shape}  클래스: {np.unique(y)}")
    return X, y


def load_splits() -> list[dict]:
    with open(SPLITS_PATH, encoding='utf-8') as f:
        config = json.load(f)
    return config['splits']


# ── SVM 평가 ──────────────────────────────────────────────────────────────────

def _make_svm() -> Pipeline:
    """v1 SVM과 동일 설정: StandardScaler + RBF SVM + balanced class weight"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=10.0, gamma='scale',
                       class_weight='balanced', random_state=42)),
    ])


def run_svm_on_splits(X: np.ndarray, y: np.ndarray, splits: list[dict]) -> pd.DataFrame:
    """
    각 split마다 labeled_idx로 SVM 학습, test_idx로 평가.
    반환: fold / ratio / trial / svm_f1 컬럼을 가진 DataFrame
    """
    rows = []
    n_total = len(splits)
    for i, s in enumerate(splits, 1):
        labeled = s['labeled_idx']
        test    = s['test_idx']

        X_train, y_train = X[labeled], y[labeled]
        X_test,  y_test  = X[test],    y[test]

        clf = _make_svm()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        rows.append({
            'fold':       s['fold'],
            'ratio':      s['seed_ratio'],
            'trial':      s['trial'],
            'svm_f1':     round(f1, 4),
        })

        if i % 4 == 0 or i == n_total:
            print(f"  [{i:>3}/{n_total}] fold={s['fold']} ratio={s['seed_ratio']:.0%} "
                  f"trial={s['trial']}  SVM F1={f1:.4f}")

    return pd.DataFrame(rows)


# ── MERT 결과 집계 ──────────────────────────────────────────────────────────────

def load_mert_results() -> pd.DataFrame:
    """all_results.csv에서 ratio별 (baseline_f1, best_st_f1 at τ=0.7) 평균 추출"""
    df = pd.read_csv(MERT_RESULTS)
    tau07 = df[df['tau'] == 0.7].copy()
    agg = tau07.groupby('ratio')[['baseline_f1', 'best_st_f1']].mean().round(4)
    agg.columns = ['mert_baseline', 'mert_st_07']
    agg.index = agg.index.map(lambda r: round(float(r), 2))
    return agg


# ── 테이블 생성 & 저장 ──────────────────────────────────────────────────────────

def build_comparison_table(svm_df: pd.DataFrame, mert_df: pd.DataFrame) -> pd.DataFrame:
    """ratio별 평균값을 하나의 비교 테이블로 조합한다."""
    svm_agg = svm_df.groupby('ratio')['svm_f1'].mean().round(4)
    svm_agg.index = svm_agg.index.map(lambda r: round(float(r), 2))

    table = mert_df.copy()
    table['svm'] = svm_agg
    table['clap'] = CLAP_F1
    table = table[['clap', 'svm', 'mert_baseline', 'mert_st_07']]
    table.index.name = 'seed_ratio'
    return table


def _ratio_label(r: float) -> str:
    return f"{int(r * 100)}%"


def save_report(table: pd.DataFrame, svm_df: pd.DataFrame) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase 1 보완 — SVM seed별 성능 비교",
        "",
        "연구과정/03_피드백_전문가.md 1순위: MERT+self-training이 '언제' SVM보다 유리한지 교차점 확인.",
        "",
        "## 비교 테이블 (Macro F1, LOSO 평균)",
        "",
        "| seed | CLAP 기준선 | SVM | MERT baseline | MERT + ST (τ=0.7) |",
        "|------|------------|-----|---------------|-------------------|",
    ]
    for ratio, row in table.iterrows():
        label = _ratio_label(ratio)
        lines.append(
            f"| {label} | {row['clap']:.3f} | **{row['svm']:.3f}** "
            f"| {row['mert_baseline']:.3f} | {row['mert_st_07']:.3f} |"
        )

    # 교차점 분석
    lines += ["", "## 교차점 분석", ""]
    crossover_found = False
    for ratio, row in table.iterrows():
        if row['mert_st_07'] > row['svm']:
            lines.append(
                f"- **{_ratio_label(ratio)}**: MERT+ST ({row['mert_st_07']:.3f}) > "
                f"SVM ({row['svm']:.3f})  ← Foundation model 유리 구간"
            )
            crossover_found = True
        else:
            lines.append(
                f"- {_ratio_label(ratio)}: SVM ({row['svm']:.3f}) ≥ "
                f"MERT+ST ({row['mert_st_07']:.3f})"
            )
    if not crossover_found:
        lines.append("")
        lines.append(
            "> 모든 seed 비율에서 SVM이 MERT+ST 이상. "
            "OrchideaSOL 규모(3,402개)에서 수동 특징이 여전히 우세."
        )

    # 상세 결과 (fold × trial)
    lines += [
        "",
        "## 상세 결과 (fold × trial별)",
        "",
        "| fold | ratio | trial | SVM F1 |",
        "|------|-------|-------|--------|",
    ]
    for _, row in svm_df.iterrows():
        lines.append(
            f"| {int(row['fold'])} | {_ratio_label(row['ratio'])} "
            f"| {int(row['trial'])} | {row['svm_f1']:.4f} |"
        )

    report = "\n".join(lines) + "\n"
    OUT_PATH.write_text(report, encoding='utf-8')
    print(f"\n저장: {OUT_PATH}")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    assert DATA_PATH.exists(),   f"features_full.csv 없음: {DATA_PATH}"
    assert SPLITS_PATH.exists(), f"split_configs.json 없음: {SPLITS_PATH}\n  먼저 실행: python src/phase1_selftraining.py --step splits"
    assert MERT_RESULTS.exists(), f"all_results.csv 없음: {MERT_RESULTS}\n  Phase 1 self-training 실행 후 재시도"

    print("=== Phase 1 보완 — SVM seed별 성능 비교 ===\n")

    print("1. 특징 행렬 로드")
    X, y = load_features()

    print("\n2. 분할 설정 로드")
    splits = load_splits()
    print(f"  총 splits: {len(splits)}  "
          f"(folds × ratios × trials = {len(set(s['fold'] for s in splits))} × "
          f"{len(SEED_RATIOS)} × {len(set(s['trial'] for s in splits))})")

    print("\n3. SVM 평가 실행 (각 split마다 학습 → LOSO 평가)")
    svm_df = run_svm_on_splits(X, y, splits)

    print("\n4. MERT 결과 로드 (all_results.csv)")
    mert_df = load_mert_results()

    print("\n5. 비교 테이블 생성")
    table = build_comparison_table(svm_df, mert_df)

    print("\n" + "=" * 60)
    print("비교 테이블 (Macro F1, LOSO 평균)\n")
    print(f"{'seed':>6}  {'CLAP':>6}  {'SVM':>6}  {'MERT-base':>10}  {'MERT+ST τ=0.7':>14}")
    print("-" * 50)
    for ratio, row in table.iterrows():
        marker = " ← crossover" if row['mert_st_07'] > row['svm'] else ""
        print(f"{_ratio_label(ratio):>6}  {row['clap']:>6.3f}  {row['svm']:>6.3f}  "
              f"{row['mert_baseline']:>10.3f}  {row['mert_st_07']:>14.3f}{marker}")

    save_report(table, svm_df)
    print("\n완료.")


if __name__ == '__main__':
    main()

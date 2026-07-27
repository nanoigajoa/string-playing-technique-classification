"""
Phase 1 보완 — Self-Training 향상폭 통계적 유의성 검정

연구과정/06_지시_연구재구성.md 1-4 + 연구과정/08_지시_누수수정후속.md "검정력 현실" 반영:
  - paired 검정 (LOSO fold별로 baseline↔self-training 짝지어짐).
  - paired 단위 = fold (n=4). trial(3회)은 독립 fold가 아니므로 fold별 trial 평균을 먼저 낸다.
  - fold 4개로는 p<0.05가 수학적으로 거의 불가능(양측 최소 p=0.125). 억지 유의성 주장 금지.
  - 주 실험(primary) 1개만 검정, 나머지 seed는 기술통계로만 보고.

실행:
    python src/phase1_significance.py

의존:
    results/phase1/all_results.csv   (누수 수정 후 재실행본)

출력:
    results/phase1/significance.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / 'results' / 'phase1' / 'all_results.csv'
OUT     = ROOT / 'results' / 'phase1' / 'significance.md'

# 사전 지정 주 실험 (pre-registered primary) — 비교표 확인 후 변경 가능
PRIMARY_RATIO = 0.20
PRIMARY_TAU   = 0.7


def load() -> pd.DataFrame:
    assert RESULTS.exists(), f"all_results.csv 없음: {RESULTS}\n  먼저 Phase 1 노트북 재실행."
    df = pd.read_csv(RESULTS)
    need = {'fold', 'ratio', 'trial', 'tau', 'baseline_f1', 'best_st_f1'}
    missing = need - set(df.columns)
    assert not missing, f"필요 컬럼 없음: {missing}"
    return df


def fold_pairs(df: pd.DataFrame, ratio: float, tau: float) -> pd.DataFrame:
    """(ratio, tau)에서 fold별 trial 평균 → fold당 (baseline, self-training) 1쌍."""
    sub = df[(np.isclose(df['ratio'], ratio)) & (np.isclose(df['tau'], tau))]
    return (sub.groupby('fold')[['baseline_f1', 'best_st_f1']]
               .mean()
               .reset_index()
               .sort_values('fold'))


def run_primary(df: pd.DataFrame) -> dict:
    pairs = fold_pairs(df, PRIMARY_RATIO, PRIMARY_TAU)
    base = pairs['baseline_f1'].to_numpy()
    st   = pairs['best_st_f1'].to_numpy()
    diff = st - base
    n = len(pairs)

    # Wilcoxon signed-rank (paired). n=4 → 양측 최소 p=0.125. 차이가 모두 0이면 검정 불가.
    p_value, stat, note = None, None, ""
    if n < 1 or np.allclose(diff, 0):
        note = "차이가 사실상 0이거나 표본 부족 — 검정 불가"
    else:
        try:
            res = wilcoxon(st, base, zero_method='wilcox', alternative='two-sided')
            stat, p_value = float(res.statistic), float(res.pvalue)
        except ValueError as e:
            note = f"Wilcoxon 계산 불가: {e}"

    return {
        'pairs': pairs, 'n': n,
        'delta_mean': float(diff.mean()), 'delta_std': float(diff.std(ddof=1)) if n > 1 else 0.0,
        'stat': stat, 'p_value': p_value, 'note': note,
    }


def descriptive_all(df: pd.DataFrame) -> pd.DataFrame:
    """primary τ에서 seed별 향상폭 기술통계 (검정 없이, fold-level)."""
    rows = []
    for ratio in sorted(df['ratio'].unique()):
        pairs = fold_pairs(df, ratio, PRIMARY_TAU)
        diff  = (pairs['best_st_f1'] - pairs['baseline_f1']).to_numpy()
        rows.append({
            'seed':        f"{int(ratio*100)}%",
            'baseline':    round(pairs['baseline_f1'].mean(), 4),
            'self_train':  round(pairs['best_st_f1'].mean(), 4),
            'delta_mean':  round(float(diff.mean()), 4),
            'delta_std':   round(float(diff.std(ddof=1)) if len(diff) > 1 else 0.0, 4),
        })
    return pd.DataFrame(rows)


def write_report(primary: dict, desc: pd.DataFrame) -> None:
    p = primary
    if p['p_value'] is not None:
        sig = "유의 (p<0.05)" if p['p_value'] < 0.05 else "유의하지 않음 (p≥0.05)"
        p_str = f"{p['p_value']:.4f}"
    else:
        sig = "판정 불가"
        p_str = p['note'] or "N/A"

    lines = [
        "# Phase 1 — Self-Training 향상폭 유의성 검정",
        "",
        f"**주 실험(primary)**: seed {int(PRIMARY_RATIO*100)}%, τ={PRIMARY_TAU} "
        "(사전 지정, 다중비교 회피 위해 1개만 검정)",
        "",
        "**검정**: Wilcoxon signed-rank (paired). "
        f"단위 = fold (n={p['n']}), trial은 fold별로 평균 후 짝지음.",
        "",
        "## Fold별 짝 (baseline vs self-training)",
        "",
        "| fold | baseline | self-training | Δ |",
        "|---|---|---|---|",
    ]
    for _, r in p['pairs'].iterrows():
        d = r['best_st_f1'] - r['baseline_f1']
        lines.append(f"| {int(r['fold'])} | {r['baseline_f1']:.4f} "
                     f"| {r['best_st_f1']:.4f} | {d:+.4f} |")

    lines += [
        "",
        f"- 향상폭: **{p['delta_mean']:+.4f} ± {p['delta_std']:.4f}** (fold-level, n={p['n']})",
        f"- Wilcoxon 통계량: {p['stat'] if p['stat'] is not None else 'N/A'}",
        f"- p-value: **{p_str}**  → {sig}",
        "",
        "> ⚠ **검정력 한계 (정직한 보고)**: fold가 4개뿐이라 Wilcoxon 양측 최소 p-value가 "
        "0.125로, 표본상 p<0.05가 수학적으로 거의 불가능하다. "
        "따라서 **\"향상은 관찰되나 fold 수가 적어 통계적 유의성은 확정할 수 없다\"** 가 "
        "정직한 결론이며, 향상폭의 방향과 크기(Δ·부호)를 기술통계로 함께 보고한다.",
        "",
        "## 참고 — seed별 향상폭 기술통계 (검정 없이, τ=%.1f)" % PRIMARY_TAU,
        "",
        "| seed | baseline | self-training | Δ mean | Δ std |",
        "|---|---|---|---|---|",
    ]
    for _, r in desc.iterrows():
        lines.append(f"| {r['seed']} | {r['baseline']:.4f} | {r['self_train']:.4f} "
                     f"| {r['delta_mean']:+.4f} | {r['delta_std']:.4f} |")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding='utf-8')
    print(f"저장: {OUT}")


def main():
    print("=== Phase 1 — 유의성 검정 (Wilcoxon, fold n=4) ===\n")
    df = load()
    print(f"주 실험: seed {int(PRIMARY_RATIO*100)}%, τ={PRIMARY_TAU}")
    primary = run_primary(df)

    print(f"\nFold 짝 (n={primary['n']}):")
    for _, r in primary['pairs'].iterrows():
        d = r['best_st_f1'] - r['baseline_f1']
        print(f"  fold {int(r['fold'])}: {r['baseline_f1']:.4f} → {r['best_st_f1']:.4f} ({d:+.4f})")
    print(f"\n향상폭: {primary['delta_mean']:+.4f} ± {primary['delta_std']:.4f}")
    if primary['p_value'] is not None:
        print(f"Wilcoxon p = {primary['p_value']:.4f} "
              f"({'유의' if primary['p_value'] < 0.05 else '유의하지 않음'})")
    else:
        print(f"검정 불가: {primary['note']}")
    print("주의: fold 4개 → 양측 최소 p=0.125, p<0.05 사실상 불가")

    desc = descriptive_all(df)
    write_report(primary, desc)
    print("\n완료.")


if __name__ == '__main__':
    main()

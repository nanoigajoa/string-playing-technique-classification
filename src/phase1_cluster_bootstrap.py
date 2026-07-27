"""
B6 — 클러스터 부트스트랩 신뢰구간 (녹음 그룹 단위)

배경(10_계획_방향재정비.md Track B):
      주 비교표는 fold n=4의 pooled std로만 불확실성을 보고해 왔다. 검정력이 낮아
      "유의하지 않다"만 반복하게 되는데, 정작 **얼마나 모르는지**는 말하지 못한다.
      Kruger(2020 JNMR)가 BCa 부트스트랩 CI를 쓰므로, 같은 엄밀성을 맞추면
      "가장 강한 선행연구와 동급"이라는 정당화가 생긴다.

왜 클러스터 부트스트랩인가:
      클립은 독립이 아니다. 같은 recording_id(같은 음고·강약의 한 녹음)에서 나온 클립들은
      서로 강하게 상관된다. 클립 단위로 재표집하면 유효 표본수를 과대평가해 CI가 좁아진다.
      → **recording_id를 클러스터로 묶어 재표집**한다. fold는 악기 단위(LOSO)이므로
      재표집은 각 fold의 test set 안에서, 그 fold의 클러스터 목록에 대해 수행한다.

프로토콜:
      - 1단계(preds): 감사(A1)와 **동일한 분할 재현** + 감사 CSV가 고른 best config만 재적합
        (24-config 그리드 재탐색 없음) → per-clip 예측 저장. 재현 점검을 위해 감사 CSV의
        점추정과의 차이(Δ)를 함께 보고한다.
      - 2단계(boot): B회 반복. 각 반복에서 fold별로 클러스터를 복원추출하고, 그 fold에 속한
        모든 (ratio, trial) split에 **동일한 재표집을 적용**한다(test set이 fold 내에서 공통).
        집계는 주 수치와 같은 경로 — trial 평균 → fold 평균.
      - 구간: percentile + **BCa**(가속도는 클러스터 단위 jackknife).
      - 모델 간 차이는 **대응(paired) 차이**의 CI로 보고한다. 같은 재표집에서 계산되므로
        상관이 보존된다.

실행:
    python src/phase1_cluster_bootstrap.py                 # preds → boot
    python src/phase1_cluster_bootstrap.py --stage boot    # 저장된 예측으로 CI만 재계산
    python src/phase1_cluster_bootstrap.py -B 5000
출력:
    results/phase1/bootstrap_preds.npz
    results/phase1/cluster_bootstrap.{md,csv}
"""

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from phase1_mert_tuning_audit import (  # noqa: E402  분할·학습 로직을 그대로 재사용
    N_CLASSES, embeddings, labels, X_feat, split_val, train_head_cfg, DEVICE,
)

META_PATH   = ROOT / 'data' / 'mert_meta.csv'
SPLITS_PATH = ROOT / 'data' / 'split_configs.json'
AUDIT_CSV   = ROOT / 'results' / 'phase1' / 'mert_tuning_audit.csv'
PRED_PATH   = ROOT / 'results' / 'phase1' / 'bootstrap_preds.npz'
RESULTS_DIR = ROOT / 'results' / 'phase1'

# 주 수치에 대응하는 세 계열 (감사 CSV의 tuned 열)
MODELS = ['mert_tuned', 'svm_tuned', 'svm_tuned_full']
LABEL  = {
    'mert_tuned':     'MERT (tuned, tr-only)',
    'svm_tuned':      'SVM (tuned, tr-only · 동일 학습데이터)',
    'svm_tuned_full': 'SVM (tuned, seed_core 80%)',
}
# 보고할 대응 차이
CONTRASTS = [
    ('mert_tuned', 'svm_tuned',      'MERT − SVM (동일 학습데이터, 주 비교)'),
    ('mert_tuned', 'svm_tuned_full', 'MERT − SVM (SVM 학습량 우위 조건)'),
]


def macro_f1(y_true, y_pred, n_classes=N_CLASSES):
    """np.bincount 기반 빠른 macro F1 (부트스트랩 내부 루프용)."""
    cm = np.bincount(y_true * n_classes + y_pred,
                     minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean()


# ─────────────────────────── 1단계: 예측 재생성 ───────────────────────────

def fit_predict_split(s, row):
    """감사와 동일한 분할을 재현하고, 감사가 고른 config만 재적합해 test 예측을 만든다."""
    seed_core, val_sel = split_val(s['labeled_idx'], val_ratio=0.2, seed=s['trial'] + 1000)
    tr_idx, es_idx     = split_val(seed_core, val_ratio=0.2, seed=s['trial'])
    test_idx = np.asarray(s['test_idx'])

    out = {}

    # MERT: 감사가 val_select에서 고른 config를 그대로 재적합
    cfg = ast.literal_eval(row['mert_best_cfg'])
    torch.manual_seed(1234 + int(s['fold']) * 100 + int(s['trial']))
    model, _ = train_head_cfg(tr_idx, es_idx, cfg)
    model.eval()
    with torch.no_grad():
        X = torch.tensor(embeddings[test_idx]).to(DEVICE)
        out['mert_tuned'] = model(X).argmax(1).cpu().numpy().astype(np.int8)

    # SVM: tr-only(주 수치) / seed_core(민감도) 각각의 선택 config
    for key, train_idx, cfg_col in [
        ('svm_tuned',      tr_idx,    'svm_best_cfg'),
        ('svm_tuned_full', seed_core, 'svm_best_cfg_full'),
    ]:
        scfg = ast.literal_eval(row[cfg_col])
        pipe = make_pipeline(StandardScaler(),
                             SVC(kernel='rbf', class_weight='balanced', **scfg))
        pipe.fit(X_feat[train_idx], labels[train_idx])
        out[key] = pipe.predict(X_feat[test_idx]).astype(np.int8)

    return out, test_idx


def stage_preds():
    audit = pd.read_csv(AUDIT_CSV)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)['splits']
    key = {(int(s['fold']), float(s['seed_ratio']), int(s['trial'])): s for s in splits}

    store, meta_rows, t0 = {}, [], time.time()
    for i, row in audit.iterrows():
        s = key[(int(row['fold']), float(row['ratio']), int(row['trial']))]
        preds, test_idx = fit_predict_split(s, row)
        tag = f"{int(row['fold'])}_{row['ratio']}_{int(row['trial'])}"
        store[f'test_idx__{tag}'] = test_idx.astype(np.int32)
        rec = {'fold': int(row['fold']), 'ratio': float(row['ratio']),
               'trial': int(row['trial']), 'tag': tag}
        for m in MODELS:
            store[f'{m}__{tag}'] = preds[m]
            f1_new = macro_f1(labels[test_idx], preds[m].astype(np.int64))
            rec[f'{m}_refit'] = f1_new
            rec[f'{m}_audit'] = float(row[m])
            rec[f'{m}_delta'] = f1_new - float(row[m])
        meta_rows.append(rec)
        print(f'  [{i + 1:>2}/{len(audit)}] fold={rec["fold"]} r={rec["ratio"]:.0%} '
              f't={rec["trial"]} | {time.time() - t0:.0f}s', flush=True)

    meta = pd.DataFrame(meta_rows)
    np.savez_compressed(PRED_PATH, **store, meta_csv=meta.to_csv(index=False))
    return meta


# ─────────────────────────── 2단계: 클러스터 부트스트랩 ───────────────────────────

def load_preds():
    z = np.load(PRED_PATH, allow_pickle=True)
    meta = pd.read_csv(pd.io.common.StringIO(str(z['meta_csv'])))
    return z, meta


def fold_clusters(z, meta, rec_id):
    """fold별 test 클러스터 목록과, 클러스터→클립 인덱스(테스트셋 내 위치) 매핑."""
    info = {}
    for fold in sorted(meta['fold'].unique()):
        tags = meta[meta['fold'] == fold]['tag'].values
        tis = [z[f'test_idx__{t}'] for t in tags]
        # fold 내 모든 split이 같은 test set을 쓰는지 확인 (재표집 공유의 전제)
        assert all(np.array_equal(tis[0], t) for t in tis), f'fold {fold}: test_idx 불일치'
        ti = tis[0]
        cl = rec_id[ti]
        uniq = np.unique(cl)
        info[fold] = {
            'test_idx': ti,
            'y': labels[ti].astype(np.int64),
            'clusters': uniq,
            'pos': {c: np.flatnonzero(cl == c) for c in uniq},
        }
    return info


def stat_from_positions(z, meta, info, ratio, model, pos_by_fold):
    """주 수치와 동일한 집계 경로: trial 평균 → fold 평균."""
    fold_means = []
    for fold, fi in info.items():
        sub = meta[(meta['fold'] == fold) & (meta['ratio'] == ratio)]
        y = fi['y'][pos_by_fold[fold]]
        vals = [macro_f1(y, z[f'{model}__{t}'][pos_by_fold[fold]].astype(np.int64))
                for t in sub['tag'].values]
        fold_means.append(np.mean(vals))
    return float(np.mean(fold_means))


def identity_positions(info):
    return {f: np.arange(len(fi['y'])) for f, fi in info.items()}


def resample_positions(info, rng):
    """fold별로 클러스터를 복원추출 → 클립 위치 배열. fold 내 모든 split에 공유된다."""
    out = {}
    for fold, fi in info.items():
        cl = rng.choice(fi['clusters'], size=len(fi['clusters']), replace=True)
        out[fold] = np.concatenate([fi['pos'][c] for c in cl])
    return out


def jackknife_positions(info, drop_fold, drop_cluster):
    out = {}
    for fold, fi in info.items():
        if fold == drop_fold:
            keep = [c for c in fi['clusters'] if c != drop_cluster]
            out[fold] = np.concatenate([fi['pos'][c] for c in keep])
        else:
            out[fold] = np.arange(len(fi['y']))
    return out


def bca_ci(theta_hat, boot, jack, alpha=0.05):
    """BCa 구간. boot: (B,) 부트스트랩 통계, jack: (J,) jackknife 통계."""
    from scipy.stats import norm
    boot = np.asarray(boot); jack = np.asarray(jack)
    prop = np.mean(boot < theta_hat)
    prop = min(max(prop, 1.0 / (len(boot) + 1)), 1 - 1.0 / (len(boot) + 1))
    z0 = norm.ppf(prop)
    jbar = jack.mean()
    num = ((jbar - jack) ** 3).sum()
    den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
    a = num / den if den > 0 else 0.0
    lo_z, hi_z = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

    def adj(zq):
        return norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))

    ql, qh = adj(lo_z), adj(hi_z)
    return (float(np.quantile(boot, ql)), float(np.quantile(boot, qh)),
            float(z0), float(a))


def stage_boot(B, alpha, seed=42):
    z, meta = load_preds()
    rec_id = pd.read_csv(META_PATH)['recording_id'].values
    info = idfy = fold_clusters(z, meta, rec_id)
    ratios = sorted(meta['ratio'].unique())
    rng = np.random.default_rng(seed)

    ident = identity_positions(info)
    jack_specs = [(f, c) for f, fi in info.items() for c in fi['clusters']]
    print(f'B={B} · ratios={len(ratios)} · jackknife 클러스터={len(jack_specs)}')

    # 재표집 위치는 모든 ratio/모델이 공유해야 대응(paired) 구조가 보존된다
    boot_pos = [resample_positions(info, rng) for _ in range(B)]

    rows, t0 = [], time.time()
    for ratio in ratios:
        cache_boot, cache_jack, point = {}, {}, {}
        for m in MODELS:
            point[m] = stat_from_positions(z, meta, info, ratio, m, ident)
            cache_boot[m] = np.array([stat_from_positions(z, meta, info, ratio, m, p)
                                      for p in boot_pos])
            cache_jack[m] = np.array([stat_from_positions(z, meta, info, ratio, m,
                                                          jackknife_positions(info, f, c))
                                      for f, c in jack_specs])
            lo, hi, z0, a = bca_ci(point[m], cache_boot[m], cache_jack[m], alpha)
            pl, ph = np.quantile(cache_boot[m], [alpha / 2, 1 - alpha / 2])
            rows.append({'ratio': ratio, 'quantity': LABEL[m], 'kind': 'model',
                         'point': point[m], 'bca_lo': lo, 'bca_hi': hi,
                         'pct_lo': float(pl), 'pct_hi': float(ph),
                         'boot_sd': float(cache_boot[m].std(ddof=1)), 'z0': z0, 'accel': a})
            print(f'  r={ratio:.0%} {LABEL[m]:<42s} {point[m]:.3f} '
                  f'[{lo:.3f}, {hi:.3f}]  {time.time() - t0:.0f}s', flush=True)

        for m1, m2, name in CONTRASTS:
            d_point = point[m1] - point[m2]
            d_boot = cache_boot[m1] - cache_boot[m2]      # 동일 재표집 → 대응 차이
            d_jack = cache_jack[m1] - cache_jack[m2]
            lo, hi, z0, a = bca_ci(d_point, d_boot, d_jack, alpha)
            pl, ph = np.quantile(d_boot, [alpha / 2, 1 - alpha / 2])
            rows.append({'ratio': ratio, 'quantity': name, 'kind': 'contrast',
                         'point': d_point, 'bca_lo': lo, 'bca_hi': hi,
                         'pct_lo': float(pl), 'pct_hi': float(ph),
                         'boot_sd': float(d_boot.std(ddof=1)), 'z0': z0, 'accel': a})
            print(f'  r={ratio:.0%} {name:<42s} {d_point:+.3f} '
                  f'[{lo:+.3f}, {hi:+.3f}]', flush=True)

    return pd.DataFrame(rows)


def write_report(df, meta, B, alpha, path, df2=None):
    L = []
    L.append('# B6 — 클러스터 부트스트랩 신뢰구간 (녹음 그룹 단위)\n')
    L.append(f'> B={B} · {100 * (1 - alpha):.0f}% CI · 재표집 단위 = **recording_id**(녹음 그룹).')
    L.append('> fold는 악기 단위 LOSO이며, fold 내 test set이 모든 split에서 동일하므로 '
             '재표집을 fold 단위로 공유한다. 집계는 주 수치와 동일 경로(trial 평균 → fold 평균).')
    L.append('> 모델 간 차이는 **동일 재표집에서 계산한 대응(paired) 차이**의 CI다.\n')
    L.append('> **왜 클립 단위가 아닌가**: 같은 recording_id의 클립들은 강하게 상관돼 있어, '
             '클립 단위 재표집은 유효 표본수를 과대평가해 CI를 좁게 만든다.\n')

    L.append('## 1. 모델별 macro F1 (BCa CI)\n')
    sub = df[df['kind'] == 'model']
    L.append('| seed | 모델 | F1 | BCa 95% CI | percentile 95% CI | boot SD |')
    L.append('|---|---|---|---|---|---|')
    for _, r in sub.iterrows():
        L.append(f"| {r['ratio']:.0%} | {r['quantity']} | {r['point']:.3f} | "
                 f"[{r['bca_lo']:.3f}, {r['bca_hi']:.3f}] | "
                 f"[{r['pct_lo']:.3f}, {r['pct_hi']:.3f}] | {r['boot_sd']:.3f} |")
    L.append('')

    L.append('## 2. 대응 차이 — 0을 포함하는가\n')
    sub = df[df['kind'] == 'contrast']
    L.append('| seed | 대비 | 차이 | BCa 95% CI | 0 포함 |')
    L.append('|---|---|---|---|---|')
    for _, r in sub.iterrows():
        inc = '**예**' if r['bca_lo'] <= 0 <= r['bca_hi'] else '아니오'
        L.append(f"| {r['ratio']:.0%} | {r['quantity']} | {r['point']:+.3f} | "
                 f"[{r['bca_lo']:+.3f}, {r['bca_hi']:+.3f}] | {inc} |")
    L.append('')

    L.append('## 3. 2수준 부트스트랩 — fold(악기)까지 재표집\n')
    L.append('> §1–2의 1수준 구간은 **test 클립 표집 변동만** 담는다. fold(악기)와 labeled 추출은')
    L.append('> 고정으로 취급하므로 "다른 악기에서도 같은 결론인가"에는 답하지 못한다.')
    L.append('> 이 논문의 주장은 악기를 넘어선 일반화이므로, 지배적 분산 성분인 **fold 간 변동**을')
    L.append('> 포함한 구간을 함께 싣는다. n_fold=4라 필연적으로 넓다 — 그것이 정직한 답이다.\n')
    if df2 is not None:
        sub = df2[df2['kind'] == 'contrast']
        L.append('| seed | 대비 | 차이 | 2수준 95% CI | 0 포함 | 1수준 대비 SD |')
        L.append('|---|---|---|---|---|---|')
        for _, r in sub.iterrows():
            inc = '**예**' if r['pct_lo'] <= 0 <= r['pct_hi'] else '아니오'
            m1 = df[(df['kind'] == 'contrast') & (df['ratio'] == r['ratio'])
                    & (df['quantity'] == r['quantity'])]
            ratio_sd = (r['boot_sd'] / m1['boot_sd'].iloc[0]) if len(m1) else float('nan')
            L.append(f"| {r['ratio']:.0%} | {r['quantity']} | {r['point']:+.3f} | "
                     f"[{r['pct_lo']:+.3f}, {r['pct_hi']:+.3f}] | {inc} | {ratio_sd:.1f}× |")
        L.append('')

    L.append('## 4. 점추정 재현 점검 (감사 CSV 대비)\n')
    L.append('> 예측 재생성 시 24-config 그리드를 재탐색하지 않고 감사가 고른 config만 재적합했다. '
             'MERT head는 확률적 학습이라 완전 동일하지 않을 수 있어 Δ를 명시한다.\n')
    L.append('| 모델 | 평균 Δ | 최대 |Δ| |')
    L.append('|---|---|---|')
    for m in MODELS:
        d = meta[f'{m}_delta']
        L.append(f'| {LABEL[m]} | {d.mean():+.4f} | {d.abs().max():.4f} |')
    L.append('')
    L.append('---\n')
    L.append('원자료: `cluster_bootstrap.csv` · 예측: `bootstrap_preds.npz`')
    path.write_text('\n'.join(L), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['preds', 'boot', 'all'], default='all')
    ap.add_argument('-B', type=int, default=2000)
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in ('preds', 'all'):
        print('[1/2] 분할 재현 + 선택 config 재적합 → per-clip 예측')
        meta = stage_preds()
        print(meta[[f'{m}_delta' for m in MODELS]].describe().loc[['mean', 'max']].to_string())

    if args.stage in ('boot', 'all'):
        print('\n[2/2] 클러스터 부트스트랩')
        df = stage_boot(args.B, args.alpha)
        df.to_csv(RESULTS_DIR / 'cluster_bootstrap.csv', index=False)
        print('\n[2b] 2수준(fold 포함) 부트스트랩')
        df2 = stage_boot2(args.B, args.alpha)
        df2.to_csv(RESULTS_DIR / 'cluster_bootstrap_2level.csv', index=False)
        _, meta = load_preds()
        write_report(df, meta, args.B, args.alpha, RESULTS_DIR / 'cluster_bootstrap.md', df2)
        print(f'\n저장: {RESULTS_DIR}/cluster_bootstrap.{{md,csv}}')


if __name__ == '__main__':
    main()


# ─────────────── 2수준(계층) 부트스트랩: fold(악기)까지 재표집 ───────────────
#
# 위의 1수준 부트스트랩은 **test 클립 표집 변동만** 담는다. fold(=악기)와 labeled 추출은
# 고정된 것으로 취급하므로, "다른 악기에서도 같은 결론이 나오는가"는 답하지 못한다.
# 이 논문의 주장은 악기를 넘어선 일반화이므로, 지배적 분산 성분인 **fold 간 변동**을
# 포함한 구간을 함께 보고해야 한다. n_fold=4라 필연적으로 넓다 — 그것이 정직한 답이다.

def stat_from_pairs(z, meta, info, ratio, model, pairs):
    """pairs: [(fold, positions), ...] — 같은 fold가 중복 등장할 수 있다."""
    fold_means = []
    for fold, pos in pairs:
        sub = meta[(meta['fold'] == fold) & (meta['ratio'] == ratio)]
        y = info[fold]['y'][pos]
        vals = [macro_f1(y, z[f'{model}__{t}'][pos].astype(np.int64))
                for t in sub['tag'].values]
        fold_means.append(np.mean(vals))
    return float(np.mean(fold_means))


def resample_pairs_two_level(info, rng):
    """fold를 복원추출하고, 뽑힌 각 fold에 대해 클러스터를 독립적으로 다시 복원추출."""
    folds = list(info.keys())
    drawn = rng.choice(folds, size=len(folds), replace=True)
    pairs = []
    for f in drawn:
        fi = info[f]
        cl = rng.choice(fi['clusters'], size=len(fi['clusters']), replace=True)
        pairs.append((f, np.concatenate([fi['pos'][c] for c in cl])))
    return pairs


def stage_boot2(B, alpha, seed=4242):
    z, meta = load_preds()
    rec_id = pd.read_csv(META_PATH)['recording_id'].values
    info = fold_clusters(z, meta, rec_id)
    ratios = sorted(meta['ratio'].unique())
    rng = np.random.default_rng(seed)
    ident_pairs = [(f, np.arange(len(fi['y']))) for f, fi in info.items()]

    boot_pairs = [resample_pairs_two_level(info, rng) for _ in range(B)]
    rows, t0 = [], time.time()
    for ratio in ratios:
        point, cache = {}, {}
        for m in MODELS:
            point[m] = stat_from_pairs(z, meta, info, ratio, m, ident_pairs)
            cache[m] = np.array([stat_from_pairs(z, meta, info, ratio, m, p)
                                 for p in boot_pairs])
            lo, hi = np.quantile(cache[m], [alpha / 2, 1 - alpha / 2])
            rows.append({'ratio': ratio, 'quantity': LABEL[m], 'kind': 'model',
                         'point': point[m], 'pct_lo': float(lo), 'pct_hi': float(hi),
                         'boot_sd': float(cache[m].std(ddof=1))})
        for m1, m2, name in CONTRASTS:
            d = cache[m1] - cache[m2]
            lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
            rows.append({'ratio': ratio, 'quantity': name, 'kind': 'contrast',
                         'point': point[m1] - point[m2],
                         'pct_lo': float(lo), 'pct_hi': float(hi),
                         'boot_sd': float(d.std(ddof=1))})
            print(f'  r={ratio:.0%} {name:<42s} {point[m1] - point[m2]:+.3f} '
                  f'[{lo:+.3f}, {hi:+.3f}]  {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(rows)

"""
D2 — descriptor probe: MERT 각 층에 "음악가의 물리량"이 얼마나 선형으로 존재하는가

배경(연구과정/12_방향성재고_정리.md §4):
      B2는 각 층으로 **라벨**을 얼마나 맞추는지 봤다(층별 기법 분류 성능 + 학습형 가중).
      여기서는 각 층으로 **handcrafted descriptor 자체**를 회귀한다. 즉
        "onset-strength / ZCR / centroid 같은, 음악가가 기법을 구분할 때 쓰는 물리량이
         MERT 어느 층에 얼마나 선형으로 담겨 있는가"
      를 직접 측정한다. B2의 학습 가중(L4 최대, L12 최소)이 '왜' 그런지를 채우는 조각.

읽는 법 — **절대 R²가 아니라 층 프로파일의 모양**을 본다:
      descriptor는 동일 오디오에서 파생됐으므로 절대 R²는 당연히 높다. 의미가 있는 건
      (a) 어느 층에서 정점을 찍는가, (b) 그 프로파일이 B2 학습 가중과 얼마나 정렬되는가,
      (c) 음성 대조군(chroma — 기법과 무관한 pitch class)과 프로파일이 갈리는가.

논문 함의(가설, 반증 가능):
      저층에서 이미 R²가 높고 깊어져도 안 오른다면 → 판별 신호가 저차원 물리량에 실려 있고
      그것이 MERT 저층에 이미 선형으로 존재 → "깊이가 살 게 없다"는 parity의 메커니즘 설명.
      **저층 R²가 낮게 나오면 이 서사는 기각된다** (12_ §6).

프로토콜(프로젝트 원칙 준수):
      - 악기 단위 GroupKFold(LOSO, 4 fold) — 같은 녹음이 train/test에 동시에 들어가지 않음
      - alpha는 **train 안에서만** RidgeCV(GCV)로 선택, target별 독립
      - R²는 fold별 평균이 아니라 **OOF 예측을 모아 한 번에** 계산
      - 입력 X는 train 통계로만 표준화(fold 누수 차단)

실행:
    python src/phase3_descriptor_probe.py
    python src/phase3_descriptor_probe.py --report-only   # 학습 없이 CSV로 리포트/그림만 재생성
출력:
    results/phase3/descriptor_probe.{md,csv}
    results/phase3/descriptor_probe.png     (층×그룹 R² + B2 학습가중 오버레이)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

np.random.seed(42)

EMB13_PATH  = ROOT / 'data' / 'mert_embeddings_13layer.npy'   # (N, 13, 768)
FEAT_PATH   = ROOT / 'data' / 'features_full.csv'
META_PATH   = ROOT / 'data' / 'mert_meta.csv'
LW_PATH     = ROOT / 'results' / 'phase2' / 'layer_probe.csv'  # B2 학습 가중 (오버레이용)
RESULTS_DIR = ROOT / 'results' / 'phase3'

ALPHAS = np.logspace(-1, 6, 15)

# descriptor 그룹 — 표시명, 컬럼 prefix, 논문에서의 역할
#   'flux'는 실체가 librosa.onset.onset_strength (A5에서 용어 통일) → "onset-strength envelope"
GROUPS = [
    ('Onset-strength envelope', r'^flux_',                 'core'),
    ('ZCR',                     r'^zcr_',                  'core'),
    ('RMS energy',              r'^rms_',                  'core'),
    ('Log first-onset time',    r'^log_first_onset_time$', 'core'),
    ('Spectral contrast',       r'^sc_',                   'mid'),
    ('MFCC',                    r'^mfcc_(?!d)',            'mid'),
    ('MFCC Δ',                  r'^mfcc_d_',               'mid'),
    ('MFCC ΔΔ',                 r'^mfcc_d2_',              'mid'),
    ('Chroma (neg. control)',   r'^chroma_',               'control'),
]


def load_data():
    emb = np.load(EMB13_PATH).astype(np.float32)          # (N, 13, 768)
    feat = pd.read_csv(FEAT_PATH)
    meta = pd.read_csv(META_PATH)
    assert len(emb) == len(feat) == len(meta), (
        f'행 수 불일치: emb={len(emb)} feat={len(feat)} meta={len(meta)}')
    # D1에서 실증한 positional alignment 재확인 (features_full.csv에 join key 컬럼이 없음)
    assert (feat['label_id'].values == meta['label_id'].values).all(), 'label_id 정렬 불일치'
    assert (feat['instrument'].astype(str).values
            == meta['instrument'].astype(str).values).all(), 'instrument 정렬 불일치'
    return emb, feat, meta


def resolve_groups(feat_cols):
    """GROUPS의 정규식을 실제 컬럼으로 해석. 미매칭 그룹은 에러."""
    out = []
    for name, pat, role in GROUPS:
        cols = [c for c in feat_cols if re.search(pat, c)]
        if not cols:
            raise ValueError(f'그룹 "{name}" 패턴 {pat} 이 어떤 컬럼과도 매칭되지 않음')
        out.append((name, cols, role))
    return out


def oof_r2(X_all, Y_all, groups):
    """악기 단위 GroupKFold OOF 예측 → target별 R².

    X_all: (N, 768) 한 층의 임베딩 / Y_all: (N, T) descriptor 행렬(표준화 전 원값)
    반환: (T,) R² 배열
    """
    gkf = GroupKFold(n_splits=4)
    oof = np.full_like(Y_all, np.nan, dtype=np.float64)

    for tr, te in gkf.split(X_all, groups=groups):
        # 표준화는 train 통계로만 (fold 누수 차단)
        mu, sd = X_all[tr].mean(0), X_all[tr].std(0) + 1e-8
        Xtr, Xte = (X_all[tr] - mu) / sd, (X_all[te] - mu) / sd
        ymu, ysd = Y_all[tr].mean(0), Y_all[tr].std(0) + 1e-8
        Ytr = (Y_all[tr] - ymu) / ysd

        # alpha는 train 내부 GCV로 target별 독립 선택
        model = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
        model.fit(Xtr, Ytr)
        pred = model.predict(Xte)
        if pred.ndim == 1:          # 단일 target 그룹(log_first_onset_time)은 1-D로 반환됨
            pred = pred[:, None]
        oof[te] = pred * ysd + ymu

    ss_res = ((Y_all - oof) ** 2).sum(0)
    ss_tot = ((Y_all - Y_all.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def run(emb, feat, meta, resolved):
    groups = meta['instrument'].values
    n_layers = emb.shape[1]
    rows = []

    for layer in range(n_layers):
        X = emb[:, layer, :].astype(np.float64)
        for name, cols, role in resolved:
            Y = feat[cols].values.astype(np.float64)
            r2 = oof_r2(X, Y, groups)
            rows.append({
                'layer':    layer,
                'group':    name,
                'role':     role,
                'n_cols':   len(cols),
                'r2_mean':  float(np.mean(r2)),
                'r2_median': float(np.median(r2)),
                'r2_max':   float(np.max(r2)),
            })
            print(f'  L{layer:<2d} {name:<24s} R²(mean)={np.mean(r2):.3f}')
    return pd.DataFrame(rows)


def load_lw_weights():
    """B2의 학습형 레이어 가중 평균 (오버레이용). 없으면 None."""
    if not LW_PATH.exists():
        return None
    df = pd.read_csv(LW_PATH)
    if 'lw_weights' not in df.columns:
        return None
    ws = []
    for s in df['lw_weights'].dropna():
        try:
            v = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', str(s))]
            if len(v) == 13:
                ws.append(v)
        except ValueError:
            continue
    return np.mean(ws, axis=0) if ws else None


def profile_stats(df, lw):
    """그룹별 층 프로파일 요약 — 정점 층, 저층/고층 대비, B2 가중과의 정렬."""
    from scipy.stats import spearmanr
    out = []
    for name, g in df.groupby('group', sort=False):
        g = g.sort_values('layer')
        r2 = g['r2_mean'].values
        rec = {
            'group':      name,
            'role':       g['role'].iloc[0],
            'peak_layer': int(g['layer'].values[int(np.argmax(r2))]),
            'r2_peak':    float(r2.max()),
            'r2_L0_L4':   float(r2[:5].mean()),
            'r2_L8_L12':  float(r2[8:].mean()),
        }
        rec['delta_high_low'] = rec['r2_L8_L12'] - rec['r2_L0_L4']
        if lw is not None:
            rho, p = spearmanr(r2, lw)
            rec['spearman_vs_B2w'] = float(rho)
            rec['p_vs_B2w'] = float(p)
        out.append(rec)
    return pd.DataFrame(out)


def make_plot(df, lw, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [n for n, _, _ in GROUPS]
    piv = df.pivot(index='group', columns='layer', values='r2_mean').reindex(names)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9),
                             gridspec_kw={'height_ratios': [2.1, 1]})

    ax = axes[0]
    im = ax.imshow(piv.values, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f'L{i}' for i in piv.columns])
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index)
    ax.set_title('Descriptor probe — layer-wise OOF $R^2$ '
                 '(instrument-wise LOSO)\n'
                 'Linear availability of musician-facing acoustic descriptors '
                 'in each MERT layer')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if v < 0.6 else 'black')
    fig.colorbar(im, ax=ax, label='OOF R²', fraction=0.025)

    ax = axes[1]
    for name in names:
        role = df[df['group'] == name]['role'].iloc[0]
        style = dict(lw=2.4, marker='o', ms=4)
        if role == 'control':
            style.update(ls='--', color='crimson', lw=2.8)
        elif role == 'core':
            style.update(alpha=0.95)
        else:
            style.update(alpha=0.35, lw=1.4, ms=3)
        ax.plot(piv.columns, piv.loc[name].values, label=name, **style)
    ax.set_xlabel('MERT layer'); ax.set_ylabel('OOF R²')
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f'L{i}' for i in piv.columns])
    ax.grid(alpha=0.3)

    if lw is not None:
        ax2 = ax.twinx()
        ax2.bar(range(len(lw)), lw, alpha=0.16, color='gray', zorder=0)
        ax2.set_ylabel('B2 learned layer weight (gray bars)', color='gray')
        ax2.tick_params(axis='y', colors='gray')
    ax.legend(fontsize=7.5, ncol=2, loc='lower left')
    ax.set_title('Layer profiles — lines: descriptor $R^2$ ; gray bars: B2 learned weights for technique classification', fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(df, stats, lw, path):
    L = []
    L.append('# D2 — Descriptor probe: MERT 층별 음향 서술자 선형 가용성\n')
    L.append('> 각 층 임베딩(768d) → handcrafted descriptor **회귀**의 OOF R².')
    L.append('> 프로토콜: 악기 단위 GroupKFold(LOSO, 4 fold), alpha는 train 내 RidgeCV(GCV)로 '
             'target별 선택, 표준화는 train 통계만 사용, R²는 OOF 예측 pooled 계산.\n')
    L.append('> **읽는 법**: descriptor는 동일 오디오 파생이라 절대 R²가 높은 것은 당연하다. '
             '의미 있는 것은 (a) 정점 층, (b) 저층↔고층 대비, (c) 음성 대조군(chroma)과의 분리, '
             '(d) B2 학습가중과의 정렬이다.\n')

    L.append('## 1. 그룹별 프로파일 요약\n')
    cols = ['group', 'role', 'peak_layer', 'r2_peak', 'r2_L0_L4', 'r2_L8_L12', 'delta_high_low']
    hdr = ['그룹', '역할', '정점층', 'R²(정점)', 'R² L0–L4', 'R² L8–L12', '고층−저층']
    if 'spearman_vs_B2w' in stats.columns:
        cols += ['spearman_vs_B2w']; hdr += ['B2가중과 ρ']
    L.append('| ' + ' | '.join(hdr) + ' |')
    L.append('|' + '---|' * len(hdr))
    for _, r in stats.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f'L{int(v)}' if c == 'peak_layer'
                         else (f'{v:+.3f}' if c in ('delta_high_low', 'spearman_vs_B2w')
                               else (f'{v:.3f}' if isinstance(v, float) else str(v))))
        L.append('| ' + ' | '.join(cells) + ' |')
    L.append('')

    if lw is not None:
        L.append('## 2. B2 학습형 레이어 가중 (대조용)\n')
        L.append('| ' + ' | '.join(f'L{i}' for i in range(len(lw))) + ' |')
        L.append('|' + '---|' * len(lw))
        L.append('| ' + ' | '.join(f'{w:.3f}' for w in lw) + ' |')
        L.append(f'\n최대 **L{int(np.argmax(lw))}** ({lw.max():.3f}) / '
                 f'최소 **L{int(np.argmin(lw))}** ({lw.min():.3f})\n')

    L.append('## 3. 전체 표 (층 × 그룹)\n')
    piv = df.pivot(index='group', columns='layer', values='r2_mean').reindex(
        [n for n, _, _ in GROUPS])
    L.append('| 그룹 | ' + ' | '.join(f'L{c}' for c in piv.columns) + ' |')
    L.append('|' + '---|' * (piv.shape[1] + 1))
    for name, row in piv.iterrows():
        L.append(f'| {name} | ' + ' | '.join(f'{v:.3f}' for v in row.values) + ' |')
    L.append('')
    L.append('---\n')
    L.append('그림: `descriptor_probe.png` · 원자료: `descriptor_probe.csv`')
    path.write_text('\n'.join(L), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-only', action='store_true',
                    help='학습 없이 기존 CSV로 리포트/그림만 재생성')
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / 'descriptor_probe.csv'

    if args.report_only:
        assert csv_path.exists(), f'CSV 없음: {csv_path}'
        df = pd.read_csv(csv_path)
    else:
        emb, feat, meta = load_data()
        resolved = resolve_groups(feat.columns)
        print(f'데이터: emb {emb.shape} / 그룹 {len(resolved)}개 / '
              f'악기 {meta.instrument.nunique()}개(LOSO fold)')
        for name, cols, role in resolved:
            print(f'  - {name:<24s} {len(cols):>3d}컬럼 ({role})')
        df = run(emb, feat, meta, resolved)
        df.to_csv(csv_path, index=False)

    lw = load_lw_weights()
    stats = profile_stats(df, lw)
    stats.to_csv(RESULTS_DIR / 'descriptor_probe_profile.csv', index=False)
    make_plot(df, lw, RESULTS_DIR / 'descriptor_probe.png')
    write_report(df, stats, lw, RESULTS_DIR / 'descriptor_probe.md')

    print('\n' + stats.to_string(index=False))
    print(f'\n저장: {RESULTS_DIR}/descriptor_probe.{{md,csv,png}}')


if __name__ == '__main__':
    main()

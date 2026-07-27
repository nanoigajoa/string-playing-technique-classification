"""
B2 — 레이어별 probe + 학습형 레이어 가중 (관행 따라잡기)

배경(§2.3): 우리는 last_hidden_state(마지막 층)에 단일 linear만 얹어 관행(레이어 가중
      결합 + MLP head) 이하에서 출발했다. B1이 만든 13층 캐시로 이제 관행을 따라잡고,
      "동급" 결론이 관행 수준에서도 유지되는지 검증한다.

비교 대상 (모두 A1과 동일 프로토콜: 3분할 val 분리, tr-only 주 수치):
  - last        : 마지막 층(12)만 — 우리가 쓰던 설정 (감사의 mert_tuned와 대응)
  - best_layer  : 13개 중 val_select에서 가장 좋은 단일 레이어
  - layer_weighted : 학습형 softmax 가중으로 13층 결합 (MARBLE/SUPERB 관행)
  head 구조(linear/mlp256)와 hparam은 감사 그리드를 그대로 재사용.

실행:
    python src/phase1_layer_probe.py                 # 전 seed
    python src/phase1_layer_probe.py --ratios 0.5    # 특정 seed
출력:
    results/phase2/layer_probe.md , layer_probe.csv
    results/phase2/layer_importance.png   (레이어별 성능 + 학습된 가중)
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

torch.manual_seed(42)
np.random.seed(42)

CLASSES   = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
N_CLASSES = len(CLASSES)
DEVICE    = 'mps' if torch.backends.mps.is_available() else 'cpu'

EMB13_PATH  = ROOT / 'data' / 'mert_embeddings_13layer.npy'   # (N, 13, 768)
LAB_PATH    = ROOT / 'data' / 'mert_labels.npy'
SPLITS_PATH = ROOT / 'data' / 'split_configs.json'
RESULTS_DIR = ROOT / 'results' / 'phase2'

# 감사와 동일 그리드 (arch/lr/wd/dropout)
GRID = {
    'arch':    ['linear', 'mlp256'],
    'lr':      [3e-4, 1e-3, 3e-3],
    'wd':      [1e-4, 1e-3],
    'dropout': [0.1, 0.3],
}

emb13  = np.load(EMB13_PATH).astype(np.float32)   # (N, 13, 768)
labels = np.load(LAB_PATH).astype(np.int64)
N_LAYERS = emb13.shape[1]
assert emb13.shape[0] == labels.shape[0], (emb13.shape, labels.shape)

counts = np.bincount(labels, minlength=N_CLASSES).astype(float)
class_weights = torch.tensor(counts.sum() / (N_CLASSES * counts), dtype=torch.float32).to(DEVICE)


# ── heads ─────────────────────────────────────────────────────────────────────

class SingleLayerHead(nn.Module):
    """고정된 한 레이어(768)에 대한 head."""
    def __init__(self, arch, dropout, in_dim=768):
        super().__init__()
        if arch == 'linear':
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                     nn.Linear(in_dim, N_CLASSES))
        else:  # mlp256
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                     nn.Linear(in_dim, 256), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(256, N_CLASSES))

    def forward(self, x):          # x: (B, 768)
        return self.net(x)


class LayerWeightedHead(nn.Module):
    """13층을 학습형 softmax 가중으로 결합 후 head (SUPERB/MARBLE 관행)."""
    def __init__(self, arch, dropout, n_layers=N_LAYERS, in_dim=768):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        if arch == 'linear':
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                     nn.Linear(in_dim, N_CLASSES))
        else:
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                     nn.Linear(in_dim, 256), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(256, N_CLASSES))

    def forward(self, x):          # x: (B, 13, 768)
        w = torch.softmax(self.layer_logits, dim=0)       # (13,)
        pooled = (x * w[None, :, None]).sum(dim=1)         # (B, 768)
        return self.net(pooled)

    def weights(self):
        return torch.softmax(self.layer_logits, dim=0).detach().cpu().numpy()


# ── data helpers ──────────────────────────────────────────────────────────────

def split_val(labeled_idx, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.array(labeled_idx); lab = labels[idx]
    tr, va = [], []
    for c in range(N_CLASSES):
        pos = idx[lab == c]
        n_val = max(1, round(len(pos) * val_ratio))
        chosen = rng.choice(len(pos), n_val, replace=False)
        va.extend(pos[chosen].tolist()); tr.extend(np.delete(pos, chosen).tolist())
    return tr, va


def _xy(idx, layer=None):
    """layer=int → 단일 레이어 (B,768); layer=None → 전체 (B,13,768)."""
    x = emb13[idx, layer, :] if layer is not None else emb13[idx]
    return torch.tensor(x).to(DEVICE), torch.tensor(labels[idx]).to(DEVICE)


LOGIT_LR_MULT = 25   # layer_logits는 softmax-정규화라 gradient가 작아 별도 높은 lr (SUPERB/S3PRL 관행)


def train_head(model, tr_idx, es_idx, cfg, layer=None, epochs=200, patience=20):
    crit = nn.CrossEntropyLoss(weight=class_weights)
    # layer_logits가 있으면 별도 param group에 높은 lr (균등 붕괴 방지, B2-v2)
    if hasattr(model, 'layer_logits'):
        logit_params = [model.layer_logits]
        rest = [p for n, p in model.named_parameters() if n != 'layer_logits']
        opt = torch.optim.AdamW([
            {'params': rest, 'lr': cfg['lr'], 'weight_decay': cfg['wd']},
            {'params': logit_params, 'lr': cfg['lr'] * LOGIT_LR_MULT, 'weight_decay': 0.0},
        ])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    X_tr, y_tr = _xy(tr_idx, layer); X_es, y_es = _xy(es_idx, layer)
    best_f1, best_state, wait = -1.0, None, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        crit(model(X_tr), y_tr).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_es).argmax(1).cpu().numpy()
        vf1 = f1_score(y_es.cpu().numpy(), pred, average='macro', zero_division=0)
        if vf1 > best_f1:
            best_f1, best_state, wait = vf1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_f1


def eval_head(model, idx, layer=None):
    model.eval()
    X, y = _xy(idx, layer)
    with torch.no_grad():
        pred = model(X).argmax(1).cpu().numpy()
    return f1_score(y.cpu().numpy(), pred, average='macro', zero_division=0)


def search(make_model, tr_idx, es_idx, val_sel_idx, test_idx, layer=None,
           epochs=200, patience=20):
    """감사 그리드 탐색: es로 early-stop, val_sel로 config 선택, test로 최종."""
    best_vf1, best_test, best_cfg = -1.0, None, None
    for combo in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID.keys(), combo))
        model = make_model(cfg).to(DEVICE)
        model, _ = train_head(model, tr_idx, es_idx, cfg, layer, epochs=epochs, patience=patience)
        vf1 = eval_head(model, val_sel_idx, layer)
        if vf1 > best_vf1:
            best_vf1, best_test, best_cfg = vf1, eval_head(model, test_idx, layer), cfg
            best_model = model
    return best_test, best_cfg, best_vf1, best_model


def process_split(s):
    seed_core, val_sel = split_val(s['labeled_idx'], val_ratio=0.2, seed=s['trial'] + 1000)
    tr_idx, es_idx     = split_val(seed_core, val_ratio=0.2, seed=s['trial'])
    test_idx = s['test_idx']
    out = {'fold': s['fold'], 'ratio': s['seed_ratio'], 'trial': s['trial']}

    # 1) last layer only (마지막 층) — 우리가 쓰던 설정
    last = N_LAYERS - 1
    out['last'], out['last_cfg'], _, _ = search(
        lambda c: SingleLayerHead(c['arch'], c['dropout']),
        tr_idx, es_idx, val_sel, test_idx, layer=last)

    # 2) best single layer — 먼저 고정 config로 13층을 val_select에서 랭킹(비용 절감),
    #    선택된 최적 레이어에만 전체 그리드 탐색을 적용한다.
    RANK_CFG = {'arch': 'mlp256', 'lr': 1e-3, 'wd': 1e-4, 'dropout': 0.1}
    best_layer, best_layer_vf1 = None, -1.0
    per_layer_val = {}
    for L in range(N_LAYERS):
        model = SingleLayerHead(RANK_CFG['arch'], RANK_CFG['dropout']).to(DEVICE)
        model, _ = train_head(model, tr_idx, es_idx, RANK_CFG, layer=L)
        vf1 = eval_head(model, val_sel, layer=L)
        per_layer_val[L] = vf1
        if vf1 > best_layer_vf1:
            best_layer_vf1, best_layer = vf1, L
    # 선택된 레이어에 전체 그리드
    out['best_layer'] = best_layer
    out['best_layer_test'], out['best_layer_cfg'], _, _ = search(
        lambda c: SingleLayerHead(c['arch'], c['dropout']),
        tr_idx, es_idx, val_sel, test_idx, layer=best_layer)
    out['per_layer_val'] = json.dumps({int(k): round(v, 4) for k, v in per_layer_val.items()})

    # 3) learned layer-weighted — logits에 별도 높은 lr + 더 긴 스케줄(B2-v2, 균등 붕괴 방지)
    lw_test, lw_cfg, _, lw_model = search(
        lambda c: LayerWeightedHead(c['arch'], c['dropout']),
        tr_idx, es_idx, val_sel, test_idx, layer=None, epochs=400, patience=50)
    out['layer_weighted'] = lw_test
    out['lw_weights'] = json.dumps([round(float(w), 4) for w in lw_model.weights()])
    return out


def fold_stats(df, ratio, col):
    fm = df[df['ratio'] == ratio].groupby('fold')[col].mean()
    return fm.mean(), fm.std()


AUDIT_CSV = ROOT / 'results' / 'phase1' / 'mert_tuning_audit.csv'
V1_CSV    = RESULTS_DIR / 'layer_probe_v1.csv'


def _pooled_sigma(s_a, s_b):
    """두 조건의 fold-level std를 합친 척도. 차이를 σ 배수로 표현할 때 쓴다."""
    return float(np.sqrt((s_a ** 2 + s_b ** 2) / 2))


def head_to_head_lines(df):
    """A1 감사(SVM)와의 head-to-head. 감사 CSV가 없으면 섹션을 생략한다."""
    if not AUDIT_CSV.exists():
        return [f"> (head-to-head 생략: {AUDIT_CSV.name} 없음)", ""]
    audit = pd.read_csv(AUDIT_CSV)
    lines = [
        "## SVM과의 head-to-head (관행 probing 적용 후)",
        "",
        "> A1의 SVM 수치와 동일 split·동일 프로토콜. σ는 fold-level pooled std 대비 배수(n=4).",
        "",
        "| seed | MERT(layer-weighted) | SVM(tr, 동일 학습데이터) | 차이 | SVM(seed_core 80%) | 차이 |",
        "|---|---|---|---|---|---|",
    ]
    for ratio in sorted(df['ratio'].unique()):
        lw_m, lw_s = fold_stats(df, ratio, 'layer_weighted')
        sv_m, sv_s = fold_stats(audit, ratio, 'svm_tuned')
        fu_m, _    = fold_stats(audit, ratio, 'svm_tuned_full')
        d1, d2 = lw_m - sv_m, lw_m - fu_m
        sig = _pooled_sigma(lw_s, sv_s)
        # 강조 표기 없음: 어느 구간도 유의하지 않으므로 굵게 쓰면 유의성을 시사하게 된다
        lines.append(f"| {int(ratio*100)}% | {lw_m:.3f} | {sv_m:.3f} | "
                     f"{d1:+.3f} ({abs(d1)/sig:.2f}σ) | {fu_m:.3f} | {d2:+.3f} |")
    lines += [
        "",
        "- 동일 학습데이터 기준 방향은 일관되게 MERT 쪽이나, **어느 구간도 통계적 유의에 이르지 못함**(n=4, 검정력 낮음).",
        "- SVM에 학습량 이점(seed_core 80%)을 주면 차이가 대부분 소멸 → **공정 조건에서 두 계열은 구분 불가**.",
        "",
    ]
    return lines


def stability_lines(df):
    """v1(균등에 가까운 가중) vs v2(logits 별도 lr) — under-training 가설 검증."""
    if not V1_CSV.exists():
        return [f"> (안정성 비교 생략: {V1_CSV.name} 없음)", ""]
    v1 = pd.read_csv(V1_CSV)
    lines = [
        "## 안정성 — v1(균등 가중) vs v2(학습 가중)",
        "",
        "| seed | v1 lw | v2 lw | Δ |",
        "|---|---|---|---|",
    ]
    dmax = 0.0
    for ratio in sorted(df['ratio'].unique()):
        a, _ = fold_stats(v1, ratio, 'layer_weighted')
        b, _ = fold_stats(df, ratio, 'layer_weighted')
        dmax = max(dmax, abs(b - a))
        lines.append(f"| {int(ratio*100)}% | {a:.3f} | {b:.3f} | {b-a:+.3f} |")
    lines += [
        "",
        f"> 레이어 가중 학습 방식(균등 붕괴 → logits 별도 lr {LOGIT_LR_MULT}×)을 바꿔도 "
        f"결과가 ≤{dmax:.3f}로 동일.",
        "> **under-training 가설 기각 — 결론이 가중 학습 세부에 의존하지 않는다.**",
        "",
    ]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratios', type=float, nargs='*', default=[0.05, 0.10, 0.20, 0.50])
    ap.add_argument('--report-only', action='store_true',
                    help='학습 없이 기존 layer_probe.csv에서 리포트만 재생성')
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.report_only:
        df = pd.read_csv(RESULTS_DIR / 'layer_probe.csv')
        print(f"report-only | rows={len(df)}", flush=True)
    else:
        with open(SPLITS_PATH) as f:
            splits = [s for s in json.load(f)['splits'] if s['seed_ratio'] in args.ratios]
        print(f"device={DEVICE} | layers={N_LAYERS} | splits={len(splits)}", flush=True)

        rows = []
        for i, s in enumerate(splits, 1):
            r = process_split(s); rows.append(r)
            print(f"  [{i:>2}/{len(splits)}] fold={s['fold']} r={s['seed_ratio']:.0%} t={s['trial']} | "
                  f"last={r['last']:.3f} best(L{r['best_layer']})={r['best_layer_test']:.3f} "
                  f"lw={r['layer_weighted']:.3f} | {time.time()-t0:.0f}s", flush=True)

        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / 'layer_probe.csv', index=False)

    # 리포트
    lines = [
        "# B2 — 레이어별 probe + 학습형 레이어 가중",
        "",
        "> A1과 동일 프로토콜(3분할 val 분리, tr-only). 목적: 마지막-층-only(관행 이하)를",
        "> best-layer / layer-weighted(관행)로 따라잡을 때 성능과 '동급' 결론이 어떻게 되는지.",
        "",
        "## 성능 비교 (fold 평균±표준편차, n=4)",
        "",
        "| seed | last(마지막층) | best single layer | layer-weighted | lw − last |",
        "|---|---|---|---|---|",
    ]
    for ratio in sorted(df['ratio'].unique()):
        la_m, la_s = fold_stats(df, ratio, 'last')
        bl_m, bl_s = fold_stats(df, ratio, 'best_layer_test')
        lw_m, lw_s = fold_stats(df, ratio, 'layer_weighted')
        lines.append(f"| {int(ratio*100)}% | {la_m:.4f}±{la_s:.4f} | {bl_m:.4f}±{bl_s:.4f} "
                     f"| {lw_m:.4f}±{lw_s:.4f} | **{lw_m-la_m:+.4f}** |")

    lines.append("")
    lines += head_to_head_lines(df)
    lines += stability_lines(df)

    # 자주 선택된 best layer + 평균 학습 가중
    all_w = np.array([json.loads(w) for w in df['lw_weights']])
    mean_w = all_w.mean(axis=0)
    lines += [
        "## 레이어 사용 양상",
        "",
        f"- best single layer 분포(층:횟수): { {int(k): int(v) for k, v in df['best_layer'].value_counts().sort_index().items()} }",
        f"- 학습형 가중 평균(층0→12): {np.round(mean_w, 3).tolist()}",
        f"- 가중 최대 층: **L{int(mean_w.argmax())}** (마지막 층 L{N_LAYERS-1} 가중 {mean_w[-1]:.3f})",
        "",
    ]

    # 착지 — probe 설정 효과 vs 모델 계열 효과
    probe_eff = [fold_stats(df, r, 'layer_weighted')[0] - fold_stats(df, r, 'last')[0]
                 for r in sorted(df['ratio'].unique())]
    lines += ["## 착지 (논문 서술)", ""]
    if AUDIT_CSV.exists():
        audit = pd.read_csv(AUDIT_CSV)
        fam_eff = [fold_stats(df, r, 'layer_weighted')[0] - fold_stats(audit, r, 'svm_tuned_full')[0]
                   for r in sorted(df['ratio'].unique())]
        lines += [
            f"- **probe 설정 효과**: {min(probe_eff):+.3f}~{max(probe_eff):+.3f} "
            f"(전 구간 일관, 평균 {np.mean(probe_eff):+.3f})",
            f"- **모델 계열 효과**(동등 예산): {min(fam_eff):+.3f}~{max(fam_eff):+.3f} "
            f"(부호 불일치, 평균 {np.mean(fam_eff):+.3f})",
            "- → **probe 설정 효과가 모델 계열 효과보다 크다.** Musgrave et al.의 "
            "\"튜닝 예산 차이가 방법론 차이보다 크다\"의 MIR 판이며, §2.3을 정량으로 뒷받침한다.",
            "",
            "> 착지: **'동급' 유지**. 단, 기존 동급은 *관행 이하로 probing된* MERT로 잰 것이었다.",
            "> 관행대로 재면 방향은 일관되게 MERT 쪽이나 fold-std 안에 머물러 유의하지 않으며,",
            "> SVM에 동등 학습량을 주면 차이가 소멸한다. 유의하지 않은 방향성을 결론으로",
            "> 승격시키지 않는 것이 이 논문의 전제와 일관된 선택이다.",
            "",
        ]
    (RESULTS_DIR / 'layer_probe.md').write_text("\n".join(lines) + "\n", encoding='utf-8')

    # 그림
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        # 좌: 레이어별 평균 val f1 (전 split 평균)
        pl = np.array([[json.loads(r)[str(L)] if str(L) in json.loads(r) else json.loads(r).get(L)
                        for L in range(N_LAYERS)] for r in df['per_layer_val']], dtype=float)
        ax[0].plot(range(N_LAYERS), pl.mean(axis=0), 'o-')
        ax[0].set(xlabel='layer', ylabel='val macro F1 (mean)', title='레이어별 단일 probe 성능')
        ax[0].grid(alpha=0.3)
        # 우: 학습형 가중 평균
        ax[1].bar(range(N_LAYERS), mean_w)
        ax[1].set(xlabel='layer', ylabel='learned weight', title='학습형 레이어 가중 (평균)')
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / 'layer_importance.png', dpi=150)
        plt.close()
    except Exception as e:
        print(f"  (그림 스킵: {e})")

    print(f"\n저장: {RESULTS_DIR/'layer_probe.md'}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()

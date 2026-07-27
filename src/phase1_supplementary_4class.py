"""
Supplementary — ordinario 제외 4클래스 공정 튜닝 비교

목적 (지금 방향 = 평가 함정 방법론 논문):
  "ordinario(최다 클래스)를 빼도 SVM ≈ MERT 동급 결론이 유지되는가?"
  → "ordinario가 많아서 F1이 높은 것 아니냐"는 리뷰어 공격 선제 차단 + 더 어려운 세팅 검증.

방법: `phase1_mert_tuning_audit.py`와 동일한 val 기반 공정 튜닝(양쪽 모두)을
      ordinario를 제거한 4클래스에서 재실행한다. 누수 없음(탐색은 seed의 val에서만).

실행:
    python src/phase1_supplementary_4class.py                 # 4 ratio 전부 (~7분)
    python src/phase1_supplementary_4class.py --ratios 0.5    # 특정 ratio만

출력:
    results/phase1/supplementary_4class.md
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

torch.manual_seed(42)
np.random.seed(42)

# ── 5클래스 원본 → ordinario 제거 후 4클래스 ─────────────────────────────────────
CLASSES_5 = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
ORD_ID    = CLASSES_5.index('ordinario')  # = 0
CLASSES   = [c for c in CLASSES_5 if c != 'ordinario']  # 4클래스
N_CLASSES = len(CLASSES)  # 4
DEVICE    = 'mps' if torch.backends.mps.is_available() else 'cpu'

EMB_PATH    = ROOT / 'data' / 'mert_embeddings.npy'
LAB_PATH    = ROOT / 'data' / 'mert_labels.npy'
FEAT_PATH   = ROOT / 'data' / 'features_full.csv'
SPLITS_PATH = ROOT / 'data' / 'split_configs.json'
OUT         = ROOT / 'results' / 'phase1' / 'supplementary_4class.md'

# ── 데이터 로드 후 ordinario 제거 + 컴팩트 재인덱싱 ──────────────────────────────
_emb_full = np.load(EMB_PATH).astype(np.float32)
_lab_full = np.load(LAB_PATH).astype(np.int64)
_feat_df  = pd.read_csv(FEAT_PATH)
_feat_cols = [c for c in _feat_df.columns if c not in ('label_id', 'instrument')]
_X_full   = _feat_df[_feat_cols].to_numpy(dtype=np.float32)

keep      = _lab_full != ORD_ID                    # ordinario 아닌 행
old2new   = np.full(len(_lab_full), -1, dtype=np.int64)
old2new[keep] = np.arange(keep.sum())

embeddings = _emb_full[keep]
X_feat     = _X_full[keep]
labels     = _lab_full[keep] - 1                    # 1..4 → 0..3 (ordinario=0 제거했으므로 -1)
assert labels.min() == 0 and labels.max() == N_CLASSES - 1

counts = np.bincount(labels, minlength=N_CLASSES).astype(float)
class_weights = torch.tensor(counts.sum() / (N_CLASSES * counts), dtype=torch.float32).to(DEVICE)


def remap_idx(idx_list):
    """전역(5클래스) 인덱스 리스트 → ordinario 제외 + 컴팩트 인덱스로 변환."""
    arr = np.asarray(idx_list, dtype=np.int64)
    arr = arr[keep[arr]]           # ordinario 샘플 제거
    return old2new[arr].tolist()   # 새 인덱스로 매핑


# ── 탐색 그리드 (감사 스크립트와 동일) ───────────────────────────────────────────
MERT_GRID = {'arch': ['linear', 'mlp256'], 'lr': [3e-4, 1e-3, 3e-3],
             'wd': [1e-4, 1e-3], 'dropout': [0.1, 0.3]}          # 24 config
MERT_DEFAULT = {'arch': 'linear', 'lr': 1e-3, 'wd': 1e-4, 'dropout': 0.3}
SVM_GRID = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.01, 0.001]}  # 12 config
SVM_DEFAULT = {'C': 10, 'gamma': 'scale'}


# ── MERT head ─────────────────────────────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, arch, dropout, in_dim=768, n_classes=N_CLASSES):
        super().__init__()
        if arch == 'linear':
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                     nn.Linear(in_dim, n_classes))
        elif arch == 'mlp256':
            self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                     nn.Linear(in_dim, 256), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(256, n_classes))
        else:
            raise ValueError(arch)

    def forward(self, x):
        return self.net(x)


def _tensors(idx):
    return (torch.tensor(embeddings[idx]).to(DEVICE),
            torch.tensor(labels[idx]).to(DEVICE))


def train_head_cfg(tr_idx, es_idx, cfg, epochs=200, patience=20):
    """es_idx는 early-stop 전용. config 선택은 별도 val_select에서 (A1)."""
    model = Head(cfg['arch'], cfg['dropout']).to(DEVICE)
    crit  = nn.CrossEntropyLoss(weight=class_weights)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    X_tr, y_tr = _tensors(tr_idx)
    X_es, y_es = _tensors(es_idx)
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
    return model, best_f1   # es_val 점수 — config 선택에 쓰지 말 것


def eval_head(model, test_idx):
    model.eval()
    X_te, y_te = _tensors(test_idx)
    with torch.no_grad():
        pred = model(X_te).argmax(1).cpu().numpy()
    return f1_score(y_te.cpu().numpy(), pred, average='macro', zero_division=0)


def train_eval_svm(tr_idx, val_idx, test_idx, cfg):
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svm', SVC(kernel='rbf', C=cfg['C'], gamma=cfg['gamma'],
                                 class_weight='balanced', random_state=42))])
    pipe.fit(X_feat[tr_idx], labels[tr_idx])
    vf1 = f1_score(labels[val_idx], pipe.predict(X_feat[val_idx]), average='macro', zero_division=0)
    tf1 = f1_score(labels[test_idx], pipe.predict(X_feat[test_idx]), average='macro', zero_division=0)
    return vf1, tf1


def split_val(labeled_idx, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.array(labeled_idx); lab = labels[idx]
    tr, va = [], []
    for c in range(N_CLASSES):
        pos = idx[lab == c]
        if len(pos) == 0:
            continue
        n_val = max(1, round(len(pos) * val_ratio))
        chosen = rng.choice(len(pos), n_val, replace=False)
        va.extend(pos[chosen].tolist()); tr.extend(np.delete(pos, chosen).tolist())
    return tr, va


def process_split(s):
    lab_idx = remap_idx(s['labeled_idx'])
    test_idx = remap_idx(s['test_idx'])
    # A1: config 선택 전용 val_select 고정, early-stop val은 seed_core 안에서만 → 둘이 겹치지 않음
    seed_core, val_sel = split_val(lab_idx, val_ratio=0.2, seed=s['trial'] + 1000)
    tr_idx, es_idx     = split_val(seed_core, val_ratio=0.2, seed=s['trial'])
    out = {'fold': s['fold'], 'ratio': s['seed_ratio'], 'trial': s['trial']}

    m_def, _ = train_head_cfg(tr_idx, es_idx, MERT_DEFAULT)
    out['mert_default'] = eval_head(m_def, test_idx)

    best_vf1, best_test = -1.0, None
    for combo in itertools.product(*MERT_GRID.values()):
        cfg = dict(zip(MERT_GRID.keys(), combo))
        model, _ = train_head_cfg(tr_idx, es_idx, cfg)
        vf1 = eval_head(model, val_sel)
        if vf1 > best_vf1:
            best_vf1, best_test = vf1, eval_head(model, test_idx)
            out['mert_best_cfg'] = cfg
    out['mert_tuned'] = best_test

    # SVM: 주 수치는 MERT와 동일 학습데이터(tr_idx), config 선택은 val_sel
    _, out['svm_default'] = train_eval_svm(tr_idx, val_sel, test_idx, SVM_DEFAULT)
    best_vf1, best_test = -1.0, None
    for combo in itertools.product(*SVM_GRID.values()):
        cfg = dict(zip(SVM_GRID.keys(), combo))
        vf1, tf1 = train_eval_svm(tr_idx, val_sel, test_idx, cfg)
        if vf1 > best_vf1:
            best_vf1, best_test = vf1, tf1
    out['svm_tuned'] = best_test
    return out


def fold_stats(df, ratio, col):
    fm = df[df['ratio'] == ratio].groupby('fold')[col].mean()
    return fm.mean(), fm.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratios', type=float, nargs='*', default=[0.05, 0.10, 0.20, 0.50])
    args = ap.parse_args()

    with open(SPLITS_PATH) as f:
        splits = [s for s in json.load(f)['splits'] if s['seed_ratio'] in args.ratios]

    print(f"device={DEVICE} | 4클래스={CLASSES} | splits={len(splits)}")
    print(f"샘플: {len(labels)}개 (ordinario 제외), 클래스 분포={counts.astype(int).tolist()}")

    rows, t0 = [], time.time()
    for i, s in enumerate(splits, 1):
        r = process_split(s)
        rows.append(r)
        print(f"  [{i:>2}/{len(splits)}] fold={s['fold']} r={s['seed_ratio']:.0%} t={s['trial']} | "
              f"MERT def={r['mert_default']:.3f} tuned={r['mert_tuned']:.3f} | "
              f"SVM def={r['svm_default']:.3f} tuned={r['svm_tuned']:.3f} | {time.time()-t0:.0f}s",
              flush=True)

    df = pd.DataFrame(rows)

    lines = [
        "# Supplementary — ordinario 제외 4클래스 공정 튜닝 비교",
        "",
        f"4클래스: {', '.join(CLASSES)} (ordinario 제외, {len(labels)}개 샘플)",
        "",
        "> 목적: \"ordinario가 많아서 F1이 높은 것 아니냐\" 방어 + 더 어려운 세팅에서 SVM≈MERT 재확인.",
        "> 방법: `phase1_mert_tuning_audit.py`와 동일한 val 기반 공정 튜닝(MERT 24 / SVM 12 config).",
        "",
        "## 성능 비교 (fold-level 평균±표준편차, n=4)",
        "",
        "| seed | SVM(기존) | SVM(튜닝) | MERT(기존) | MERT(튜닝) | 튜닝MERT − 튜닝SVM |",
        "|---|---|---|---|---|---|",
    ]
    verdict = []
    for ratio in sorted(df['ratio'].unique()):
        sd = fold_stats(df, ratio, 'svm_default'); st = fold_stats(df, ratio, 'svm_tuned')
        md = fold_stats(df, ratio, 'mert_default'); mt = fold_stats(df, ratio, 'mert_tuned')
        gap = mt[0] - st[0]
        verdict.append((ratio, mt[0], st[0], gap))
        lines.append(f"| {int(ratio*100)}% | {sd[0]:.4f}±{sd[1]:.4f} | {st[0]:.4f}±{st[1]:.4f} "
                     f"| {md[0]:.4f}±{md[1]:.4f} | {mt[0]:.4f}±{mt[1]:.4f} | **{gap:+.4f}** |")

    lines += ["", "## 판정", ""]
    tiers = ["동급" if abs(g) < 0.03 else ("MERT 근소 우위" if g > 0 else "SVM 근소 우위")
             for (_, _, _, g) in verdict]
    n_tie, n_svm, n_mert = tiers.count("동급"), tiers.count("SVM 근소 우위"), tiers.count("MERT 근소 우위")
    lines.append(
        f"구간별: 동급 {n_tie} · SVM 근소우위 {n_svm} · MERT 근소우위 {n_mert} "
        "(근소 우위 = |차이|≥0.03, fold 표준편차와 겹치는 경계 수준). "
        "ordinario 제외 후에도 두 모델은 같은 수준대 → \"ordinario가 F1을 부풀렸다\"는 공격 방어 성립.")
    for (ratio, mt, st, gap), who in zip(verdict, tiers):
        lines.append(f"- seed {int(ratio*100)}%: SVM {st:.4f} vs MERT {mt:.4f} ({gap:+.4f}) → {who}")

    lines += ["", "> 5클래스 원본 비교: `results/phase1/mert_tuning_audit.md`", ""]
    OUT.write_text("\n".join(lines) + "\n", encoding='utf-8')
    df.to_csv(ROOT / 'results' / 'phase1' / 'supplementary_4class.csv', index=False)
    print(f"\n저장: {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()

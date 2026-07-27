"""
B3 — LoRA 부분 미세조정 (튜닝 예산 스펙트럼의 마지막 점)

목적: "frozen head로만 비교해서 MERT가 진 것 아니냐 / MERTech처럼 fine-tune하면
      이긴다"는 반박을 선제 차단. 튜닝 예산을 **끝까지**(백본 LoRA 미세조정) 올려도
      SVM과 동급이 유지되는지 확인한다. 성능 우위 주장이 아니라 함정 방어용.

스펙트럼: frozen-linear → frozen-mlp(=frozen tuned) → layer-weighted → **LoRA**
         앞 3개는 mert_tuning_audit / layer_probe에 이미 있음. 이 스크립트는 LoRA 점만 채운다.

설계:
  - MERT-v1-95M 백본에 LoRA 어댑터(q_proj·v_proj) + last_hidden_state time-mean-pool → linear head.
    백본 본체는 동결, LoRA 어댑터와 head만 학습.
  - 분할은 감사(phase1_mert_tuning_audit.py)와 동일한 3분할 컨벤션:
      labeled → seed_core(80%)/val_select(20%);  seed_core → tr(80%)/es_val(20%)
    tr로 학습, es_val로 early-stop, test로 평가. (단일 LoRA config — 예산은 이미 최대이므로
    config 그리드 없이 하나로 고정. val_select는 사용하지 않음.)
  - batch=1 forward + gradient accumulation → 패딩 평균 편향 없이 캐시 추출과 동일한 mean-pool.
  - split별 결과를 CSV에 append → 중단 시 재개 가능.

계획서 조건: 최소 seed 20%·50%만. 기본 trial 0 (fold당 1회, fold-level n=4).

실행:
    python src/phase1_lora.py --smoke                 # 1 split만, correctness/타이밍 측정
    python src/phase1_lora.py                          # ratio 0.2·0.5, trial 0 (8 split)
    python src/phase1_lora.py --ratios 0.5 --trials 0 1 2
    python src/phase1_lora.py --report-only            # 기존 CSV로 스펙트럼 표만 재생성

출력:
    results/phase1/lora_spectrum.md , lora_results.csv
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
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

torch.manual_seed(42)
np.random.seed(42)

CLASSES   = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
N_CLASSES = len(CLASSES)
MERT_MODEL_ID = 'm-a-p/MERT-v1-95M'
MERT_SR   = 24000

LAB_PATH    = ROOT / 'data' / 'mert_labels.npy'
META_PATH   = ROOT / 'data' / 'mert_meta.csv'
SPLITS_PATH = ROOT / 'data' / 'split_configs.json'
AUDIT_CSV   = ROOT / 'results' / 'phase1' / 'mert_tuning_audit.csv'
LP_CSV      = ROOT / 'results' / 'phase2' / 'layer_probe.csv'
OUT_CSV     = ROOT / 'results' / 'phase1' / 'lora_results.csv'
OUT_MD      = ROOT / 'results' / 'phase1' / 'lora_spectrum.md'

# 단일 LoRA config (예산 최대점 — 그리드 없음)
LORA_CFG = dict(r=8, alpha=16, dropout=0.1, target=['q_proj', 'v_proj'],
                lr=5e-4, wd=1e-4, epochs=25, patience=5, accum=16)

# 에폭 경계 냉각 대기(초). 팬리스 M-series 쓰로틀링 완화용. --rest 로 조절, 0이면 비활성.
# 클립 단위가 아니라 에폭 사이에서만 쉼. 기본 60초(사용자 지정 — 냉각 우선).
REST_SEC = 60.0


def get_device():
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device='mps') + 1
            return 'mps'
        except Exception:
            pass
    return 'cpu'


# ── 데이터 ───────────────────────────────────────────────────────────────────────
labels = np.load(LAB_PATH).astype(np.int64)
_meta  = pd.read_csv(META_PATH)
PATHS  = _meta['filepath'].tolist()   # mert_labels / split_configs와 동일 행 순서
assert len(PATHS) == len(labels), (len(PATHS), len(labels))

counts = np.bincount(labels, minlength=N_CLASSES).astype(float)


# ── 분할 (감사와 동일) ────────────────────────────────────────────────────────────
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


# ── MERT + LoRA ─────────────────────────────────────────────────────────────────
def load_backbone(device):
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    proc = Wav2Vec2FeatureExtractor.from_pretrained(MERT_MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MERT_MODEL_ID, trust_remote_code=True)
    return proc, model


def wrap_lora(model):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=LORA_CFG['r'], lora_alpha=LORA_CFG['alpha'],
                     lora_dropout=LORA_CFG['dropout'],
                     target_modules=LORA_CFG['target'], bias='none')
    return get_peft_model(model, cfg)


class Head(nn.Module):
    def __init__(self, in_dim=768, n_classes=N_CLASSES, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                 nn.Linear(in_dim, n_classes))

    def forward(self, x):
        return self.net(x)


def tokenize(proc, idxs, cache):
    """각 클립을 processor로 1회 토큰화해 CPU 텐서로 캐시 (에폭 간 재사용)."""
    import librosa
    for i in idxs:
        if i in cache:
            continue
        y, _ = librosa.load(PATHS[i], sr=MERT_SR, mono=True)
        iv = proc(y, sampling_rate=MERT_SR, return_tensors='pt', padding=True)['input_values']
        cache[i] = iv   # (1, T)


def pool_forward(backbone, input_values):
    """(1,T) → (768,) : last_hidden_state time-mean-pool (캐시 추출과 동일, batch=1 무패딩)."""
    out = backbone(input_values=input_values, output_hidden_states=False)
    return out.last_hidden_state[0].mean(dim=0)   # (768,)


def train_lora(backbone, head, tr_idx, es_idx, cache, device, cw, log):
    params = [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LORA_CFG['lr'], weight_decay=LORA_CFG['wd'])
    crit = nn.CrossEntropyLoss(weight=cw)
    accum = LORA_CFG['accum']

    best_f1, best_state, wait = -1.0, None, 0
    for ep in range(LORA_CFG['epochs']):
        if ep > 0 and REST_SEC > 0:      # 에폭 사이 냉각 대기 (첫 에폭·마지막 후엔 안 쉼)
            time.sleep(REST_SEC)
        backbone.train(); head.train()
        order = np.random.permutation(tr_idx)
        opt.zero_grad()
        for step, i in enumerate(order, 1):
            iv = cache[i].to(device)
            feat = pool_forward(backbone, iv)
            logit = head(feat.unsqueeze(0))
            loss = crit(logit, torch.tensor([labels[i]], device=device)) / accum
            loss.backward()
            if step % accum == 0 or step == len(order):
                opt.step(); opt.zero_grad()
        vf1 = eval_lora(backbone, head, es_idx, cache, device)
        if vf1 > best_f1 + 1e-6:
            best_f1, wait = vf1, 0
            best_state = ({k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()
                           if 'lora' in k.lower()},
                          {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        else:
            wait += 1
        log(f"    ep{ep:>2} es_f1={vf1:.3f} best={best_f1:.3f} wait={wait}")
        if wait >= LORA_CFG['patience']:
            break
    # best 복원
    if best_state is not None:
        bb, hd = best_state
        sd = backbone.state_dict(); sd.update({k: v.to(device) for k, v in bb.items()})
        backbone.load_state_dict(sd)
        head.load_state_dict({k: v.to(device) for k, v in hd.items()})
    return best_f1


@torch.no_grad()
def eval_lora(backbone, head, idxs, cache, device):
    backbone.eval(); head.eval()
    preds, ys = [], []
    for i in idxs:
        iv = cache[i].to(device)
        feat = pool_forward(backbone, iv)
        preds.append(int(head(feat.unsqueeze(0)).argmax(1).item()))
        ys.append(int(labels[i]))
    return f1_score(ys, preds, average='macro', zero_division=0)


def process_split(s, proc, base_model, device, log):
    seed_core, _val_sel = split_val(s['labeled_idx'], 0.2, seed=s['trial'] + 1000)
    tr_idx, es_idx = split_val(seed_core, 0.2, seed=s['trial'])
    test_idx = s['test_idx']

    cw = torch.tensor(counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=device)

    # 매 split마다 LoRA 어댑터 새로 (백본 가중은 공유 base에서 복제)
    import copy
    backbone = wrap_lora(copy.deepcopy(base_model)).to(device)
    head = Head().to(device)

    cache = {}
    t0 = time.time()
    tokenize(proc, list(tr_idx) + list(es_idx) + list(test_idx), cache)
    log(f"  tokenize {len(cache)} clips {time.time()-t0:.0f}s")

    best_es = train_lora(backbone, head, tr_idx, es_idx, cache, device, cw, log)
    lora_test = eval_lora(backbone, head, test_idx, cache, device)

    del backbone, head, cache
    if device == 'mps':
        torch.mps.empty_cache()

    return {'fold': s['fold'], 'ratio': s['seed_ratio'], 'trial': s['trial'],
            'n_tr': len(tr_idx), 'n_es': len(es_idx), 'n_test': len(test_idx),
            'lora_es_f1': best_es, 'lora_test': lora_test}


# ── 스펙트럼 리포트 ──────────────────────────────────────────────────────────────
def _fold_stat(df, ratio, col):
    sub = df[np.isclose(df['ratio'], ratio)]
    if sub.empty or col not in sub:
        return None, None
    fm = sub.groupby('fold')[col].mean()
    return fm.mean(), fm.std()


def build_report():
    if not OUT_CSV.exists():
        print("결과 CSV 없음 — 먼저 실행 필요"); return
    lora = pd.read_csv(OUT_CSV)
    audit = pd.read_csv(AUDIT_CSV) if AUDIT_CSV.exists() else None
    lp = pd.read_csv(LP_CSV) if LP_CSV.exists() else None

    ratios = sorted(lora['ratio'].unique())
    lines = [
        "# B3 — 튜닝 예산 스펙트럼 (frozen → LoRA)",
        "",
        "> 질문: 튜닝 예산을 백본 LoRA 미세조정까지 끝까지 올려도 SVM과 동급이 유지되는가?",
        "> 성능 우위 주장이 아니라 함정③(백본 언더튜닝) 방어의 마지막 점. 계획서: seed 20%·50%만.",
        f"> LoRA config: r={LORA_CFG['r']} α={LORA_CFG['alpha']} target={LORA_CFG['target']} "
        f"lr={LORA_CFG['lr']} epochs≤{LORA_CFG['epochs']}(patience {LORA_CFG['patience']}).",
        "",
        "## 스펙트럼 표 (macro F1, fold-level 평균±표준편차, n=4)",
        "",
        "| seed | SVM(튜닝) | frozen-linear | frozen-tuned | layer-weighted | **LoRA** | LoRA − SVM |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in ratios:
        def cell(m, sd):
            return f"{m:.4f}±{sd:.4f}" if m is not None else "—"
        svm_m, svm_s   = _fold_stat(audit, r, 'svm_tuned') if audit is not None else (None, None)
        fl_m, fl_s     = _fold_stat(audit, r, 'mert_default') if audit is not None else (None, None)
        ft_m, ft_s     = _fold_stat(audit, r, 'mert_tuned') if audit is not None else (None, None)
        lw_m, lw_s     = _fold_stat(lp, r, 'layer_weighted') if lp is not None else (None, None)
        lo_m, lo_s     = _fold_stat(lora, r, 'lora_test')
        gap = (lo_m - svm_m) if (lo_m is not None and svm_m is not None) else None
        gap_s = f"**{gap:+.4f}**" if gap is not None else "—"
        lines.append(f"| {int(r*100)}% | {cell(svm_m,svm_s)} | {cell(fl_m,fl_s)} | {cell(ft_m,ft_s)} "
                     f"| {cell(lw_m,lw_s)} | {cell(lo_m,lo_s)} | {gap_s} |")

    lines += ["", "## 판정", ""]
    verdict = []
    for r in ratios:
        lo_m, _ = _fold_stat(lora, r, 'lora_test')
        svm_m, svm_s = _fold_stat(audit, r, 'svm_tuned') if audit is not None else (None, None)
        if lo_m is None or svm_m is None:
            continue
        gap = lo_m - svm_m
        sig = "" if svm_s in (None, 0) or np.isnan(svm_s) else f" ({gap/svm_s:+.2f}σ)"
        verdict.append((r, lo_m, svm_m, gap, sig))
    if verdict and all(abs(g) <= (s if s else 1) for _, _, _, g, s in [(r,a,b,g, _fold_stat(audit,r,'svm_tuned')[1]) for r,a,b,g,_ in verdict]):
        lines.append("✅ **LoRA도 SVM과 fold-표준편차 이내 — 동급 결론이 최대 예산에서도 유지.**")
    for r, lo_m, svm_m, gap, sig in verdict:
        lines.append(f"- seed {int(r*100)}%: LoRA {lo_m:.4f} vs SVM(튜닝) {svm_m:.4f} ({gap:+.4f}{sig})")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print(f"저장: {OUT_MD}")
    print("\n".join(lines[-len(verdict)-3:]))


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    global REST_SEC
    ap = argparse.ArgumentParser()
    ap.add_argument('--ratios', type=float, nargs='*', default=[0.2, 0.5])
    ap.add_argument('--trials', type=int, nargs='*', default=[0])
    ap.add_argument('--smoke', action='store_true', help='1 split만 (첫 ratio·fold0·trial0)')
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--rest', type=float, default=REST_SEC,
                    help='에폭 사이 냉각 대기(초). 기본 10, 0이면 비활성. 냉각 목적이면 20~30 권장.')
    args = ap.parse_args()
    REST_SEC = args.rest

    if args.report_only:
        build_report(); return

    device = get_device()
    with open(SPLITS_PATH) as f:
        splits = json.load(f)['splits']
    sel = [s for s in splits
           if any(np.isclose(s['seed_ratio'], r) for r in args.ratios) and s['trial'] in args.trials]
    if args.smoke:
        r0 = args.ratios[0]
        sel = [s for s in splits if np.isclose(s['seed_ratio'], r0)
               and s['fold'] == 0 and s['trial'] == 0][:1]

    # 이미 완료된 split은 건너뜀 (재개)
    done = set()
    if OUT_CSV.exists() and not args.smoke:
        prev = pd.read_csv(OUT_CSV)
        done = {(int(x.fold), round(float(x.ratio), 3), int(x.trial)) for x in prev.itertuples()}
    sel = [s for s in sel if (s['fold'], round(s['seed_ratio'], 3), s['trial']) not in done]

    print(f"device={device} | 실행할 split={len(sel)} | LoRA r={LORA_CFG['r']} | rest={REST_SEC}s/epoch", flush=True)
    if not sel:
        print("남은 split 없음 → 리포트만 생성"); build_report(); return

    print("MERT 로드...", flush=True)
    proc, base_model = load_backbone(device)
    base_model.eval()
    # 모듈 이름 확인 (target 존재 여부)
    names = {n.split('.')[-1] for n, _ in base_model.named_modules()}
    miss = [t for t in LORA_CFG['target'] if t not in names]
    if miss:
        print(f"⚠ LoRA target 모듈 없음: {miss} — 사용 가능 후보:",
              sorted(n for n in names if 'proj' in n)[:10])

    t_all = time.time()
    for k, s in enumerate(sel, 1):
        log = lambda m: print(f"[{k}/{len(sel)}] f{s['fold']} r{s['seed_ratio']:.0%} t{s['trial']} {m}", flush=True)
        t0 = time.time()
        row = process_split(s, proc, base_model, device, log)
        row['sec'] = round(time.time() - t0, 1)
        hdr = not OUT_CSV.exists()
        pd.DataFrame([row]).to_csv(OUT_CSV, mode='a', header=hdr, index=False)
        print(f"[{k}/{len(sel)}] ✓ LoRA test={row['lora_test']:.4f} "
              f"(es={row['lora_es_f1']:.3f}) {row['sec']:.0f}s", flush=True)

    print(f"\n총 {time.time()-t_all:.0f}s", flush=True)
    if not args.smoke:
        build_report()


if __name__ == '__main__':
    main()

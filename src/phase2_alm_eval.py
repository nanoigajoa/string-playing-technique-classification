"""
Phase 2 — Audio-Language Model Hallucination Evaluation

Evaluates whether Qwen2-Audio-7B-Instruct can classify fine-grained string
instrument bowing techniques, and measures where it fails.

Usage:
    python src/phase2_alm_eval.py                   # all steps
    python src/phase2_alm_eval.py --step build-eval  # 2-2: build data/eval_set.json
    python src/phase2_alm_eval.py --step run         # 2-3: run inference
    python src/phase2_alm_eval.py --step analyze     # 2-4: metrics + report
"""

import argparse
import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT        = Path(__file__).parent.parent
DATA_ROOT   = ROOT / '_OrchideaSOL2020_release' / 'OrchideaSOL2020' / 'Strings'
RESULTS_DIR = ROOT / 'results' / 'phase2'
EVAL_SET    = ROOT / 'data' / 'eval_set.json'
EVAL_RESULT = ROOT / 'data' / 'eval_results.json'

CLASSES     = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']
INSTRUMENTS = ['Violin', 'Viola', 'Violoncello', 'Contrabass']
MODEL_ID    = 'Qwen/Qwen2-Audio-7B-Instruct'
N_PER_CLASS = 25

# Instrument code → natural name (from OrchideaSOL filename prefix)
INST_NAME = {
    'Vn': 'violin', 'Va': 'viola', 'Vc': 'violoncello', 'Cb': 'contrabass'
}

# Technique → natural language for questions
TECH_NAME = {
    'ordinario':         'ordinario (normal arco bowing)',
    'pizzicato_secco':   'pizzicato',
    'tremolo':           'tremolo',
    'sul_ponticello':    'sul ponticello',
    'col_legno_battuto': 'col legno battuto',
}

# Keywords for open-ended response parsing
KEYWORDS = {
    'ordinario':         ['arco', 'ordinary', 'normal', 'standard', 'regular', 'ordinario'],
    'pizzicato_secco':   ['pizzicato', 'pluck', 'pizz'],
    'tremolo':           ['tremolo', 'trembling', 'rapid bow'],
    'sul_ponticello':    ['ponticello', 'bridge', 'sul pont'],
    'col_legno_battuto': ['col legno', 'col_legno', 'wood', 'stick', 'battuto'],
}


# ── Device ────────────────────────────────────────────────────────────────────

def _get_device():
    if torch.backends.mps.is_available():
        try:
            t = torch.zeros(1).to('mps')
            _ = t + 1
            return 'mps'
        except Exception:
            pass
    return 'cpu'


# ── Model load ────────────────────────────────────────────────────────────────

def load_model():
    """
    Load Qwen2-Audio-7B-Instruct.
    Priority: float16 device_map=auto → quanto int8 → CPU float16
    Model is ~15GB (float16), fitting within 24GB M5 unified memory.
    """
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    device = _get_device()
    print(f'\nLoading {MODEL_ID}')
    print(f'Device: {device}  |  Weights: ~15GB float16')
    print('(First run downloads ~15GB — subsequent runs use cache)')

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Attempt 1: float16 with device_map="auto"
    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f'Loaded float16  ({n_params:.1f}B params)')
        return processor, model
    except Exception as e:
        if 'memory' not in str(e).lower() and 'oom' not in str(e).lower():
            raise
        print(f'OOM on float16: {e}')

    # Attempt 2: optimum-quanto int8 (~8GB)
    print('Trying optimum-quanto int8 quantization...')
    try:
        from optimum.quanto import quantize, qint8
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        )
        quantize(model, weights=qint8)
        model = model.to(device)
        print('Loaded quanto int8')
        return processor, model
    except ImportError:
        print('optimum-quanto not available')
    except Exception as e:
        print(f'quanto failed: {e}')

    # Attempt 3: CPU fallback
    print('Loading on CPU (slow inference, ~15 min per sample)')
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to('cpu')
    return processor, model


# ── Inference helpers ─────────────────────────────────────────────────────────

def _processor_call(processor, text, audio_array, sr):
    """Pass audio to Qwen2Audio processor (transformers 5.x uses audio=)."""
    return processor(
        text=text,
        audio=[audio_array],
        return_tensors='pt',
        padding=True,
        sampling_rate=sr,
    )


def _run_one(processor, model, filepath, question, max_new_tokens=64):
    """Run a single audio + text query, return the model's response string."""
    sr    = processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(str(filepath), sr=sr, mono=True)

    messages = [{"role": "user", "content": [
        {"type": "audio", "audio_url": f"file://{filepath}"},
        {"type": "text",  "text": question},
    ]}]
    text   = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    device = next(model.parameters()).device
    inputs = _processor_call(processor, text, audio, sr)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    new_ids  = gen_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    return response


# ── Step 2-2: Build eval set ──────────────────────────────────────────────────

def build_eval_set():
    """
    Sample 25 per class (125 total).
    Each sample gets:
      - 1 yes/no question (TP or FP trap, alternating)
      - 1 open-ended question
    """
    rng = np.random.default_rng(42)

    all_files = {cls: [] for cls in CLASSES}
    for inst in INSTRUMENTS:
        for cls in CLASSES:
            folder = DATA_ROOT / inst / cls
            if folder.exists():
                all_files[cls].extend(sorted(folder.glob('*.wav')))

    records = []
    for cls in CLASSES:
        files  = all_files[cls]
        chosen = rng.choice(len(files), min(N_PER_CLASS, len(files)), replace=False)

        for i, idx in enumerate(chosen):
            fp        = files[int(idx)]
            inst_code = fp.name.split('-')[0]
            inst_name = INST_NAME.get(inst_code, 'string instrument')

            # Yes/No: first half TP (correct technique), second half FP (wrong technique)
            if i < len(chosen) // 2:
                yn_tech     = cls
                expected_yn = 'yes'
                yn_type     = 'tp'
            else:
                wrong = [c for c in CLASSES if c != cls]
                yn_tech     = wrong[i % len(wrong)]
                expected_yn = 'no'
                yn_type     = 'fp'

            q_yn = (
                f"This is a recording of a {inst_name} playing a single note. "
                f"Does this audio contain {TECH_NAME[yn_tech]} bowing technique? "
                f"Answer with only 'yes' or 'no'."
            )
            q_open = (
                "This is a recording of a string instrument playing a single note. "
                "Describe the bowing technique used in this audio. "
                "Be specific about whether you hear arco, pizzicato, tremolo, "
                "sul ponticello, or col legno battuto."
            )

            records.append({
                'id':           len(records),
                'file':         fp.name,
                'filepath':     str(fp),
                'true_label':   cls,
                'instrument':   inst_name,
                'question_yn':  q_yn,
                'yn_type':      yn_type,
                'yn_technique': yn_tech,
                'expected_yn':  expected_yn,
                'question_open': q_open,
            })

    (ROOT / 'data').mkdir(exist_ok=True)
    with open(EVAL_SET, 'w') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    df_r = pd.DataFrame(records)
    print(f'\nEval set saved: {EVAL_SET}  ({len(records)} samples)')
    print(df_r.groupby(['true_label','yn_type']).size().to_string())
    return records


# ── Step 2-3: Run inference ───────────────────────────────────────────────────

def _parse_yn(response: str):
    resp  = response.lower().strip()
    first = re.split(r'[\s,.]', resp)[0] if resp else ''
    if first == 'yes':
        return 'yes'
    if first == 'no':
        return 'no'
    yes_n = len(re.findall(r'\byes\b', resp))
    no_n  = len(re.findall(r'\bno\b',  resp))
    if yes_n > no_n:
        return 'yes'
    if no_n > yes_n:
        return 'no'
    return 'ambiguous'


def _parse_open(response: str, true_cls: str):
    resp          = response.lower()
    correct_hit   = any(kw in resp for kw in KEYWORDS[true_cls])
    wrong_hit     = any(
        any(kw in resp for kw in KEYWORDS[c])
        for c in CLASSES if c != true_cls
    )
    return correct_hit, wrong_hit


def run_inference(processor, model):
    """Run all 125 samples × 2 question types = 250 inference calls."""
    with open(EVAL_SET) as f:
        records = json.load(f)

    results = []
    for rec in tqdm(records, desc='Qwen2-Audio inference'):
        row = dict(rec)

        # Yes/No
        try:
            resp_yn   = _run_one(processor, model, rec['filepath'], rec['question_yn'], 32)
            parsed_yn = _parse_yn(resp_yn)
            yn_ok     = (parsed_yn == rec['expected_yn'])
        except Exception as e:
            resp_yn, parsed_yn, yn_ok = f'ERROR: {e}', 'error', False

        # Open-ended
        try:
            resp_open             = _run_one(processor, model, rec['filepath'], rec['question_open'], 128)
            open_correct, open_wrong = _parse_open(resp_open, rec['true_label'])
        except Exception as e:
            resp_open, open_correct, open_wrong = f'ERROR: {e}', False, False

        row.update({
            'response_yn':   resp_yn,
            'parsed_yn':     parsed_yn,
            'yn_correct':    yn_ok,
            'response_open': resp_open,
            'open_correct':  open_correct,
            'open_wrong':    open_wrong,
        })
        results.append(row)

        if len(results) % 10 == 0:
            # Checkpoint
            with open(EVAL_RESULT, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            print(f'  checkpoint: {len(results)}/{len(records)} done')

    with open(EVAL_RESULT, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved: {EVAL_RESULT}')
    return results


# ── Step 2-4: Analyze ─────────────────────────────────────────────────────────

def analyze_results():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_RESULT) as f:
        results = json.load(f)
    df = pd.DataFrame(results)

    # Per-class stats
    stats = []
    for cls in CLASSES:
        sub     = df[df['true_label'] == cls]
        n       = len(sub)
        yn_miss = (~sub['yn_correct'].astype(bool)).mean()
        op_miss = (~sub['open_correct'].astype(bool)).mean()
        amb     = (sub['parsed_yn'] == 'ambiguous').mean()
        stats.append({
            'class': cls, 'n': n,
            'yn_miss_rate':   round(float(yn_miss), 3),
            'open_miss_rate': round(float(op_miss), 3),
            'ambiguous_rate': round(float(amb), 3),
        })
    df_stats = pd.DataFrame(stats)

    print('\n=== Per-class Misclassification Rate ===')
    print(df_stats.to_string(index=False))
    print(f'\nOverall yes/no accuracy  : {df["yn_correct"].astype(bool).mean():.3f}')
    print(f'Overall open-ended accuracy: {df["open_correct"].astype(bool).mean():.3f}')

    # Phase 1 worst classes
    p1_csv = ROOT / 'results' / 'phase1' / 'all_results.csv'
    phase1_worst = []
    if p1_csv.exists():
        df_p1 = pd.read_csv(p1_csv)
        cols  = [f'base_f1_{c}' for c in CLASSES]
        p1_f1 = df_p1[cols].mean()
        p1_f1.index = CLASSES
        phase1_worst = p1_f1.sort_values().head(2).index.tolist()
        print(f'\nPhase 1 worst classes : {phase1_worst}')

    phase2_worst = df_stats.sort_values('open_miss_rate', ascending=False).head(2)['class'].tolist()
    intersection = sorted(set(phase1_worst) & set(phase2_worst))
    print(f'Phase 2 worst classes : {phase2_worst}')
    print(f'Intersection (Phase 3 targets): {intersection}')

    # Heatmap
    heat = df_stats[['yn_miss_rate', 'open_miss_rate']].values.astype(float)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(heat, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Yes/No', 'Open-ended'], fontsize=11)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontsize=10)
    ax.set_title('Misclassification Rate by Technique', fontsize=12)
    plt.colorbar(im, ax=ax, label='Misclassification rate')
    for i in range(len(CLASSES)):
        for j in range(2):
            ax.text(j, i, f'{heat[i,j]:.2f}', ha='center', va='center',
                    color='white' if heat[i,j] > 0.5 else 'black', fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'misclassification_heatmap.png', dpi=150)
    plt.close()
    print('saved: misclassification_heatmap.png')

    # Report
    yn_acc   = df['yn_correct'].astype(bool).mean()
    open_acc = df['open_correct'].astype(bool).mean()
    amb_cnt  = (df['parsed_yn'] == 'ambiguous').sum()

    lines = [
        '# Phase 2 Results — Qwen2-Audio Hallucination Evaluation\n\n',
        f'**Model**: {MODEL_ID}\n\n',
        f'**Eval set**: {len(df)} samples × 2 question types = {len(df)*2} queries\n\n',
        '## Overall Accuracy\n\n',
        '| Question type | Accuracy |\n|---|---|\n',
        f'| Yes/No        | {yn_acc:.3f} |\n',
        f'| Open-ended    | {open_acc:.3f} |\n\n',
        '## Per-class Misclassification Rate\n\n',
        df_stats.to_markdown(index=False), '\n\n',
        '## Phase 1 / Phase 2 Intersection\n\n',
        f'Phase 1 worst classes : {phase1_worst}\n\n' if phase1_worst else '',
        f'Phase 2 worst classes : {phase2_worst}\n\n',
        f'Intersection (Phase 3 case study targets): **{intersection}**\n\n',
        '## Notes\n\n',
        f'- Ambiguous yes/no responses: {amb_cnt} / {len(df)}\n',
        f'- Open-ended wrong technique mentioned alongside correct: {df["open_wrong"].astype(bool).sum()} / {len(df)}\n',
    ]
    (RESULTS_DIR / 'report.md').write_text(''.join(lines), encoding='utf-8')
    print('saved: report.md')

    return df_stats


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Phase 2 — ALM Hallucination Evaluation')
    parser.add_argument(
        '--step',
        choices=['build-eval', 'run', 'analyze', 'all'],
        default='all',
        help='Step to run (default: all)',
    )
    args = parser.parse_args()

    if args.step in ('build-eval', 'all'):
        build_eval_set()

    if args.step in ('run', 'all'):
        processor, model = load_model()
        run_inference(processor, model)

    if args.step in ('analyze', 'all'):
        analyze_results()


if __name__ == '__main__':
    main()

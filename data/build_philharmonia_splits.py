"""
Phase 4 — Philharmonia Pool A / Pool B 분할

Philharmonia Strings 폴더에서 우리 5클래스에 해당하는 파일을 수집하고
stratified 70/30 split으로 Pool A(self-training용)와 Pool B(holdout 평가용)를 생성한다.

실행:
    python data/build_philharmonia_splits.py
    python data/build_philharmonia_splits.py --dry-run   # 파일 저장 없이 통계만 출력

출력:
    data/philharmonia_pool_a.json
    data/philharmonia_pool_b.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent

# ── 상수 ──────────────────────────────────────────────────────────────────────

STRINGS_DIR = ROOT / 'data' / 'philharmonia' / 'Strings'

INSTRUMENT_DIRS = ['violin', 'viola', 'cello', 'double bass']

# Philharmonia 파일명 마지막 토큰 → 우리 클래스 매핑
TECHNIQUE_MAP = {
    'arco-normal':            'ordinario',
    'pizz-normal':            'pizzicato_secco',
    'arco-col-legno-battuto': 'col_legno_battuto',
    'arco-tremolo':           'tremolo',
    'arco-sul-ponticello':    'sul_ponticello',
}

CLASSES = ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']

ORDINARIO_CAP = 300   # ordinario 과잉 샘플 억제 (3,072개 → 300개)
POOL_B_RATIO  = 0.30  # 30% holdout
RANDOM_SEED   = 42

POOL_A_PATH = ROOT / 'data' / 'philharmonia_pool_a.json'
POOL_B_PATH = ROOT / 'data' / 'philharmonia_pool_b.json'


# ── 파싱 ──────────────────────────────────────────────────────────────────────

def parse_technique(stem: str) -> str | None:
    """파일명 stem의 마지막 '_' 토큰에서 기법 추출. 매핑 없으면 None."""
    technique_token = stem.split('_')[-1]
    return TECHNIQUE_MAP.get(technique_token)


def collect_samples() -> list[dict]:
    """
    STRINGS_DIR 아래 모든 악기 폴더를 순회해 매핑 가능한 샘플을 수집한다.
    ordinario는 ORDINARIO_CAP개로 제한 (무작위 선택, seed 고정).

    반환: [{'path': str(ROOT 기준 상대 경로), 'label': str, 'instrument': str}, ...]
    """
    samples = []

    for inst_name in INSTRUMENT_DIRS:
        inst_dir = STRINGS_DIR / inst_name
        if not inst_dir.exists():
            print(f'  [경고] 폴더 없음: {inst_dir}')
            continue

        for mp3 in sorted(inst_dir.glob('*.mp3')):
            label = parse_technique(mp3.stem)
            if label is None:
                continue
            samples.append({
                'path':       str(mp3.relative_to(ROOT)),
                'label':      label,
                'instrument': inst_name,
            })

    # ordinario 상한 적용 (stratified split 전에 적용)
    ordinario = [s for s in samples if s['label'] == 'ordinario']
    others    = [s for s in samples if s['label'] != 'ordinario']

    if len(ordinario) > ORDINARIO_CAP:
        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.choice(len(ordinario), size=ORDINARIO_CAP, replace=False)
        ordinario = [ordinario[i] for i in sorted(indices)]

    return others + ordinario


# ── 분할 ──────────────────────────────────────────────────────────────────────

def stratified_split(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    클래스 비율 유지 stratified 70/30 split.
    반환: (pool_a, pool_b)
    """
    labels = [s['label'] for s in samples]

    idx_a, idx_b = train_test_split(
        range(len(samples)),
        test_size=POOL_B_RATIO,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    pool_a = [samples[i] for i in sorted(idx_a)]
    pool_b = [samples[i] for i in sorted(idx_b)]
    return pool_a, pool_b


# ── 리포트 ────────────────────────────────────────────────────────────────────

def print_stats(name: str, pool: list[dict]) -> None:
    print(f'\n{name}  (총 {len(pool)}개)')
    for cls in CLASSES:
        n = sum(1 for s in pool if s['label'] == cls)
        print(f'  {cls:<22} {n:>4}')


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    print('Philharmonia 샘플 수집 중...')
    samples = collect_samples()
    print(f'  매핑된 샘플 수: {len(samples)}개')

    # 전체 분포 확인
    print('\n전체 분포 (ordinario 상한 적용 후):')
    for cls in CLASSES:
        n = sum(1 for s in samples if s['label'] == cls)
        print(f'  {cls:<22} {n:>4}')

    # 클래스별 최솟값 확인
    counts = {cls: sum(1 for s in samples if s['label'] == cls) for cls in CLASSES}
    min_cls = min(counts, key=counts.get)
    min_n   = counts[min_cls]
    pool_b_min = int(min_n * POOL_B_RATIO)
    print(f'\n최솟값 클래스: {min_cls} ({min_n}개) → Pool B 최소 {pool_b_min}개 확보')

    # 분할
    pool_a, pool_b = stratified_split(samples)
    print_stats('Pool A (self-training용, 라벨 숨김)', pool_a)
    print_stats('Pool B (holdout 평가용)', pool_b)

    if dry_run:
        print('\n[dry-run] 파일 저장 생략.')
        return

    # 저장
    with open(POOL_A_PATH, 'w', encoding='utf-8') as f:
        json.dump(pool_a, f, ensure_ascii=False, indent=2)

    with open(POOL_B_PATH, 'w', encoding='utf-8') as f:
        json.dump(pool_b, f, ensure_ascii=False, indent=2)

    print(f'\n저장 완료:')
    print(f'  {POOL_A_PATH}')
    print(f'  {POOL_B_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='파일 저장 없이 통계만 출력')
    args = parser.parse_args()
    main(dry_run=args.dry_run)

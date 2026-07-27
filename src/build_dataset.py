"""
특징 추출 실행 스크립트.
OrchideaSOL 현악기 오디오 전체를 읽어 특징 벡터를 추출하고 data/features_full.csv를 재현한다.

이 스크립트의 기본 실행 결과는 저장소에 커밋된 `data/features_full.csv`와 **행·열이 완전히 일치**한다:
  - 행: 다운샘플 없이 전체 3,402개. 순서 = INSTRUMENTS × CLASSES × sorted(glob) (scan 순서).
    → 13층 임베딩 캐시(mert_embeddings_13layer.npy 등)와 positional 정렬이 유지된다.
  - 열: FEATURE_NAMES(245) + [label_id, instrument]  (특징 먼저, 메타는 뒤 2열).

grouping 키(recording_id)는 features_full.csv에 없어 그동안 positional 재계산에 의존했다.
이를 없애기 위해 같은 행 순서의 사이드카 `data/features_full_groups.csv`
(recording_id, instrument, label_id)를 함께 저장한다.

실행:
    python src/build_dataset.py            # features_full.csv 재현 (전체)
    python src/build_dataset.py --sample 200   # 클래스당 200개로 다운샘플(빠른 실험용, 다른 파일명)
"""

import argparse

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from features import extract_features, make_feature_names, parse_recording_id, FEATURE_NAMES

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT    = PROJECT_ROOT / '_OrchideaSOL2020_release' / 'OrchideaSOL2020' / 'Strings'
OUTPUT_PATH  = PROJECT_ROOT / 'data' / 'features_full.csv'          # 커밋된 canonical 파일과 동일 스키마
GROUPS_PATH  = PROJECT_ROOT / 'data' / 'features_full_groups.csv'   # 사이드카: recording_id 등 grouping 키

# ── 클래스 / 악기 설정 ───────────────────────────────────────────────────────
CLASSES = [
    'ordinario',
    'pizzicato_secco',
    'tremolo',
    'sul_ponticello',
    'col_legno_battuto',
]
INSTRUMENTS = ['Violin', 'Viola', 'Violoncello', 'Contrabass']
LABEL2ID    = {c: i for i, c in enumerate(CLASSES)}

# 클래스당 최대 샘플 수 (None = 전체 사용 → features_full.csv 재현). --sample로 덮어씀.
MAX_SAMPLES_PER_CLASS = None


def scan_files() -> pd.DataFrame:
    """DATA_ROOT를 순회하며 대상 파일 목록을 DataFrame으로 반환"""
    rows = []
    for instrument in INSTRUMENTS:
        for technique in CLASSES:
            folder = DATA_ROOT / instrument / technique
            if not folder.exists():
                print(f'  [없음] {instrument}/{technique}')
                continue
            for f in sorted(folder.glob('*.wav')):
                rows.append({
                    'filepath':     str(f),
                    'label':        technique,
                    'label_id':     LABEL2ID[technique],
                    'instrument':   instrument,
                    'recording_id': parse_recording_id(f.name),
                })
    return pd.DataFrame(rows)


def downsample(df: pd.DataFrame, max_per_class: int) -> pd.DataFrame:
    return (
        df.groupby('label', group_keys=False)
          .apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42))
          .reset_index(drop=True)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=MAX_SAMPLES_PER_CLASS,
                    help='클래스당 최대 샘플 수 (기본: 전체). 지정 시 features_sampled.csv로 저장')
    args = ap.parse_args()
    max_per_class = args.sample

    print(f'데이터 경로: {DATA_ROOT}')
    assert DATA_ROOT.exists(), f'데이터 폴더가 없습니다: {DATA_ROOT}'

    # 파일 스캔
    df_all = scan_files()
    print(f'\n전체 파일 수: {len(df_all)}')
    print(df_all.groupby('label')['filepath'].count().to_string())

    # 다운샘플링 — 지정 시에만. 전체 실행이 canonical(features_full.csv) 재현 경로.
    if max_per_class:
        df = downsample(df_all, max_per_class)
        out_path = PROJECT_ROOT / 'data' / 'features_sampled.csv'
        print(f'\n다운샘플({max_per_class}/클래스) 후: {len(df)} 파일 → {out_path.name}')
    else:
        df = df_all
        out_path = OUTPUT_PATH
        print(f'\n전체 사용: {len(df)} 파일 → {out_path.name}')

    # 특징 추출
    records, errors = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='특징 추출'):
        try:
            vec    = extract_features(row['filepath'])
            record = dict(zip(FEATURE_NAMES, vec))
            record.update({
                'label':        row['label'],
                'label_id':     row['label_id'],
                'instrument':   row['instrument'],
                'recording_id': row['recording_id'],
            })
            records.append(record)
        except Exception as e:
            errors.append((row['filepath'], str(e)))

    print(f'\n성공: {len(records)} / 실패: {len(errors)}')
    for path, msg in errors[:5]:
        print(f'  오류: {Path(path).name} — {msg}')

    # 저장 — canonical 스키마: 특징(245) 다음에 [label_id, instrument]
    df_rec = pd.DataFrame(records)
    df_feat = df_rec[FEATURE_NAMES + ['label_id', 'instrument']]

    # 기본 검증
    nan_count = df_feat[FEATURE_NAMES].isna().sum().sum()
    if nan_count > 0:
        print(f'경고: NaN {nan_count}개 발견')

    out_path.parent.mkdir(exist_ok=True)
    df_feat.to_csv(out_path, index=False)
    print(f'\n저장 완료: {out_path}')
    print(f'shape: {df_feat.shape}')

    # grouping 사이드카 (canonical 전체 실행일 때만): recording_id를 같은 행 순서로 보존
    if out_path == OUTPUT_PATH:
        df_rec[['recording_id', 'instrument', 'label_id']].to_csv(GROUPS_PATH, index=False)
        print(f'grouping 사이드카 저장: {GROUPS_PATH}')


if __name__ == '__main__':
    main()

"""Inventory existing audio and label coverage without training or changing splits.

Only file-byte duplicates are detected; this is not an acoustic fingerprint audit.
Filename-derived pitch/dynamic keys are never interpreted as player/session IDs.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['ordinario', 'pizzicato_secco', 'tremolo',
           'sul_ponticello', 'col_legno_battuto']
PHIL_MAP = {
    'arco-normal': 'ordinario', 'pizz-normal': 'pizzicato_secco',
    'arco-tremolo': 'tremolo', 'arco-sul-ponticello': 'sul_ponticello',
    'arco-col-legno-battuto': 'col_legno_battuto',
}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def coverage(rows):
    return {instrument: dict(Counter(row['label'] for row in rows
                                    if row['instrument'] == instrument))
            for instrument in sorted({row['instrument'] for row in rows})}


def inventory(out_dir):
    orch = read_csv(ROOT / 'data/mert_meta.csv')
    phil_meta = read_csv(ROOT / 'data/philharmonia_mert_meta.csv')
    files = sorted((ROOT / 'data/philharmonia/Strings').rglob('*.mp3'))
    phil = [{'path': str(path.relative_to(ROOT)), 'instrument': path.parent.name,
             'label': PHIL_MAP[path.stem.split('_')[-1]]}
            for path in files if path.parent.name in
            ['violin', 'viola', 'cello', 'double bass']
            and path.stem.split('_')[-1] in PHIL_MAP]
    manifest = []
    for dataset, rows, key in [('OrchideaSOL', orch, 'filepath'),
                               ('Philharmonia', phil, 'path')]:
        for row in rows:
            path = Path(row[key])
            if not path.is_absolute():
                path = ROOT / path
            exists = path.is_file()
            try:
                relative = str(path.relative_to(ROOT))
            except ValueError:
                relative = str(path)
            manifest.append({'dataset': dataset, 'path': relative,
                             'instrument': row['instrument'], 'label': row['label'],
                             'exists': exists, 'bytes': path.stat().st_size if exists else 0,
                             'sha256': sha256(path) if exists else '',
                             'performer_id': 'unknown', 'session_id': 'unknown'})
    duplicate_groups = defaultdict(list)
    for row in manifest:
        if row['sha256']:
            duplicate_groups[row['sha256']].append(row['path'])
    # These keys indicate only matching instrument/pitch/dynamic metadata.
    proxy = defaultdict(set)
    for row in orch:
        if Path(row['filepath']).is_file():
            proxy[(row['instrument'], row['recording_id'])].add(row['label'])
    pair_counts = []
    for inst in sorted({row['instrument'] for row in orch}):
        for target in ['tremolo', 'sul_ponticello']:
            count = sum({'ordinario', target}.issubset(labels)
                        for (instrument, _), labels in proxy.items() if instrument == inst)
            pair_counts.append({'instrument': inst, 'contrast': 'ordinario/' + target,
                                'matching_proxy_keys': count})
    erhu_paths = sorted((ROOT / 'data/erhu').rglob('*.wav'))
    erhu_hashes = defaultdict(list)
    for path in erhu_paths:
        erhu_hashes[sha256(path)].append(str(path.relative_to(ROOT)))
    pools = {}
    for name in ['a', 'b']:
        path = ROOT / f'data/philharmonia_pool_{name}.json'
        rows = json.loads(path.read_text())
        pools[name] = {'rows': len(rows), 'coverage': coverage(rows),
                       'missing_paths': [r['path'] for r in rows
                                         if not (ROOT / r['path']).is_file()]}
    result = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256(Path(__file__)),
        'metadata_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                            [ROOT / 'data/mert_meta.csv',
                             ROOT / 'data/philharmonia_mert_meta.csv']},
        'scope': 'Local file existence, byte hashes, filename labels; no decoding or training.',
        'orchidea': {'metadata_rows': len(orch), 'coverage': coverage(orch),
                    'proxy_key_count': len(proxy),
                    'proxy_keys_with_multiple_labels': sum(len(v) > 1 for v in proxy.values()),
                    'metadata_matched_contrasts': pair_counts},
        'philharmonia': {'all_local_strings_mp3': len(files), 'mapped_rows': len(phil),
                        'coverage': coverage(phil), 'embedded_rows': len(phil_meta),
                        'pools': pools},
        'missing_mapped_files': [r['path'] for r in manifest if not r['exists']],
        'byte_duplicate_groups_mapped': [v for v in duplicate_groups.values() if len(v) > 1],
        'erhu': {'wav_paths': len(erhu_paths), 'unique_file_hashes': len(erhu_hashes),
                 'directory_counts': dict(Counter(str(p.parent.relative_to(ROOT / 'data/erhu'))
                                                  for p in erhu_paths)),
                 'note': '4/7/11-class views must not be counted as independent observations.'},
        'quartet_audio_files': sum(p.suffix.lower() in ['.wav', '.mp3', '.flac', '.aif', '.aiff']
                                   for p in (ROOT / 'data/quartet').rglob('*') if p.is_file()),
        'limitations': ['No performer or session ID in inspected OrchideaSOL/Philharmonia metadata.',
                        'Byte hashes do not detect transcoded or acoustically similar recordings.',
                        'Filename label mappings have not been validated by expert listening.',
                        'No new independent confirmation cohort established.'],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'local_inventory.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (out_dir / 'local_audio_manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (out_dir / 'class_coverage.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'instrument'] + CLASSES)
        writer.writeheader()
        for name, rows in [('OrchideaSOL', orch), ('Philharmonia', phil)]:
            for inst, counts in coverage(rows).items():
                writer.writerow({'dataset': name, 'instrument': inst,
                                 **{c: counts.get(c, 0) for c in CLASSES}})
    print(json.dumps({
        'orchidea_rows': len(orch), 'philharmonia_mapped_rows': len(phil),
        'missing_mapped_files': len(result['missing_mapped_files']),
        'mapped_duplicate_groups': len(result['byte_duplicate_groups_mapped']),
        'erhu_wav_paths': len(erhu_paths), 'erhu_unique_byte_hashes': len(erhu_hashes),
        'output': str(out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/phase5/data_feasibility')
    args = parser.parse_args()
    inventory(args.out)

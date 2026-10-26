"""Check Korean manuscript tables against canonical CSVs and export artifacts."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import statistics
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/manuscripts/v2"
BASE = ROOT / "results/phase1/canonical"
SOURCE = OUT / "주법분류_논문초안_국문_v2.md"


def read(relative):
    with (BASE / relative).open() as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = SOURCE.read_text()
    groups = re.findall(r"(?:^\|.*\|\n)+", source, re.M)
    assert len(groups) == 7
    tables = [[list(map(str.strip, line.strip().strip("|").split("|"))) for line in table.splitlines()[2:]] for table in groups]
    summary = read("mert_svm/canonical_summary.csv")
    intervals = read("verification/bootstrap_sensitivity.csv")
    for row, expected in zip(tables[3], summary, strict=True):
        ratio = float(expected["seed_ratio"])
        interval = next(r for r in intervals if float(r["seed_ratio"]) == ratio and r["unit"] == "instrument_type" and r["measure"] == "mert_minus_svm")
        values = [f"{ratio:.0%}", f"{float(expected['mert_macro_f1_fold_mean']):.4f}", f"{float(expected['svm_macro_f1_fold_mean']):.4f}",
                  f"{float(expected['mert_minus_svm']):+.4f}", f"[{float(interval['ci_low']):+.4f}, {float(interval['ci_high']):+.4f}]"]
        values[4] = values[4].replace("[+", "[")
        assert [v.replace('−','-') for v in row] == values, (row, values)
    st = read("self_training/canonical_summary.csv")
    for row in tables[4]:
        ratio = float(row[0].strip('%')) / 100
        values = {r['condition']: float(r['macro_f1_fold_mean']) for r in st if float(r['ratio']) == ratio}
        expected = [row[0], f"{values['supervised']:.4f}", f"{values['pseudo']:.4f}", f"{values['pseudo']-values['supervised']:+.4f}", f"{values['oracle']:.4f}"]
        assert [v.replace('−','-') for v in row] == expected
    cnn = read("cnn/canonical_results.csv")
    labels = dict(zip(["바이올린", "비올라", "첼로", "콘트라베이스"], ["Violin", "Viola", "Violoncello", "Contrabass"]))
    for row in tables[5]:
        value = statistics.mean(float(r['macro_f1']) for r in cnn) if row[0] == "네 악기종 평균" else float(next(r for r in cnn if r['held_out_instrument'] == labels[row[0]])['macro_f1'])
        assert row[1] == f"{value:.4f}"
    manifests = [json.loads(p.read_text()) for p in (BASE / 'mert_svm/manifests').glob('*.json')]
    for row in tables[6]:
        ratio = float(row[0].strip('%')) / 100
        subset = [m for m in manifests if m['seed_ratio'] == ratio]
        for cell, key in zip(row[1:], ['train','early_stop','val_select'], strict=True):
            values = [m['partition_sizes'][key] for m in subset]
            assert cell == f"{min(values)}–{max(values)}"
    metadata = read("release/canonical-reproduction/data/release_meta.csv")
    counts = {label: sum(r['label'].replace('_',' ') == label for r in metadata) for label in [row[0] for row in tables[0][:-1]]}
    for row in tables[0][:-1]:
        assert int(row[2].replace(',','')) == counts[row[0]]
    assert int(tables[0][-1][2].replace(',','')) == len(metadata) == 3402
    assert sum(int(row[2]) for row in tables[1][:-1]) == int(tables[1][-1][2]) == 245
    figure_manifest = json.loads((OUT/'figures/manifest.json').read_text())
    for rel,digest in figure_manifest['inputs'].items():
        assert sha(ROOT/rel)==digest,rel
    for rel,digest in figure_manifest['outputs'].items():
        assert sha(OUT/rel)==digest,rel
    figures = re.findall(r'!\[(.*?)\]\((.*?)\)',source)
    assert len(figures)==6
    for i,(caption,path) in enumerate(figures,1):
        assert caption.startswith(f'그림 {i}.') and (OUT/path).is_file()
    assert source.count('Table: 표 ')==7
    references = source.split('## 참고문헌\n\n')[1].split('## 부록 A.')[0].strip().split('\n\n')
    assert len(references) == 11
    # Check the class-level changes discussed in the content review, not only tables.
    class_rows = read('self_training/per_class_f1.csv')
    class_deltas = {}
    for label in ['ordinario', 'pizzicato_secco', 'tremolo', 'sul_ponticello', 'col_legno_battuto']:
        values = {r['condition']: float(r['f1']) for r in class_rows
                  if float(r['ratio']) == 0.05 and r['class'] == label}
        delta = values['pseudo'] - values['supervised']
        assert delta < 0
        assert f"{label.replace('_', ' ')} {delta:+.4f}".replace('-', '−') in source
        class_deltas[label] = delta
    assert min(class_deltas, key=class_deltas.get) == 'pizzicato_secco'
    docx=SOURCE.with_suffix('.docx')
    with zipfile.ZipFile(docx) as archive:
        assert archive.testzip() is None
        xml=ET.fromstring(archive.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        assert len(xml.findall('.//w:tbl',ns))==7
        assert len(xml.findall('.//w:drawing',ns))==6
        docx_text = ''.join(node.text or '' for node in xml.findall('.//w:t', ns))
    pdf=OUT/'review_ko_v2.pdf'
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
    pdf_text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
    for n in range(1,7):
        assert f'그림 {n}.' in pdf_text
    assert '부록 A.' in pdf_text and '부록 B.' in pdf_text and '참고문헌' in pdf_text
    assert '\ufffd' not in pdf_text
    key_values=[cell for table in [tables[3],tables[4],tables[5]] for row in table for cell in row if re.fullmatch(r'0\.\d{4}',cell)]
    assert all(value in pdf_text for value in key_values)
    # These content markers detect stale exports after manuscript prose changes.
    markers = ['다섯 클래스 모두 음수였다.', '가장 큰 F1 하락은 pizzicato secco에서 관찰됐다.',
               '프레임별 순서를 명시적으로 전달하지 않는다.',
               'MERT 사전학습 자료에 해당 음원이나 관련 녹음이 포함됐는지는 이번 감사에서 확인하지 않았다.',
               'MERTech']
    html_text = html.unescape(re.sub(r'<[^>]+>', '', (OUT/'review_ko_v2.html').read_text()))
    for text in [source, docx_text, html_text, pdf_text]:
        normalized = re.sub(r'\s+', '', text)
        assert all(re.sub(r'\s+', '', marker) in normalized for marker in markers)
    report={"passed":True,"mode":"manuscript_table_and_artifact_consistency","numeric_tables_checked":6,
            "training_configuration_table":"manually checked against source code", "figures":6,"tables":7,"references":len(references),
            "self_training_class_deltas_checked":class_deltas,"content_markers_checked_across_exports":len(markers),
            "pdf_pages":pages,"manuscript_characters":len(source),"whitespace_tokens":len(source.split()),
            "figure_inputs_and_outputs_hash_verified":True,"docx_zip_and_structure":"passed",
            "pdf_korean_and_caption_text":"passed","new_model_training":False,
            "manual_layout_review":"see REVIEW.md; not established by this script"}
    (OUT/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    artifact_paths=[SOURCE,docx,pdf,OUT/'review_ko_v2.html',OUT/'review_ko.css',OUT/'figures/manifest.json',OUT/'VERIFICATION.json']
    for name in ['phase1_paper_v2_figures.py','phase1_paper_v2_export.py','phase1_paper_v2_verify.py']:
        artifact_paths.append(ROOT/'src'/name)
    (OUT/'artifact_manifest.json').write_text(json.dumps({"root":"repository root","files":{str(p.relative_to(ROOT)):sha(p) for p in artifact_paths}},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()

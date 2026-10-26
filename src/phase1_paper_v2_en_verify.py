"""Verify the English translation, numeric parity, figures, and PDF exports."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import html
import json
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/manuscripts/v2"
KO = OUT / "주법분류_논문초안_국문_v2.md"
EN = OUT / "주법분류_논문초안_영문_v2.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return re.sub(r"\s+", "", text.replace("−", "-"))


def tables(text):
    blocks = re.findall(r"(?:^\|.*\|\n)+", text, re.M)
    return [[list(map(str.strip, row.strip().strip("|").split("|")))
             for row in block.splitlines()[2:]] for block in blocks]


def numbers(text):
    return re.findall(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", text.replace("−", "-"))


def main():
    korean, english = KO.read_text(), EN.read_text()
    assert not re.search(r"[가-힣]", english)
    ko_blocks, en_blocks = korean.strip().split("\n\n"), english.strip().split("\n\n")
    assert len(ko_blocks) == len(en_blocks) == 141
    decimal_pattern = r"(?<!\d)[+-]?(?:0|1)\.\d{2,}(?!\d)"
    for index, (ko, en) in enumerate(zip(ko_blocks, en_blocks, strict=True)):
        assert Counter(re.findall(decimal_pattern, ko.replace("−", "-"))) == Counter(
            re.findall(decimal_pattern, en.replace("−", "-"))), index
    kt, et = tables(korean), tables(english)
    assert len(kt) == len(et) == 7
    for table_index, (ko_rows, en_rows) in enumerate(zip(kt, et, strict=True)):
        assert len(ko_rows) == len(en_rows)
        for ko_row, en_row in zip(ko_rows, en_rows, strict=True):
            assert len(ko_row) == len(en_row)
            # Definitions are translated; all numeric/data cells retain their values.
            start = 2 if table_index in (0, 1) else 1
            assert [numbers(cell) for cell in ko_row[start:]] == [numbers(cell) for cell in en_row[start:]], (
                table_index, ko_row, en_row)
    figure_pattern = r"!\[(.*?)\]\((.*?)\)"
    kf, ef = re.findall(figure_pattern, korean), re.findall(figure_pattern, english)
    assert len(kf) == len(ef) == 6
    for index, ((_, ko_path), (caption, en_path)) in enumerate(zip(kf, ef, strict=True), 1):
        assert caption.startswith(f"Figure {index}.")
        assert Path(ko_path).name == Path(en_path).name and (OUT / en_path).is_file()
    ko_refs = korean.split("## 참고문헌\n\n")[1].split("## 부록 A.")[0].strip().split("\n\n")
    en_refs = english.split("## References\n\n")[1].split("## Appendix A.")[0].strip().split("\n\n")
    assert len(ko_refs) == len(en_refs) == 11
    for ko, en in zip(ko_refs, en_refs, strict=True):
        assert ko.split("(")[0] == en.split("(")[0]
        assert re.findall(r"https?://[^>\s]+", ko) == re.findall(r"https?://[^>\s]+", en)
    assert re.findall(r"\$\$(.*?)\$\$", korean, re.S) == re.findall(r"\$\$(.*?)\$\$", english, re.S)
    ko_headings = re.findall(r"^#{2,3} ", korean, re.M)
    en_headings = re.findall(r"^#{2,3} ", english, re.M)
    assert len(ko_headings) == len(en_headings) == 30
    csv_count = 0
    for path in sorted((OUT / "tables_en").glob("*.csv")):
        with path.open() as a, (OUT / "tables" / path.name).open() as b:
            assert list(csv.reader(a)) == list(csv.reader(b)), path.name
        csv_count += 1
    assert csv_count == 9
    manifest = json.loads((OUT / "figures_en/manifest.json").read_text())
    for path, expected in manifest["inputs"].items():
        assert digest(ROOT / path) == expected, path
    for path, expected in manifest["outputs"].items():
        assert digest(OUT / path) == expected, path
    pdf = OUT / "review_en_v2.pdf"
    pdf_text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    page_count = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    assert "\ufffd" not in pdf_text and not re.search(r"[가-힣]", pdf_text)
    for i in range(1, 7):
        assert f"Figure {i}." in pdf_text
    for label in ["Table 1.", "Table 2.", "Table 3.", "Table 4.", "Table 5.",
                  "Table A1.", "Table B1.", "Appendix A.", "Appendix B.", "References"]:
        assert label in pdf_text, label
    for row in et[3] + et[4] + et[5]:
        for value in row:
            for decimal in re.findall(r"0\.\d{4}", value):
                assert decimal in pdf_text, decimal
    for i in range(1, 7):
        figure_text = subprocess.check_output(
            ["pdftotext", str(OUT / ef[i - 1][1]).replace(".png", ".pdf"), "-"], text=True)
        assert not re.search(r"[가-힣]", figure_text)
    docx = EN.with_suffix(".docx")
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(docx) as archive:
        assert archive.testzip() is None
        xml = ET.fromstring(archive.read("word/document.xml"))
        assert len(xml.findall(".//w:tbl", ns)) == 7
        assert len(xml.findall(".//w:drawing", ns)) == 6
        docx_text = "".join(node.text or "" for node in xml.findall(".//w:t", ns))
    html_path = OUT / "review_en_v2.html"
    html_text = html.unescape(re.sub(r"<[^>]+>", "", html_path.read_text()))
    markers = ["all five are negative", "not a deployable semi-supervised method or performance ceiling",
               "three independent test datasets", "pretraining data included these audio files",
               "not guarantee higher test performance", "Appendix B."]
    for text in [english, pdf_text, docx_text, html_text]:
        for marker in markers:
            assert normalize(marker) in normalize(text), marker
    # The Korean artifacts and email are protected inputs, not export targets.
    ko_manifest = json.loads((OUT / "artifact_manifest.json").read_text())
    protected = {path: expected for path, expected in ko_manifest["files"].items()
                 if path.startswith("paper/manuscripts/v2/")}
    for path, expected in protected.items():
        assert digest(ROOT / path) == expected, path
    report = {"passed": True, "mode": "complete_english_translation_parity_and_exports",
              "review_date": "2026-10-08", "pdf_pages": page_count,
              "whitespace_words_including_metadata_and_references": len(english.split()),
              "parallel_markdown_blocks": 141, "numeric_table_parity": 7,
              "blockwise_decimal_token_parity": True, "unchanged_equations": 2,
              "headings": 30, "references": 11, "english_figures": 6,
              "identical_detail_csvs": 9, "docx_zip_and_structure": "passed",
              "export_content_markers": len(markers), "korean_artifacts_unchanged": True,
              "new_model_training": False, "manual_layout_review": "see REVIEW_EN.md"}
    report_path = OUT / "VERIFICATION_EN.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    paths = [KO, OUT / "review_ko_v2.pdf", EN, docx, pdf, html_path, OUT / "review_en.css",
             OUT / "figures_en/manifest.json", report_path]
    if (OUT / "REVIEW_EN.md").exists():
        paths.append(OUT / "REVIEW_EN.md")
    paths += [ROOT / "src" / name for name in ["phase1_paper_v2_en_figures.py",
              "phase1_paper_v2_en_export.py", "phase1_paper_v2_en_verify.py"]]
    output_manifest = {"root": "repository root", "files": {str(p.relative_to(ROOT)): digest(p) for p in paths}}
    (OUT / "artifact_manifest_en.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

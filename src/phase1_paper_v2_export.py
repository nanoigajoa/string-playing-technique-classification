"""Export the Korean v2 source to editable DOCX and self-contained HTML.

PDF is printed separately with a local browser so its layout can be inspected.
"""

from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/manuscripts/v2"
SOURCE = OUT / "주법분류_논문초안_국문_v2.md"
CSS = """
@page {
  size: A4; margin: 20mm 19mm 20mm;
  @bottom-center { content: counter(page) " / " counter(pages); font: 9pt Arial; color: #666; }
}
html { color: #171717; background: white; }
body { font-family: 'Apple SD Gothic Neo', 'Nanum Gothic', 'Malgun Gothic', sans-serif;
  font-size: 10.5pt; line-height: 1.65; max-width: 172mm; margin: auto; }
h1 { font-size: 21pt; line-height: 1.4; font-weight: 700; margin: 0 0 18px; word-break: keep-all; }
h2 { font-size: 15pt; line-height: 1.45; margin: 24px 0 10px; }
h3 { font-size: 11.5pt; line-height: 1.5; margin: 20px 0 8px; }
h1, h2, h3 { break-after: avoid; }
p { margin: 0 0 11px; text-align: justify; word-break: normal; overflow-wrap: anywhere; orphans: 3; widows: 3; }
blockquote { margin: 12px 0 20px; padding: 6px 12px; border-left: 2px solid #999; color: #555; font-size: 9pt; }
table { width: 100%; border-collapse: collapse; font-size: 9pt; line-height: 1.55; margin: 14px 0 18px; break-inside: avoid; }
caption { text-align: left; font-size: 9.1pt; font-weight: 600; margin-bottom: 7px; }
th, td { padding: 6px 7px; border-bottom: .5pt solid #bbb; vertical-align: top; }
thead { border-top: 1pt solid #444; border-bottom: 1pt solid #444; }
tr { break-inside: avoid; }
figure { margin: 18px 0; break-inside: avoid; }
figure img { display: block; width: 100%; height: auto; max-height: 177mm; object-fit: contain; }
figcaption { margin-top: 9px; font-size: 9pt; line-height: 1.55; text-align: justify; }
a { color: inherit; text-decoration: none; overflow-wrap: anywhere; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.5pt; }
math[display=block] { margin: 14px auto; font-size: 11pt; }
.page-break { break-before: page; }
.references p { font-size: 9.3pt; line-height: 1.5; text-align: left; break-inside: avoid; }
@media screen { body { padding: 30px; } }
"""


def style_docx(path):
    """Set Korean fonts, A4 and paragraph keep rules without changing content."""
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ns = {"w": w}

    def q(name):
        return f"{{{w}}}{name}"

    def child(parent, tag):
        item = parent.find(q(tag))
        return item if item is not None else ET.SubElement(parent, q(tag))

    with zipfile.ZipFile(path) as source:
        files = {name: source.read(name) for name in source.namelist()}
    styles = ET.fromstring(files["word/styles.xml"])
    defaults = child(child(child(styles, "docDefaults"), "rPrDefault"), "rPr")
    fonts = child(defaults, "rFonts")
    for attr in list(fonts.attrib):
        if attr.lower().endswith("theme"):
            del fonts.attrib[attr]
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(q(attr), "Apple SD Gothic Neo")
    child(defaults, "lang").set(q("eastAsia"), "ko-KR")
    child(defaults, "lang").set(q("val"), "ko-KR")
    child(defaults, "sz").set(q("val"), "21")
    for style in styles.findall("w:style", ns):
        sid = style.get(q("styleId"), "")
        if sid in {"Normal", "BodyText", "FirstParagraph", "Caption", "ImageCaption", "TableCaption"}:
            spacing = child(child(style, "pPr"), "spacing")
            spacing.set(q("line"), "350")
            spacing.set(q("lineRule"), "auto")
            child(child(style, "rPr"), "sz").set(q("val"), "21" if "Caption" not in sid else "18")
        if sid.startswith("Heading") or sid in {"Title", "TableCaption"}:
            child(child(style, "pPr"), "keepNext")
        if sid in {"Heading1", "Heading2", "Heading3"}:
            child(child(style, "rPr"), "sz").set(q("val"), {"Heading1": "38", "Heading2": "28", "Heading3": "23"}[sid])
    document = ET.fromstring(files["word/document.xml"])
    for section in document.findall(".//w:sectPr", ns):
        size = child(section, "pgSz")
        size.set(q("w"), "11906"); size.set(q("h"), "16838")
        margin = child(section, "pgMar")
        for attr in ("top", "bottom", "left", "right"):
            margin.set(q(attr), "1134")
    for paragraph in document.findall(".//w:p", ns):
        text = "".join(paragraph.itertext())
        if text.startswith(("1. 서론", "참고문헌", "부록 A.")):
            child(child(paragraph, "pPr"), "pageBreakBefore")
        if paragraph.find(".//w:drawing", ns) is not None:
            child(child(paragraph, "pPr"), "keepNext")
    for row in document.findall(".//w:tr", ns):
        child(child(row, "trPr"), "cantSplit")
    files["word/styles.xml"] = ET.tostring(styles, encoding="utf-8", xml_declaration=True)
    files["word/document.xml"] = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for name, data in files.items():
            target.writestr(name, data)


def main():
    css_path = OUT / "review_ko.css"
    css_path.write_text(CSS, encoding="utf-8")
    docx = SOURCE.with_suffix(".docx")
    subprocess.run(["pandoc", str(SOURCE), f"--resource-path={OUT}", "--metadata", "lang=ko-KR", "-o", str(docx)], check=True)
    style_docx(docx)
    html_path = OUT / "review_ko_v2.html"
    subprocess.run(["pandoc", str(SOURCE), f"--resource-path={OUT}", "--standalone", "--embed-resources", "--mathml",
                    f"--css={css_path}", "--metadata", "lang=ko-KR", "--metadata", "pagetitle=현악 주법 분류 — 국문 v2",
                    "-o", str(html_path)], check=True)
    html = html_path.read_text()
    # Start the main text after the title/abstract and group bibliography styling.
    html = re.sub(r'(<h2[^>]*>1\. 서론</h2>)', r'<div class="page-break"></div>\1', html, count=1)
    html = re.sub(r'(<h2[^>]*>참고문헌</h2>)', r'<div class="references">\1', html, count=1)
    html = re.sub(r'(<h2[^>]*>부록 A\.)', r'</div><div class="page-break"></div>\1', html, count=1)
    html_path.write_text(html)
    assert len(re.findall(r"<figure[ >]", html)) == 6
    assert len(re.findall(r"<figcaption", html)) == 6
    assert len(re.findall(r"<table[ >]", html)) == 7
    assert len(re.findall(r'src="data:image/png;base64,', html)) == 6
    with zipfile.ZipFile(docx) as archive:
        assert archive.testzip() is None
        document = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        assert len(document.findall(".//w:tbl", ns)) == 7
        assert len(document.findall(".//w:drawing", ns)) == 6
    print(json.dumps({"docx": str(docx.relative_to(ROOT)), "html": str(html_path.relative_to(ROOT)),
                      "figures": 6, "tables": 7, "characters": len(SOURCE.read_text()),
                      "whitespace_tokens": len(SOURCE.read_text().split())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

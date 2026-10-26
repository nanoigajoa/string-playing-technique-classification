"""Export the complete English v2 manuscript to DOCX, HTML, and optionally PDF."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from phase1_paper_v2_export import CSS as KOREAN_CSS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/manuscripts/v2"
SOURCE = OUT / "주법분류_논문초안_영문_v2.md"
CSS = KOREAN_CSS.replace(
    "'Apple SD Gothic Neo', 'Nanum Gothic', 'Malgun Gothic', sans-serif",
    "'Times New Roman', Georgia, serif",
).replace("line-height: 1.65", "line-height: 1.5").replace(
    "overflow-wrap: anywhere", "overflow-wrap: break-word"
).replace("word-break: keep-all", "word-break: normal")


def style_docx(path):
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ns = {"w": namespace}

    def child(parent, tag):
        item = parent.find(f"{{{namespace}}}{tag}")
        return item if item is not None else ET.SubElement(parent, f"{{{namespace}}}{tag}")

    def set_attr(item, key, value):
        item.set(f"{{{namespace}}}{key}", value)

    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    styles = ET.fromstring(files["word/styles.xml"])
    defaults = child(child(child(styles, "docDefaults"), "rPrDefault"), "rPr")
    fonts = child(defaults, "rFonts")
    for key in list(fonts.attrib):
        if "theme" in key.lower():
            del fonts.attrib[key]
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        set_attr(fonts, key, "Times New Roman")
    set_attr(child(defaults, "lang"), "val", "en-US")
    set_attr(child(defaults, "sz"), "val", "21")
    for style in styles.findall("w:style", ns):
        sid = style.get(f"{{{namespace}}}styleId", "")
        if sid.startswith("Heading") or sid in {"Title", "TableCaption"}:
            child(child(style, "pPr"), "keepNext")
    document = ET.fromstring(files["word/document.xml"])
    for section in document.findall(".//w:sectPr", ns):
        size = child(section, "pgSz")
        set_attr(size, "w", "11906")
        set_attr(size, "h", "16838")
        margins = child(section, "pgMar")
        for key in ("top", "bottom", "left", "right"):
            set_attr(margins, key, "1134")
    for paragraph in document.findall(".//w:p", ns):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns))
        if text.startswith(("1. Introduction", "References", "Appendix A.")):
            child(child(paragraph, "pPr"), "pageBreakBefore")
        if paragraph.find(".//w:drawing", ns) is not None:
            child(child(paragraph, "pPr"), "keepNext")
    for row in document.findall(".//w:tr", ns):
        child(child(row, "trPr"), "cantSplit")
    files["word/styles.xml"] = ET.tostring(styles, encoding="utf-8", xml_declaration=True)
    files["word/document.xml"] = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def print_pdf(html):
    """Print through local headless Chrome; close the task browser after export."""
    from playwright.sync_api import sync_playwright

    target = OUT / "review_en_v2.pdf"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=["--disable-background-networking", "--disable-component-update", "--no-first-run"],
        )
        page = browser.new_page()
        page.goto(html.as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready")
        page.emulate_media(media="print")
        page.pdf(path=str(target), format="A4", print_background=True,
                 prefer_css_page_size=True, display_header_footer=False)
        browser.close()
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="Also print PDF using local Chrome")
    args = parser.parse_args()
    css = OUT / "review_en.css"
    css.write_text(CSS)
    docx = SOURCE.with_suffix(".docx")
    subprocess.run(["pandoc", str(SOURCE), f"--resource-path={OUT}", "--metadata", "lang=en-US",
                    "-o", str(docx)], check=True)
    style_docx(docx)
    html_path = OUT / "review_en_v2.html"
    subprocess.run(["pandoc", str(SOURCE), f"--resource-path={OUT}", "--standalone", "--embed-resources",
                    "--mathml", f"--css={css}", "--metadata", "lang=en-US", "--metadata",
                    "pagetitle=String Playing Technique Classification — English v2", "-o", str(html_path)], check=True)
    html = html_path.read_text()
    html = re.sub(r'(<h2[^>]*>1\. Introduction</h2>)', r'<div class="page-break"></div>\1', html, count=1)
    html = re.sub(r'(<h2[^>]*>References</h2>)', r'<div class="references">\1', html, count=1)
    html = re.sub(r'(<h2[^>]*>Appendix A\.)', r'</div><div class="page-break"></div>\1', html, count=1)
    html_path.write_text(html)
    assert len(re.findall(r"<figure[ >]", html)) == 6
    assert len(re.findall(r"<table[ >]", html)) == 7
    assert html.count('src="data:image/png;base64,') == 6
    with zipfile.ZipFile(docx) as archive:
        assert archive.testzip() is None
    result = {"docx": str(docx.relative_to(ROOT)), "html": str(html_path.relative_to(ROOT)),
              "whitespace_words": len(SOURCE.read_text().split()), "figures": 6, "tables": 7}
    if args.pdf:
        result["pdf"] = str(print_pdf(html_path).relative_to(ROOT))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

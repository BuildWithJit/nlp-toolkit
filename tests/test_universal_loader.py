from pathlib import Path
import pytest
import shutil
from nlp_toolkit.loaders.universal import load, LoaderConfig

def test_txt_roundtrip(tmp_path:Path):
    p = tmp_path / "test.txt"
    p.write_text("Hello, world!")
    docs, rep = load(p)
    assert len(docs) == 1
    assert docs[0]["text"] == "Hello, world!"
    assert rep["pages_total"] == 1
    assert rep["pages_ocr"] == 0
    assert rep["engines"] == ["text"]
    
@pytest.mark.skipif(shutil.which("tesseract") is None or shutil.which("pdftoppm") is None,
                    reason="requires tesseract & poppler")
def test_pdf_selective_ocr(tmp_path):
    # tiny one-page PDF with no text forces OCR
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    pdf = tmp_path / "o.pdf"
    c = canvas.Canvas(str(pdf), pagesize=A4)
    # Create a blank page (no text, no images - forces OCR)
    c.showPage()
    c.save()
    docs, rep = load(pdf, LoaderConfig(prefer="auto"))
    assert rep["pages_total"] == 1 and rep["pages_ocr"] == 1 and docs[0]["meta"]["method"] == "ocr"
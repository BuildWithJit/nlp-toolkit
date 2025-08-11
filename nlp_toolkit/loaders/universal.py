from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO, IOBase
from pathlib import Path
from typing import Any, TypedDict
import re
import time

Docs = list[dict[str, Any]]

class LoadReport(TypedDict):
    pages_total:int
    pages_ocr:int
    time_ms: int
    engines: list[str]

@dataclass
class LoaderConfig:
    prefer:str = 'auto' # 'auto'| 'ocr'| 'text'
    min_text_ratio: float = 0.5 # non-whitespace ratio to accept native text
    dpi: int = 300 # dpi to use for OCR
    ocr_lang:str = 'eng' # language for OCR
    deskew:bool = False
    denoise:bool = False


def _to_bytes(src: str|Path|bytes|IOBase) -> bytes:
    if isinstance(src, (str, Path)): 
        return Path(src).read_bytes()
    if isinstance(src, bytes) : 
        return src
    if isinstance(src, IOBase) : 
        return src.read()
    raise ValueError(f"Unsupported type: {type(src)}")


def _ratio(text:str) -> float:
    length = len(text)
    return 0.0 if length == 0 else len(re.sub(r'\s+', '', text)) / length

def _looks_image(data: bytes) -> bool:
    sig = data[:10]
    return sig.startswith(b"\x89PNG") or sig.startswith(b"\xff\xd8")

def _ocr_pil(img, config:LoaderConfig) -> str:
    try:
        import pytesseract as pt
    except Exception as e:
        raise RuntimeError("Tesseract not available. Install nlp-toolkit[ocr].") from e
    
    if config.deskew or config.denoise:
        from nlp_toolkit.preproc.vision import preprocess_pil
        img = preprocess_pil(img, deskew=config.deskew, denoise=config.denoise)
    return pt.image_to_string(img, lang=config.ocr_lang)

def _ocr_pdf_page(page, i:int, pdf_bytes:bytes, config:LoaderConfig) -> str:
    from pdf2image import convert_from_bytes
    imgs = convert_from_bytes(pdf_bytes, dpi=config.dpi, first_page=i+1, last_page=i+1)
    text = _ocr_pil(imgs[0], config)
    return {
        'text': text,
        "metadata": {
            "page":i,
            "dpi": config.dpi,
            "method": "ocr",
        }
    }
    
def load(src:str|Path|bytes|IOBase, config:LoaderConfig|None=None) -> Docs:
    """Return docs: [{"text", "meta"}], plus telemetry report."""
    t0 = time.perf_counter()
    config = config or LoaderConfig()
    data = _to_bytes(src)
    engines:list[str] = []
    docs:Docs = []
    pages_total = pages_ocr = 0

    is_pdf = data[:4] == b"%PDF"
    if is_pdf and config.prefer in ("auto", "text"):
        import pdfplumber
        engines.append("pymupdf")
        with pdfplumber.open(BytesIO(data)) as pdf:
            pages_total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text().strip()
                if config.prefer == "text" or (config.prefer == "auto" and _ratio(text) >= config.min_text_ratio):
                    docs.append({
                        "text": text,
                        "metadata": {
                            "page": i,
                            "method": "text",
                            "dpi": config.dpi,
                        }
                    })
                else:
                    docs.append(_ocr_pdf_page(page, i, data, config))
                    pages_ocr += 1
                    if pages_ocr: 
                        engines.append("ocr")
    elif is_pdf and config.prefer =='ocr':
        import pdfplumber
        engines.append("ocr")
        with pdfplumber.open(BytesIO(data)) as pdf:
            pages_total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                docs.append(_ocr_pdf_page(page, i, data, config))
                pages_ocr += 1
                engines.append("ocr")
    else:
        if _looks_image(data):
            from PIL import Image
            text = _ocr_pil(Image.open(BytesIO(data)), config)
            docs.append({
                "text": text,
                "metadata": {"method": "ocr"}
            })
            pages_total = pages_ocr = 1
            engines.append("ocr")
        else:
            txt = data.decode('utf-8', errors='ignore')
            docs.append({"text": txt, "metadata": {"method": "text"}})
            pages_total = 1
    
    rep: LoadReport = {
        "pages_total": int(pages_total or 1),
        "pages_ocr": pages_ocr,
        "time_ms": int((time.perf_counter() - t0) * 1000),
        "engines": engines or ["text"],
    }
    return docs, rep
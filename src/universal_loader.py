from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import (
    TextLoader, CSVLoader, JSONLoader, UnstructuredExcelLoader,
    PyPDFLoader, UnstructuredFileLoader
)
from langchain.docstore.document import Document

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile


class UniversalDocumentLoader:
    """Loads text from multiple file formats into LangChain Document objects."""

    def __init__(self, ocr_lang: str = "eng"):
        self.ocr_lang = ocr_lang

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return TextLoader(str(path), encoding="utf-8").load()

        elif suffix == ".csv":
            return CSVLoader(str(path)).load()

        elif suffix == ".json":
            return JSONLoader(str(path), jq_schema=".").load()

        elif suffix in [".xls", ".xlsx"]:
            return UnstructuredExcelLoader(str(path)).load()

        elif suffix == ".pdf":
            docs = PyPDFLoader(str(path)).load()
            if not any(doc.page_content.strip() for doc in docs):
                # OCR fallback for scanned PDFs
                return self._ocr_pdf(path)
            return docs

        elif suffix in [".png", ".jpg", ".jpeg", ".tiff"]:
            return self._ocr_image(path)

        else:
            # Try unstructured loader as a last resort
            return UnstructuredFileLoader(str(path)).load()

    def _ocr_pdf(self, path: Path) -> List[Document]:
        images = convert_from_path(str(path))
        temp_dir = Path(tempfile.mkdtemp())
        docs = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang=self.ocr_lang)
            docs.append(Document(page_content=text, metadata={"source": str(path), "page": i+1}))
        return docs

    def _ocr_image(self, path: Path) -> List[Document]:
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang=self.ocr_lang)
        return [Document(page_content=text, metadata={"source": str(path)})]

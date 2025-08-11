from pathlib import Path
from nlp_toolkit.loaders.universal import load

def test_txt_roundtrip(tmp_path:Path):
    p = tmp_path / "test.txt"
    p.write_text("Hello, world!")
    docs, rep = load(p)
    assert len(docs) == 1
    assert docs[0]["text"] == "Hello, world!"
    assert rep["pages_total"] == 1
    assert rep["pages_ocr"] == 0
    assert rep["engines"] == ["text"]
    
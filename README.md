# nlp-toolkit
A collection of reusable, production-ready Python modules covering everything from text ingestion and cleaning to embeddings, classification, summarization, and deployment — designed to save you hours of setup and accelerate real-world NLP projects.

# nlp-toolkit (v0.1 work-in-progress)
Goal: Documents → clean text → YAML/API. Quickstart arrives in v0.1.

Install dev:
```
uv pip install -e ".[all]"
```

Run tests:
```
uv run pytest -q
```


### OS dependencies
macOS:  `brew install tesseract poppler`
Ubuntu: `sudo apt-get update && sudo apt-get install -y tesseract-ocr poppler-utils`

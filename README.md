# TraceLens AI v1.8 — Full-Source Copy Matching & AI-Model Comparison

**Start with [UPGRADE_V18.md](UPGRADE_V18.md).** This release addresses a reported failure to find text copied from Wikipedia. It adds source-discovery improvements, full-source exact token alignment, a Thorough web mode, independent n-gram/MiniLM AI-writing classifier comparison, and clearer uncertainty reporting.

**Verification:** [RELEASE_VERIFICATION_V18.md](RELEASE_VERIFICATION_V18.md) — offline tests passed, but live APIs/browser/independent AI accuracy require your Windows verification.

TraceLens is a college research project. It is **not** a substitute for an institution's validated plagiarism policies, exhaustive source databases, or reliable AI-authorship determination. No match means only that no qualifying overlap was found in the *specific successfully retrieved sources*.

## Main features

- PDF/DOCX/TXT upload and paste-text input (text-based PDFs, not scanned PDF OCR).
- Reference library in SQLite, text extraction, word-position alignment.
- Exact, TF-IDF and optional MiniLM semantic similarity detection.
- Independently selectable **Off**, **Wikipedia only**, **Tavily only**, or **Wikipedia + Tavily** search modes.
- Direct public webpage URL lookup; only verified extracted webpage content is compared, not search snippets.
- Combined-provider partial failure handling, URL deduplication and source provenance.
- Side-by-side exact-word highlighting, citation patterns, low-information overlap filters, source-level human review.
- Separate AI-writing experimental window analysis with flagged analyzed-text coverage, **per-window human review and notes**, and source-independent reporting.
- PDF evidence export with both types of human review and provider search statuses.
- Offline regression tests and QA fixtures.

## Run on Windows (Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then visit **http://localhost:8501**.

To enable Wikipedia, set `WIKIMEDIA_CONTACT=your-contact-email` in `.env` (no API key). For Tavily, set `TAVILY_API_KEY=your-private-key`. Copy `.env.example` to `.env` before editing. External queries are opt-in.

## Known limitations

- The seven-feature AI-writing classifier is *experimental* and has missed known AI-generated text. v1.8 adds optional comparison models and OOD warnings; it does **not** claim better model accuracy.
- Textual similarity is evidence of overlap, not proof of academic dishonesty. Citation signals need human contextual review.
- Online retrieval is not comprehensive and some sites deny scraping or full-text extraction.
- No OCR on scanned PDF, no authenticated multi-user accounts, no cross-language/code plagiarism support in this version.

Official sources: https://www.mediawiki.org/wiki/Wikimedia_APIs/Access_policy and https://guides.turnitin.com/hc/en-us/articles/22774058814093-Using-the-AI-Writing-Report

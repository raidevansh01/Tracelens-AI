# TraceLens AI v1.8 — Source Discovery & Multi-Model Research Upgrade

**Purpose:** Address a real reported failure: user copied Wikipedia text, but automatic search or matching did not identify its source. Also provide alternative experimental models to compare with the existing 7-feature stylometric classifier.

## Improvements

1. **Global exact-copy anchor index:** Unlike v1.7, exact text matching scans the full extracted source; no dependency on TF-IDF top-k or sentence-window selection. Words are aligned back to the original character spans for safe highlighting. The reference text can be very long. Coverage is unique original submitted words verified in source alignments, *not a misconduct verdict*.
2. **Wikipedia article discovery:** Uses exact quoted phrases drawn from the beginning, middle and end of the submission, plus a topic fallback. MediaWiki supports phrase searching through its full-text index. Separately retrieves full page body using the API. Searches additional candidates in Thorough mode.
3. **Tavily:** Diversified quoted and keyword queries; search and full-text extraction remain separate steps. Thorough mode chooses advanced search and advanced extraction, potentially consuming more API credits. Up to 10 candidate pages (15 in Thorough mode) are extracted.
4. **Observed source matches vs retrieved pages:** Source inventory and per-provider status distinguish fetched pages from sources containing matches. An empty/failed search is never interpreted as proof of originality.
5. **AI writing:** Current seven-feature model now marks severe feature-distribution shift **inconclusive**, rather than presenting its low scores as human evidence. New **optional and independently evaluated** supervised classifiers: word+character n-gram TF-IDF and MiniLM embeddings with Logistic Regression. Each candidate reports scores independently. Neither is validated as accurate just by installing it. MiniLM semantic source similarity is distinct from AI-writing classification.
6. **Human review and PDF reporting:** Preserved existing reviews and notes. PDF distinguishes verified exact-copy lengths from TF-IDF similarities.

## Installation on Windows (recommended parallel installation)

1. Keep your existing Stage-9 v1.7 intact.
2. Extract this ZIP to `C:\Code\Plagrism\Stage-10\TraceLens-AI` (ensure `app.py` is at the folder root).
3. Copy only your own existing `.env`, `storage\tracelens.sqlite3`, trusted `storage\ai_pattern_model.joblib`, and `datasets\ai_writing.csv` from Stage-9 to the matching relative locations in Stage-10. You may need to create `datasets` in Stage-10. **Do not copy `.venv`. Do not upload your API key to anyone.**
4. Open PowerShell in the new project directory and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe qa\self_check.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Optional MiniLM:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
```

## Diagnose the actual Wikipedia-copy issue

### A. Direct check of a known source

This tests whether the matching engine can identify a **known copied excerpt** in a **known Wikipedia article**, separately from whether search can discover the article:

```powershell
.\.venv\Scripts\python.exe qa\wiki_copy_acceptance.py --url https://en.wikipedia.org/wiki/Machine_learning
```

The script pulls the page with the official Wikipedia API, copies ~85 words from the last third, and verifies matching. Requires `WIKIMEDIA_CONTACT` in `.env` (your own email or public HTTPS project URL). It does **not** use an API key.

### B. Automatic article rediscovery

```powershell
.\.venv\Scripts\python.exe qa\wiki_copy_acceptance.py --url https://en.wikipedia.org/wiki/Machine_learning --search
```

If direct matching passes but automatic search fails, this is a **source-discovery miss**, not a broken matching algorithm. The console prints the article list and any warnings.

### C. Repeat using YOUR failed Wikipedia excerpt

Save the copied paragraph to `my_wiki_excerpt.txt` (at least 30–50 consecutive words) and use the actual original Wikipedia article URL:

```powershell
.\.venv\Scripts\python.exe qa\wiki_copy_acceptance.py --url https://en.wikipedia.org/wiki/Machine_learning --file my_wiki_excerpt.txt --search
```

Replace the URL with the actual article you copied from. In the Streamlit Analyze page, you can also paste the source URL into **Optional direct public webpage URL**. This bypasses source-discovery uncertainty and compares against that exact article.

## New AI-writing classifier comparisons

The existing model stored at `storage\ai_pattern_model.joblib` remains compatible. The new optional models do **not** overwrite it.

Train a word+character n-gram baseline on your *own* labeled CSV (must have `text,label,group_id`, labels 0=human, 1=AI):

```powershell
.\.venv\Scripts\python.exe train_ai_comparator.py --data datasets\ai_writing.csv --method ngram
```

Optional separate MiniLM embedding classifier (downloads a pretrained embedding model on first use, CPU processing may take longer):

```powershell
.\.venv\Scripts\python.exe train_ai_comparator.py --data datasets\ai_writing.csv --method minilm
```

The Analyze page shows any installed candidates separately. They **are not ensemble-voted** into an authorship verdict and may disagree. They use group-separated train/validation/test partitions, but a *same-corpus split* does not demonstrate generalization to new generators, languages or student writing.

### Independent evaluation — essential before claiming improvement

Prepare `datasets\independent_test.csv` with **genuinely unseen** labeled human and AI samples, both preferably >=120 words. Use multiple authors, subjects and AI generators. Do not include material present in HC3 or training prompts.

```powershell
.\.venv\Scripts\python.exe qa\evaluate_external_writing.py --data datasets\independent_test.csv --training-data datasets\ai_writing.csv --model stylometry
.\.venv\Scripts\python.exe qa\evaluate_external_writing.py --data datasets\independent_test.csv --training-data datasets\ai_writing.csv --model ngram
.\.venv\Scripts\python.exe qa\evaluate_external_writing.py --data datasets\independent_test.csv --training-data datasets\ai_writing.csv --model minilm
```

The evaluator checks exact duplicate texts and group IDs when provided, but cannot independently guarantee author provenance or that prompts/subjects aren't leaked. Only compare metrics on a known independent dataset. Do not tune thresholds on the independent test data.

## Known limitations — important

- Automated public web search is **not exhaustive**; unindexed, private, inaccessible, OCR-only, subscription or truncated pages can still be missed. It can find a source only when it retrieves and extracts it. A quoted phrase is a better discovery strategy but not a guarantee.
- Wikipedia searches **English Wikipedia**, not the entire internet.
- Some very long fetched pages are clipped at 250,000 characters. Network request limits, robots rules and API quotas still apply.
- Thorough Tavily mode can be slower and consume more credits. Send only content you are authorized to search externally.
- Short common text can match independently; human review of matching, quotations, bibliographies, fair use and citations is essential.
- Writing-style classification is **not a reliable determination of AI authorship**. We have one confirmed false-negative test on a known AI document. New candidates are experiment controls until externally evaluated.
- This environment did not run your live APIs, installed MiniLM model, or Streamlit browser. Run the Windows diagnostic commands above to verify your actual external services.

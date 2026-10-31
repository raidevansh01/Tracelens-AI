# TraceLens AI v1.7 — Web Search Reliability & AI Model Diagnostics

## What this release fixes

- Tavily discovery no longer requires a long exact quote. It uses **short quoted phrases plus keyword searches** and retrieves full text before comparing.
- Wikipedia uses its MediaWiki **full rendered article parse** endpoint first and only falls back to the older TextExtracts API where needed. Queries are focused on shorter topic concepts.
- Long online articles are searched for relevant passages **throughout** the source, rather than limiting source comparison to the first 140 passages. Passage and computational limits still exist; this is not exhaustive Internet coverage.
- Wikipedia + Tavily search continues independently, with deduplication of tracking-parameter URLs across providers.
- The Analyze page has a **Preview web-search queries** panel so you know which phrases would be sent externally. Neither source provider is contacted until you explicitly opt in and run analysis.
- AI writing analysis stores training-feature ranges when you retrain, and annotates text windows whose features are unusually far from those ranges. Existing v1.4/v1.6 models continue to load, but cannot provide these new distribution checks until retrained.
- `qa/health_check_v17.py` checks local matching and optionally tests real Wikipedia or Tavily access. `qa/evaluate_external_writing.py` evaluates the writing detector against **independent labeled text**, without retraining.

## Important: this does NOT solve AI-authorship detection accuracy

Your previous trained model missed the supplied known AI-written sample. This release does not claim a measured recall improvement. Detecting AI authorship from text alone is uncertain; low classifier scores are **not proof of human authorship**. For a meaningful accuracy assessment, collect independent labeled documents (including current AI systems and real human student writing) and evaluate false positives/recall.

## Windows upgrade

1. Keep your existing Stage-8 directory as a backup. Extract this ZIP into a **new** directory, for example `C:\Code\Plagrism\Stage-9\TraceLens-AI`.
2. With both copies of Streamlit stopped, copy these trusted local files from the previous version if present:
   - `.env` (contains secrets: never post or commit it)
   - `storage\tracelens.sqlite3` (your previous documents/reviews)
   - `storage\ai_pattern_model.joblib` (your trained model; only load a file you trust)
   - `datasets\ai_writing.csv` (your training data)
3. Open PowerShell in the new `TraceLens-AI` directory and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe qa\health_check_v17.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

4. Open `http://localhost:8501`.
5. In **Analyze**, choose Wikipedia, Tavily, or combined, click **Preview selected phrases**, check external-text sharing consent, then Run analysis. If any source is missing, inspect **Results & Review → provider breakdown and warnings**.

## Test the live APIs (on your internet-connected PC)

These commands require the relevant entries in `.env` and do not print your Tavily key. The Tavily option can consume credits.

```powershell
.\.venv\Scripts\python.exe qa\health_check_v17.py --live-wikipedia
.\.venv\Scripts\python.exe qa\health_check_v17.py --live-tavily
```

A successful Wikipedia direct-source test shows article text was retrieved. The search test separately reports discovered Wikipedia articles. Tavily reports retrieved full-text pages; zero matching pages does **not** mean the document was original.

## Review the AI detector

To test the existing classifier on the known AI-written sample:

```powershell
.\.venv\Scripts\python.exe qa\health_check_v17.py
.\.venv\Scripts\python.exe check_ai_feature.py
```

If the model again flags 0 of 3 windows, **that is a documented false negative**. Do not lower its decision threshold just to force a positive result.

To add training-distribution warnings (does not itself improve accuracy), retrain with the labeled dataset:

```powershell
.\.venv\Scripts\python.exe train_ai_patterns.py --data datasets\ai_writing.csv
```

Evaluate on independent samples not used in training or threshold selection. CSV columns are `text,label` with human=`0`, AI=`1`, and optional `group_id`:

```powershell
.\.venv\Scripts\python.exe qa\evaluate_external_writing.py --data datasets\independent_test.csv --training-data datasets\ai_writing.csv
```

This prints confusion counts, precision, recall, and false-positive rate. It warns on small samples and rejects duplicate text or group IDs overlapping the training file. It cannot detect near-duplicates; ensure the split is truly independent.

## Current verification

- 47 automated Python regression tests passed in this environment.
- 20 controlled offline acceptance checks passed.
- Source retrieval on live Tavily and Wikipedia, Streamlit UI rendering, CPU MiniLM runtime, and the model's accuracy against independently labeled writing require verification on your own computer.

## Security

- Rotate any Tavily API key previously shown in public screenshots.
- Never commit `.env`, student submissions, or your untrusted `.joblib` files.
- Combined search is opt-in. It sends selected text phrases outside your computer.

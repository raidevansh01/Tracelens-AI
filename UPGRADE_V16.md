# TraceLens AI v1.6 — Upgrade and hands-on verification

## What has changed

1. Three mutually exclusive online choices are now available in **Analyze**: **Wikipedia only**, **Tavily only**, and **Wikipedia + Tavily** (plus Off).
2. Combined mode queries BOTH services independently. If one fails, other available sources and local references are still analyzed. Duplicate URLs are merged; the source inventory reports the discovery services.
3. **Results & Review** now contains human review forms for **every AI-writing window**, including non-flagged ones. Notes are saved in a new `writing_reviews` SQLite table, independently of source-matching reviews and ML scores.
4. The AI-writing section shows a clearly labeled *flagged analyzed-text share*. It counts unique analyzed words in flagged windows, NOT a reliable AI-authorship percentage.
5. The PDF includes provider retrieval status, separate model signals, human review outcomes, and reviewer notes.
6. All public-source matches still require extracted webpage text. Search result snippets by themselves are not accepted as verified evidence.

## Upgrade your Windows installation safely

**Do not overwrite your old application.** The old classifier and library belong to your existing copy.

1. Extract this ZIP to a **new** folder, for example `C:\Code\Plagrism\Stage-8\TraceLens-AI`.
2. Stop your old Streamlit server with `Ctrl+C`, and keep the original folder untouched as a backup.
3. Copy these files from your existing Stage-7 (or Stage-6) TraceLens-AI folder into the same relative positions in the new folder:
   - `.env` (Wikipedia contact and Tavily key; never send the key in screenshots)
   - `storage\tracelens.sqlite3` (reference library, analyses and previous source reviews)
   - `storage\ai_pattern_model.joblib` (only if it was trained by **you**, using the compatible v1.4+ training script)
   - `datasets\ai_writing.csv` (if you want the dataset available for retraining)
4. Open PowerShell **inside the new `TraceLens-AI` directory** and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe qa\self_check.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

5. Open `http://localhost:8501`. Your existing database is migrated automatically by creating the new `writing_reviews` table; **old analyses and source review notes remain preserved**.

Do not copy `.venv`, `__pycache__`, or an untrusted `.joblib` file.

## Test combined web search

1. In **Analyze**, upload a **nonconfidential** document with public content, or paste a passage from an English Wikipedia article.
2. Select **Wikipedia + Tavily · combined** under **Online search provider**. This is one selector; you do not need two separate checkboxes.
3. Enter a **real operator contact** in the Wikimedia contact field, or set `WIKIMEDIA_CONTACT=` in `.env`. Wikipedia needs no API key.
4. Configure `TAVILY_API_KEY=` in `.env` using your **new, private key**. If your old key was exposed, revoke/rotate it first. `TAVILY_KEYLESS=1` is an optional rate-limited experiment; it may not work reliably.
5. Enable the external-sharing consent checkbox. Search queries sent to providers can include selected phrases from your document.
6. Run the analysis. In **Results & Review**, look at **Public web search coverage** and **All sources actually compared**. The provider boxes show how many full-text pages each retrieved. Different sources with the same URL are counted once for comparison.
7. Also test **Wikipedia only** and **Tavily only** independently; these are different runs. One provider may retrieve no accessible pages while the other succeeds.
8. For a deterministic live Wikipedia test, paste a distinct paragraph from `https://en.wikipedia.org/wiki/Machine_learning` and supply that same URL in **Optional direct public webpage URL**. This checks whether the article's full text can be accessed, subject to service availability.

**Important:** The public web is not exhaustively indexed by either provider. Zero retrieved pages is *not* evidence that a document is original. Tavily calls may cost credits. A web-provider status of `retrieved` means full-text sources were returned, not that plagiarism was proven.

## Test AI-writing reviews and PDF

1. Confirm `storage\ai_pattern_model.joblib` exists and is from your own training run. Otherwise train with `train_ai_patterns.py` using your properly labeled CSV.
2. On **Analyze**, enable **Analyze AI-writing patterns (experimental)**. Web search can remain off if you only want to test the writing classifier.
3. Upload `TEST_AI_GENERATED.txt`, select **Run analysis**, then **Open full results & AI-writing details**.
4. In **B · AI-writing analysis**, note **Windows flagged**, **Flagged analyzed-text share**, and **Review threshold**. The test text is known AI-generated, but the previous experimental classifier missed it; no flag does not mean human writing.
5. Open **Review writing passages**. Choose a **Human review outcome** for Window 1, enter a note, and click **Save writing review**. Repeat on a different window, including one the model did not flag.
6. Change pages, return to **Results & Review**, and verify that both review decisions persisted. The model score should remain unchanged.
7. Download the evidence PDF and inspect the **Independent AI-writing pattern experiment** section. Your notes and human decisions should appear alongside the model scores.
8. For the command-line smoke test: ` .\.venv\Scripts\python.exe check_ai_feature.py` (run from project root).

### What is deliberately **not** claimed

- A score of 0.90 is **not** a 90% chance of AI authorship.
- Human stylistic features can resemble AI-generated writing. Never use the classifier alone for penalties or accusations.
- A flagged analyzed-text share is not an AI-writing percentage; overlapping windows are combined only for a display aid.
- The v1.6 release **does not retrain or improve the 7-feature stylistic classifier itself**. Re-evaluate with modern, representative writing before relying on it.
- Scanned-PDF OCR, paywalled article retrieval, cross-language matching, code plagiarism, and large-scale batch submission checking are not implemented.

## Debugging / validation

- Run ` .\.venv\Scripts\python.exe -m pytest -q` to check the offline regression suite.
- Run ` .\.venv\Scripts\python.exe qa\self_check.py` for the local source test.
- Run ` .\.venv\Scripts\python.exe qa\wiki_live_check.py` to check Wikipedia with a configured Wikimedia contact.
- Tavily must be checked manually using a valid current API key. Do **not** share API credentials or private student text in a bug report.
- If the writing classifier is unavailable, install its dependencies and ensure `storage\ai_pattern_model.joblib` was trained by your own v1.4+ pipeline.

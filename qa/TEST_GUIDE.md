# TraceLens AI v1.2 — One-File Feature Verification Guide

This guide uses **only one submitted file** (`qa/ONE_FILE_FEATURE_TEST.txt`) to exercise the application's matching, explanations, citation signals, review, and PDF report. It requires **three known local reference texts** to measure what the detector finds. They are included and can be loaded with one button. No public web API or GPU is required for the baseline.

**Important:** Never upload `ONE_FILE_FEATURE_TEST.txt` into Reference Library. Doing so causes a self-comparison, often resulting in near-100% exact coverage. Start with a fresh project folder (or remove unrelated local references) for the precise baseline below.

## A. Start the application (Windows PowerShell)

1. Extract the v1.2 ZIP into a **new** folder. Don't put it inside your old running project's folder. Open that folder and type `powershell` in the Explorer address bar.
2. Install dependencies (first run only):

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Start it:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

4. Open `http://localhost:8501`. Keep PowerShell open while the app runs.

## B. Establish the controlled baseline (one uploaded file)

1. Sidebar → **Reference Library** → **One-file verification kit (TraceLens QA)** → **Load QA verification references**.
2. Confirm the library contains exactly: `library_ml.txt`, `library_energy.txt`, `qa_reference_common.txt`. If other references exist, the expected numbers below may differ.
3. Sidebar → **Analyze** → **Upload a file** → select `qa/ONE_FILE_FEATURE_TEST.txt`.
4. Settings: ✅ Local reference library; ❌ Public web search; ❌ Semantic AI (MiniLM); ❌ Experimental writing patterns. Leave direct URL empty. Leave advanced thresholds unchanged (lexical 0.32, semantic 0.70), and identical-reference exclusion off.
5. Click **Run document analysis**.

**Expected baseline, using the bundled unmodified fixture and three references:**

| Metric | Expected |
| --- | ---: |
| Exact matched-word coverage | **26.07%** |
| Source passages flagged | **6** |
| References checked | **3** |
| Word count in Results | **280** |
| Semantic AI status in Results | **NOT RUN** (deliberately off) |

These deterministic values were verified against the bundled code. A mismatch may indicate additional references, changed test content, altered thresholds, or different software versions; it does not necessarily prove a malfunction.

## C. Verify EACH feature in the same analyzed file

Open **Results & Review**. Use the evidence-type filter and expand matches.

| Feature | Where to check | What counts as a pass |
| --- | --- | --- |
| Document upload | Analyze | TXT accepted; analysis starts |
| Text extraction | Results | Word count is 280; recognizable sentences shown |
| Preprocessing | Results | Correct sentence segments; no offset drift |
| Local reference library | Results | 3 references checked |
| Exact-match detection | Evidence list | Two or more identical passages from `library_ml.txt` and `library_energy.txt` |
| TF-IDF | Expand a match | Numerical TF-IDF cosine, e.g. 1.000 for verbatim text |
| False-positive filtering | Filter **Low-information overlap** | `Machine learning is a subset of artificial intelligence.` appears separately and is excluded from headline coverage |
| Source identification | Expand a match | Correct local source file is named |
| Highlighting | Source comparison panels | Actual aligned words highlighted, not unrelated characters |
| Side-by-side comparison | Expand a match | Uploaded and reference passages visible together |
| Citation detection | Matching neural-network quotation | Explanation notes quotation/citation; this is a fake example citation and should **not** be considered verified attribution |
| Unrelated content | Find TEST F in the input | `cobalt umbrellas` should not be flagged as a matching passage |
| Human review | Expand match → reviewer decision | Select `Needs investigation`, add note, **Save review**, reopen result; decision should persist |
| PDF report | Bottom of Results | **Download evidence report (PDF)** opens with correct sources, scores, and saved reviewer note |
| Analysis history | Results → Analysis history | Saved analysis can be selected again |
| Model comparison | Sidebar → Model Comparison | TF-IDF scores display; **Run toy evaluation** returns educational metrics |
| Error handling | Analyze → Paste text | Submitting empty text produces a useful error rather than a crash |
| Delete history | Bottom of Results | Optional: with confirmation, saved analysis deletes; **avoid** doing this until other checks are completed |

The file also contains **TEST D**, a paraphrase that helps inspect semantic retrieval. Exact paragraph classification is not guaranteed until the MiniLM model is installed and the thresholds are validated.

## D. Verify that the transformer AI ACTUALLY RUNS

1. In a **second PowerShell window** in the project folder, install:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
   ```

2. Restart Streamlit (Ctrl+C, then the startup command) so it loads newly installed packages.
3. Analyze the **same** `qa/ONE_FILE_FEATURE_TEST.txt` again, with **Semantic AI (MiniLM, if installed)** checked. Keep local references on; keep web off.
4. Open Results & Review → **Semantic AI: ACTIVE (MiniLM executed)**. Expanded matches should show numerical **MiniLM cosine** scores, not `Not run`.
5. Open **Model Comparison**. Compare TF-IDF and MiniLM scores. The paraphrase in TEST D may be found as a semantic candidate; whether it is flagged depends on model outputs and the configured threshold.
6. Verify by command line (see section G): `python qa/self_check.py --with-ai` using the virtual environment.

**A successful TF-IDF test does not prove the neural model ran.** If a warning says the semantic model could not run, inspect the full PowerShell error and check available disk/RAM and the model download.

## E. Online web search and direct URL (LIVE, independent checks)

1. Copy `.env.example` to `.env`, put `TAVILY_API_KEY=YOUR_KEY` in it, then restart the app. Never share the key publicly.
2. Analyze the same file with **Public web search** on, and check the external-sharing consent box. This fixture is synthetic and mostly not published online, so **zero web matches is a valid outcome**.
3. Confirm the result's **retrieved web sources** count or warnings. For a positive live-source test, use a separate public source whose actual text is known and permission to fetch is granted. Copy a distinctive sentence into your test document, or use the direct URL field to compare an accessible public page. Never equate a search snippet with checked webpage content.
4. To test **direct URL**, paste a real publicly accessible HTML article URL in the direct URL field (while enabled/consented). Some sites block automated extraction; a clear retrieval warning is expected in that case, not a 'clean internet' conclusion.
5. Remove the API key from screenshots. Tavily calls may consume credits.

**No offline script can prove a live Tavily request or a real webpage fetch worked.** The automated QA script includes a **mocked** web path and a failure-fallback test, both explicitly labeled as simulations.

## F. Experimental AI-generated writing detection

Toggle **Experimental writing pattern analysis** while analyzing the same file. Without a separately trained, documented classifier, the expected result is an **unavailable** notice. That is correct behavior, not a bug. Do not expect an AI-generated percentage from the provided ZIP. Training would need a properly labeled dataset and held-out evaluation, especially false-positive tests on human writing.

## G. Run the automatic test suite

In PowerShell, in the extracted project folder:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe qa\self_check.py
```

The second script writes:

- `qa/output/feature_test_results.json` — detailed PASS / FAIL / NOT TESTED results
- `qa/output/feature_evidence_report.pdf` — generated two-page example evidence report

After installing MiniLM:

```powershell
.\.venv\Scripts\python.exe qa\self_check.py --with-ai
```

The `--with-ai` check fails intentionally if semantic inference cannot run.

## H. How to interpret the test outcome

- **PASS:** The specified automated behavior was observed in a controlled sample.
- **FAIL:** An expected invariant was not met; inspect the reported feature and the console traceback.
- **NOT TESTED:** A live service, model, or validated classifier was unavailable or not requested.
- **MANUAL:** Human observation in the Streamlit interface is still needed.
- **NOT SUPPORTED:** Not implemented, e.g. scanned-PDF OCR in this version.

**A complete acceptance check requires** the baseline tests, the AI-enabled run, UI button checks, and a separate internet-connected web test. Do not report 100% verified just because the offline suite passes.


## Live Wikipedia source check (v1.3, free and no API key)

- Enter your own contact email or HTTPS project URL under **Wikimedia contact** in the Analyze screen (or set WIKIMEDIA_CONTACT in `.env`). This is an identifying User-Agent, not a secret API key.
- Choose **Wikipedia (free, no key)** and opt in to sharing your selected query phrases. Run the QA fixture: Wikipedia may find no matching articles because the fixture is synthetic.
- For a positive direct-source test, paste a distinctive paragraph from https://en.wikipedia.org/wiki/Machine_learning into TraceLens, and paste the same URL in the direct-URL field. Verify that Retrieved web sources > 0 and that a source passage is actually shown.
- Run `python qa/wiki_live_check.py` for a diagnostic without opening Streamlit.
- A failed network call should produce a warning; it is not evidence that the document is original. Live Wikimedia connectivity has not been verified by the packaged offline tests.

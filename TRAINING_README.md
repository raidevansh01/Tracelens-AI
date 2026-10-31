# TraceLens AI — v2 ChatGPT-style writing research training kit

This is an **add-on for the Stage-10 TraceLens AI v1.8.1 project**. It changes **neither the plagiarism engine nor the working SQLite database**. It deliberately saves new AI-writing models under `storage/v2_experiments/`, so the existing app keeps its present classifiers until you explicitly select a replacement.

## What this trains

- **N-gram candidate**: character- and word-level TF-IDF with Logistic Regression.
- **MiniLM candidate**: pretrained `all-MiniLM-L6-v2` embeddings (CPU-compatible) with Logistic Regression. MiniLM provides text representations; the separate Logistic Regression layer is what learns the AI-vs-human labels.
- Both candidates are trained from `datasets/ai_v2_train.csv`, formed from balanced labeled **TRAIN-split** essays plus your existing HC3 human/older-ChatGPT pairs. The dataset is labeled by external publishers; model identity and author identity are not fully verifiable.
- Both use document-group-held-out validation and an observed 5% human-review false-positive *target*. That is not a promised population false-positive rate or an accuracy result.

**Do not put `datasets/independent_test.csv` into training, even if it improves recall.** The currently retained 120 test texts come from the **TEST split of the same upstream essay dataset**, so they are split-held-out but *not domain-independent*. For an honest ChatGPT-specific assessment also test new ChatGPT outputs and truly human texts produced outside these corpora.

## Setup (run inside `C:\Code\Plagrism\Stage-10\TraceLens-AI`)

1. Copy the five `.py` files and one `.ps1` file from the ZIP **beside** `app.py`. Do not copy the README into the `.venv` folder.
2. **Back up your project folder and trained models.** Keep `.env`, SQLite, and previous models untouched.
3. Install optional MiniLM dependencies if necessary:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
```

4. Ensure the frozen independent test data and older HC3 data are present:

```powershell
Test-Path .\datasets\independent_test.csv
Test-Path .\datasets\ai_writing.csv
```

Both should be `True`. If the second file is absent, the builder still trains from the new essay dataset, but without the ChatGPT-labeled HC3 supplement.

5. Download the public TRAIN split sample and train/evaluate both candidate models:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_V2_TRAINING.ps1
```

The process accesses a public Hugging Face dataset endpoint and needs internet access. The installer does **not** need OpenAI or Tavily API keys and does not automatically contact ChatGPT. It may take some time on a CPU-only PC.

To train just the faster n-gram candidate first:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_V2_TRAINING.ps1 -Method ngram
```

To skip a *previously successful* dataset download on a later run:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_V2_TRAINING.ps1 -SkipDownload -Method minilm
```

## Read the results

```powershell
Get-Content .\storage\v2_experiments\training_report.json
Get-Content .\datasets\evaluation_reports\v2_external_comparison.json
.\.venv\Scripts\python.exe score_one_v2.py --file TEST_AI_GENERATED.txt
```

Evaluate the **human false-positive rate**, **AI recall**, and scores on fresh ChatGPT-generated and genuinely human texts. Training-set and validation scores are not enough.

If the results are acceptable and you *want* the model visible in the existing Streamlit **experimental model comparison** section, preview:

```powershell
.\.venv\Scripts\python.exe promote_v2_candidate.py --method minilm
```

Then explicitly install **only if the observed external false-positive rate is at most 10%**:

```powershell
.\.venv\Scripts\python.exe promote_v2_candidate.py --method minilm --install
```

The script backs up the old `storage/ai_minilm_candidate.joblib`. It never replaces `storage/ai_pattern_model.joblib`, the original seven-feature classifier. Restart Streamlit to see the new candidate.

If the benchmark FPR exceeds 10%, **do not install it as the recommended detector**; keep it as a research experiment.

## Suggested acceptance criteria for a college demonstration

These are aspirational **evaluation criteria**, not promises:

- Controlled and live source-plagiarism workflows remain unchanged and pass existing tests.
- On fresh human-written and ChatGPT-generated academic texts not used for training or threshold selection, aim for **human false-positive rate ≤5%**, with **AI recall ≥75%** under the same threshold. If the trade-off is worse, report the measured limitation.
- Have at least 40+ samples per class in an initial evaluation; more are needed for narrow confidence intervals and for subgroups such as non-native English writers.
- Test raw ChatGPT output, edited ChatGPT text, short inputs, non-native English writing, and subjects beyond essays. Record separate subgroup metrics.
- Show "experimental / inconclusive" rather than stating that AI authorship is proven.

## Technical limitations

- The train and existing 120-document test subsets are from the same publisher's upstream essay dataset, with limited provenance. The test split must remain frozen to avoid repeated tuning to it.
- There can be near-duplicate essays or shared prompts across splits despite exact and first-45-word overlap checks. A second independent corpus and fresh, human-verified ChatGPT examples are essential before making broader claims.
- The downloaded samples are not guaranteed to come from recent ChatGPT versions. HC3 covers an older ChatGPT writing style. This package **cannot guarantee** reliably detecting current ChatGPT output, even after a successful training run.
- This is text classification, not author identification and not a misconduct decision. Never use the model's score as sole proof.

Source dataset: <https://huggingface.co/datasets/andythetechnerd03/AI-human-text> (Apache-2.0, according to its dataset card). Historic ChatGPT dataset: <https://huggingface.co/datasets/Hello-SimpleAI/HC3> (CC BY-SA 4.0, according to its dataset card). Inspect redistribution license obligations when sharing training data.

## Optional offline regression checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_v2_training_kit.py
```

These tests use synthetic sample texts only, not real AI detector accuracy.

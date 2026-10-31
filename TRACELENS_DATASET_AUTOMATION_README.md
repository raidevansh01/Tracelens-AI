# TraceLens Stage-10 automatic external benchmark

## Purpose

Prepare an independent-*source* evaluation CSV without having to create text files manually. The sample comes from the public **test split** of [`andythetechnerd03/AI-human-text`](https://huggingface.co/datasets/andythetechnerd03/AI-human-text), listed as Apache-2.0 on the publisher's dataset card. The upstream source is a processed essay dataset, not the HC3 dataset used for TraceLens's initial training. The upstream label field `generated` uses `0` for human and `1` for AI.

This is **evaluation-only**. Do not train on `datasets/independent_test.csv` or use it to select thresholds. Labels may contain errors, source splits may share unseen material, and dataset performance does not imply performance on real student writing or recent models. The script blocks exact normalized duplicates of your current `datasets/ai_writing.csv` and near-identical prefixes within the downloaded test sample. This **does not prove complete leakage avoidance**.

## Installation and usage

1. Extract `prepare_tracelens_external_benchmark.py` and `RUN_DATASET_EVALUATION.ps1` into `C:\Code\Plagrism\Stage-10\TraceLens-AI` (beside `app.py`).
2. Open **PowerShell** and run:

```powershell
cd C:\Code\Plagrism\Stage-10\TraceLens-AI
Write-Host "Project folder:" (Get-Location)
Write-Host "Dataset script present:" (Test-Path .\prepare_tracelens_external_benchmark.py)
Write-Host "Existing training data:" (Test-Path .\datasets\ai_writing.csv)
powershell -ExecutionPolicy Bypass -File .\RUN_DATASET_EVALUATION.ps1
```

The runner automatically downloads a **balanced 120-document** external benchmark, checks local training overlap, saves a dataset provenance file, preserves existing models, trains missing models if necessary, evaluates all three installed models, and saves reports. Model training uses `ai_writing.csv` only; independent evaluation uses `independent_test.csv` only.

If you only want the dataset and wish to run model evaluation later:

```powershell
.\.venv\Scripts\python.exe prepare_tracelens_external_benchmark.py --per-class 60
```

Files created:

- `datasets/independent_test.csv`: `text,label,group_id`, with 60 examples per class by default
- `datasets/independent_test_provenance.json`: dataset/source/split, sampling seed, counts, rejection reasons, checksum, limitations
- `datasets/evaluation_reports/external_{stylometry,ngram,minilm}.txt`: available models' metrics when using the runner

## Troubleshooting

- `Test-Path` normally prints `True` or `False`. If PowerShell shows `>>` and nothing runs, press `Ctrl+C` to return to a `PS ...>` prompt. Commands containing `| Out-Null` hide output on purpose.
- No internet / 403 / 429: open `https://huggingface.co/datasets/andythetechnerd03/AI-human-text` in a browser. Check network access to `https://datasets-server.huggingface.co`, and retry if rate limited. No API key required for this public dataset.
- `datasets/ai_writing.csv` missing: copy the **training** CSV from Stage-9. Never substitute `independent_test.csv`.
- Not enough examples due to filtering: `python prepare_tracelens_external_benchmark.py --max-pages 120`.
- Existing benchmark: runner reuses it to keep evaluation stable unless `-RebuildDataset` is supplied.
- Slow optional MiniLM training: CPU inference can be resource intensive; use `-SkipModelTraining` to evaluate only existing models.
- Never publish actual training data or model files without checking permission and provenance.

Dataset API: https://huggingface.co/docs/dataset-viewer/rows

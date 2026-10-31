param(
  [int]$PerClass = 1000,
  [ValidateSet('ngram','minilm','both')][string]$Method = 'both',
  [switch]$SkipDownload
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw 'Missing .venv Python; install requirements.txt and requirements-ai.txt first.' }
if (!(Test-Path '.\datasets\independent_test.csv')) { throw 'Missing protected external benchmark: datasets\independent_test.csv.' }
if (-not $SkipDownload) {
  Write-Host 'Preparing a NEW training set from the publisher TRAIN split; protected TEST split will not be used.'
  & $python 'prepare_v2_training.py' '--per-class' $PerClass
  if ($LASTEXITCODE -ne 0) { throw 'New training data preparation failed. Existing models are untouched.' }
} else {
  if (!(Test-Path '.\datasets\ai_v2_train.csv')) { throw 'Missing ai_v2_train.csv. Do not use -SkipDownload before preparing it.' }
}
Write-Host "Training isolated $Method candidates (not replacing your original models) ..."
& $python 'train_v2_candidates.py' '--method' $Method
if ($LASTEXITCODE -ne 0) { throw 'Model training failed. Existing models are untouched.' }
Write-Host 'Running one external benchmark evaluation, without changing thresholds ...'
& $python 'evaluate_v2_candidates.py' '--method' $Method
if ($LASTEXITCODE -ne 0) { throw 'External evaluation failed. Inspect the error; do not promote the candidate.' }
Write-Host 'Checking the known ChatGPT example ...'
& $python 'score_one_v2.py' '--file' 'TEST_AI_GENERATED.txt'
if ($LASTEXITCODE -ne 0) { throw 'One-document smoke check failed.' }
Write-Host 'Completed research training. Review reports BEFORE any optional manual installation.'

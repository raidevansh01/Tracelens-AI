# Stage-10 TraceLens: external dataset preparation + three model evaluations.
# Run this script from the TraceLens project directory after extracting it there.
param(
    [int]$PerClass = 60,
    [switch]$SkipModelTraining,
    [switch]$RebuildDataset
)
$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$dataset = Join-Path $root 'datasets\independent_test.csv'
$train = Join-Path $root 'datasets\ai_writing.csv'

if (-not (Test-Path (Join-Path $root 'app.py'))) { throw 'Run from C:\Code\Plagrism\Stage-10\TraceLens-AI (must contain app.py).' }
if (-not (Test-Path $python)) { throw 'Virtual environment missing. Create .venv first.' }
if (-not (Test-Path $train)) { throw 'datasets\ai_writing.csv is missing. Copy the existing HC3 TRAINING CSV from Stage-9; do not use the external benchmark as training data.' }
if (-not (Test-Path (Join-Path $root 'prepare_tracelens_external_benchmark.py'))) { throw 'Dataset preparation script missing. Extract all files from the kit to your Stage-10 project folder.' }

if ((Test-Path $dataset) -and -not $RebuildDataset) {
    Write-Host "Using existing held-out test dataset: $dataset" -ForegroundColor Yellow
    Write-Host 'Use -RebuildDataset only if you deliberately want to make a new benchmark.'
} else {
    Write-Host 'Downloading a balanced public TEST split for external evaluation ...' -ForegroundColor Cyan
    & $python 'prepare_tracelens_external_benchmark.py' --per-class $PerClass
    if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed. See the detailed message above.' }
}
if (-not (Test-Path $dataset)) { throw 'Dataset still missing. Cannot evaluate.' }

$models = @(
    @{ Name='stylometry'; Path='storage\ai_pattern_model.joblib'; Trainer='train_ai_patterns.py'; Method='' },
    @{ Name='ngram'; Path='storage\ai_ngram_candidate.joblib'; Trainer='train_ai_comparator.py'; Method='ngram' },
    @{ Name='minilm'; Path='storage\ai_minilm_candidate.joblib'; Trainer='train_ai_comparator.py'; Method='minilm' }
)

$reportDir = Join-Path $root 'datasets\evaluation_reports'
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
foreach ($m in $models) {
    $modelFile = Join-Path $root $m.Path
    if (-not (Test-Path $modelFile)) {
        if ($SkipModelTraining) {
            Write-Host "SKIP $($m.Name): model file absent: $($m.Path)" -ForegroundColor Yellow
            continue
        }
        Write-Host "Training MISSING $($m.Name) model from HC3 TRAINING data only ..." -ForegroundColor Cyan
        if ($m.Name -eq 'stylometry') {
            & $python $m.Trainer --data 'datasets\ai_writing.csv'
        } else {
            & $python $m.Trainer --data 'datasets\ai_writing.csv' --method $m.Method
        }
        if ($LASTEXITCODE -ne 0) { Write-Warning "Failed training $($m.Name); continuing to other models."; continue }
    }
    Write-Host "Evaluating $($m.Name) against the external held-out dataset ..." -ForegroundColor Cyan
    $log = Join-Path $reportDir ("external_" + $m.Name + ".txt")
    & $python 'qa\evaluate_external_writing.py' --data 'datasets\independent_test.csv' --training-data 'datasets\ai_writing.csv' --model $m.Name | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { Write-Warning "Evaluation FAILED for $($m.Name), see $log" }
    else { Write-Host "Saved results: $log" -ForegroundColor Green }
}
Write-Host "Completed the available evaluations. Reports: $reportDir" -ForegroundColor Green
Write-Host 'These results characterize this benchmark only; predictions are NOT proof of human/AI authorship.' -ForegroundColor Yellow

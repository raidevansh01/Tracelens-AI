param(
    [string]$ProjectDir = "C:\Code\Plagrism\Stage-10\TraceLens-AI"
)
$ErrorActionPreference = 'Stop'
$patchRoot = $PSScriptRoot
$project = (Resolve-Path $ProjectDir).Path
foreach ($relative in @('db\store.py','qa\self_check.py')) {
    $source = Join-Path $patchRoot $relative
    $target = Join-Path $project $relative
    if (-not (Test-Path $source)) { throw "Missing patch file: $source" }
    if (-not (Test-Path $target)) { throw "Missing project file: $target" }
}
$backupDir = Join-Path $project ("hotfix_backup_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
foreach ($relative in @('db\store.py','qa\self_check.py')) {
    $target = Join-Path $project $relative
    $backup = Join-Path $backupDir ($relative -replace '\\', '_')
    Copy-Item $target $backup -Force
    Copy-Item (Join-Path $patchRoot $relative) $target -Force
    Write-Host "Patched $relative"
}
$testsFolder = Join-Path $project 'tests'
New-Item -ItemType Directory -Path $testsFolder -Force | Out-Null
Copy-Item (Join-Path $patchRoot 'tests\test_sqlite_lifecycle.py') (Join-Path $testsFolder 'test_sqlite_lifecycle.py') -Force
Write-Host "Backups: $backupDir"
Write-Host 'Patch applied. Run:'
Write-Host "cd `"$project`""
Write-Host '.\.venv\Scripts\python.exe -m pytest -q'
Write-Host '.\.venv\Scripts\python.exe qa\self_check.py'
Write-Host '.\.venv\Scripts\python.exe qa\self_check.py --with-ai'

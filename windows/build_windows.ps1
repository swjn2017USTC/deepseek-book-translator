$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

py -3.11 -m venv .venv-build
& .\.venv-build\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-build\Scripts\python.exe -m pip install `
  -e .\chapter-structure-recovery-lab `
  -e .\structured-book-translation-pipeline `
  "pyinstaller>=6.10,<7" `
  pytest
& .\.venv-build\Scripts\python.exe -m PyInstaller --clean --noconfirm .\windows\DeepSeekBookTranslator.spec
$env:DEEPSEEK_SELF_TEST_REPORT = "$Repo\self-test-report.json"
$Process = Start-Process `
  -FilePath "$Repo\dist\DeepSeekBookTranslator.exe" `
  -ArgumentList "--self-test" `
  -Wait `
  -PassThru
Get-Content "$Repo\self-test-report.json"
Remove-Item "$Repo\self-test-report.json"
if ($Process.ExitCode -ne 0) { exit $Process.ExitCode }

Write-Host "Built: $Repo\dist\DeepSeekBookTranslator.exe"

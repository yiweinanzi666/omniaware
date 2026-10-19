$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$env:PYTHONPATH = Join-Path $root 'tools'
$env:PYTHONUTF8 = '1'
$env:OMNIAWARE_ASSETS = Join-Path $root 'assets'
$env:OMNIAWARE_CACHE = Join-Path $root 'cache'
$env:OMNIAWARE_UNIVERSALRAG_DIR = Join-Path $root 'omniaware\external\UniversalRAG'
$python = (Get-Command python.exe).Source
$jobs = @()
foreach ($name in @('okvqa','hybridqa','gs','videorag')) {
    $script = Join-Path $root "download_$name.py"
    $jobs += Start-Process -FilePath $python -ArgumentList @('-u', $script) -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $root "logs\resume_$name.log") -RedirectStandardError (Join-Path $root "logs\resume_$name.errors.log")
}
$jobs += Start-Process -FilePath $python -ArgumentList @('-u',(Join-Path $root 'download_dpr.py')) -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $root 'logs\resume_dpr.log') -RedirectStandardError (Join-Path $root 'logs\resume_dpr.errors.log')
$jobs | Wait-Process
& $python -u (Join-Path $root 'finalize_downloads.py') *> (Join-Path $root 'finalize_batch.log')
Write-Output (Join-Path $root '下载报告.md')


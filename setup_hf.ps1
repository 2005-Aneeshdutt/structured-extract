<#
.SYNOPSIS
    One-shot: save the HF token, detect the username from it, push the dataset.

Written as a script rather than a sequence of pasted commands because the
sequence had three separate failure points -- a token that silently did not
paste, a venv path that differs from the repo root, and a username that had to
be typed correctly in two places. Each one failed quietly and looked like a
different problem. Here the token is validated before anything else runs, the
interpreter is located rather than assumed, and the username is read from the
token instead of typed.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Repo = $PSScriptRoot
Set-Location $Repo

# --- locate the interpreter (repo, then parent -- the venv lives one level up)
$Py = $null
foreach ($c in @(
    (Join-Path $Repo ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path $Repo -Parent) ".venv\Scripts\python.exe")
)) { if (Test-Path $c) { $Py = $c; break } }
if (-not $Py) { $Py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $Py) { throw "No Python found." }
Write-Host "python: $Py" -ForegroundColor DarkGray

# --- token -----------------------------------------------------------------
Write-Host ""
Write-Host "Paste your HuggingFace token, then press Enter." -ForegroundColor Cyan
Write-Host "  Get one at https://huggingface.co/settings/tokens (type: Write)"
Write-Host "  NOTE: Ctrl+V may do nothing in this console. RIGHT-CLICK to paste." -ForegroundColor Yellow
$token = Read-Host "token"

if ([string]::IsNullOrWhiteSpace($token)) {
    throw "Nothing was pasted. Right-click pastes in PowerShell, not Ctrl+V."
}
$token = $token.Trim()
if (-not $token.StartsWith("hf_")) {
    throw "That does not look like an HF token (expected it to start with 'hf_', got '$($token.Substring(0,[Math]::Min(6,$token.Length)))...')."
}

# --- verify the token and READ THE USERNAME FROM IT ------------------------
# Deriving the username removes the last thing that had to be typed correctly
# in two places; a wrong one only shows up later as a 403 on push.
$env:HF_TOKEN = $token
$who = & $Py -c @"
import os, sys
from huggingface_hub import HfApi
try:
    info = HfApi(token=os.environ['HF_TOKEN']).whoami()
except Exception as e:
    print('ERR:' + str(e)[:200]); sys.exit(1)
print(info['name'])
"@
if ($LASTEXITCODE -ne 0 -or $who -like "ERR:*") { throw "Token rejected by HuggingFace: $who" }
$user = $who.Trim()
Write-Host "authenticated as: $user" -ForegroundColor Green

# --- persist to .env -------------------------------------------------------
$envPath = Join-Path $Repo ".env"
(Get-Content $envPath) `
    -replace '^HF_TOKEN=.*', "HF_TOKEN=$token" `
    -replace '^HF_USER=.*',  "HF_USER=$user" |
    Set-Content $envPath -Encoding utf8
Write-Host "saved HF_TOKEN and HF_USER to .env" -ForegroundColor Green

# --- push ------------------------------------------------------------------
$repoId = "$user/structured-extract-jobs"
Write-Host ""
Write-Host "pushing dataset -> $repoId" -ForegroundColor Cyan
$env:PYTHONPATH = $Repo
& $Py -m data.prepare_dataset `
    --in data/interim/labeled.jsonl `
    --gold-from data/interim/labeled_gold.jsonl `
    --out data/processed `
    --push-to-hub $repoId
if ($LASTEXITCODE -ne 0) { throw "push failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "DONE. On Kaggle, replace the data cell with:" -ForegroundColor Green
Write-Host ""
Write-Host "    from datasets import load_dataset" -ForegroundColor White
Write-Host "    ds = load_dataset('$repoId')" -ForegroundColor White
Write-Host "    ds.save_to_disk('/kaggle/working/structured-extract/data/processed/hf')" -ForegroundColor White
Write-Host "    print(ds)" -ForegroundColor White
Write-Host ""

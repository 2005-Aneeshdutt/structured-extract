<#
.SYNOPSIS
    Windows equivalent of the Makefile. Same target names, same commands.

.DESCRIPTION
    `make` does not ship with Windows, so every `make <target>` in the README has
    a `.\run.ps1 <target>` counterpart here. The Makefile stays authoritative for
    Linux, Kaggle and CI; this file mirrors it for local Windows work.

    If the two ever drift, the Makefile is correct.

.EXAMPLE
    .\run.ps1 help
    .\run.ps1 label-gold
    .\run.ps1 smoke
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = "help",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

# NOTE: this file is deliberately ASCII-only. Windows PowerShell 5.1 reads .ps1
# files as ANSI unless they carry a UTF-8 BOM, so a stray em dash or arrow turns
# into mojibake and produces a parse error that points at the wrong line.

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

# Prefer a project venv. Checked in the repo first, then one level up -- the repo
# is often cloned into a workspace folder that already owns the venv, and
# silently falling back to system Python means missing packages later.
$Py = $null
foreach ($candidate in @(
    (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path $RepoRoot -Parent) ".venv\Scripts\python.exe")
)) {
    if (Test-Path $candidate) { $Py = $candidate; break }
}
if (-not $Py) {
    $Py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $Py) { throw "No Python found. Create a venv: python -m venv .venv" }
    Write-Host "note: no .venv found, falling back to $Py" -ForegroundColor Yellow
}

# data/ and eval/ are imported as packages, so the repo root must be importable.
$env:PYTHONPATH = $RepoRoot

$DataDir = "data/processed"
$Adapter = "outputs/qwen2.5-1.5b-r16-a32/adapter"
$Gguf    = "models/qwen2.5-1.5b-r16-a32-Q4_K_M.gguf"
$Pred    = "results/raw_predictions"

function Invoke-Step {
    param([string]$Description, [string[]]$Arguments)
    Write-Host "`n> $Description" -ForegroundColor Cyan

    # Python's logging module writes to stderr by design, and Windows PowerShell
    # 5.1 turns any native-command stderr into a terminating NativeCommandError
    # while $ErrorActionPreference is "Stop". Progress logs would therefore look
    # like crashes. Exit code is the only trustworthy success signal for a native
    # process, so relax the preference around the call and judge by that.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Py @Arguments
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): $Description" }
}

switch ($Target) {

    "help" {
        Write-Host @"

  structured-extract - Windows task runner

  SETUP
    install        Install local dependencies
    test           Run the test suite
    lint           Run ruff
    smoke          Full data pipeline on 120 postings, mock teacher, no API key

  DATA  (needs GOOGLE_API_KEY in .env - run these in order)
    label-gold     PHASE 1: held-out labels, 3-sample self-consistency vote
    label-bulk     PHASE 2: training labels, single pass, skips phase-1 postings
                   Both accept extra flags, e.g.
                     .\run.ps1 label-gold --teacher openrouter --requests-per-day 1000
    prepare        Clean, split, format for SFT, export HF + JSONL

  EVALUATION  (needs a trained adapter)
    eval-all       All four arms on the held-out test split
    eval-ablation  Rank 8/16/32 on the validation split
    robustness     Perturbation suite
    compare        Build results/comparison_table.md
    report         Charts, ablation table, failure analysis

  SHIP
    verify         GGUF vs fp16-adapter regression check
    app            Run the Gradio demo locally

  Training and quantization run on Kaggle - see training/KAGGLE.md

"@ -ForegroundColor White
    }

    "install"  { Invoke-Step "installing dependencies" @("-m", "pip", "install", "-r", "requirements.txt") }
    "test"     { Invoke-Step "running tests" @("-m", "pytest", "tests/", "-q") }
    "lint"     { Invoke-Step "linting" @("-m", "ruff", "check", ".") }

    # 250 postings, not 120: prepare_dataset refuses to build splits with fewer
    # than 100 training examples left after test+val, and a smoke test that trips
    # that guard tests nothing.
    "smoke" {
        Invoke-Step "labeling 250 postings with the mock teacher" @(
            "-m", "data.generate_synthetic", "--n", "250", "--max-corpus-rows", "4000",
            "--teacher", "mock", "--out", "data/interim/smoke_labeled.jsonl",
            "--cache", "data/interim/smoke_cache.jsonl", "--audit-out", "results/label_audit_smoke.md")
        Invoke-Step "preparing splits" @(
            "-m", "data.prepare_dataset", "--in", "data/interim/smoke_labeled.jsonl",
            "--out", "data/processed_smoke", "--stats-out", "results/dataset_stats_smoke.md",
            "--test-size", "25", "--val-size", "25")
        Write-Host "`nSmoke test passed. The pipeline works end to end." -ForegroundColor Green
    }

    # -- Phase 1. Run FIRST: phase 2 excludes these postings by id. ------------
    # Extra arguments are appended after the defaults, and argparse honours the
    # LAST occurrence of an option, so anything here can be overridden ad hoc:
    #     .\run.ps1 label-gold --teacher openrouter --requests-per-day 1000
    # --teacher is NOT hardcoded here: it resolves from $TEACHER in .env, so a
    # run that spans days does not depend on remembering a flag each time.
    "label-gold" {
        Invoke-Step "PHASE 1 - held-out labels (3-sample vote)" (@(
            "-m", "data.generate_synthetic", "--n", "1000",
            "--n-samples", "3", "--out", "data/interim/labeled_gold.jsonl",
            "--audit-out", "results/label_audit.md") + $Rest)
        Write-Host "`nPhase 1 chunk done. Re-run this same command to continue if it stopped early." -ForegroundColor Green
    }

    # --n is the CORPUS SLICE, not the number of new labels: phase 1's postings
    # are inside it and get skipped by --exclude-from. 4225 - 1000 = 3225 new.
    "label-bulk" {
        Invoke-Step "PHASE 2 - training labels (single pass)" (@(
            "-m", "data.generate_synthetic", "--n", "4225",
            "--n-samples", "1", "--exclude-from", "data/interim/labeled_gold.jsonl",
            "--out", "data/interim/labeled.jsonl",
            "--audit-out", "results/label_audit_bulk.md") + $Rest)
        Write-Host "`nPhase 2 chunk done. Re-run this same command to continue if it stopped early." -ForegroundColor Green
    }

    "prepare" {
        Invoke-Step "preparing dataset" @(
            "-m", "data.prepare_dataset", "--in", "data/interim/labeled.jsonl",
            "--gold-from", "data/interim/labeled_gold.jsonl", "--out", $DataDir)
    }

    "eval-all" {
        Invoke-Step "base, 0-shot"    @("-m", "eval.run_eval", "--backend", "hf", "--split", "test", "--out", "$Pred/base_0shot.json")
        Invoke-Step "base, 3-shot"    @("-m", "eval.run_eval", "--backend", "hf", "--few-shot", "3", "--split", "test", "--out", "$Pred/base_3shot.json")
        Invoke-Step "fine-tuned"      @("-m", "eval.run_eval", "--backend", "hf", "--adapter", $Adapter, "--split", "test", "--out", "$Pred/finetuned_r16.json")
        Invoke-Step "Gemini ceiling"  @("-m", "eval.run_eval", "--backend", "gemini", "--split", "test", "--out", "$Pred/gemini.json")
    }

    "eval-ablation" {
        foreach ($r in 8, 16, 32) {
            $alpha = $r * 2
            Invoke-Step "rank $r on validation" @(
                "-m", "eval.run_eval", "--backend", "hf",
                "--adapter", "outputs/qwen2.5-1.5b-r$r-a$alpha/adapter",
                "--split", "val", "--out", "$Pred/val_r$r.json")
        }
    }

    "robustness" {
        Invoke-Step "robustness, fine-tuned" @("-m", "eval.robustness_test", "--backend", "hf", "--adapter", $Adapter, "--out", "$Pred/robustness_finetuned.json")
        Invoke-Step "robustness, base"       @("-m", "eval.robustness_test", "--backend", "hf", "--out", "$Pred/robustness_base.json")
    }

    "compare" {
        Invoke-Step "building comparison table" @(
            "-m", "eval.compare_models",
            "--run", "Base 0-shot=$Pred/base_0shot.json",
            "--run", "Base 3-shot=$Pred/base_3shot.json",
            "--run", "LoRA r=16 (ours)=$Pred/finetuned_r16.json",
            "--run", "Gemini 2.0 Flash=$Pred/gemini.json",
            "--baseline", "Base 0-shot", "--ceiling", "Gemini 2.0 Flash", "--ours", "LoRA r=16 (ours)")
    }

    "report" {
        Invoke-Step "building charts and analysis" @(
            "-m", "eval.generate_report",
            "--run", "Base 0-shot=$Pred/base_0shot.json",
            "--run", "Base 3-shot=$Pred/base_3shot.json",
            "--run", "LoRA r=16 (ours)=$Pred/finetuned_r16.json",
            "--run", "Gemini 2.0 Flash=$Pred/gemini.json",
            "--ours", "LoRA r=16 (ours)", "--baseline", "Base 0-shot",
            "--ablation", "r=8=$Pred/val_r8.json",
            "--ablation", "r=16=$Pred/val_r16.json",
            "--ablation", "r=32=$Pred/val_r32.json",
            "--robustness", "$Pred/robustness_finetuned.json",
            "--robustness-baseline", "$Pred/robustness_base.json")
    }

    "verify" { Invoke-Step "verifying quantization" @("-m", "quantize.verify_quantized", "--gguf", $Gguf, "--adapter", $Adapter, "--n", "50") }
    "app"    { Invoke-Step "starting Gradio" @("app/app.py", "--local", "models/") }

    default {
        Write-Host "Unknown target '$Target'. Run '.\run.ps1 help' for the list." -ForegroundColor Red
        exit 1
    }
}

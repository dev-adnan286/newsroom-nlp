<#
.SYNOPSIS
    End-to-end Newsroom NLP demo: corpus -> classifier -> evaluation -> search index -> queries -> topics.
.EXAMPLE
    .\scripts\demo_pipeline.ps1 -Artifacts artifacts -NDocs 2000
#>
param(
    [string]$Artifacts = "artifacts",
    [int]$NDocs = 2000
)

$ErrorActionPreference = "Stop"

$Corpus = Join-Path $Artifacts "corpus.csv"
$Model = Join-Path $Artifacts "classifier.joblib"
$Index = Join-Path $Artifacts "index"
$Reports = Join-Path $Artifacts "reports"

if (-not (Get-Command newsroom -ErrorAction SilentlyContinue)) {
    Write-Error "The 'newsroom' CLI is not on PATH. Install with: pip install -e ."
}

function Invoke-Step([string]$Label, [scriptblock]$Action) {
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Label" }
}

Invoke-Step "[1/7] Generating $NDocs synthetic articles" { newsroom generate-corpus --n-docs $NDocs --output $Corpus }
Invoke-Step "[2/7] Training classifier" { newsroom train --input $Corpus --output $Model --config configs/classifier.yaml }
Invoke-Step "[3/7] Building hybrid search index" { newsroom build-index --input $Corpus --output $Index --config configs/search.yaml }
Invoke-Step "[4/7] Evaluating classifier and retrieval" { newsroom evaluate --input $Corpus --model $Model --index $Index --report-dir $Reports }

Invoke-Step "[5/7] Example queries" {
    newsroom search --index $Index "central bank raises interest rates amid inflation" --top-k 5
    newsroom search --index $Index "goalkeeper saves late penalty in cup final" --top-k 5 --mode bm25
    newsroom search --index $Index "telescope observes exoplanet atmosphere" --top-k 5 --mode vector
}

$FirstText = python -c "import pandas as pd,sys; print(pd.read_csv(sys.argv[1])['text'].iloc[0])" $Corpus
Invoke-Step "[6/7] Keywords and summary for the first article" {
    newsroom keywords "$FirstText" --top-k 6
    newsroom summarize "$FirstText" --sentences 2
}

Invoke-Step "[7/7] Topic clustering and near-duplicate scan" {
    newsroom cluster --input $Corpus --output (Join-Path $Reports "clusters.json")
    newsroom dedup --input $Corpus --threshold 0.6 --limit 5
}

Write-Host ""
Write-Host "Done. Start the API with: newsroom serve --artifacts $Artifacts" -ForegroundColor Green

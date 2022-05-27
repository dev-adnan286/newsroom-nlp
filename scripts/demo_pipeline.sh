#!/usr/bin/env bash
# End-to-end demo: corpus -> classifier -> evaluation -> search index -> queries -> topics.
# Usage: scripts/demo_pipeline.sh [artifacts_dir] [n_docs]
set -euo pipefail

ARTIFACTS="${1:-artifacts}"
N_DOCS="${2:-2000}"
CORPUS="$ARTIFACTS/corpus.csv"
MODEL="$ARTIFACTS/classifier.joblib"
INDEX="$ARTIFACTS/index"
REPORTS="$ARTIFACTS/reports"

if ! command -v newsroom >/dev/null 2>&1; then
  echo "The 'newsroom' CLI is not on PATH. Install with: pip install -e ." >&2
  exit 1
fi

echo "==> [1/7] Generating $N_DOCS synthetic articles"
newsroom generate-corpus --n-docs "$N_DOCS" --output "$CORPUS"

echo "==> [2/7] Training classifier (configs/classifier.yaml)"
newsroom train --input "$CORPUS" --output "$MODEL" --config configs/classifier.yaml

echo "==> [3/7] Building hybrid search index (configs/search.yaml)"
newsroom build-index --input "$CORPUS" --output "$INDEX" --config configs/search.yaml

echo "==> [4/7] Evaluating classifier + retrieval, writing reports to $REPORTS"
newsroom evaluate --input "$CORPUS" --model "$MODEL" --index "$INDEX" --report-dir "$REPORTS"

echo "==> [5/7] Example queries"
newsroom search --index "$INDEX" "central bank raises interest rates amid inflation" --top-k 5
newsroom search --index "$INDEX" "goalkeeper saves late penalty in cup final" --top-k 5 --mode bm25
newsroom search --index "$INDEX" "telescope observes exoplanet atmosphere" --top-k 5 --mode vector

echo "==> [6/7] Keywords and summary for the first article"
FIRST_TEXT="$(python -c "import pandas as pd,sys; print(pd.read_csv(sys.argv[1])['text'].iloc[0])" "$CORPUS")"
newsroom keywords "$FIRST_TEXT" --top-k 6
newsroom summarize "$FIRST_TEXT" --sentences 2

echo "==> [7/7] Topic clustering + near-duplicate scan"
newsroom cluster --input "$CORPUS" --output "$REPORTS/clusters.json"
newsroom dedup --input "$CORPUS" --threshold 0.6 --limit 5

echo
echo "Done. Start the API with: newsroom serve --artifacts $ARTIFACTS"

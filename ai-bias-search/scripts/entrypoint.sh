#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  /app/data/raw \
  /app/data/enriched \
  /app/data/cache \
  /app/results/metrics \
  /app/results/reports \
  /app/results/figs

if [ "$#" -eq 0 ]; then
  exec python -m ai_bias_search.cli
fi

exec "$@"

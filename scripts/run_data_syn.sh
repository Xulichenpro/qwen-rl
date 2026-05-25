#!/usr/bin/env bash
# Launch data synthesis with sane defaults.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/pgroup/pxd-team/miniconda3/etc/profile.d/conda.sh
conda activate qwenrl

# Internal vLLM endpoints must NOT go through SOCKS/HTTP proxy.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

WORKERS="${WORKERS:-3}"
LIMIT="${LIMIT:--1}"
INPUT="${INPUT:-datasets/raw_train/train.json}"
OUTPUT="${OUTPUT:-datasets/syn_train/train_cot.jsonl}"

mkdir -p "$(dirname "$OUTPUT")" outputs/logs

exec python -m src.data_syn.run_synth \
    --input  "$INPUT" \
    --output "$OUTPUT" \
    --workers "$WORKERS" \
    --limit "$LIMIT"

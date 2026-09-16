#!/usr/bin/env bash
# Offline end-to-end demo: Dream-RSI vs Recursive Fixed Exploration.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m dreamrsi.cli compare --config configs/synthetic.yaml "$@"

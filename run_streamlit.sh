#!/usr/bin/env bash
set -euo pipefail

# Configurable via env, with safe defaults
PORT="${PORT:-8888}"
ADDRESS="${ADDRESS:-0.0.0.0}"

exec python -m streamlit run streamlit_app.py \
  --server.port "$PORT" \
  --server.address "$ADDRESS"



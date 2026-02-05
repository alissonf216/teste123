#!/bin/sh

set -e

export PORT="${PORT:-8013}"

echo "Starting Streamlit"
exec streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port ${PORT} \
  --server.headless true \
  --server.enableCORS=false \
  --server.runOnSave true

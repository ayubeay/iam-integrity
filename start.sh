#!/usr/bin/env bash
set -e
INDEX_REQUIRE_ONCHAIN=0 python3 verity_indexer.py || true
exec uvicorn verity_api_gated:app --host 0.0.0.0 --port ${PORT:-8000}

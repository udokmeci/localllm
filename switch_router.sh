#!/bin/bash
# Switch loaded model in router mode via /models/unload + /models/load API
# Usage: bash switch_router.sh <model_id>
#   model_id = section name from models.ini (e.g. Qwen36-35B, Qwen38-27B-Q4)
#
# With --models-max 1, requesting a new model auto-evicts the old one.
# This script does it explicitly so you can see what's happening.

set -e
cd "$(dirname "$0")"

SERVER="http://localhost:8085"
MODEL_ID="$1"

if [ -z "$MODEL_ID" ]; then
    echo "Usage: $0 <model_id> [--unload-only]"
    echo ""
    echo "Available models:"
    curl -s "$SERVER/models" | jq -r '.data[] | "  \(.id)  \(.status)"' 2>/dev/null || echo "  (server not running)"
    exit 1
fi

# Check if --unload-only flag
if [ "$MODEL_ID" = "--unload-only" ]; then
    echo "==> Unloading all loaded models..."
    curl -s "$SERVER/models" | jq -r '.data[] | select(.status == "loaded") | .id' | while read -r m; do
        echo "  Unloading: $m"
        curl -s -X POST "$SERVER/models/unload" -H "Content-Type: application/json" -d "{\"model\":\"$m\"}"
        echo
    done
    echo "==> Done."
    exit 0
fi

# Get currently loaded model
CURRENT=$(curl -s "$SERVER/models" | jq -r '.data[] | select(.status == "loaded") | .id' 2>/dev/null)

if [ -n "$CURRENT" ] && [ "$CURRENT" != "$MODEL_ID" ]; then
    echo "==> Unloading current model: $CURRENT"
    curl -s -X POST "$SERVER/models/unload" -H "Content-Type: application/json" -d "{\"model\":\"$CURRENT\"}"
    sleep 1
fi

echo "==> Loading model: $MODEL_ID"
date +%s > /tmp/llama-last-request
curl -s -X POST "$SERVER/models/load" -H "Content-Type: application/json" -d "{\"model\":\"$MODEL_ID\"}" | jq . 2>/dev/null || true
sleep 2

echo ""
echo "==> Current state:"
curl -s "$SERVER/models" | jq -r '.data[] | "  \(.id)  \(.status)"' 2>/dev/null

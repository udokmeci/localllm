#!/bin/bash
# Idle unload watchdog for llama-router mode
# Polls /models every 30s, unloads if no requests in last N seconds
# Usage: bash idle_unload.sh [idle_seconds] [server_url]

IDLE_THRESHOLD="${1:-300}"
SERVER="${2:-http://localhost:8085}"
LAST_REQ_FILE="/tmp/llama-last-request"

touch "$LAST_REQ_FILE"

# Update last request time on each API call (called by switch_router.sh)
# Or set initial time
if [ ! -s "$LAST_REQ_FILE" ]; then
    date +%s > "$LAST_REQ_FILE"
fi

echo "==> idle_unload.sh: threshold=${IDLE_THRESHOLD}s, server=$SERVER"
echo "==> Press Ctrl+C to stop."

while true; do
    STATUS=$(curl -s "$SERVER/models" 2>/dev/null | jq -r '.data[] | select(.status == "loaded") | .id' 2>/dev/null)
    
    if [ -n "$STATUS" ]; then
        LAST_REQ=$(cat "$LAST_REQ_FILE" 2>/dev/null || echo 0)
        NOW=$(date +%s)
        IDLE=$((NOW - LAST_REQ))
        
        if [ "$IDLE" -gt "$IDLE_THRESHOLD" ]; then
            echo "[$(date '+%H:%M:%S')] Unloading $STATUS after ${IDLE}s idle"
            curl -s -X POST "$SERVER/models/unload" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"$STATUS\"}" 2>/dev/null
        else
            echo "[$(date '+%H:%M:%S')] $STATUS loaded, idle=${IDLE}s/${IDLE_THRESHOLD}s"
        fi
    else
        echo "[$(date '+%H:%M:%S')] No loaded models"
    fi
    
    sleep 30
done

#!/bin/bash
# Switch which run_*.sh (and optional quant) the llama-server systemd service
# launches, and restart it.
# Usage: bash switch_model.sh run_qwen36_35b_a3b.sh [QUANT_TAG]
set -e
cd "$(dirname "$0")"

TARGET="$1"
if [ -z "$TARGET" ] || [ ! -f "$TARGET" ]; then
    echo "Usage: $0 <run_script.sh> [QUANT_TAG]"
    echo "Available:"
    ls run_*.sh
    exit 1
fi
QUANT="$2"

printf '#!/bin/bash\nexec "%s/%s" %s\n' "$(pwd)" "$TARGET" "$QUANT" > current_model.sh
chmod +x current_model.sh

echo "==> current_model.sh -> $TARGET${QUANT:+ (quant: $QUANT)}"
systemctl --user restart llama-server
sleep 2
systemctl --user status llama-server --no-pager | head -6

#!/bin/bash
# Run llama-server with Qwen3.6-27B
export LC_NUMERIC=C
# Includes hardware estimation: auto-selects best quant that fits in VRAM

MODEL_DIR="/home/ugur/localllm"
# turboquant fork: faster TurboQuant kernels; no MTP model support
SERVER_BIN="/home/ugur/localllm/llama-cpp-turboquant/build_vulkan/bin/llama-server"
# upstream llama.cpp: required for -MTP.gguf models (draft heads)
SERVER_BIN_MTP="/home/ugur/localllm/llama.cpp/build_vulkan/bin/llama-server"

HOST="0.0.0.0"
PORT=8085
CTX_SIZE=80000
N_GPU_LAYERS=9999
THREADS=14
BATCH_SIZE=2048
UBATCH_SIZE=512
PARALLEL=1
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="q8_0"
TEMP=0.6
TOP_P=0.9
TOP_K=20
MIN_P=0.0
TIMEOUT=600
MTP_ARGS=""   # set below: prefers -MTP.gguf when available

# ── Quant catalogue (name  size_gb  has_mtp) ────────────────────────────────
# Format: "QUANT_TAG:SIZE_GB:HAS_MTP"  sorted best→worst quality
QUANTS=(
    "UD-Q8_K_XL:35.8:0"
    "Q8_0:29.0:0"
    "UD-Q6_K_XL:26.0:0"
    "Q6_K:22.9:0"
    "UD-Q5_K_XL:20.4:1"
    "Q5_K_M:19.8:0"
    "Q5_K_S:19.3:0"
    "UD-Q4_K_XL:17.9:1"
    "Q4_K_M:17.1:0"
    "Q4_1:17.5:0"
    "IQ4_NL:16.3:0"
    "Q4_K_S:16.1:0"
    "Q4_0:16.1:0"
    "IQ4_XS:15.7:0"
    "UD-Q3_K_XL:14.8:0"
    "Q3_K_M:13.8:0"
    "Q3_K_S:12.6:0"
    "UD-IQ3_XXS:12.2:0"
    "UD-Q2_K_XL:12.0:0"
    "UD-IQ2_M:11.0:0"
    "UD-IQ2_XXS:9.57:0"
)

# ── VRAM detection ────────────────────────────────────────────────────────────
get_vram_gb() {
    local bytes
    bytes=$(rocm-smi --showmeminfo vram 2>/dev/null \
        | awk '/GPU\[0\].*VRAM Total Memory/ {print $NF; exit}')
    if [ -n "$bytes" ]; then
        echo "scale=1; $bytes / 1073741824" | bc
    else
        echo "32"   # fallback
    fi
}

get_vram_used_gb() {
    local bytes
    bytes=$(rocm-smi --showmeminfo vram 2>/dev/null \
        | awk '/GPU\[0\].*VRAM Total Used/ {print $NF; exit}')
    if [ -n "$bytes" ]; then
        echo "scale=1; $bytes / 1073741824" | bc
    else
        echo "1.5"
    fi
}

# KV cache estimate for q8_0 at given context
# Qwen3.6-27B: 28 layers, GQA 8 heads, head_dim 128, 2 (K+V), 1 byte per element (q8)
kv_cache_gb() {
    local ctx=$1
    echo "scale=2; $ctx * 28 * 8 * 128 * 2 / 1073741824" | bc
}

OVERHEAD_GB=1.5   # Vulkan runtime + activations

# ── Hardware report ──────────────────────────────────────────────────────────
VRAM_TOTAL=$(get_vram_gb)
VRAM_USED=$(get_vram_used_gb)
VRAM_FREE=$(echo "scale=1; $VRAM_TOTAL - $VRAM_USED" | bc)
KV_GB=$(kv_cache_gb $CTX_SIZE)
BUDGET=$(echo "scale=1; $VRAM_FREE - $KV_GB - $OVERHEAD_GB" | bc)

echo "╔══════════════════════════════════════════════════════╗"
echo "║         Qwen3.6-27B — Hardware Estimation            ║"
echo "╠══════════════════════════════════════════════════════╣"
printf "║  VRAM total     : %5.1f GB                           ║\n" "$VRAM_TOTAL"
printf "║  VRAM used now  : %5.1f GB                           ║\n" "$VRAM_USED"
printf "║  VRAM free      : %5.1f GB                           ║\n" "$VRAM_FREE"
printf "║  KV cache @%5dk: %5.1f GB (q8_0)                  ║\n" "$CTX_SIZE" "$KV_GB"
printf "║  Runtime overhead: %4.1f GB                           ║\n" "$OVERHEAD_GB"
printf "║  Budget for model: %4.1f GB                           ║\n" "$BUDGET"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Quant              Size    Fits?  MTP  Quality      ║"
echo "╠══════════════════════════════════════════════════════╣"

BEST_QUANT=""
BEST_MTP=0

for entry in "${QUANTS[@]}"; do
    IFS=':' read -r tag size mtp <<< "$entry"
    fits=$(echo "$size <= $BUDGET" | bc 2>/dev/null)
    if [ "$fits" = "1" ]; then
        status=" YES  "
        [ -z "$BEST_QUANT" ] && { BEST_QUANT="$tag"; BEST_MTP="$mtp"; }
    else
        status="  -   "
    fi
    mtp_str=$([ "$mtp" = "1" ] && echo " yes " || echo "  -  ")
    # crude quality label
    if   echo "$size > 25" | bc -q | grep -q 1; then qlabel="highest  "
    elif echo "$size > 18" | bc -q | grep -q 1; then qlabel="very high"
    elif echo "$size > 15" | bc -q | grep -q 1; then qlabel="high     "
    elif echo "$size > 12" | bc -q | grep -q 1; then qlabel="medium   "
    else                                               qlabel="low      "
    fi
    printf "║  %-18s %5.1f GB  %s  %s  %s  ║\n" "$tag" "$size" "$status" "$mtp_str" "$qlabel"
done

echo "╠══════════════════════════════════════════════════════╣"
if [ -n "$BEST_QUANT" ]; then
    printf "║  >> Auto-selected : %-31s  ║\n" "$BEST_QUANT"
else
    echo "║  >> WARNING: No quant fits in available VRAM!        ║"
fi
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Override quant from argument: ./run_qwen27b_80k.sh Q4_K_M ───────────────
if [ -n "$1" ] && [ "$1" != "--dry-run" ]; then
    BEST_QUANT="$1"
    # check MTP for override
    for entry in "${QUANTS[@]}"; do
        IFS=':' read -r tag size mtp <<< "$entry"
        [ "$tag" = "$BEST_QUANT" ] && BEST_MTP="$mtp" && break
    done
    echo "==> Using user-specified quant: $BEST_QUANT"
fi

if [ "$1" = "--dry-run" ]; then
    echo "Dry run complete. Exiting."
    exit 0
fi

if [ -z "$BEST_QUANT" ]; then
    echo "ERROR: No suitable quant found. Free up VRAM or use a smaller quant:"
    echo "  $0 UD-IQ2_XXS"
    exit 1
fi

# ── Model file + binary selection ────────────────────────────────────────────
# -MTP.gguf has embedded draft heads; requires upstream llama.cpp (not turboquant)
if [ "$BEST_MTP" = "1" ] && [ -f "$MODEL_DIR/Qwen3.6-27B-${BEST_QUANT}-MTP.gguf" ]; then
    MODEL_FILE="$MODEL_DIR/Qwen3.6-27B-${BEST_QUANT}-MTP.gguf"
    MTP_ARGS="--draft-max 4"
    SERVER_BIN="$SERVER_BIN_MTP"
    echo "==> MTP model detected: ${BEST_QUANT}-MTP.gguf  (--draft-max 4, upstream binary)"
else
    MODEL_FILE="$MODEL_DIR/Qwen3.6-27B-${BEST_QUANT}.gguf"
    MTP_ARGS=""
fi

# ── Model presence check ──────────────────────────────────────────────────────
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model not found:"
    echo "  $MODEL_FILE"
    echo ""
    echo "Download it with:"
    echo "  mkdir -p $MODEL_DIR"
    echo "  huggingface-cli download unsloth/Qwen3.6-27B-GGUF \\"
    echo "    Qwen3.6-27B-${BEST_QUANT}.gguf \\"
    echo "    --local-dir $MODEL_DIR"
    exit 1
fi

if [ ! -f "$SERVER_BIN" ]; then
    echo "ERROR: llama-server not found at $SERVER_BIN"
    if [ -n "$MTP_ARGS" ]; then
        echo "MTP model requires upstream llama.cpp. Build it:"
        echo "  git clone https://github.com/ggerganov/llama.cpp /home/ugur/localllm/llama.cpp"
        echo "  bash /home/ugur/localllm/build_vulkan.sh"
    else
        echo "Build it: bash /home/ugur/localllm/build_rocm.sh"
    fi
    exit 1
fi

ulimit -l unlimited 2>/dev/null || true
export GGML_VK_DISABLE_VALIDATION=1
export AMD_VULKAN_ICD=RADV

echo "==> Starting llama-server"
echo "    Model  : $MODEL_FILE"
echo "    Context: ${CTX_SIZE} tokens"
echo "    KV     : ${CACHE_TYPE_K}"
echo "    MTP    : $([ -n "$MTP_ARGS" ] && echo enabled || echo disabled)"
echo "    Port   : $PORT"
echo ""

exec "$SERVER_BIN" \
    --model        "$MODEL_FILE" \
    --host         "$HOST" \
    --port         "$PORT" \
    --ctx-size     "$CTX_SIZE" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --threads      "$THREADS" \
    --batch-size   "$BATCH_SIZE" \
    --ubatch-size  "$UBATCH_SIZE" \
    --parallel     "$PARALLEL" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    --flash-attn auto \
    --temp         "$TEMP" \
    --top-p        "$TOP_P" \
    --top-k        "$TOP_K" \
    --min-p        "$MIN_P" \
    --timeout      "$TIMEOUT" \
    --cont-batching \
    --metrics      \
    --reasoning-budget -1 \
    $MTP_ARGS \
    --chat-template-kwargs '{"preserve_thinking": true}'

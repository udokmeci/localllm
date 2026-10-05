#!/bin/bash
# Run llama-server with Qwen3.6-35B-A3B — 262k context, no MTP
# Usage: bash run_qwen36_35b_a3b.sh [QUANT_TAG]
#   e.g. bash run_qwen36_35b_a3b.sh UD-Q5_K_S
export LC_NUMERIC=C

MODEL_BASENAME="Qwen3.6-35B-A3B"

MODEL_DIR="/home/ugur/localllm/models"
# turboquant fork: faster TurboQuant kernels; this model has no MTP draft head
SERVER_BIN="/home/ugur/localllm/llama-cpp-turboquant/build_vulkan/bin/llama-server"

HOST="0.0.0.0"
PORT=8085
CTX_SIZE=262144
N_GPU_LAYERS=9999
THREADS=14
BATCH_SIZE=1024
UBATCH_SIZE=256
PARALLEL=1
CACHE_TYPE_K=q8_0
CACHE_TYPE_V=q8_0
TEMP=0.6
TOP_P=0.9
TOP_K=20
MIN_P=0.0
TIMEOUT=600
REASONING_BUDGET=800

# ── Quant catalogue (name  size_gb) ─────────────────────────────────────────
# Format: "QUANT_TAG:SIZE_GB"  sorted best→worst quality
# Sizes taken from unsloth/Qwen3.6-35B-A3B-GGUF (HF API, GiB).
QUANTS=(
    "UD-Q8_K_XL:38.5"
    "Q8_0:36.9"
    "UD-Q6_K_XL:31.8"
    "UD-Q6_K:29.3"
    "UD-Q5_K_XL:26.6"
    "UD-Q5_K_M:26.5"
    "UD-Q5_K_S:24.9"
    "UD-Q4_K_XL:22.4"
    "UD-Q4_K_M:22.1"
    "MXFP4_MOE:21.7"
    "UD-Q4_K_S:20.9"
    "UD-IQ4_NL_XL:19.5"
    "UD-IQ4_NL:18.0"
    "UD-IQ4_XS:17.7"
    "UD-Q3_K_XL:16.8"
    "UD-Q3_K_M:16.6"
    "UD-Q3_K_S:15.4"
    "UD-IQ3_S:13.7"
    "UD-IQ3_XXS:13.2"
    "UD-Q2_K_XL:12.3"
    "UD-IQ2_M:11.5"
    "UD-IQ2_XXS:10.8"
    "UD-IQ1_M:10.0"
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

# KV cache estimate for q8_0 at given context.
# Qwen3.6-35B-A3B is a hybrid linear-attention MoE model (40 layers total,
# only every 4th layer is full attention: 10 full-attention layers, 2 KV
# heads, head_dim 256). The other 30 layers use a fixed-size linear/SSM
# recurrent state that does NOT grow with context — its small, ~constant
# memory cost is folded into OVERHEAD_GB below. Only 3B of the 35B params
# activate per token (MoE), so this loads/runs fully on GPU with no expert
# offload needed at any quant that fits below.
kv_cache_gb() {
    local ctx=$1
    echo "scale=2; $ctx * 10 * 2 * 256 * 2 / 1073741824" | bc
}

# Measured empirically: UD-Q5_K_S at 262144 ctx used 27.43GB actual vs 27.40GB
# nominal (weights + KV) — MoE compute buffers are small with no MTP draft
# model in play, so overhead here is much lower than the dense 27B script.
OVERHEAD_GB=1.0

# ── Hardware report ──────────────────────────────────────────────────────────
VRAM_TOTAL=$(get_vram_gb)
VRAM_USED=$(get_vram_used_gb)
VRAM_FREE=$(echo "scale=1; $VRAM_TOTAL - $VRAM_USED" | bc)
KV_GB=$(kv_cache_gb $CTX_SIZE)
BUDGET=$(echo "scale=1; $VRAM_FREE - $KV_GB - $OVERHEAD_GB" | bc)

echo "╔══════════════════════════════════════════════════════╗"
echo "║    Qwen3.6-35B-A3B — Hardware Estimation (262k)      ║"
echo "╠══════════════════════════════════════════════════════╣"
printf "║  VRAM total     : %5.1f GB                           ║\n" "$VRAM_TOTAL"
printf "║  VRAM used now  : %5.1f GB                           ║\n" "$VRAM_USED"
printf "║  VRAM free      : %5.1f GB                           ║\n" "$VRAM_FREE"
printf "║  KV cache @%5dk: %5.1f GB (q8_0)                  ║\n" "$CTX_SIZE" "$KV_GB"
printf "║  Runtime overhead: %4.1f GB                           ║\n" "$OVERHEAD_GB"
printf "║  Budget for model: %4.1f GB                           ║\n" "$BUDGET"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Quant              Size    Fits?  Quality            ║"
echo "╠══════════════════════════════════════════════════════╣"

BEST_QUANT=""

for entry in "${QUANTS[@]}"; do
    IFS=':' read -r tag size <<< "$entry"
    fits=$(echo "$size <= $BUDGET" | bc 2>/dev/null)
    if [ "$fits" = "1" ]; then
        status=" YES  "
        [ -z "$BEST_QUANT" ] && BEST_QUANT="$tag"
    else
        status="  -   "
    fi
    # crude quality label
    if   echo "$size > 30" | bc -q | grep -q 1; then qlabel="highest  "
    elif echo "$size > 24" | bc -q | grep -q 1; then qlabel="very high"
    elif echo "$size > 18" | bc -q | grep -q 1; then qlabel="high     "
    elif echo "$size > 13" | bc -q | grep -q 1; then qlabel="medium   "
    else                                               qlabel="low      "
    fi
    printf "║  %-18s %5.1f GB  %s  %s        ║\n" "$tag" "$size" "$status" "$qlabel"
done

echo "╠══════════════════════════════════════════════════════╣"
if [ -n "$BEST_QUANT" ]; then
    printf "║  >> Auto-selected : %-31s  ║\n" "$BEST_QUANT"
else
    echo "║  >> WARNING: No quant fits in available VRAM!        ║"
fi
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Override quant from argument ─────────────────────────────────────────────
if [ -n "$1" ] && [ "$1" != "--dry-run" ]; then
    BEST_QUANT="$1"
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
MODEL_FILE="$MODEL_DIR/${MODEL_BASENAME}-${BEST_QUANT}.gguf"

# ── Model presence check ──────────────────────────────────────────────────────
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model not found:"
    echo "  $MODEL_FILE"
    echo ""
    echo "Download it with:"
    echo "  mkdir -p $MODEL_DIR"
    echo "  hf download unsloth/Qwen3.6-35B-A3B-GGUF \\"
    echo "    ${MODEL_BASENAME}-${BEST_QUANT}.gguf \\"
    echo "    --local-dir $MODEL_DIR"
    exit 1
fi

if [ ! -f "$SERVER_BIN" ]; then
    echo "ERROR: llama-server not found at $SERVER_BIN"
    echo "Build it: bash /home/ugur/localllm/build_rocm.sh"
    exit 1
fi

ulimit -l unlimited 2>/dev/null || true
export GGML_VK_DISABLE_VALIDATION=1
export AMD_VULKAN_ICD=RADV
export GGML_VK_DEVICE=0          # pin to R9700 (Vulkan0), not iGPU (Vulkan1)

echo "==> Starting llama-server"
echo "    Model  : $MODEL_FILE"
echo "    Context: ${CTX_SIZE} tokens"
echo "    KV     : ${CACHE_TYPE_K}"
echo "    MTP    : disabled (no draft head for this repo)"
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
    --kv-unified   \
    --reasoning-budget "$REASONING_BUDGET" \
    --reasoning-budget-message "

Considering the limited time by the user, I have to give the solution based on the thinking directly now.

" \
    --chat-template-kwargs '{"preserve_thinking": true}'

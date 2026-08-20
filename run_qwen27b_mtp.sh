#!/bin/bash
# Run llama-server with Qwen3.8-27B — 258k context, MTP preferred
# Usage: bash run_qwen27b_mtp.sh [QUANT_TAG]
#   e.g. bash run_qwen27b_mtp.sh UD-Q4_K_XL
# MTP draft head (mtp-Qwen3.8-27B-Q4_0.gguf) is a single file shared across
# all main quants (unlike Qwen3.6's fused per-quant -MTP.gguf files).
export LC_NUMERIC=C

MODEL_BASENAME="Qwen3.8-27B"
# path is relative to MODEL_DIR; matches the MTP/ subfolder in the HF repo
MTP_DRAFT_FILE_NAME="MTP/mtp-Qwen3.8-27B-Q4_0.gguf"

MODEL_DIR="/home/ugur/localllm/models"
# turboquant fork: faster TurboQuant kernels; no MTP model support
SERVER_BIN="/home/ugur/localllm/llama-cpp-turboquant/build_vulkan/bin/llama-server"
# upstream llama.cpp: required for -MTP.gguf models (draft heads)
SERVER_BIN_MTP="/home/ugur/localllm/llama.cpp/build_vulkan/bin/llama-server"

HOST="0.0.0.0"
PORT=8085
CTX_SIZE=258000
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
MTP_ARGS=""   # set below: prefers -MTP.gguf when available

# ── Quant catalogue (name  size_gb) ─────────────────────────────────────────
# Format: "QUANT_TAG:SIZE_GB"  sorted best→worst quality
# Sizes taken from unsloth/Qwen3.8-27B-GGUF (HF API, GiB). MTP draft head is a
# single shared file (see MTP_DRAFT_FILE_NAME) usable with any quant below.
QUANTS=(
    "UD-Q8_K_XL:29.3"
    "UD-Q8_K_L:26.1"
    "Q8_0:27.1"
    "UD-Q6_K_XL:23.6"
    "UD-Q6_K_L:22.5"
    "UD-Q6_K_M:21.5"
    "UD-Q6_K:20.5"
    "UD-Q5_K_XL:19.4"
    "UD-Q5_K_M:18.4"
    "UD-Q5_K_S:17.4"
    "UD-Q4_K_XL:16.4"
    "Q4_1:16.3"
    "UD-Q4_K_M:15.3"
    "Q4_0:15.0"
    "UD-Q4_K_S:14.3"
    "UD-IQ4_XS:13.3"
    "UD-Q3_K_XL:12.2"
    "UD-IQ3_S:11.2"
    "UD-IQ3_XXS:10.2"
    "UD-Q2_K_XL:9.2"
    "UD-IQ2_S:7.8"
    "UD-IQ2_XXS:6.8"
    "UD-IQ1_M:6.3"
    "UD-IQ1_S:5.8"
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
# Qwen3.8-27B is a hybrid linear-attention model (64 layers total, only every
# 4th layer is full attention: 16 full-attention layers, 4 KV heads, head_dim
# 256). The other 48 layers use a fixed-size linear/SSM recurrent state that
# does NOT grow with context — its small, ~constant memory cost is folded
# into OVERHEAD_GB below rather than modeled per-token.
kv_cache_gb() {
    local ctx=$1
    echo "scale=2; $ctx * 16 * 4 * 256 * 2 / 1073741824" | bc
}

OVERHEAD_GB=2.0   # Vulkan runtime + activations + linear-attention SSM state

# ── Hardware report ──────────────────────────────────────────────────────────
VRAM_TOTAL=$(get_vram_gb)
VRAM_USED=$(get_vram_used_gb)
VRAM_FREE=$(echo "scale=1; $VRAM_TOTAL - $VRAM_USED" | bc)
KV_GB=$(kv_cache_gb $CTX_SIZE)
BUDGET=$(echo "scale=1; $VRAM_FREE - $KV_GB - $OVERHEAD_GB" | bc)

echo "╔══════════════════════════════════════════════════════╗"
echo "║     Qwen3.8-27B — Hardware Estimation (258k MTP)    ║"
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

BEST_QUANT="UD-Q4_K_XL"

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
    if   echo "$size > 25" | bc -q | grep -q 1; then qlabel="highest  "
    elif echo "$size > 18" | bc -q | grep -q 1; then qlabel="very high"
    elif echo "$size > 15" | bc -q | grep -q 1; then qlabel="high     "
    elif echo "$size > 12" | bc -q | grep -q 1; then qlabel="medium   "
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
# MTP draft head is a single file shared across all quants (unlike Qwen3.6's
# fused per-quant -MTP.gguf); pass it separately via --model-draft.
MODEL_FILE="$MODEL_DIR/${MODEL_BASENAME}-${BEST_QUANT}.gguf"
MTP_DRAFT_FILE="$MODEL_DIR/$MTP_DRAFT_FILE_NAME"

if [ -f "$MTP_DRAFT_FILE" ]; then
    MTP_ARGS="--spec-type draft-mtp --spec-draft-n-max 4 --model-draft $MTP_DRAFT_FILE"
    SERVER_BIN="$SERVER_BIN_MTP"
    echo "==> MTP draft head found: $MTP_DRAFT_FILE_NAME  (--spec-draft-n-max 4, upstream binary)"
else
    MTP_ARGS=""
    echo "==> No MTP draft head found ($MTP_DRAFT_FILE_NAME); running without speculative decoding"
fi

# ── Model presence check ──────────────────────────────────────────────────────
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model not found:"
    echo "  $MODEL_FILE"
    echo ""
    echo "Download it with:"
    echo "  mkdir -p $MODEL_DIR"
    echo "  hf download unsloth/Qwen3.8-27B-GGUF \\"
    echo "    ${MODEL_BASENAME}-${BEST_QUANT}.gguf ${MTP_DRAFT_FILE_NAME} \\"
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
export GGML_VK_DEVICE=0          # pin to R9700 (Vulkan0), not iGPU (Vulkan1)

echo "==> Starting llama-server"
echo "    Model  : $MODEL_FILE"
echo "    Context: ${CTX_SIZE} tokens"
echo "    KV     : ${CACHE_TYPE_K}"
echo "    MTP    : $([ -n "$MTP_ARGS" ] && echo "enabled (draft-mtp, max 4 tokens)" || echo disabled)"
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
    --reasoning-budget -1 \
    $MTP_ARGS \
    --chat-template-kwargs '{"preserve_thinking": true}' \
    --cache-reuse 256 --cache-ram 32000 --ctx-checkpoints 24

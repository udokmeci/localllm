#!/bin/bash
# Adapted config for Radeon AI Pro R9700 32GB VRAM
# Performance: ~80-180 tok/s generation with Vulkan/ROCm backend
# VRAM usage: Full 35B model fits in 32GB VRAM

# Environment variables and calculations
GPU_VRAM_MB=32768  # Radeon AI Pro R9700 32GB
CPU_THREADS=16     # Ryzen 7 7700
MODEL_SIZE_MB=24438  # Qwen3.6-35B-A3B-APEX-I-Balanced.gguf actual size in MB
KV_CACHE_MB=3000   # Approx for 262144 ctx, turbo4 (larger for 32GB VRAM)
OVERHEAD_MB=2000   # ROCm/Vulkan + runtime
MODEL_LAYERS=40  # Total layers in model
# Optimized for 32GB VRAM: use all layers (MoE fits in VRAM)
N_GPU_LAYERS=40  # All 40 layers on GPU
N_CPU_MOE=0     # No CPU offload
PROJECTED_VRAM_MB=$((MODEL_SIZE_MB + KV_CACHE_MB + OVERHEAD_MB))
ACTUAL_VRAM_USAGE_MB=$((OVERHEAD_MB + KV_CACHE_MB + (MODEL_SIZE_MB * N_GPU_LAYERS / MODEL_LAYERS)))

MODEL_SIZE_GB_DISPLAY=$(echo "scale=2; $MODEL_SIZE_MB / 1024" | bc)

echo "Hardware: GPU VRAM ${GPU_VRAM_MB}MB, CPU ${CPU_THREADS} threads"
echo "GPU layers: ${N_GPU_LAYERS}, CPU MoE experts: ${N_CPU_MOE}"
echo "Model: ${MODEL_SIZE_GB_DISPLAY}GB Q5_K (${MODEL_LAYERS} layers)"
echo "Projected VRAM: ${PROJECTED_VRAM_MB}MB (with offloading: ${ACTUAL_VRAM_USAGE_MB}MB used)"
echo "Free VRAM: $((GPU_VRAM_MB - ACTUAL_VRAM_USAGE_MB))MB"

# If --dry-run, exit after printing calculations
if [ "$1" == "--dry-run" ]; then
    echo "Dry run: Calculations printed. Server not started."
    exit 0
fi

# Model and server config
SELECTED_MODEL="/home/ugur/localllm/llama-cpp-turboquant/models/Qwen3.6-35B-A3B-APEX-I-Balanced.gguf"
HOST="0.0.0.0"
PORT=8085
CTX_SIZE=262144
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="q8_0"
FLASH_ATTN="auto"
BATCH_SIZE=2048
PARALLEL=1
NO_MMAP=""
MLOCK=""
UBATCH_SIZE=512
THREADS=16
CONT_BATCHING="--cont-batching"
TIMEOUT=300
TEMP=0.2
TOP_P=0.95
MIN_P=0.05
TOP_K=20
METRICS="--metrics"
CHAT_TEMPLATE_KWARGS='{"preserve_thinking": true}'

# Increase memory lock limit to prevent mlock warnings
ulimit -l unlimited 2>/dev/null || echo "Warning: Could not set ulimit -l unlimited (run as root if needed)"

# Set environment for Vulkan backend (AMD R9700)
export HSA_OVERRIDE_SKIP_EOF=1
export ROCM_PATH=/opt/rocm

echo "Starting Llama Server with Vulkan backend..."
echo "Model: $SELECTED_MODEL"
echo "Config: ${N_GPU_LAYERS} GPU layers, ${N_CPU_MOE} CPU MoE experts, context ${CTX_SIZE}, ${CACHE_TYPE_K} KV cache"
echo "Expected performance: ~50-80 tok/s generation (262K context)"

# Run llama-server with Vulkan backend (AMD R9700 support)
/home/ugur/localllm/llama-cpp-turboquant/build_vulkan/bin/llama-server \
--model "$SELECTED_MODEL" \
--host "$HOST" \
--port "$PORT" \
--ctx-size "$CTX_SIZE" \
--n-gpu-layers "$N_GPU_LAYERS" \
--n-cpu-moe "$N_CPU_MOE" \
--cache-type-k "$CACHE_TYPE_K" \
--cache-type-v "$CACHE_TYPE_V" \
--flash-attn "$FLASH_ATTN" \
--batch-size "$BATCH_SIZE" \
--parallel "$PARALLEL" \
--ubatch-size "$UBATCH_SIZE" \
--threads "$THREADS" \
"$CONT_BATCHING" \
--timeout "$TIMEOUT" \
--temp "$TEMP" \
--top-p "$TOP_P" \
--min-p "$MIN_P" \
--top-k "$TOP_K" \
"$METRICS" \
--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS"

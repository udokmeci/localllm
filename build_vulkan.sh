#!/bin/bash
# Build standard llama.cpp with Vulkan backend
# Required for MTP (Multi-Token Prediction) model support
# Target: AMD Radeon AI Pro R9700 (gfx1201 / RDNA 4) via Vulkan
#
# Produces: llama.cpp/build_vulkan/bin/llama-server

set -e

PROJECT_DIR="/home/ugur/localllm/llama.cpp"
BUILD_DIR="$PROJECT_DIR/build_vulkan"

echo "==> Building llama.cpp (upstream) with Vulkan — MTP support"
echo "    Project : $PROJECT_DIR"
echo "    Build   : $BUILD_DIR"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: $PROJECT_DIR not found."
    echo "Clone it first:"
    echo "  git clone https://github.com/ggerganov/llama.cpp $PROJECT_DIR"
    exit 1
fi

mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DLLAMA_BUILD_SERVER=ON

echo "==> Compiling (using $(nproc) cores)..."
cmake --build "$BUILD_DIR" --target llama-server -j "$(nproc)"

echo ""
echo "==> Build complete."
echo "    Binary : $BUILD_DIR/bin/llama-server"

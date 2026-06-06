#!/bin/bash
# Build llama-cpp-turboquant with ROCm/HIP backend
# Target: AMD Radeon RX 9700 (gfx1201 / RDNA 4)

set -e

PROJECT_DIR="/home/ugur/localllm/llama-cpp-turboquant"
BUILD_DIR="$PROJECT_DIR/build_rocm"

echo "==> Building llama-cpp-turboquant with ROCm/HIP..."
echo "    Project : $PROJECT_DIR"
echo "    Build   : $BUILD_DIR"

# Verify ROCm is available
if ! command -v hipcc &>/dev/null; then
    echo "ERROR: hipcc not found. Make sure ROCm is installed and /opt/rocm/bin is in PATH."
    exit 1
fi

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="$ROCM_PATH/bin:$PATH"

# RX 9700 is gfx1201 (RDNA 4). Force target so CMake doesn't auto-detect wrong arch.
export AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1201}"

mkdir -p "$BUILD_DIR"

cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DGGML_HIP_FA=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DAMDGPU_TARGETS="$AMDGPU_TARGETS"

echo "==> Compiling (using $(nproc) cores)..."
cmake --build "$BUILD_DIR" --target llama-server -j "$(nproc)"

echo ""
echo "==> Build complete."
echo "    Binary : $BUILD_DIR/bin/llama-server"

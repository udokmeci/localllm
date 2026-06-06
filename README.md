# Local LLM — Qwen3.6-27B on AMD Radeon AI Pro R9700

Self-hosted llama.cpp inference server for **Qwen3.6-27B** with Multi-Token Prediction (MTP), running on AMD Radeon AI Pro R9700 (RDNA 4) via Vulkan/RADV.

## Hardware

| Component | Detail |
|-----------|--------|
| GPU | AMD Radeon AI Pro R9700 (32 GB VRAM, gfx1201) |
| Driver | Vulkan via RADV (Mesa) |
| Model | Qwen3.6-27B (UD-Q4_K_XL quant, ~17.9 GB) |
| Context | 96k tokens, q8_0 KV cache |
| MTP | `--spec-type draft-mtp --spec-draft-n-max 4` |

## Quick Start

```bash
# Build upstream llama.cpp with Vulkan (required for MTP models)
bash build_vulkan.sh

# Run with auto-detected best quant and MTP
bash run_qwen27b_96k_mtp.sh

# Override quant
bash run_qwen27b_96k_mtp.sh UD-Q5_K_XL

# Dry run (show VRAM estimation only)
bash run_qwen27b_96k_mtp.sh --dry-run
```

## Scripts

| Script | Context | Description |
|--------|---------|-------------|
| `run_qwen27b_96k_mtp.sh` | 96k | MTP-first, auto-selects best quant fitting in VRAM |
| `run_qwen27b_80k.sh` | 80k | Legacy launcher, MTP auto-detection |
| `build_vulkan.sh` | — | Builds upstream llama.cpp with Vulkan backend |
| `build_rocm.sh` | — | Builds turboquant fork with ROCm backend |

## MTP (Multi-Token Prediction)

MTP models (`*-MTP.gguf`) contain embedded draft heads for speculative decoding. Requires **upstream llama.cpp** (not turboquant fork).

The run scripts auto-detect `-MTP.gguf` files and switch to the upstream binary with `--spec-type draft-mtp --spec-draft-n-max 4`.

### Available MTP Quantizations

| Quant | Size | Quality |
|-------|------|---------|
| UD-Q5_K_XL | 20.4 GB | Very high |
| UD-Q4_K_XL | 17.9 GB | High (default) |

Download from [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF):

```bash
huggingface-cli download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf \
  --local-dir /home/ugur/localllm
```

## Vulkan GPU Pinning

The R9700 is exposed as `Vulkan0`. The iGPU is `Vulkan1`. The run scripts pin to the correct device:

```bash
export GGML_VK_DEVICE=0          # R9700
export AMD_VULKAN_ICD=RADV       # Mesa Vulkan driver
```

## Systemd Service

A user service is available for auto-start:

```
~/.config/systemd/user/llama-server-qwen27b.service
```

```bash
systemctl --user enable --now llama-server-qwen27b
systemctl --user status llama-server-qwen27b
journalctl --user -u llama-server-qwen27b -f
```

## VRAM Budget

The run scripts estimate available VRAM and auto-select the best quantization:

- **Model weights** — varies by quant (9.6–35.8 GB)
- **KV cache** — ~11 GB at 96k context (q8_0)
- **Runtime overhead** — ~1.5 GB (Vulkan + activations)

## Submodules

| Submodule | Purpose |
|-----------|---------|
| `llama-cpp-turboquant` | TurboQuant fork (faster kernels, no MTP) |
| `turboquant_plus` | Benchmark profiles and quantization tools |
| `llama.cpp` | Upstream llama.cpp (required for MTP models) |

## Kilo / OpenCode Integration

Configure Kilo to point at the llama-server:

```jsonc
// ~/.config/kilo/kilo.jsonc
{
  "models": {
    "local-qwen": {
      "provider": "ollama",
      "model": "qwen3.6-27b-ud-q4k-xl-mtp",
      "contextLength": 96000,
      "maxTokens": 32000
    }
  }
}
```

## Quantization Catalogue

All available quantizations for Qwen3.6-27B (best → worst quality):

| Quant | Size (GB) | MTP | Quality |
|-------|-----------|-----|---------|
| UD-Q8_K_XL | 35.8 | — | Highest |
| Q8_0 | 29.0 | — | Highest |
| UD-Q6_K_XL | 26.0 | — | Highest |
| Q6_K | 22.9 | — | Very high |
| UD-Q5_K_XL | 20.4 | ✅ | Very high |
| Q5_K_M | 19.8 | — | Very high |
| Q5_K_S | 19.3 | — | High |
| UD-Q4_K_XL | 17.9 | ✅ | High |
| Q4_K_M | 17.1 | — | High |
| Q4_1 | 17.5 | — | High |
| IQ4_NL | 16.3 | — | High |
| Q4_K_S | 16.1 | — | High |
| Q4_0 | 16.1 | — | High |
| IQ4_XS | 15.7 | — | Medium |
| UD-Q3_K_XL | 14.8 | — | Medium |
| Q3_K_M | 13.8 | — | Medium |
| Q3_K_S | 12.6 | — | Medium |
| UD-IQ3_XXS | 12.2 | — | Medium |
| UD-Q2_K_XL | 12.0 | — | Low |
| UD-IQ2_M | 11.0 | — | Low |
| UD-IQ2_XXS | 9.57 | — | Low |

## License

This repository contains configuration and scripts only. Model weights are licensed under their respective licenses (Qwen3.6 uses Qwen License). llama.cpp is MIT licensed.
# Local LLM — Qwen3.8-27B on AMD Radeon AI Pro R9700

Self-hosted llama.cpp inference server for **Qwen3.8-27B** with Multi-Token Prediction (MTP), running on AMD Radeon AI Pro R9700 (RDNA 4) via Vulkan/RADV.

## Hardware

| Component | Detail |
|-----------|--------|
| GPU | AMD Radeon AI Pro R9700 (32 GB VRAM, gfx1201) |
| Driver | Vulkan via RADV (Mesa) |
| Model | Qwen3.8-27B (UD-Q4_K_XL quant, ~16.4 GB) |
| Context | 258k tokens, q8_0 KV cache |
| MTP | `--spec-type draft-mtp --spec-draft-n-max 4 --model-draft <file>` |

## Quick Start

```bash
# Build upstream llama.cpp with Vulkan (required for MTP models)
bash build_vulkan.sh

# Run with auto-detected best quant and MTP
bash run_qwen27b_mtp.sh

# Override quant
bash run_qwen27b_mtp.sh UD-Q5_K_XL

# Dry run (show VRAM estimation only)
bash run_qwen27b_mtp.sh --dry-run
```

## Scripts

| Script | Context | Description |
|--------|---------|-------------|
| `run_qwen27b_mtp.sh` | 258k | MTP-first, auto-selects best quant fitting in VRAM |
| `run_qwen27b_80k.sh` | 80k | Legacy launcher, MTP auto-detection |
| `run_server.sh` | 262k | Legacy Qwen3.6-35B-A3B launcher (kept for 3.6 testing) |
| `build_vulkan.sh` | — | Builds upstream llama.cpp with Vulkan backend |
| `build_rocm.sh` | — | Builds turboquant fork with ROCm backend |

## MTP (Multi-Token Prediction)

Unlike Qwen3.6, Qwen3.8's MTP draft head is **not fused into the main quant file** — it ships as a single file (`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`) shared across every quant, and the main model gguf actually requires it: its metadata declares an extra "nextn" layer whose tensors only exist in that separate file, so loading the main model without pairing it fails with `missing tensor 'blk.64.ssm_conv1d.weight'`.

The run scripts auto-detect `models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf` and, if present, switch to the upstream binary and add `--spec-type draft-mtp --spec-draft-n-max 4 --model-draft <file>`.

### Available Quantizations

| Quant | Size | Quality |
|-------|------|---------|
| UD-Q5_K_XL | 19.4 GB | Very high |
| UD-Q4_K_XL | 16.4 GB | High (default) |

Download from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF):

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
  --local-dir /home/ugur/localllm/models
```

## Vulkan GPU Pinning

The R9700 is exposed as `Vulkan0`. The iGPU is `Vulkan1`. The run scripts pin to the correct device:

```bash
export GGML_VK_DEVICE=0          # R9700
export AMD_VULKAN_ICD=RADV       # Mesa Vulkan driver
```

## Systemd Service

The user service is tracked in this repo at `systemd/llama-server-qwen27b.service`.

Install it:

```bash
install -Dm644 systemd/llama-server-qwen27b.service ~/.config/systemd/user/llama-server-qwen27b.service
systemctl --user daemon-reload
```

Manage it:

```bash
systemctl --user enable --now llama-server-qwen27b
systemctl --user status llama-server-qwen27b
journalctl --user -u llama-server-qwen27b -f
```

## VRAM Budget

The run scripts estimate available VRAM and auto-select the best quantization. Qwen3.8-27B is a hybrid linear-attention model: of its 64 layers, only every 4th (16 total) is full attention — the other 48 use a fixed-size linear/SSM recurrent state that does not grow with context, so KV cache scales far more gently than a dense model of the same size.

- **Model weights** — varies by quant (5.8–29.3 GB)
- **KV cache** — ~7.9 GB at 258k context (q8_0), full-attention layers only
- **Runtime overhead** — ~2.0 GB (Vulkan + activations + linear-attention state)

## Submodules

| Submodule | Purpose |
|-----------|---------|
| `llama-cpp-turboquant` | TurboQuant fork (faster kernels, no MTP) |
| `turboquant_plus` | Benchmark profiles and quantization tools |
| `llama.cpp` | Upstream llama.cpp (required for MTP models) |

## Kilo / OpenCode Integration

Both `~/.config/kilo/kilo.jsonc` and `~/.config/opencode/opencode.json` point at the llama-server via an OpenAI-compatible provider:

```jsonc
{
  "model": "local-qwen/qwen3.8-27b-ud-q4k-xl-mtp",
  "provider": {
    "local-qwen": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Qwen (llama.cpp)",
      "options": { "baseURL": "http://localhost:8085/v1" },
      "models": {
        "qwen3.8-27b-ud-q4k-xl-mtp": {
          "name": "Qwen3.8 27B UD-Q4_K_XL MTP 96k (local)",
          "tools": true,
          "limit": { "context": 258000, "output": 20000 }
        }
      }
    }
  }
}
```

## Quantization Catalogue

All available quantizations for Qwen3.8-27B (best → worst quality), from `unsloth/Qwen3.8-27B-GGUF`:

| Quant | Size (GB) | Quality |
|-------|-----------|---------|
| UD-Q8_K_XL | 29.3 | Highest |
| UD-Q8_K_L | 26.1 | Highest |
| Q8_0 | 27.1 | Highest |
| UD-Q6_K_XL | 23.6 | Very high |
| UD-Q6_K_L | 22.5 | Very high |
| UD-Q6_K_M | 21.5 | Very high |
| UD-Q6_K | 20.5 | Very high |
| UD-Q5_K_XL | 19.4 | Very high |
| UD-Q5_K_M | 18.4 | Very high |
| UD-Q5_K_S | 17.4 | High |
| UD-Q4_K_XL | 16.4 | High |
| Q4_1 | 16.3 | High |
| UD-Q4_K_M | 15.3 | High |
| Q4_0 | 15.0 | Medium |
| UD-Q4_K_S | 14.3 | Medium |
| UD-IQ4_XS | 13.3 | Medium |
| UD-Q3_K_XL | 12.2 | Medium |
| UD-IQ3_S | 11.2 | Low |
| UD-IQ3_XXS | 10.2 | Low |
| UD-Q2_K_XL | 9.2 | Low |
| UD-IQ2_S | 7.8 | Low |
| UD-IQ2_XXS | 6.8 | Low |
| UD-IQ1_M | 6.3 | Low |
| UD-IQ1_S | 5.8 | Low |

## License

This repository contains configuration and scripts only. Model weights are licensed under their respective licenses (Qwen3.8 uses the Qwen License). llama.cpp is MIT licensed.

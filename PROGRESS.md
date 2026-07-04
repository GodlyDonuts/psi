# Psi progress and handoff

_Where the project stands. Last updated 2026-07-03._

Repo: https://github.com/GodlyDonuts/Psi. Everything below is committed and pushed to `main`.

---

## What Psi is

A small language model on a fully custom stack: my own autograd, runtime, and GPU kernels, no PyTorch. The model is the byproduct. The real goals are mastering the full stack, building something novel, and the craft of it. North-star metric is capability-per-bit, the smallest model in bits to clear a capability bar.

Active goal: the smallest model (params, then bits) that still does what Microsoft's TinyStories did, writing coherent, grammatical, consistent simple children's stories, using 2026 techniques on my radical stack. TinyStories (2023) did it at ~2.5M params. I'm going below 1M, then crushing the bits with ternary (~1.58-bit) weights on my own kernel.

---

## Status by layer (what's built and working)

| Layer | State | Where |
|---|---|---|
| **Step 0** scalar autograd | done, grad-checked | `src/step0_scalar_autograd/` |
| **Step 1** tensor autograd (the engine everything builds on) | done, 12/12 ops grad-checked ~1e-12 | `src/step1_tensor_autograd/tensor.hpp` |
| **Step 2** psi-nano (char-level GPT, the prototype) | trains, generates fluent corpus English | `src/step2_psi_nano/` (`main.cpp`, `model.hpp`, `nn.hpp`) |
| **Step 3** GPU kernels (Metal) | matmul parity-class with MLX (99% on the key shape) plus a novel ternary GEMM (16× smaller weights, full speed) | `src/step3_metal/`, writeup `docs/GPU_KERNELS.md` |
| **Step 4** CUDA backend (NVIDIA) | same 4-fn interface on H100/V100, cuBLAS plus a hand-written kernel, both validated. Training is bit-identical to CPU and ~10× faster. Runs on UCF Newton HPC | `src/step4_cuda/`, writeup `docs/CUDA_BACKEND.md` |
| **psi-stories** modern sub-1M model | built, trains/saves/loads/generates, all techniques grad-checked | `src/step2_psi_nano/stories.cpp` + `model_stories.hpp` |
| **Capability bar** (the eval) | prompts plus rubric, an LLM grades | `eval/tinystories_prompts.txt`, `docs/EVAL.md` |

### The modern psi-stories architecture (`model_stories.hpp`, `ModernGPT`)

Every capability-per-param technique from the research, all on grad-checked ops: small-BPE (`bpe.hpp`), multi-head plus GQA, RoPE, SwiGLU, block-wise weight-sharing (depth at ~0 param cost), tied embeddings, RMSNorm, and a WSD LR schedule.

### The GPU kernel contribution (`docs/GPU_KERNELS.md`)

Matmul tuned from ~28% to 45–54% of M1 peak (float4, 64×64/8sg tiling, bank-conflict padding), bit-exact, matching MLX's method (verified against their source), parity on the key shape.

The ternary-weight GEMM (`ternary_gemm.mm`) is the unique edge: weights {−1,0,+1} at ~16× less memory, running at full fp32-GEMM speed. That's a capability-per-bit path MLX has no equivalent for.

Two findings fell out of the kernel work: M1 matrix units are precision-independent (fp16 isn't faster), and epilogue fusion doesn't pay on M1.

---

## Current models

Four trained models so far, all sub-1M:

| model | params |
|---|---:|
| femto | 115K |
| nano | 215K |
| small | 354K |
| mid | 574K |

The result I care about: small at 354K gets the same 8/6/6/5 as TinyStories-1M, at a third of the size. None of my sub-1M models clear the coherence bar cleanly, so the honest read is a capability-per-parameter win, not a raw-quality win. Quality climbs cleanly with size, and the vocab-512 models beat the vocab-1024 ones per parameter at this scale.

Each trained model is reproducible from `models/<name>/MODEL.md` (exact config, command, and git commit).

---

## Immediate next steps

1. Ternary-QAT the smallest passing config (the BitNet 16-to-1.58 recipe, see `docs/RESEARCH.md`) for the capability-per-bit record. A passing ~300K model at 1.58 bits is ~60 KB.
2. Keep pushing the sweep to lower loss and re-grade against the bar to find the real smallest config that clears it.

---

## How to build and run

```sh
# psi-stories (modern sub-1M model), float build links the Metal GPU matmul backend
clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \
  src/step2_psi_nano/stories.cpp src/step3_metal/metal_backend.mm \
  -framework Metal -framework Foundation -o psi_stories

# data (gitignored, fetch per data/README.md)
curl -L -o data/tinystories-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# train  [data] [steps] [vocab d layers ctx hidden heads n_kv n_unique]
./psi_stories train data/tinystories-valid.txt 4000 512 64 4 64 192 4 2 2
./psi_stories eval  psi_stories.bin eval/tinystories_prompts.txt 0.7      # completions to grade
./psi_stories gen   psi_stories.bin "Once upon a time"

# grad-checks (must stay 12/12 PASS before relying on any op)
clang++ -std=c++17 -O2 src/step2_psi_nano/gradcheck.cpp -o step2_gradcheck && ./step2_gradcheck
```

---

## Doc map

- `docs/RESEARCH.md`: every technique considered plus the verdict for my regime (including the ternary recipe), frontier-checked 2026-06-18.
- `docs/GPU_KERNELS.md`: the Metal kernel journey, the ternary GEMM, and the findings.
- `docs/CUDA_BACKEND.md`: the NVIDIA backend on Newton.
- `docs/EVAL.md`: the capability bar (rubric plus method).
- `models/README.md`: the models and the capability-per-bit frontier table.
- `data/README.md`: how to fetch the TinyStories data.
- `docs/DESIGN.md`, `docs/RADICAL.md`, `docs/00-charter.md`: vision and strategy.

## Principles

Everything is written to be read. Novelty is first-class. No reaching for PyTorch, the custom stack is the point. Correctness is never traded for speed, grad-checks gate every op, and "better" has to be a measured number.

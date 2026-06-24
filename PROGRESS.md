# Ψ Psi — Progress & Handoff

_Single-source status for picking up the project (e.g. in a new session). Last updated 2026-06-18._

Repo: https://github.com/GodlyDonuts/Psi · everything below is committed + pushed to `main`.

---

## What Psi is

A personal **state-of-the-art Small Language Model on a fully custom stack** — own autograd, runtime, and
GPU kernels, **no PyTorch**. The model is the byproduct; the real goals are **mastering the full stack,
producing something genuinely novel, and the craft.** North-star metric: **capability-per-bit** (smallest
model in *bits* to clear a capability bar).

**🎯 Active concrete goal:** the **smallest model (params → bits) that still does what Microsoft's
TinyStories did** — write coherent, grammatical, consistent simple children's stories — using 2026
techniques + our radical stack. TinyStories (2023) did it at ~2.5M params; we're trying to go **below 1M**,
then crush the *bits* with ternary (~1.58-bit) weights on our own kernel.

---

## Status by layer (what's built & working)

| Layer | State | Where |
|---|---|---|
| **Step 0** scalar autograd | ✅ done, grad-checked | `src/step0_scalar_autograd/` |
| **Step 1** tensor autograd (the engine everything builds on) | ✅ done, 12/12 ops grad-checked ~1e-12 | `src/step1_tensor_autograd/tensor.hpp` |
| **Step 2** psi-nano (char-level GPT, the prototype) | ✅ trains, generates fluent corpus English | `src/step2_psi_nano/` (`main.cpp`, `model.hpp`, `nn.hpp`) |
| **Step 3** GPU kernels (Metal) | ✅ matmul **parity-class with MLX** (99% on key shape) + a **novel ternary GEMM** (16× smaller weights, full speed) | `src/step3_metal/`, writeup `docs/GPU_KERNELS.md` |
| **psi-stories** modern sub-1M model | ✅ built, trains/saves/loads/generates; all techniques grad-checked | `src/step2_psi_nano/stories.cpp` + `model_stories.hpp` |
| **Capability bar** (the eval) | ✅ prompts + rubric; Claude grades | `eval/tinystories_prompts.txt`, `docs/EVAL.md` |
| **First training run** | ⚠️ done but **memory-capped** (see below) | `docs/OVERNIGHT_REPORT.md`, `models/` |

### The modern psi-stories architecture (`model_stories.hpp`, `ModernGPT`)
Every capability-per-param technique from the research, all on grad-checked ops:
**small-BPE** (`bpe.hpp`) · **multi-head + GQA** · **RoPE** · **SwiGLU** · **block-wise weight-sharing**
(depth at ~0 param cost) · **tied embeddings** · **RMSNorm** · **WSD** LR schedule.

### The GPU kernel contribution (`docs/GPU_KERNELS.md`)
- Matmul tuned from ~28% → **45–54% of M1 peak** (float4 + 64×64/8sg tiling + bank-conflict padding),
  **bit-exact**, matching MLX's *method* (verified against their source), parity on the key shape.
- **Ternary-weight GEMM** (`ternary_gemm.mm`) — the unique edge: weights {−1,0,+1} at ~16× less memory,
  full fp32-GEMM speed. Capability-per-bit, which MLX has no path for.
- Findings: M1 matrix units are **precision-independent** (fp16 ≠ faster); epilogue **fusion doesn't pay**
  on M1.

---

## Latest result (overnight 2026-06-18) — read `docs/OVERNIGHT_REPORT.md`

**Headline:** `nano_130k` — a **131K-param** model writes grammatical, TinyStories-flavored English with
dialogue + named characters after **4 min / 4000 steps** (loss 2.70). Graded **5/3/2/2** — it *drifts*
(undertrained), doesn't clear the bar yet, but the shape is very promising for the size.

| model | params | steps reached | loss | outcome |
|---|---:|---:|---:|---|
| **nano_130k** | 131K | **4000 ✓** | 2.70 | graded 5/3/2/2 (undertrained) |
| tiny_215k | 215K | ~700 (OOM) | 3.21 | no checkpoint |
| small_350k | 354K | ~400 (OOM) | 3.68 | no checkpoint |
| mid_570k | 574K | ~200 (OOM) | 5.44 | no checkpoint |
| flagship_900k | 918K | ~0 (OOM) | — | no checkpoint |

**Nothing cleared the bar — that's UNDERTRAINING, not architecture.** Loss curves were all still falling.

---

## ⛔ The current blocker: 8GB Mac memory

The campaign OOM'd repeatedly. Diagnosed + fixed along the way (gradient accumulation; ctx 128→64; and the
real one — a **per-step autograd memory leak ~6 MB/step**, cut **15×** via graph-teardown-after-backward in
`tensor.hpp` + rare held-out eval in `stories.cpp`). A **~0.4 MB/step residual** (likely allocator
high-water) + a **~1.8 GB Metal/framework baseline** still cap trainable steps on this 8 GB machine, and
the cost **scales with model size** — so only the smallest model trained meaningfully.

---

## ➡️ Immediate next steps (in order)

1. **Unblock full training**, pick one:
   - **Chunked checkpoint-resume**: train in ~600-step chunks, each a *fresh process* (memory resets),
     saving + reloading the checkpoint between chunks. Needs a small "resume from checkpoint" mode in
     `stories.cpp` + a wrapper loop. Bounds memory regardless of the residual leak. **(~½ day, works on 8GB.)**
   - **Move to the GH200** (the planned scale-up) — sidesteps the 8 GB ceiling entirely.
2. **Re-run the frontier sweep properly** to loss ~1.5–2.0, grade against the bar → the *real* smallest
   config that clears it.
3. **Ternary-QAT** the smallest passing config (BitNet **16-to-1.58** recipe, see `docs/RESEARCH.md`) →
   the capability-per-bit record (a passing ~300K model at 1.58 bits ≈ ~60 KB).
4. _(optional)_ chase the ~0.4 MB/step residual (pooled allocator / `malloc_trim`) so full runs fit on 8 GB.

---

## How to build & run

```sh
# psi-stories (modern sub-1M model) — float build links the Metal GPU matmul backend
clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \
  src/step2_psi_nano/stories.cpp src/step3_metal/metal_backend.mm \
  -framework Metal -framework Foundation -o psi_stories

# data (gitignored — fetch per data/README.md)
curl -L -o data/tinystories-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# train  [data] [steps] [vocab d layers ctx hidden heads n_kv n_unique]   (nano_130k config shown)
./psi_stories train data/tinystories-valid.txt 4000 512 64 4 64 192 4 2 2
./psi_stories eval  psi_stories.bin eval/tinystories_prompts.txt 0.7      # completions to grade
./psi_stories gen   psi_stories.bin "Once upon a time"

# grad-checks (must stay 12/12 PASS before relying on any op)
clang++ -std=c++17 -O2 src/step2_psi_nano/gradcheck.cpp -o step2_gradcheck && ./step2_gradcheck
```

Each trained model is reproducible from `models/<name>/MODEL.md` (exact config + command + git commit).

---

## Doc map

- **`docs/OVERNIGHT_REPORT.md`** — the latest run + the full OOM-debugging saga.
- **`docs/RESEARCH.md`** — every technique considered + verdict for our regime (incl. ternary recipe);
  frontier-checked 2026-06-18.
- **`docs/GPU_KERNELS.md`** — the Metal kernel journey + ternary GEMM + findings.
- **`docs/EVAL.md`** — the capability bar (rubric + method).
- **`models/README.md`** — the model zoo + the capability-per-bit frontier table.
- **`data/README.md`** — how to fetch the TinyStories data.
- **`docs/DESIGN.md` · `docs/RADICAL.md` · `docs/SHOWCASE.md` · `docs/00-charter.md`** — vision/strategy.

## Working agreement (how we operate)
Claude implements, user steers & learns; everything written to be read; novelty first-class; **never
suggest "just use PyTorch"** (the custom stack IS the point); correctness is never traded for speed
(grad-checks gate every op); "better" must be a *measured* number.

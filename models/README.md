# Models — the psi-stories capability-per-bit zoo

Each subfolder here is **one trained model variation**, self-contained and reproducible. The mission
([docs/EVAL.md](../docs/EVAL.md)): find the **smallest model — in params, then in bits — that still
clears the TinyStories bar** (coherent, grammatical, consistent simple stories). TinyStories (2023) did
it at ~2.5M params; the lowest reported is ~1M. We're trying to go **below 1M** with 2026 techniques
(small BPE, multi-head attention, and ultimately ternary ~1.58-bit weights via our own kernel).

## What's in each `models/<name>/` folder

| file | what |
|---|---|
| `MODEL.md` | manifest — exact config, the one-line **reproduce command**, the git commit it was trained at, final loss, and the capability-bar grade |
| `train.txt` | full training curve (train + held-out val loss) |
| `eval.txt` | the model's completions on the bar prompts ([eval/tinystories_prompts.txt](../eval/tinystories_prompts.txt)) |
| `model.bin` | the trained checkpoint (not committed — large; regenerate via `MODEL.md`) |

The data is shared and documented in [data/README.md](../data/README.md). The whole sweep is produced by
[`run_overnight.sh`](../run_overnight.sh).

## The frontier (filled in as models are graded)

Grades are 1–10 per the [rubric](../docs/EVAL.md); a model "clears the bar" at roughly ≥7 on Grammar,
Coherence, and Consistency. The **smallest row that passes is the result.**

All variations use the **full modern stack**: small-BPE · multi-head + **GQA** · **RoPE** · **SwiGLU** ·
**block-wise weight-sharing** (deep-and-thin) · tied embeddings · **WSD** schedule (see [RESEARCH.md](../docs/RESEARCH.md)).

> **First-pass result (8GB M1) — see [docs/OVERNIGHT_REPORT.md](../docs/OVERNIGHT_REPORT.md).** A memory
> ceiling capped trainable steps, and the per-step cost **scales with model size**, so only the smallest
> model (nano_130k) trained meaningfully — the rest OOM'd undertrained (no checkpoint). Nothing cleared the
> bar: that's **undertraining, not architecture**. Full training (chunked resume / roomier machine) is next.

| model | params | size | steps reached | loss | Gram | Coh | Cons | Plot | clears bar? |
|---|---|---|---|---|---|---|---|---|---|
| **nano_130k** | 131K | 0.5 MB | **4000 ✓** | 2.70 | 5 | 3 | 2 | 2 | ❌ grammatical + right vocab, drifts (undertrained) |
| tiny_215k | 215K | 0.9 MB | ~700 (OOM) | 3.21 | — | — | — | — | ❌ OOM, no checkpoint |
| small_350k | 354K | 1.4 MB | ~400 (OOM) | 3.68 | — | — | — | — | ❌ OOM, no checkpoint |
| mid_570k | 574K | 2.3 MB | ~200 (OOM) | 5.44 | — | — | — | — | ❌ OOM, no checkpoint |
| flagship_900k | 918K | 3.7 MB | ~0 (OOM) | — | — | — | — | — | ❌ OOM, no checkpoint |

_Next after the fp32 frontier is found: re-train the smallest passing config with **ternary (~1.58-bit)**
weights → the same capability at ~16× fewer bits (e.g. a passing 400K model → ~0.08 MB)._

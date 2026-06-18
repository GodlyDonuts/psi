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

| model | params | bits/wt | size | Gram | Coh | Cons | Plot | clears bar? |
|---|---|---|---|---|---|---|---|---|
| flagship_965k | ~965K | 32 (fp32) | ~3.9 MB | — | — | — | — | _pending overnight run_ |
| ablate_1head  | ~965K | 32 | ~3.9 MB | — | — | — | — | _ablation: multi-head off_ |
| mid_600k      | ~600K | 32 | ~2.4 MB | — | — | — | — | _pending_ |
| small_450k    | ~450K | 32 | ~1.8 MB | — | — | — | — | _pending_ |
| tiny_250k     | ~250K | 32 | ~1.0 MB | — | — | — | — | _pending_ |
| nano_150k     | ~150K | 32 | ~0.6 MB | — | — | — | — | _pending_ |

_Next after the fp32 frontier is found: re-train the smallest passing config with **ternary (~1.58-bit)**
weights → the same capability at ~16× fewer bits (e.g. a passing 400K model → ~0.08 MB)._

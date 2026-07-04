# Models

Each subfolder here is one trained model, self-contained and reproducible. The goal (see [docs/EVAL.md](../docs/EVAL.md)) is to find the smallest model, in params first and then in bits, that still clears the TinyStories bar: coherent, grammatical, consistent simple stories. TinyStories (2023) did it at about 2.5M params, and the lowest reported is around 1M. I'm pushing below 1M with 2026 techniques: small BPE, multi-head attention, and eventually ternary ~1.58-bit weights via my own kernel.

## What's in each `models/<name>/` folder

| file | what |
|---|---|
| `MODEL.md` | manifest: exact config, the one-line reproduce command, the git commit it was trained at, final loss, and the capability-bar grade |
| `train.txt` | full training curve (train + held-out val loss) |
| `eval.txt` | the model's completions on the bar prompts ([eval/tinystories_prompts.txt](../eval/tinystories_prompts.txt)) |
| `model.bin` | the trained checkpoint, small enough to ship, so the model runs straight from the repo |

The data is shared and documented in [data/README.md](../data/README.md).

## The current models

There are four: femto (115K), nano (215K), small (354K), mid (574K). Grades are 1-10 per the [rubric](../docs/EVAL.md), and a model clears the bar at roughly 7 or higher on Grammar, Coherence, and Consistency. The smallest row that passes is the result.

All four use the same modern stack: small-BPE, multi-head with GQA, RoPE, SwiGLU, block-wise weight-sharing (deep-and-thin), tied embeddings, and a WSD schedule. See [RESEARCH.md](../docs/RESEARCH.md) for the details.

The headline so far: small at 354K gets the same 8/6/6/5 as TinyStories-1M at a third of the size. None of my sub-1M models clear the coherence bar cleanly, so the honest read is a capability-per-parameter win, not a raw-quality win.

Once the fp32 frontier is nailed down, the next step is to re-train the smallest passing config with ternary (~1.58-bit) weights for the same capability at roughly 16x fewer bits (a passing 400K model would land near 0.08 MB).

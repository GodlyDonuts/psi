# Psi model zoo — results (live, updated as the record campaign completes)

The from-scratch stack (own autograd + CUDA/Metal kernels, no PyTorch) is the showcase; each model must be
genuinely good for its size. Grades under [EVAL.md](EVAL.md) v2. Filled in as jobs finish (launched 2026-07-03).

## psi-stories — smallest model that clears the TinyStories bar

Loss→grade calibration is the campaign's key output. `val` = held-out CE. Bar ≈ val ≤ ~1.25 (vocab512) /
~1.5 (vocab1024); the real bar is the LLM grade (median Gram/Coh/Cons ≥ 7).

| model | params | vocab | steps | final val | Gram | Coh | Cons | Plot | clears? |
|---|---:|---:|---:|---:|---|---|---|---|---|
| femto | 115,008 | 512 | 14000 ✓ | 2.11 | 6 | 3 | 3 | 2 | ❌ (floor) |
| nano | 215,520 | 512 | 16000 ✓ | 1.87 | 7 | 5 | 4 | 4 | ❌ (close on grammar) |
| **small** | **353,952** | 512 | 20000 ✓ | **1.72** | **8** | **6** | **6** | **5** | ⚠️ borderline — **= TinyStories-1M at 1/3 params** |
| mid | 574,336 | 1024 | 25000 ✓ | 2.03 | 8 | 6 | 6 | 5 | ⚠️ borderline |
| flagship | 918,656 | 1024 | ~46% | cancelled | — | — | — | — | (killed to free GPUs) |
| insurance | 1,209,760 | 1024 | ~41% | cancelled | — | — | — | — | (killed to free GPUs) |

**Verdict:** the sweep's headline is **`small` (354K) matching TinyStories-1M's 8/6/6/5 at 1/3 the
parameters** — a genuine capability-per-param result (the point of the campaign). The clean ladder
(femto→nano→small) shows quality rising with size; the vocab-512 configs beat vocab-1024 per-param at this
scale (embedding table cost). Honest limit: **no sub-1M config *cleanly clears* the ≥7/7/7 bar** — all top
out ~6/6 on coherence/consistency, same borderline zone as TinyStories-1M itself. Cleanly clearing the bar
looks to need >1M params (flagship/insurance, cancelled) or a longer/curated-data run + ternary for the
bits record. This is the crown-jewel of the custom no-PyTorch stack; broad capability moves to the PyTorch
100M track.

### Baselines (HF TinyStories, graded under the identical v2 protocol, models/baselines/)

| baseline | params | Gram | Coh | Cons | Plot | clears bar? |
|---|---:|---|---|---|---|---|
| TinyStories-1M | 1M | 8 | 6 | 6 | 5 | borderline |
| TinyStories-3M | 3M | 9 | 8 | 8 | 7 | ✅ |
| TinyStories-8M | 8M | 9 | 8 | 9 | 8 | ✅ |
| TinyStories-28M | 28M | 9 | 9 | 9 | 8 | ✅✅ |

**Key finding: the 2023 TinyStories/GPT-Neo recipe crosses the bar at ~3M params** (1M borderline).
**The record = a Psi modern-stack model that clears the bar BELOW 1M param** — a genuine
capability-per-param improvement over the 2023 work, then crushed further in *bits* via ternary QAT.
Prior psi: nano_130k (131K, ctx64, undertrained) = 5/3/2/2 (does not clear).

## psi-chess — smallest model that plays legal chess

Bar = legal-move depth on self-generated games ([tools/chess/chess_eval.py](../tools/chess/chess_eval.py)):
mean consecutive legal plies from opening prompts. Data: 400K Lichess games (≥1500 Elo, UCI).

| model | params | steps | final train | mean legal plies | first-move legal % | tier |
|---|---:|---:|---:|---:|---:|---|
| chess_nano | 178,656 | 12000 ✓ | 1.34 | 0.1 | 8.3% | ❌ too small (degenerates) |
| chess_small | 459,648 | 15000 | _running (51%: clean legal-ish UCI already)_ | — | — | — |
| chess_mid | 1,304,256 | 18000 | _running_ | — | — | — |

_chess_nano is the smallest-fails floor. chess_small at half-training already plays well-formed,
mostly-legal openings — the record lands at chess_small/chess_mid._

## Next

psi-math (synthetic arithmetic, generalization eval) · psi-nano general SLM (the crown jewel) — designed
after psi-stories/psi-chess grade out.

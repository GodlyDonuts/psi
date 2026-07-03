# Psi model zoo — results (live, updated as the record campaign completes)

The from-scratch stack (own autograd + CUDA/Metal kernels, no PyTorch) is the showcase; each model must be
genuinely good for its size. Grades under [EVAL.md](EVAL.md) v2. Filled in as jobs finish (launched 2026-07-03).

## psi-stories — smallest model that clears the TinyStories bar

Loss→grade calibration is the campaign's key output. `val` = held-out CE. Bar ≈ val ≤ ~1.25 (vocab512) /
~1.5 (vocab1024); the real bar is the LLM grade (median Gram/Coh/Cons ≥ 7).

| model | params | vocab | steps | final val | Gram | Coh | Cons | Plot | clears? |
|---|---:|---:|---:|---:|---|---|---|---|---|
| femto | 115,008 | 512 | 14000 | _running_ | — | — | — | — | — |
| nano | 215,520 | 512 | 16000 | _running_ | — | — | — | — | — |
| small | 353,952 | 512 | 20000 | _running_ | — | — | — | — | — |
| mid | 574,336 | 1024 | 25000 | _running_ | — | — | — | — | — |
| flagship | 918,656 | 1024 | 26000 | _running_ | — | — | — | — | — |
| insurance | 1,209,760 | 1024 | 26000 | _running_ | — | — | — | — | — |

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

| model | params | steps | final val | mean legal plies | first-move legal % | tier |
|---|---:|---:|---:|---:|---:|---|
| chess_nano | 178,656 | 12000 | _running_ | — | — | — |
| chess_small | 459,648 | 15000 | _running_ | — | — | — |
| chess_mid | 1,304,256 | 18000 | _running_ | — | — | — |

## Next

psi-math (synthetic arithmetic, generalization eval) · psi-nano general SLM (the crown jewel) — designed
after psi-stories/psi-chess grade out.

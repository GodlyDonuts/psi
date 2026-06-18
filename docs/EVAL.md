# Psi — the capability bar for `psi-stories`

> The goal is **the smallest model (in bits) that still does what TinyStories did**: write coherent,
> grammatical, consistent simple children's stories. You can't search for "smallest that clears the bar"
> without the bar. This is it — the measurement that turns the capability-per-bit search into a number.

## The method (TinyStories' "GPT-Eval", 2026 edition)

The original TinyStories paper graded models by feeding a strong LLM a story beginning + the small
model's completion, and scoring it. We do the same — **the grader is Claude.** The harness generates
completions; a strong model reads them and scores the rubric below.

```sh
psi_nano eval <model.bin> [eval/tinystories_prompts.txt]   # generates a completion per prompt
```

The prompts ([`eval/tinystories_prompts.txt`](../eval/tinystories_prompts.txt)) are TinyStories-style
openings — a named character, a simple setup, an unfinished sentence the model must continue. They probe
the three things that separate "speaks English" from "tells a story": grammar, local coherence, and
**consistency with the setup** (does it keep the same characters/objects and finish the thought?).

## The rubric (each 1–10, graded per completion)

| Dimension | Question |
|---|---|
| **Grammar** | Is it grammatically correct, well-formed English? |
| **Coherence** | Does it flow and make sense sentence-to-sentence? |
| **Consistency** | Does it stay true to the prompt — same characters, objects, situation — and finish the thought? |
| **Plot / creativity** | Is there a sensible little arc (a beginning→middle→end), not just a run-on? |

## The bar — "clears TinyStories"

A model **clears the bar** when, averaged over the prompt set, it scores roughly **≥ 7/10 on
Grammar, Coherence, and Consistency** — i.e. a layperson reading the completion would believe it's a
real (if simple) children's story, the way TinyStories' ~10–33M models did. Plot/creativity is the
stretch dimension (it's what the largest TinyStories models added).

Reference points from the 2023 paper: ~1–3M params already produce **grammatical** text; **coherent and
consistent** stories emerge around **~10M+**. Our target is to hit that coherence/consistency bar at
**fewer params and far fewer bits** (modern architecture + distillation + ternary QAT) — see
[GPU_KERNELS.md](GPU_KERNELS.md) for the ternary kernel and [RADICAL.md](RADICAL.md) for the thesis.

## Recording results

Each model we train gets a row: `params · bits · Grammar/Coherence/Consistency/Plot · pass?`. The
**smallest (in bits) row that passes is the result** — the capability-per-bit frontier. Tracked here as
we shrink.

| model | params | bits/wt | size | Gram | Coh | Cons | Plot | pass? |
|---|---|---|---|---|---|---|---|---|
| psi-nano (char, ctx 32, 8s train) | 106K | 32 | 0.4 MB | 1 | 1 | 1 | 1 | ❌ floor — char stats only, no valid words |

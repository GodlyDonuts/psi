# Psi: the capability bar for `psi-stories`

> The goal is the smallest model (in bits) that still does what TinyStories did: write coherent,
> grammatical, consistent simple children's stories. You can't search for "smallest that clears the bar"
> without a bar. This is it, the measurement that turns the capability-per-bit search into a number.

## The method (TinyStories' "GPT-Eval", 2026 edition)

The original TinyStories paper graded models by feeding a strong LLM a story beginning plus the small
model's completion, then scoring it. I do the same: the grader is a strong LLM. The harness generates
completions, and a strong model reads them and scores the rubric below.

```sh
psi_nano eval <model.bin> [eval/tinystories_prompts.txt]   # generates a completion per prompt
```

The prompts ([`eval/tinystories_prompts.txt`](../eval/tinystories_prompts.txt)) are TinyStories-style
openings: a named character, a simple setup, an unfinished sentence the model has to continue. They probe
the three things that separate "speaks English" from "tells a story": grammar, local coherence, and
consistency with the setup (does it keep the same characters and objects and finish the thought?).

## The rubric (each 1-10, graded per completion)

| Dimension | Question |
|---|---|
| Grammar | Is it grammatically correct, well-formed English? |
| Coherence | Does it flow and make sense sentence-to-sentence? |
| Consistency | Does it stay true to the prompt (same characters, objects, situation) and finish the thought? |
| Plot / creativity | Is there a sensible little arc (a beginning, middle, end), not just a run-on? |

## The bar: "clears TinyStories"

A model clears the bar when, averaged over the prompt set, it scores roughly ≥ 7/10 on Grammar,
Coherence, and Consistency. That's the point where a layperson reading the completion would believe it's
a real (if simple) children's story, the way TinyStories' ~10-33M models did. Plot/creativity is the
stretch dimension, the thing the largest TinyStories models added.

Reference points from the 2023 paper: ~1-3M params already produce grammatical text, and coherent,
consistent stories emerge around ~10M+. My target is to hit that coherence/consistency bar at fewer
params and far fewer bits (modern architecture plus ternary QAT). See
[GPU_KERNELS.md](GPU_KERNELS.md) for the ternary kernel and [RADICAL.md](RADICAL.md) for the thesis.

## Recording results

Each model I train gets a row: `params · bits · Grammar/Coherence/Consistency/Plot · pass?`. The
smallest row (in bits) that passes is the result, the capability-per-bit frontier. Tracked here as I
shrink it.

| model | params | bits/wt | size | Gram | Coh | Cons | Plot | pass? |
|---|---|---|---|---|---|---|---|---|
| psi-nano (char, ctx 32, 8s train) | 106K | 32 | 0.4 MB | 1 | 1 | 1 | 1 | ❌ floor: char stats only, no valid words |

---

## Protocol v2: FROZEN for the record campaign (2026-07-03)

The record claim ("smallest that clears the bar") is only defensible if every model is graded under one
fixed protocol. This freezes it. Grades are only comparable within v2.

Generation (fixed, applies to every model including baselines):
- `eval <model> eval/tinystories_prompts.txt 0.7 --k 3 --seed 100 --nnew 256`
- temperature 0.7, full softmax (no top-k/top-p).
- k = 3 completions per prompt, seeds 100, 101, 102 (reproducible). Generation stops at the EOS token
  (`<|endoftext|>`), so each completion is one story, no multi-story spew.
- prompt set: `eval/tinystories_prompts.txt` (12 openings). Preflight verified 0 verbatim occurrences in
  the training slice (contamination-clean). Models train on `data/slice500.txt` with the valid split
  excluded.

Scoring and the bar:
- Grade each of the 12×3 = 36 completions 1-10 on Grammar / Coherence / Consistency / Plot.
- Per model, the score for a dimension is the median over all 36 completions (report median plus IQR).
  The median is robust to the occasional degenerate sample.
- Clears the bar ⇔ median Grammar, Coherence, Consistency all ≥ 7. Plot is the stretch dimension.
- Grading runs off-cluster: a strong LLM reads `models/<name>/eval.txt`, graded blind to model size
  where feasible.

Loss ↔ grade calibration (the campaign's key output): every finished model is graded regardless of size,
building the val-loss → grade curve so the smallest bar-clearing size can be pinned (and undertrained
small configs extended via resume). Estimated clear threshold ≈ val CE ≤ ~1.25 (vocab 512) / ~1.5 (vocab
1024), i.e. ~0.55-0.60 bits/char, to be confirmed empirically.

Baselines (record comparison): Huggingface `roneneldan/TinyStories-1M/3M/8M/28M` graded under this exact
sampler (temp 0.7, k=3, EOS-stop, same prompts). PyTorch is used only as an external yardstick, never in
Psi. This establishes what param-count the 2023 recipe needed versus mine.

Reproducibility: completions are binary-specific (`-march=native -ffast-math`), so each `models/<name>/`
archives the exact `psi_stories.exe` that produced its `eval.txt`.

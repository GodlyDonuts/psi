# Results

The goal is the smallest model that can still do what TinyStories showed: write short, grammatical,
coherent children's stories. I trained a sweep of sizes on the custom stack and graded each one against
the original TinyStories models under the same protocol ([EVAL.md](EVAL.md)).

## The sweep

Every model uses the same modern architecture (small BPE, grouped-query attention, RoPE, SwiGLU,
RMSNorm, tied embeddings, block-wise weight sharing) trained on ~200M tokens of TinyStories V2 with a
warmup-stable-decay schedule. Grades are 1 to 10 on grammar, coherence, consistency, and plot, from 36
completions each.

| model | params | vocab | val loss | grammar | coherence | consistency | plot |
|---|---:|---:|---:|:--:|:--:|:--:|:--:|
| femto | 115K | 512 | 2.11 | 6 | 3 | 3 | 2 |
| nano | 215K | 512 | 1.87 | 7 | 5 | 4 | 4 |
| **small** | **354K** | 512 | **1.72** | **8** | **6** | **6** | **5** |
| mid | 574K | 1024 | 2.03 | 8 | 6 | 6 | 5 |

## Comparison to the original TinyStories models

I ran the published TinyStories models through the identical grading protocol as a baseline.

| model | params | grammar | coherence | consistency | plot |
|---|---:|:--:|:--:|:--:|:--:|
| TinyStories-1M | 1M | 8 | 6 | 6 | 5 |
| TinyStories-3M | 3M | 9 | 8 | 8 | 7 |
| TinyStories-8M | 8M | 9 | 8 | 9 | 8 |
| TinyStories-28M | 28M | 9 | 9 | 9 | 8 |

The result I care about: `small` at 354K parameters gets the same 8/6/6/5 as TinyStories-1M, at a third
of the size. The 2023 recipe needed about 3M parameters to fully clear the coherence bar (1M is
borderline). None of my sub-1M models clear it cleanly either, so the honest read is a capability-per-
parameter win, not a raw-quality win: I match the smallest published TinyStories model at a fraction of
the parameters, using the modern architecture on a from-scratch stack.

A couple of things the sweep showed:

- Quality climbs cleanly with size (femto to nano to small), and the samples degrade in a readable way:
  femto repeats itself, nano writes grammatical sentences that drift off topic, small holds a simple
  thread with dialogue.
- The vocab-512 models beat the vocab-1024 ones per parameter at this scale. At 574K, `mid` spends about
  131K parameters on its embedding table for vocab 1024, which is why it barely edges the much smaller
  vocab-512 models. `small` (354K, vocab 512) is the better point on the frontier.

## What a completion looks like

`small`, prompted with the start of a story:

> Once upon a time, there was a little girl named Lily. She found a shiny red ball in the park. She
> wanted to play with it, so she asked her mom if she could have the ball. Her mom said, "Okay, but you
> must be careful..."

## Reproduce

Each model folder has a `MODEL.md` with the exact training command and config, plus its `train.txt`
(loss curve), `eval.txt` (graded completions), and `model.bin` (the checkpoint). The models are runnable
straight from the repo:

```sh
./psi_stories gen models/small/model.bin "Once upon a time"
```

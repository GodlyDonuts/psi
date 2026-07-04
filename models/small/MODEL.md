# psi-stories / small

Sweep config in the sub-1M **capability-per-bit** TinyStories record campaign
(see [docs/RECORD_CAMPAIGN.md](../../docs/RECORD_CAMPAIGN.md), [docs/EVAL.md](../../docs/EVAL.md)).

| field | value |
|---|---|
| parameters | **353952** |
| architecture (args) | `512 96 9 256 256 4 2 3` = vocab d layers block hidden heads n_kv n_unique |
| training | 20000 steps · batch 32 · ctx 256 · AdamW lr=0.0025 wd 0.01 · WSD · grad-clip 1.0 |
| tokens seen | ~163840000 (463 tok/param) |
| data | data/slice500.txt (500MB dedup, valid-excluded — see data/slice500.txt.manifest) |
| tokenizer | s512 (BPE, in-band EOS, boundary-aware sampling) |
| code version | git `e8265f7` |
| final | step  19900   train 1.3760   val   —    |g|=0.36  rss=1992MB  (71750.0s) |

## Reproduce
```sh
module load cuda/cuda-12.6.0 && bash src/step4_cuda/build_cuda.sh
./psi_stories_cuda train data/slice500.txt 20000 512 96 9 256 256 4 2 3 --ids data/s512.ids --tok data/s512.tok \
   --batch 32 --lr 0.0025 --clip 1.0 --out model.bin
./psi_stories_cuda eval model.bin eval/tinystories_prompts.txt 0.7 --k 3 --seed 100
```

## Capability-bar grade (docs/EVAL.md v2 rubric, graded off-cluster)

Final val CE **1.72** (vocab 512 — best of the vocab-512 sweep). Graded on 36 completions:
**Grammar 8 · Coherence 6 · Consistency 6 · Plot 5 · clears bar? ⚠️ borderline (no)**

**The capability-per-param headline.** This grade *matches TinyStories-1M (roneneldan) — 8/6/6/5 — at
**1/3 the parameters** (354K vs 1M).* Grammatical with real dialogue ("Okay, but you must be careful..."),
mostly coherent, with some drift (a stray "I am Max", bird/girl mixups). Like TinyStories-1M it's borderline
and doesn't cleanly clear the ≥7/7/7 bar — but it reaches that quality at a third the size, the intended
capability-per-param result. The ladder: femto 6/3/3/2 → nano 7/5/4/4 → mid(574K,v1024) 8/6/6/5 →
**small(354K,v512) 8/6/6/5** — small is the frontier point (best quality-per-param). Clearing the bar cleanly
appears to need >1M params (flagship/insurance, cancelled) or a longer/curated-data run.

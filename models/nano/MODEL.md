# psi-stories / nano

Sweep config in the sub-1M **capability-per-bit** TinyStories record campaign
(see [docs/RECORD_CAMPAIGN.md](../../docs/RECORD_CAMPAIGN.md), [docs/EVAL.md](../../docs/EVAL.md)).

| field | value |
|---|---|
| parameters | **215520** |
| architecture (args) | `512 96 8 256 192 4 2 2` = vocab d layers block hidden heads n_kv n_unique |
| training | 16000 steps · batch 32 · ctx 256 · AdamW lr=0.0025 wd 0.01 · WSD · grad-clip 1.0 |
| tokens seen | ~131072000 (608 tok/param) |
| data | data/slice500.txt (500MB dedup, valid-excluded — see data/slice500.txt.manifest) |
| tokenizer | s512 (BPE, in-band EOS, boundary-aware sampling) |
| code version | git `e8265f7` |
| final | step  15900   train 1.5827   val   —    |g|=0.51  rss=1976MB  (48687.0s) |

## Reproduce
```sh
module load cuda/cuda-12.6.0 && bash src/step4_cuda/build_cuda.sh
./psi_stories_cuda train data/slice500.txt 16000 512 96 8 256 192 4 2 2 --ids data/s512.ids --tok data/s512.tok \
   --batch 32 --lr 0.0025 --clip 1.0 --out model.bin
./psi_stories_cuda eval model.bin eval/tinystories_prompts.txt 0.7 --k 3 --seed 100
```

## Capability-bar grade (docs/EVAL.md v2 rubric, graded off-cluster)

Final val CE **1.87** (vocab 512). Graded on 36 completions:
**Grammar 7 · Coherence 5 · Consistency 4 · Plot 4 · clears bar? ❌ NO (close on grammar)**

Second rung of the ladder (femto 6/3/3/2 → **nano 7/5/4/4** → small ~8/7/7/6). Clean grammar + real
dialogue ("Why are you sad, Lily?"), but still drifts into non-sequiturs (a prince, "a big, dangerous
sun" in Ben's fridge). ~TinyStories-1M-class quality at **1/5 the params** (215K vs 1M) — a per-param win
on efficiency, though it doesn't yet clear the coherence/consistency bar. The record candidate is `small`.

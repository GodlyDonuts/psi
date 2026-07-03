# psi-stories / femto

Sweep config in the sub-1M **capability-per-bit** TinyStories record campaign
(see [docs/RECORD_CAMPAIGN.md](../../docs/RECORD_CAMPAIGN.md), [docs/EVAL.md](../../docs/EVAL.md)).

| field | value |
|---|---|
| parameters | **115008** |
| architecture (args) | `512 64 8 256 160 4 1 2` = vocab d layers block hidden heads n_kv n_unique |
| training | 14000 steps · batch 32 · ctx 256 · AdamW lr=0.003 wd 0.01 · WSD · grad-clip 1.0 |
| tokens seen | ~114688000 (997 tok/param) |
| data | data/slice500.txt (500MB dedup, valid-excluded — see data/slice500.txt.manifest) |
| tokenizer | s512 (BPE, in-band EOS, boundary-aware sampling) |
| code version | git `e8265f7` |
| final | step  13900   train 1.7933   val   —    |g|=0.67  rss=1959MB  (36554.3s) |

## Reproduce
```sh
module load cuda/cuda-12.6.0 && bash src/step4_cuda/build_cuda.sh
./psi_stories_cuda train data/slice500.txt 14000 512 64 8 256 160 4 1 2 --ids data/s512.ids --tok data/s512.tok \
   --batch 32 --lr 0.003 --clip 1.0 --out model.bin
./psi_stories_cuda eval model.bin eval/tinystories_prompts.txt 0.7 --k 3 --seed 100
```

## Capability-bar grade (docs/EVAL.md v2 rubric, graded off-cluster)

Final val CE **2.11** (vocab 512). Graded on 36 completions (12 prompts × 3):
**Grammar 6 · Coherence 3 · Consistency 3 · Plot 2 · clears bar? ❌ NO**

The **smallest stories config** and the record's lower floor. Phrase-level grammar is decent, but at 115K
params it can't hold a thread — obsessive repetition ("the ball… the ball…") and non-sequiturs (Ben's
fridge → "a big ball" → "a tree"). Better than the old nano_130k (5/3/2/2) at a smaller size and lower
loss, but far from coherent. Coherence needs more capacity — the record lands in the bigger configs.

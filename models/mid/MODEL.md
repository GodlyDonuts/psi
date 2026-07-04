# psi-stories / mid

Sweep config in the sub-1M capability-per-bit TinyStories record campaign
(see [docs/RECORD_CAMPAIGN.md](../../docs/RECORD_CAMPAIGN.md), [docs/EVAL.md](../../docs/EVAL.md)).

| field | value |
|---|---|
| parameters | **574336** |
| architecture (args) | `1024 128 6 256 256 4 2 3` = vocab d layers block hidden heads n_kv n_unique |
| training | 25000 steps · batch 32 · ctx 256 · AdamW lr=0.002 wd 0.01 · WSD · grad-clip 1.0 |
| tokens seen | ~204800000 (357 tok/param) |
| data | data/slice500.txt (500MB dedup, valid-excluded, see data/slice500.txt.manifest) |
| tokenizer | s1024 (BPE, in-band EOS, boundary-aware sampling) |
| code version | git `e8265f7` |
| final | step 24900, train 1.5488, val n/a, |g|=0.38, rss=1628MB (69752.9s) |

## Reproduce
```sh
module load cuda/cuda-12.6.0 && bash src/step4_cuda/build_cuda.sh
./psi_stories_cuda train data/slice500.txt 25000 1024 128 6 256 256 4 2 3 --ids data/s1024.ids --tok data/s1024.tok \
   --batch 32 --lr 0.002 --clip 1.0 --out model.bin
./psi_stories_cuda eval model.bin eval/tinystories_prompts.txt 0.7 --k 3 --seed 100
```

## Capability-bar grade (docs/EVAL.md v2 rubric, graded off-cluster)

Final val CE 2.03. The vocab is 1024 here, so this loss isn't comparable to the vocab-512 models' scale.
Graded on 36 completions: Grammar 8, Coherence 6, Consistency 6, Plot 5. Clears the bar? Borderline, no.

Roughly TinyStories-1M quality at 574K params. It's grammatical and some completions hold together ("a big
bowl of jelly on the table. Ben was very happy and ran to show his mom"), but it still drifts elsewhere ("a
big skeleton").

The catch: vocab-1024 spends about 131K params on the embedding table, so mid (574K) barely edges out the
vocab-512 nano (215K). At this scale the vocab-512 configs get more capability per parameter, which makes
`small` (354K, vocab 512) my stronger record candidate.

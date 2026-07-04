# The sub-1M TinyStories record campaign

The goal is the smallest model that clears the TinyStories bar ([EVAL.md](EVAL.md) v2), and the smartest per parameter. Runs on the UCF Newton cluster via the CUDA backend ([CUDA_BACKEND.md](CUDA_BACKEND.md)). The strategy is bracket-and-extend: sweep across sizes, pin the exact size where the grade bar crosses, and extend undertrained small configs via resume.

## Why this can work now

My earlier 131K run on the Mac (4000 steps, ctx 64) came out grammatical but drifting, and badly undertrained (~15 tok/param, loss still falling, OOM-capped on the 8GB Mac). The cluster removes the memory ceiling and the CUDA backend makes steps about 10x faster, so I can train the right token budgets at a context length that fits a whole story.

## The recipe (frozen)

ctx 256, because TinyStories run about 200 to 260 tokens and ctx 64 never saw a whole story. batch 32 via grad-accumulation, so memory stays flat and wall-clock tracks tokens (8192 tok/step cuts gradient noise 16x versus the old 512).

LR by width (muP-flavored): d64 to 3e-3, d96 to 2.5e-3, d128 to 2e-3, d160 to 1.5e-3, with a global grad-clip of 1.0.

WSD schedule with 3% warmup and last-20% decay to 0.1·lr. Steps are fixed at launch and the schedule is a fraction of planned steps, stored in the checkpoint so resume is stable.

Overtrain hard: from roughly 1000 tok/param at the smallest end down to a few hundred at the top. Data lives in `data/slice500.txt`: 500 MB, 641,541 stories, exact-dedup, valid split excluded, SHA-manifested. In-band EOS plus story-boundary-aware sampling, so a training sequence is one story from position 0 (matching generation).

## The sweep (params exact-formula-verified)

| name | args `vocab d L blk hid heads kv uniq` | params | steps | lr | tok/param | role |
|---|---|---:|---:|---:|---:|---|
| femto | `512 64 8 256 160 4 1 2` | 115,008 | 14000 | 3e-3 | ~1000 | calibration / extend |
| nano | `512 96 8 256 192 4 2 2` | 215,520 | 16000 | 2.5e-3 | ~608 | calibration / extend |
| small | `512 96 9 256 256 4 2 3` | 353,952 | 20000 | 2.5e-3 | ~463 | likely record band |
| mid | `1024 128 6 256 256 4 2 3` | 574,336 | 25000 | 2e-3 | ~357 | record band; vocab-1024 |

External anchors set expectations. llama2.c `stories260K` needed about 7000 tok/param for borderline coherence, and the TinyStories paper puts emergence near 10M. So the smallest configs probably won't clear the bar at their launch budgets, and the record most plausibly lands in the 354K to 574K range. femto and nano are calibration points and extension candidates: if their val curve is still falling, resume for 2 to 3x the steps. That's why full checkpoint-resume (bit-exact, preflight-verified) is the scientific backbone here, not ops plumbing.

## Running it

```sh
# once: tokenizer caches + data slice (preflight does this)
python3 tools/make_slice.py data/tinystories-train-full.txt data/tinystories-valid.txt data/slice500.txt 500
sbatch src/step4_cuda/preflight.sbatch        # build, tokenize s512/s1024, resume-equivalence, timing+RSS probes
# launch
bash   src/step4_cuda/submit_campaign.sh      # record band first, long jobs on 'normal'
# watch
nssh 'bash ~/Psi/src/step4_cuda/status.sh'    # queue + latest loss/grad-norm/rss per model
```

Each job writes a reproducible `models/<name>/`: MODEL.md (config, exact cmd, git commit, tok/param), train.txt, eval.txt, model.bin, and the exact binary. I grade off-cluster under [EVAL.md](EVAL.md) v2, fill in the grade, and the smallest bar-clearing row is the result. From there I can ternary-QAT it for the bits record.

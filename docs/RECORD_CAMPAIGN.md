# The sub-1M TinyStories record campaign

**Goal:** the **smallest model that clears the TinyStories bar** ([EVAL.md](EVAL.md) v2), and smartest-
per-param — a defensible record. Runs on the UCF Newton cluster via the CUDA backend
([CUDA_BACKEND.md](CUDA_BACKEND.md)). Strategy = **bracket-and-extend**: sweep 115K→1.21M params, pin the
exact size where the grade bar crosses, extend undertrained small configs via resume.

## Why this can work now

nano_130k (131K, 4000 steps, ctx 64) graded 5/3/2/2 — grammatical but drifting, and **badly
undertrained** (~15 tok/param; loss still falling; OOM-capped on the 8GB Mac). The cluster removes the
memory ceiling and the CUDA backend makes steps ~10× faster, so we can train the right token budgets at a
context length that fits a whole story.

## The recipe (frozen)

- **ctx 256** — TinyStories are ~200–260 tokens; ctx 64 never saw a whole story (nano_130k's 2/2
  Consistency/Plot). **batch 32** (grad-accumulation ⇒ memory flat, wall-clock ~ tokens; 8192 tok/step
  cuts gradient noise 16× vs the old 512).
- **LR by width** (muP-flavored): d64→3e-3, d96→2.5e-3, d128→2e-3, d160→1.5e-3; **global grad-clip 1.0**.
- **WSD** schedule (3% warmup, last-20% decay to 0.1·lr); steps fixed at launch (schedule is a fraction of
  planned steps — stored in the checkpoint so resume is stable).
- **Overtrain**: ~1000 tok/param (smallest) → ~175 (insurance). Data: `data/slice500.txt` — 500 MB,
  641,541 stories, exact-dedup, valid split excluded, SHA-manifested. In-band **EOS** + **story-boundary-
  aware sampling** so a training sequence is one story from position 0 (matches generation).

## The sweep (params exact-formula-verified)

| name | args `vocab d L blk hid heads kv uniq` | params | steps | lr | tok/param | role |
|---|---|---:|---:|---:|---:|---|
| femto | `512 64 8 256 160 4 1 2` | 115,008 | 14000 | 3e-3 | ~1000 | calibration / extend |
| nano | `512 96 8 256 192 4 2 2` | 215,520 | 16000 | 2.5e-3 | ~608 | calibration / extend |
| small | `512 96 9 256 256 4 2 3` | 353,952 | 20000 | 2.5e-3 | ~463 | likely record band |
| mid | `1024 128 6 256 256 4 2 3` | 574,336 | 25000 | 2e-3 | ~357 | record band; vocab-1024 |
| flagship | `1024 128 8 256 384 4 2 4` | 918,656 | 26000 | 2e-3 | ~232 | upper bracket |
| insurance | `1024 160 8 256 384 4 2 4` | 1,209,760 | 26000 | 1.5e-3 | ~176 | headline (must clear) |

**Calibration reality:** external anchors (llama2.c `stories260K` needed ~7000 tok/param for *borderline*
coherence; TinyStories paper emergence ~10M) say the smallest configs likely **won't** clear the bar at
launch budgets — the record most plausibly lands **354K–918K**. femto/nano are calibration points and
extension candidates (resume → 2–3× steps if their val curve is still falling). This is why full
checkpoint-resume (bit-exact, preflight-verified) is the campaign's scientific backbone, not ops plumbing.

## Running it

```sh
# once: tokenizer caches + data slice (preflight does this)
python3 tools/make_slice.py data/tinystories-train-full.txt data/tinystories-valid.txt data/slice500.txt 500
sbatch src/step4_cuda/preflight.sbatch        # build, tokenize s512/s1024, resume-equivalence, timing+RSS probes
# launch
bash   src/step4_cuda/submit_campaign.sh      # 6 jobs, record band + insurance first, long jobs on 'normal'
# watch
nssh 'bash ~/Psi/src/step4_cuda/status.sh'    # queue + latest loss/grad-norm/rss per model
```

Each job writes a reproducible `models/<name>/` = MODEL.md (config + exact cmd + git commit + tok/param) +
train.txt + eval.txt + model.bin + the exact binary. Grade off-cluster under [EVAL.md](EVAL.md) v2, fill in
the grade, and the smallest bar-clearing row is the result — then ternary-QAT it for the *bits* record.

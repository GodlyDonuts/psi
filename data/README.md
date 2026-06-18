# Data — TinyStories

The training corpus for `psi-stories`. The data files themselves are **not committed** (large), but
here's exactly how to get the ones our models use, so any run is reproducible.

## Source

**TinyStories V2 (GPT-4-generated)** — synthetic short stories in a ~3–4-year-old's vocabulary.
Hugging Face: [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) ·
paper [arXiv:2305.07759](https://arxiv.org/abs/2305.07759). We use the **V2-GPT4** files (cleaner than
the v1 GPT-3.5 mix). Plain text, stories separated by `<|endoftext|>`.

## Download the exact files we use

```sh
base=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main

# validation slice (~21 MB) — quick smoke tests / held-out eval
curl -L -o data/tinystories-valid.txt $base/TinyStoriesV2-GPT4-valid.txt

# ~100 MB train slice — what the overnight sub-1M campaign trains on (range request, first 100 MB)
curl -L -r 0-104857600 -o data/tinystories-train-100mb.txt $base/TinyStoriesV2-GPT4-train.txt

# (optional) the FULL train set (~2–3 GB, ~0.5–1B tokens) for a serious run
curl -L -o data/tinystories-train.txt $base/TinyStoriesV2-GPT4-train.txt
```

## Why a small slice

A sub-1M-parameter model can't absorb the full ~0.5–1B-token corpus — its *capacity*, not the data,
is the bottleneck. ~100 MB (~40M tokens) gives plenty of diversity for the frontier search and keeps a
run to a few hours on an M1. `run_overnight.sh` downloads the 100 MB slice automatically if it's absent.

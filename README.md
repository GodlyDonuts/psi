# Psi

Psi is a small language model and the training stack it runs on, both written from scratch in C++.
Own autograd, own GPU kernels, no PyTorch and no other ML framework.

I built it to learn how these systems work all the way down, by writing them instead of reading about
them. Autograd, the transformer, the GPU matmul, the training loop. The model stays small on purpose. I
wanted to see how much a model can do per parameter when you control every layer underneath it.

## What's in here

The stack, built up one layer at a time, each checked against the one below it:

- `src/step0_scalar_autograd` scalar reverse-mode autograd, gradient-checked against finite differences
- `src/step1_tensor_autograd` tensor autograd in double precision, the reference every other path is verified against (12/12 ops match analytic gradients to ~1e-12)
- `src/step2_psi_nano` the model: a GPT with a small BPE tokenizer, grouped-query attention, RoPE, SwiGLU, RMSNorm, tied embeddings, and block-wise weight sharing
- `src/step3_metal` hand-written Metal GPU kernels for Apple Silicon, including a ternary-weight matmul
- `src/step4_cuda` a CUDA backend so the same model trains on NVIDIA GPUs

And the trained models in `models/`: a sweep of TinyStories writers from 115K to 574K parameters.

## Results

The target is TinyStories: can a tiny model write short, grammatical, coherent children's stories. I
grade completions on grammar, coherence, consistency, and plot (see [docs/EVAL.md](docs/EVAL.md)), and
compare against the original TinyStories models under the same protocol.

| model | params | grammar | coherence | consistency | plot |
|---|---:|:--:|:--:|:--:|:--:|
| femto | 115K | 6 | 3 | 3 | 2 |
| nano | 215K | 7 | 5 | 4 | 4 |
| **small** | **354K** | **8** | **6** | **6** | **5** |
| mid | 574K | 8 | 6 | 6 | 5 |

The headline is `small`: at 354K parameters it scores the same as the original TinyStories-1M model
(8/6/6/5) at a third of the size. Full write-up in [docs/RESULTS.md](docs/RESULTS.md).

On the kernel side, the Metal matmul reaches 883 GFLOP/s at 2048³ on an M1 (34% of peak, 6x the naive
kernel), bit-exact against the CPU and within a few percent of MLX on the shapes I tuned for. The ternary
matmul stores weights as {-1, 0, +1}, which is 16x smaller than fp32, and runs at full fp32-matmul speed.
The kernel journey is in [docs/GPU_KERNELS.md](docs/GPU_KERNELS.md).

## Build and run

C++17, no dependencies for the CPU path. The GPU paths need Metal (Apple) or CUDA (NVIDIA).

```sh
# Step 0: scalar autograd + an XOR MLP + gradient check
clang++ -std=c++17 -O2 src/step0_scalar_autograd/main.cpp -o step0 && ./step0

# Step 1: tensor autograd, per-op gradient checks
clang++ -std=c++17 -O2 src/step1_tensor_autograd/main.cpp -o step1 && ./step1

# Step 2: grad-check the transformer ops (all pass)
clang++ -std=c++17 -O2 src/step2_psi_nano/gradcheck.cpp -o step2_gradcheck && ./step2_gradcheck
```

Build the TinyStories model (Apple Silicon, Metal backend):

```sh
clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \
  src/step2_psi_nano/stories.cpp src/step3_metal/metal_backend.mm \
  -framework Metal -framework Foundation -o psi_stories
```

## Try the trained models

The checkpoints in `models/` are ready to run. Point the binary at one and give it a prompt:

```sh
./psi_stories gen models/small/model.bin "Once upon a time, there was a little girl who"
```

Or use the helper, which picks a sensible prompt per model:

```sh
bash try.sh small
```

## Notes

The point of Psi was never to beat anyone's benchmark. It was to build the whole thing myself and find
out what happens. Two things I did not expect going in: how much of the work is measurement rather than
cleverness (three "smarter" matmul kernels lost to a simpler one that I only found by benchmarking), and
how far a few hundred thousand parameters can go when the training recipe is right.

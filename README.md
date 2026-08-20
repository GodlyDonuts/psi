# Psi

**A small language model and the training stack beneath it, built from scratch in C++17. Custom autograd, transformer, Metal kernels, and CUDA backend—no PyTorch or external ML framework.**

I built Psi to understand how these systems work all the way down by writing the stack instead of only reading about it: reverse-mode autodiff, tensor operations, the transformer, GPU matmul, tokenization, training, and evaluation.

The model stays small on purpose. I wanted every layer to remain inspectable, then to see how far a few hundred thousand parameters could go when I controlled the entire stack.

## What's in here

The stack is built one layer at a time, with each layer checked against the one below it:

- `src/step0_scalar_autograd` — scalar reverse-mode autograd, gradient-checked against finite differences
- `src/step1_tensor_autograd` — double-precision tensor autograd and the reference path for later backends; 12/12 operations match analytic gradients to about `1e-12`
- `src/step2_psi_nano` — a GPT-style model with a small BPE tokenizer, grouped-query attention, RoPE, SwiGLU, RMSNorm, tied embeddings, and block-wise weight sharing
- `src/step3_metal` — handwritten Metal kernels for Apple Silicon, including a ternary-weight matmul
- `src/step4_cuda` — a CUDA backend for training the same model on NVIDIA GPUs

The trained checkpoints in `models/` are a sweep of TinyStories writers from 115K to 574K parameters.

## Results

The target is TinyStories: can a tiny model write short, grammatical, coherent children's stories? I grade completions on grammar, coherence, consistency, and plot using the protocol in [`docs/EVAL.md`](docs/EVAL.md), then compare them with the original TinyStories models under the same rubric.

| model | params | grammar | coherence | consistency | plot |
|---|---:|:--:|:--:|:--:|:--:|
| femto | 115K | 6 | 3 | 3 | 2 |
| nano | 215K | 7 | 5 | 4 | 4 |
| **small** | **354K** | **8** | **6** | **6** | **5** |
| mid | 574K | 8 | 6 | 6 | 5 |

The headline result is `small`: at 354K parameters, it matches the original TinyStories-1M model under this evaluation rubric while using roughly one-third as many parameters. The complete methodology and outputs are in [`docs/RESULTS.md`](docs/RESULTS.md).

On the kernel side, the Metal matmul reaches 883 GFLOP/s at `2048³` on an M1—34% of theoretical peak and 6× the naive kernel—while remaining bit-exact against the CPU reference and within a few percent of MLX on the shapes I tuned. The ternary matmul stores weights in `{-1, 0, +1}`, reducing weight storage 16× relative to fp32 while running at full fp32-matmul speed. The optimization record is in [`docs/GPU_KERNELS.md`](docs/GPU_KERNELS.md).

## Build and run

The CPU path is dependency-free C++17. The GPU paths require Metal on Apple Silicon or CUDA on NVIDIA hardware.

```bash
# Scalar autograd, XOR MLP, and finite-difference gradient check
clang++ -std=c++17 -O2 src/step0_scalar_autograd/main.cpp -o step0
./step0

# Tensor autograd and per-operation gradient checks
clang++ -std=c++17 -O2 src/step1_tensor_autograd/main.cpp -o step1
./step1

# Transformer operation gradient checks
clang++ -std=c++17 -O2 src/step2_psi_nano/gradcheck.cpp -o step2_gradcheck
./step2_gradcheck
```

Build the TinyStories model with the Metal backend:

```bash
clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \
  src/step2_psi_nano/stories.cpp src/step3_metal/metal_backend.mm \
  -framework Metal -framework Foundation -o psi_stories
```

## Try a trained model

The checkpoints in `models/` are ready to run:

```bash
./psi_stories gen models/small/model.bin \
  "Once upon a time, there was a little girl who"
```

Or use the helper, which selects a sensible prompt for each checkpoint:

```bash
bash try.sh small
```

## What I learned

Psi was never about winning a benchmark with a tiny model. It was about owning the full path from a scalar derivative to a token coming out of a GPU-backed transformer.

The biggest surprise was how often measurement beat cleverness. Three matmul kernels that looked smarter on paper lost to a simpler design I found by benchmarking. The other surprise was how capable a few hundred thousand parameters became once the training recipe and implementation details were under control.

# Psi Design Doc

This is the design doc I drafted early on, kept here mostly as a record of how I was thinking before I built anything. A lot of the original version planned tracks I later cut (a reasoning/math specialist, a ternary weight bet, distillation from a big teacher, a 300-500M base model). None of that shipped. What did ship is a from-scratch C++ training stack and a set of sub-1M TinyStories models. So I've trimmed this down to the parts that still describe the real project.

For what actually got built and how it scored, see the README and RESULTS.

## What Psi is

As small as possible while as capable as possible, on a training stack I wrote myself. The prizes are mastery, novelty, and craft: own autograd, own GPU kernels, no PyTorch, no other ML framework. The model is the byproduct. The understanding is the point.

I can't win the absolute small-LM crown by brute force. The sub-2B leaders (SmolLM2-1.7B, Qwen2.5-1.5B, Llama-3.2-1B/3B) train on 10-18 trillion tokens with industrial data pipelines, and I'm not going to out-token them. So the target I actually care about is capability-per-parameter: sit on or above the Pareto frontier of quality per parameter at a tiny scale, on a stack I built by hand.

## Architecture

A decoder-only transformer with the standard small-model toolkit. Nothing exotic in the base. The models are trained on TinyStories, with vocabularies of 512 and 1024 BPE tokens.

| Component | Choice | Why at small scale |
|---|---|---|
| Norm | RMSNorm, pre-norm | Cheaper than LayerNorm, stable |
| Attention | GQA + QK-norm | GQA cuts KV-cache; QK-norm stabilizes training |
| Positional | RoPE (θ=10000) | Standard, no learned position params |
| MLP | SwiGLU, hidden ≈ (8/3)·d_model | Best quality per param for the FFN |
| Embeddings | Tied input/output | At this scale embeddings are a big share of params, so tying is free capability |
| Depth vs width | Deep and thin | At tiny scale, more layers beats more width at fixed params |

The current models are femto (115K), nano (215K), small (354K), and mid (574K) parameters.

## Training

Over-trained / inference-optimal: I keep going while val loss improves rather than stopping at compute-optimal, because inference cost is what I'm optimizing for.

Schedule is Warmup-Stable-Decay. Short warmup, a long stable phase at peak LR, then a short sharp decay. WSD lets you branch continuation runs from the stable checkpoint and gives a clean place to anneal the data mix.

Optimizer is AdamW (β=(0.9, 0.95), wd 0.1, grad-clip 1.0). Precision is bf16 mixed-precision. Stability comes from z-loss on the softmax, QK-norm, careful scaled init, and gradient clipping. Every backward pass gets validated against finite-difference gradients on the smallest model before I trust the stack.

## The custom stack

The core of the project. Each step runs end-to-end before the next begins. I hand-write the autograd, the model, the training loop, and the kernels. I reuse a tokenizer and trivial file IO, which aren't the point.

| Step | Deliverable | Done when |
|---|---|---|
| 0. Scalar autograd | reverse-mode autodiff + tiny MLP | it learns XOR; grads match finite-diff |
| 1. Tensor autograd | ndarray + broadcasting + the ~20 ops a transformer needs (matmul, softmax, RMSNorm, SwiGLU, embedding gather, cross-entropy, RoPE) with backward | a small forward+backward matches a reference (MLX) within tolerance |
| 2. Training loop | data loader, AdamW, checkpointing | a model overfits a tiny set, then trains on real data and val-loss drops |
| 3. Real kernels | fused Metal kernels replacing the slow ops | same loss curve, multiples faster, profiled |
| 4. CUDA backend | port to NVIDIA, cuBLAS plus a hand-written kernel, bit-identical to CPU | trains on the cluster |

The autograd is a dynamic tape: record ops in the forward pass, replay them in reverse. Simple, debuggable, and enough for a specialized model. I deliberately didn't build a general graph compiler, because a reusable framework was never the goal. Specialize and move.

References I read closely: `micrograd` (autodiff in ~150 lines), `nanoGPT` (a minimal real GPT), `llm.c` (a GPT in raw C/CUDA, basically the end state), and `modded-nanoGPT` for efficiency tricks.

## Kernels: Apple Silicon then NVIDIA

The kernels that matter, by share of training time: GEMM (matmul, dominates), then fused attention (FlashAttention-style, never materialize the full score matrix), then fused cross-entropy (fuse softmax and CE so you don't materialize the huge `[seq, vocab]` logits), then fused elementwise (RMSNorm+residual, SwiGLU).

On Apple Silicon I use MLX as the reference baseline (unified memory, Metal-backed, lazy) and write custom kernels in Metal Shading Language. M-series is for bring-up and correctness on small models, not large runs.

On NVIDIA I start GEMM on cuBLAS rather than hand-writing a competitive GEMM first, and hand-write the hot kernels around it. The portability rule: keep the math identical between Metal and CUDA so only the backend swaps, and diff every kernel against a CPU/MLX reference.

## Evaluation

The models are graded on TinyStories-style coherence, grammar, and consistency. The result I care about: small at 354K parameters gets the same 8/6/6/5 as TinyStories-1M at a third of the size. None of the sub-1M models clear the coherence bar cleanly, so the honest read is a capability-per-parameter win, not a raw-quality win. The sweep also showed quality climbing cleanly with size, and the vocab-512 models beating the vocab-1024 ones per parameter at this scale.

See RESULTS for the full table.

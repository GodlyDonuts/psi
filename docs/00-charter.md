# Psi, Project Charter

_Last updated: 2026-06-16_

## One line

Build a small language model, as small as possible while as capable as possible, on a training stack I write from scratch. It's a personal challenge to learn the field by building it, and to produce something novel along the way.

## Why I'm doing this

What would make this a success even if the final model were only mediocre:

I want a first-principles understanding of autograd, kernels, training dynamics, and scaling, the kind you only get from writing them yourself. I want to produce something original: a new kernel or architecture or training idea worth sharing, not a reproduction of a paper. And I want the craft of building the whole thing end-to-end, the hard way, on purpose.

One thing I've deliberately rejected as a goal: building a lasting, reusable, general-purpose framework. I don't pay the abstraction tax. I specialize, hard-code, and throw code away freely. The codebase is a sharp instrument aimed at one model, not a mini-PyTorch.

## What "SoTA" means here

A solo builder can't out-token the frontier labs; leading sub-2B models see multiple trillions of tokens. So I don't mean beating everyone on every benchmark. I mean pushing the Pareto frontier of quality-per-parameter and quality-per-training-FLOP at my compute class, which is measurable, defensible, and directly helped by the custom kernels.

The metric I actually care about is depth and originality. The Pareto numbers are secondary.

## Constraints

The stack is fully custom: my own autograd engine, runtime, and compute kernels. No PyTorch or HF Trainer on the core path.

Compute trajectory: development happens on Apple Silicon (M-series) through a Metal/MLX loop, and scale-up targets NVIDIA GPUs with CUDA. The design has to split cleanly between the Apple dev loop and the GPU scale-up, with kernels written to port across both.

## Working agreement

I implement; the point is to keep it serving mastery and craft rather than producing a black box.

Nothing is a black box. Every nontrivial piece is a readable, first-principles reference with the reasoning and math exposed inline. If a library one-liner would hide something worth understanding, I write it out longhand.

Novelty is first-class. I actively hunt for places to do something original, most likely in the kernels, possibly in the training method.

## Build philosophy: always-working incremental path

Never a big-bang framework. Each step produces something that runs end-to-end:

1. Scalar autograd (micrograd-class): reverse-mode autodiff from scratch.
2. Tensor autograd: n-dimensional arrays, broadcasting, the ops a transformer needs.
3. Naive GPT training loop: a small model trains and loss goes down, correctness first.
4. Real fused kernels: attention, GEMM, fused norm+residual, fused optimizer, fused cross-entropy. Apple/Metal first, ported to CUDA.
5. Scale-up: larger token budgets, the over-trained small regime.

## Status

The custom stack is built and training real models. The current family is femto (115K), nano (215K), small (354K), and mid (574K), all trained from scratch on TinyStories. See RESULTS for the numbers.

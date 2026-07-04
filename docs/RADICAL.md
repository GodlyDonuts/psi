# Psi: Radicalism Map

_How Psi tries to be radical in every layer of the stack without collapsing under its own ambition.
Companion to [DESIGN.md](DESIGN.md), which covers the what. This one covers how I stay original._

---

## The unifying thesis

Being radical in eight layers at once is eight independent gambles unless they all serve one bet. Mine is:

> Smallest-possible. Push the frontier of _capability-per-bit_.

Every radical choice below serves that thesis, concretely through ternary (~1.58-bit) weights, co-designed
end to end. That turns "radical everywhere" from scattershot cleverness into one idea expressed eight ways,
where each layer's radical choice makes the next one make sense. A general framework like PyTorch or JAX
can't do that cross-layer co-design: their layers have hard walls, mine don't.

## The rule that keeps it debuggable

The danger of "radical everywhere" isn't ambition, it's debuggability. If everything is novel at once and
the loss is wrong, you can't tell which novelty broke it. So:

1. Reproduce first. Build the conventional, correct version of each layer (the always-working ladder in
   DESIGN §7). It's the reference oracle.
2. Swap one layer at a time. Bring the radical version up behind a flag, keeping both implementations side
   by side.
3. Make it prove itself. The radical version must pass the same gradient check, match or beat the baseline
   loss curve, and win on the metric before it replaces the reference.

You end up radical in all layers, but you only ever debug one novelty at a time, and every claim of
"better" is a measured number instead of a hope.

## The honesty test

Run every "novel" idea through this question: can I name the specific assumption I'm breaking, and the
metric I beat them on?

If yes, it's real innovation. If no, I'm reimplementing an existing tool, worse. Each layer below names
both explicitly.

---

## The layers (bottom to top)

### 1. Numerics: how a number is represented
- Conventional: bf16 / fp32 weights and activations.
- Radical: ternary weights {−1, 0, +1} (1.58 bits) as the headline (BitNet b1.58 lineage); explore fp4 /
  microscaling (MX) formats, stochastic rounding for low-bit stability.
- Assumption broken: "weights need ≥8 bits to train well." For a deployment-first small model, maybe they
  don't.
- Local metric: bits/weight at a fixed capability; training stability at low bit-width.

### 2. Memory / tensor layout: how arrays are stored and moved
- Conventional: dense row-major tensors, one heap allocation per tensor, every intermediate materialized.
- Radical: sub-byte bit-packing for ternary weights; an arena allocator plus ahead-of-time memory plan for
  the one fixed graph; aggressive in-place / aliasing.
- Assumption broken: "the framework can't know your shapes or lifetimes ahead of time." Mine has exactly
  one model, so it can plan everything.
- Local metric: peak memory (bytes) per token of context; allocations per step.

### 3. Autograd / IR: how gradients and the graph are represented
- Conventional: dynamic, eager, define-by-run tape (PyTorch), which pays dispatch overhead for generality
  it doesn't need here.
- Radical: a static, shape-specialized graph built once and optimized (whole-graph fusion plus the memory
  plan from layer 2); quantization-aware autodiff with the straight-through estimator (STE) as a
  first-class graph concept rather than a bolt-on.
- Assumption broken: "autodiff must be dynamic." My model's shape never changes, so the graph can be
  compiled.
- Local metric: host overhead per step (µs not spent in kernels); fusion coverage.

### 4. Kernels / ops: the actual compute
- Conventional: cuBLAS + cuDNN + FlashAttention; "ternary" usually dequantizes to fp16 and does an ordinary
  matmul.
- Radical: a ternary-native GEMM that never dequantizes (stays packed through the accumulate); fused
  attention co-designed to my exact shapes (zero padding waste); a Metal flash-attention tuned to Apple's
  threadgroup-memory model; the optimizer step fused into the backward.
- Assumption broken: "low-bit weights still need a full-precision matmul," and "attention kernels must be
  shape-general."
- Local metric: MFU (model-FLOPs utilization) on my model; tokens/sec/watt; effective bit-throughput of
  the GEMM.

### 5. Architecture: the model itself
- Conventional: Llama-style transformer (RMSNorm, RoPE, GQA, SwiGLU).
- Radical: a ternary-native architecture (BitLinear in place of Linear, with the normalization placement
  low-bit training needs); shapes snapped to Tensor-Core / Apple SIMD-group tile boundaries so kernels run
  with no waste; novel attention (DeepSeek-style MLA).
- Assumption broken: "architecture and kernels are designed by different people behind a wall." I co-design
  them.
- Local metric: capability (loss / benchmark) at a fixed bit budget; kernel tile efficiency.

### 6. Optimizer: how weights update
- Conventional: AdamW.
- Radical: Muon (Newton–Schulz-orthogonalized momentum) as a fused kernel; quantization-aware updates
  (latent high-precision weights + STE, or low-bit optimizer state to shrink memory).
- Assumption broken: "AdamW is the only safe default," and "optimizer state must be fp32 and twice the
  model size."
- Local metric: loss-per-FLOP and loss-per-wallclock vs AdamW; optimizer-state bytes.

### 7. Schedule / training loop: how the run is orchestrated
- Conventional: cosine LR, fixed batch size.
- Radical: WSD (warmup-stable-decay) with a co-designed anneal/data swap; curriculum; dynamic batch sizing;
  regularization tricks for low-bit stability (a real open research surface, since ternary training is
  twitchy).
- Assumption broken: "the schedule is independent of the numerics." Low-bit training needs its own
  stabilization.
- Local metric: steps-to-target-loss; training stability (gradient/loss spikes) at low bit-width.

### 8. Data: what the model learns from
- Conventional: pre-tokenized web shards, fixed mix.
- Radical: a co-designed synthetic + reasoning-trace pipeline (quality dominates at small scale); a
  math-aware tokenizer (digit/LaTeX-friendly) if the target is math; curriculum ordering tied to the anneal
  phase.
- Assumption broken: "more tokens win." At my scale, better tokens and a smart pipeline win.
- Local metric: capability per training-token; trace quality vs quantity (s1/LIMO effect).

### (edges, deferred)
- Eval: radical in framing, not compute. Make capability-per-bit a first-class, plotted metric rather than
  an afterthought. (See DESIGN §9.)
- Distributed runtime: later. Exploit GH200 NVLink-C2C coherent memory (Grace CPU offload, unified
  addressing) in ways DDP/FSDP don't. Only after single-device works.

---

## How the layers couple (the co-design chain)

This is why the eight choices are one bet, not eight:

```
ternary numerics (1)
  → forces sub-byte bit-packing & ahead-of-time planning (2)
    → which a static, format-aware graph can compile and fuse (3)
      → demanding a ternary-native GEMM that never dequantizes (4)
        → which only pays off with a ternary-native architecture (5)
          → trained by a quantization-aware optimizer (6)
            → kept stable by a low-bit-aware schedule (7)
              → fed by tight, curated data (8)
                → measured as capability-per-bit (eval)
```

Break any link and the chain still teaches you something. Keep them all and the radicalism compounds
instead of fragmenting.

## North-star vs local metrics

- North-star: capability-per-bit, meaning benchmark score vs total model size in megabytes. Plant a flag
  where no record exists.
- Local: each layer above has its own metric so a radical swap can be judged in isolation before it touches
  the north-star number.

## Sequencing (maps onto DESIGN §10 phases)

- P1–P2 (now): conventional baseline, scalar ✅ then tensor autograd + naive loop. The reference oracle.
  Deliberately not radical yet.
- P3: first radical layer, kernels (layer 4) on Apple/Metal, measured vs the baseline.
- P4: scale the baseline to GH200; port kernels.
- P5: data + reasoning pipeline (layer 8), the "intelligence."
- P6: the deep radical stack. Numerics, memory, architecture, and optimizer (layers 1, 2, 5, 6) come
  online together as the ternary model, validated layer-by-layer against the bf16 baseline.
- P7: write it up, the capability-per-bit result and the kernels.

## Risks and honest caveats

- Low-bit training is unstable. Ternary-from-scratch is twitchy; layer 7 exists to fight this and it may
  eat real time. Mitigation: bf16 baseline first; ternary only once everything else is proven.
- Doing all of this is a multi-month project. That's expected and fine, since mastery and craft are the
  point. Mitigation: the one-layer-at-a-time discipline keeps every week shippable and debuggable.
- "Radical" can become "broken-but-novel." Mitigation: the honesty test plus local metrics. A radical layer
  that can't beat the baseline on its metric does not ship.

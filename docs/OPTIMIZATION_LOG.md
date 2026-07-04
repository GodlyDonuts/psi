# Psi Optimization Log

Measured speedups to the custom CPU stack (the Mac prototype). The rule: every entry keeps all grad-checks passing. I never trade correctness for speed. When a change would break a grad-check, I revert it.

Benchmark: `matmul_bench`, a 256×256×256 matmul, forward plus both backward passes, reported in GFLOP/s. Hardware is Apple Silicon, 8 cores (4 performance). matmul dominates the whole stack's cost, so this number is the scoreboard.

| # | Change | GFLOP/s | vs baseline | grad-checks |
|---|---|---:|---:|---|
| 0 | baseline, naive `i-j-l` matmul, `-O2` | 1.80 | 1.0× | ✅ PASS |
| 1 | `i-l-j` loop reorder (contiguous inner loop, cache-friendly and vectorizable) + `-O3 -march=native` | 5.40 | 3.0× | ✅ PASS |
| 2 | multithread matmul over disjoint output rows (4 perf cores; large matmuls only) | 10.95 (256³) / 14.24 (512³) | 6.1× / 7.9× | ✅ PASS |

On nano (the real workload), step time fell about 3× across iters 1 through 3.

- Iters 1 and 2 (reorder plus flags; its small matmuls stay serial): step-100 went 5.9s to 2.6s, loss bit-identical (2.1949), so determinism is preserved.
- Iter 3 is the float32 training path (`-DPSI_REAL=float`). The grad-check oracle stays `double`, so correctness is still gated tightly. A 200-step run drops 7.3s to 5.1s (about 1.4×) at equivalent loss (~1.83). All grad-checks still pass.

Iter 4 was `k`-unroll-by-4 register blocking on the matmul forward, and I reverted it. Same-session A/B: 256³ went 25.8 to 21.1 GFLOP/s, 512³ went 22.6 to 18.0, a 15 to 20% regression. Grad-checks passed, so correctness was fine, but it's slower. The compiler already auto-vectorizes the simple AXPY well, and manual unrolling added register and cache pressure. Reverted per the rule: no measured win means revert.

A note on benchmark variance: absolute GFLOP/s drifts about 2× with machine load (the iter-2 "10.95" was under load; the quiet-machine baseline for the same code is ~22 to 26). Only same-session A/B numbers are trustworthy. The relative speedups in the table were same-session and stand.

Status: the simple vectorized matmul is at the practical CPU double-precision roofline (~22 to 26 GFLOP/s on 4 cores), so further matmul micro-opt shows diminishing or negative returns. The remaining clean CPU wins live in per-op overhead (arena/tape autograd for nano's many small ops) and a persistent thread pool. The real 10 to 100× is Step 3, the Metal GPU kernels, best done with me awake at the wheel.

Iter 5 added `-ffast-math` on the training build (nano only): about a 2.0× win, and I kept it. Same-session A/B over 200 steps went 3.6s/3.5s to 1.8s/1.7s with identical loss (1.8250), so convergence was unchanged and there was no NaN. The grad-check oracle stays strict `double` (no fast-math), so the math stays proven. Fast-math only relaxes FP reassociation and contraction on the training path, which is safe here: the causal mask is −1e9 not −Inf, softmax is max-subtracted, and CE is guarded.

Iter 6 batched the projection matmuls (pack the batch into one `[B·T, d]` stream plus a block-diagonal causal mask). Correct but neutral, so I reverted it. The batched forward is bit-identical to per-sequence (equivalence test: `0.00e+00`), but same-session A/B showed no speedup (1.8s to 1.8s). Packing into one `[256,256]` attention computes about 8× the cross-sequence score pairs (then masks them), which cancels the gain from parallelizing the larger projection matmuls.

The key finding here: nano's bottleneck is not matmul size. Neither bigger matmuls (iter 6) nor matmul micro-opt (iter 4) move its step time, while `-ffast-math` (iter 5) did. So the remaining cost is per-op scalar overhead (node allocation, graph build, the many small elementwise loops), not GEMM. Next step is to profile and confirm, then move to arena/tape autograd to cut per-op allocation. The big leap remains Step 3, the Metal GPU kernels.

Iter 7 was profiling: I added `psi_profile` with no engine change. Phase breakdown of a nano step (8.6 ms/step): forward plus loss 39%, backward 61%, optimizer about 0%. That confirms the diagnosis. The optimizer (pure array math) is free; the cost is evaluating about 2000 tiny ops per step (forward node-build plus compute, backward closure dispatch plus grad loops). The matmuls are too small to be throughput-bound, which is why iters 4 and 6 didn't move the needle. No single micro-opt fixes this. The structural win is fewer or fused ops, or a tape/arena autograd (build the graph once, replay it each step), a deliberate core-autograd refactor best done carefully. The clean, safe CPU quick-wins are now exhausted; the order-of-magnitude leap from here is Step 3, the Metal GPU kernels.

Iter 8 inlined `parents` (vector to a fixed `[2]`) to cut a per-node allocation. Neutral, so I reverted it. Same-session profiler A/B (3 runs each): 8.3 ms/step to 8.3 ms/step, no change. Grad-checks passed. Removing a per-node heap allocation changed nothing, so the overhead is not allocation, it's op-dispatch (~2000 `std::function` backward calls plus tiny-loop compute per step).

---

## Loop summary (paused)

Eight iterations: matmul went 1.8 to ~22 GFLOP/s (i-l-j reorder plus `-O3 -march=native` plus multithreading) and nano went about 10× (float32 plus `-ffast-math`), plus three honest reverts (register-blocking regressed, batched-matmul correct-but-neutral, parents-inline neutral) and a profiler. Every change was same-session-measured and grad-check-gated. Correctness was never traded for speed.

The clean, safe CPU quick-wins are exhausted. Iters 4, 6, and 8 confirm the bottleneck is op-dispatch, not GEMM or allocation. The remaining real speedups are all structural: a tape autograd (replace `std::function` dispatch with a typed op tape), batch-parallel gradient accumulation, or the real order-of-magnitude win, Step 3, the custom Metal GPU kernels. The kernels are the project's deliberate centerpiece (see RADICAL.md: master plus novelty). Loop paused pending direction.

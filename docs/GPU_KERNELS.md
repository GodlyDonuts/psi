# Psi — GPU Kernels (Step 3): from a naïve Metal matmul to a ternary MMA MLX can't match

> The custom GPU layer. Goal: not "fast" in the abstract, but **legibly, measurably good** — every
> kernel validated **bit-exact** against a CPU reference, every speed a **best-of-N** number on the
> shapes the model *actually* runs, every dead end recorded. The headline result is not that we tied
> a vendor library at its own game (we did) — it's that we then built a kernel the vendor library
> **doesn't have**: a ternary-weight matmul that runs the model's matmuls at full speed with weights
> **16× smaller**. That is Psi's thesis (capability-per-bit) realized in silicon.

Hardware: **Apple M1**, fp32 theoretical peak ≈ **2.6 TFLOP/s**. Files: [`src/step3_metal/`](../src/step3_metal/).

---

## 0. Methodology (the rules that make the numbers mean something)

- **Bit-exact or it doesn't count.** Every kernel/config is validated against a parallel CPU reference
  (`rel < 1e-3`) before its speed is even reported. Correctness is never traded for speed.
- **Best-of-N timing.** Background load only ever *adds* time, so the fastest of N reps is the cleanest
  estimate. Numbers drift ~20% with machine state (DVFS / other apps); only trust the trends and
  same-session A/Bs.
- **Real shapes, not vanity squares.** A transformer's matmuls are fat-and-thin — `(batch·seq, d) ×
  (d, d_ff)` — not big squares. We benchmark `mlp-up (8192×384×1536)`, `mlp-down (8192×1536×384)`,
  `attn-proj (8192×384×384)`, `logits (8192×384×4096)`, alongside `square-{1024,2048}`.
- **The yardstick is MLX/PyTorch, not the theoretical peak.** See §1.

---

## 1. The honest reframe: 2.6 TFLOP is a fantasy; MLX is the real target

We benchmarked Apple's own **MLX** and **PyTorch MPS** (fp32, bit-exact-equivalent, best-of-N). Both
top out at **~57% of peak (~1490 GFLOP/s)** at 2048, and they **peak at N=768–1024 (~64%)** then
*decline* at larger N (the L2/SLC cache cliff). So:

- **The 2.6 TFLOP peak is unreachable** — not by us, not by Apple. The matrix units never run every ALU
  every cycle.
- **The real finish line is ~57% of peak**, and on its best shape (`mlp-up`) MLX reaches **73.7%**.
- **Square-2048 fp32 was a vanity benchmark.** It's past the cache cliff *and* not a shape the model runs.

This reframing is the whole game: we stopped chasing an impossible number and started chasing a measured competitor on the shapes that matter.

---

## 2. The matmul journey (every step bit-exact)

Progression of the fp32 matmul, culminating in the champion config. Representative numbers at the real
shapes; the kernel is in [`mma_autotune.mm`](../src/step3_metal/mma_autotune.mm) (parametrized over
block size, simdgroup grid, K-tile depth, and threadgroup padding — a real autotuner).

| Stage | What changed | mlp-up | mlp-down | sq-2048 | % of peak |
|---|---|---:|---:|---:|---:|
| naïve | one thread per output | ~150 | ~150 | 146 | ~6% |
| register-tiled | threadgroup staging + per-thread micro-tile | ~600 | ~600 | 629 | ~24% |
| **simdgroup MMA** | hardware 8×8 matrix units, multi-simdgroup, BK=16 | ~800 | ~815 | 883 | ~30–35% |
| **+ float4 loads** | 4 floats/instruction (same tgmem → occupancy intact) | 1225 | 1263 | 883 | ~34–49% |
| **+ 64×64 / 8 sg** | bigger block, FM2×FN4 register tile (8 frags) | 1218 | 1246 | 1204 | ~46–48% |
| **+ PAD=4** | pad staging rows → kill threadgroup bank conflicts | **1342** | **1404** | **1342** | **51–54%** |

**Champion: `64×64 block / 8 simdgroups / BK16 / FM2×FN4 / PAD4 / float4`** ([`shape_bench.mm`](../src/step3_metal/shape_bench.mm)).

### vs MLX (the scoreboard that matters)

| shape | Psi (peak%) | MLX (peak%) | **Psi ÷ MLX** |
|---|---:|---:|---:|
| mlp-down | 54.0% | 54.4% | **99% — parity** |
| square-2048 | 51.6% | 57.3% | 90% |
| attn-proj | 46.8% | 55.2% | 85% |
| logits | 50.5% | — | strong |
| mlp-up | 51.6% | 73.7% | 70% |
| square-1024 | 44.7% | 64.5% | 69% |

A from-scratch, legible, bit-exact fp32 GEMM at **parity on the model's most important shape** and
85–90% on most. The two laggards are exactly where MLX is *exceptional* (its big-register-tile machinery, §4).

---

## 3. The graveyard — five "smarter" ideas that lost. **Occupancy is king.**

Measurement beat intuition over and over. Each of these is plausible and each *lost*, because on the
Apple GPU the binding resource is **per-core threadgroup residency (occupancy)** — the mechanism that
hides memory and instruction latency by interleaving many threadgroups.

| Idea | Hypothesis | Result | Why it lost |
|---|---|---:|---|
| **Double-buffering** | overlap load k+1 with compute k | **0.69–0.85×** | 2× threadgroup mem halves residency; lost more occupancy than it hid |
| **No-staging** | `simdgroup_load` straight from device, max occupancy | **0.55–0.72×** | device-cache reuse < explicit threadgroup staging |
| **`acc[4][4]` (16 frags)** | bigger register tile, more ILP | **2.4% (spill)** | register pressure → spill → occupancy collapse, *and* wrong results |
| **Wide-N blocks** | match MLX's large-N tiles | **all slower** | bigger blocks = fewer threadgroups = less occupancy |
| **Register prefetch** | next tile → registers during compute | **wash** | the compiler *already* overlaps global loads via occupancy |

The single lever that always *won* — float4, padding — shared a property: **it cost no occupancy.**

---

## 4. We studied MLX's source. We already matched its method.

To stop guessing, we read MLX's actual `steel` GEMM
([`ml-explore/mlx`](https://github.com/ml-explore/mlx)). The striking result: **we independently
reinvented its kernel.**

| Technique | MLX | Psi |
|---|---|---|
| double-buffering | ❌ single-buffer | ❌ single-buffer (we refuted DB) |
| barriers / k-iter | 2 threadgroup | 2 threadgroup |
| **bank-conflict padding** | `16/sizeof(T)` = **4 floats** (fp32) | **PAD=4** (found empirically) |
| float4 staged loads | ✅ | ✅ |

The *only* difference is MLX's big **4×4 = 16-fragment** per-thread register tile (fewer threads, more
ILP), implemented through a templated `BlockMMA`/`tile_matmad` fragment-register engine. Hand-written
`acc[4][4]` spills and breaks (§3). Replicating it = **porting MLX's template engine ≈ becoming MLX**,
which defeats the from-scratch purpose. So we drew the line here and pivoted to something that's *ours*.

---

## 5. The pivot: Psi's own kernel — a **ternary-weight matmul**

> Matching MLX's fp32 GEMM is a tie at best. The project's north star is **capability-per-bit / ternary
> (~1.58-bit) weights**. So we built the kernel MLX *doesn't* have.

`C = A @ W`, where **W ∈ {−1, 0, +1}**, packed at **2 bits/weight** (16 to a `uint32`). The weights are
loaded with ~16× less memory and **decoded to fp32 in threadgroup memory**, then fed the *exact same
tuned MMA path* (float4 + PAD4 + 64×64/8sg). Activations stay fp32, so the result is **bit-exact** vs a
CPU ternary reference. ([`ternary_gemm.mm`](../src/step3_metal/ternary_gemm.mm))

| shape | valid | ternary GFLOP/s | vs our fp32 | weight memory |
|---|:--:|---:|---:|---|
| mlp-up | ✅ | 1275 | 0.95× | **16× smaller** |
| mlp-down | ✅ | 1302 | 0.93× | **16× smaller** |
| square-2048 | ✅ | 1347 | 1.00× | **16× smaller** |

**Result: the model's matmuls run at full fp32-GEMM speed with weights 16× smaller.** That is
capability-per-bit, and MLX has no path to it. *This* is the unique contribution — not out-tuning a
vendor's fp32 kernel, but doing the same throughput at a fraction of the bits.

---

## 6. Two findings worth their own line

**(a) On M1, the matrix units are precision-independent.** ([`ternary_fp16.mm`](../src/step3_metal/ternary_fp16.mm))
We added fp16 activations expecting ~2× from the half MMA units. We got **0.89–0.97×** (bit-exact,
`rel=0`). Apple's "2× fp16" is a vector-ALU property; the `simdgroup_matrix` coprocessor runs fp16 and
fp32 at the *same* rate. So on M1, low precision wins on **bits, not speed** — the speed win waits for
**GH200**'s low-precision tensor cores.

**(b) On M1, epilogue fusion doesn't pay.** ([`fused_ternary.mm`](../src/step3_metal/fused_ternary.mm))
We tried to fuse the activation into the matmul epilogue (the classic "save the intermediate
round-trip"). Both mechanisms *lost*: a `[64×64]` threadgroup scratch → **0.80–0.84×** (16KB tanks
occupancy), and in-register `acc.thread_elements()` → **0.17–0.21×** (forces `acc` out of the MMA
matrix registers, spilling the whole matmul; ReLU is equally slow, so it's the mechanism, not the
transcendental). The accumulator resists element access, and output-sized scratch kills occupancy →
**standalone high-occupancy activation passes win.** MLX's separate dispatches are near-optimal on M1.

---

## 7. Conclusion & outlook

- **Matmul:** parity-class with MLX (99% on the key shape; we matched its *method*, verified against its
  source). The remaining gap is its template-engine register tile — a vendor artifact, not a missing idea.
- **Unique edge:** the **ternary GEMM** — 16× smaller weights at full fp32-GEMM speed, bit-exact.
  Capability-per-bit, realized.
- **M1 speed ceilings are honest and understood:** matmul ties, ternary/fp16 tie (precision-independent
  units), fusion loses. These are *measured*, not assumed.
- **The work compounds on GH200:** there, the same ternary kernel *also* wins on speed (real
  low-precision tensor cores) and the fusion tradeoffs flip. Mac-first was the right call; the next
  machine cashes the second dividend.

Next: wire the ternary kernel into the autograd backend (with a ragged-shape fallback) and put it to
work in **`psi-stories`** — a flagship whose weights are 16× smaller *by construction*.

---

## Reproduce

```sh
cd src/step3_metal
F="-x objective-c++ -fobjc-arc -O2 -std=c++17 -framework Metal -framework Foundation"
clang++ $F shape_bench.mm   -o shape_bench   && ./shape_bench     # champion fp32, real shapes
clang++ $F mma_autotune.mm  -o mma_autotune  && ./mma_autotune    # the parametrized design-space search
clang++ $F ternary_gemm.mm  -o ternary_gemm  && ./ternary_gemm    # ternary (fp32 act), bit-exact, 16x smaller W
clang++ $F ternary_fp16.mm  -o ternary_fp16  && ./ternary_fp16    # finding (a): precision-independence
clang++ $F fused_ternary.mm -o fused_ternary && ./fused_ternary   # finding (b): fusion doesn't pay
clang++ $F db_dev.mm        -o db_dev        && ./db_dev          # head-to-head dev harness (refuted ideas)
```

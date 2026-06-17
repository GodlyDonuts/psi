# Ψ (Psi)

A personal challenge to build a **state-of-the-art Small Language Model** — as small as
possible while as intelligent as possible — on a **fully custom training stack**: our own
autograd engine, runtime, and compute kernels. No PyTorch.

## Why this exists

The model is the byproduct, not the point. The real goals:

- **Master the full stack** — understand every layer from first principles: autograd, kernels,
  training dynamics, scaling.
- **Produce something novel** — a genuinely new kernel / architecture / training idea, not a
  reproduction.
- **The challenge & craft** — build it end-to-end, the hard way, on purpose.

Explicitly *not* a goal: a reusable, general-purpose framework. We optimize for understanding and
originality, and we stay free to specialize, hard-code, and throw things away.

Success is measured by **depth and originality**, not leaderboard optics. A useful secondary metric
is **quality-per-parameter / quality-per-training-FLOP** — the Pareto frontier for our compute
class — since a solo builder can't out-token the labs.

## Constraints

- **Fully custom stack** — own autograd → runtime → kernels. The deliberate uniqueness lever, and
  the main scope risk. Mitigated by an always-working incremental build.
- **Compute trajectory** — Apple Silicon (Metal / MLX) for the dev loop *now* → NVIDIA **GH200**
  (Grace Hopper) for scale → a future **AWS GPU cluster**. The design splits cleanly into a
  dev loop and a scale-up.

## Status

Research-first: landscape sweep → design doc → build.

- [x] Project charter & goals — see [docs/00-charter.md](docs/00-charter.md)
- [x] Design doc — see [docs/DESIGN.md](docs/DESIGN.md)
- [x] Radicalism map — radical in every layer, unified by capability-per-bit: see [docs/RADICAL.md](docs/RADICAL.md)
- [x] Quality bar — definition-of-done per component: see [docs/QUALITY.md](docs/QUALITY.md)
- [x] Target & showcase — a **model zoo** (`psi-stories`, `psi-chess`, …) that showcases the framework: see [docs/SHOWCASE.md](docs/SHOWCASE.md)
- [~] Custom stack — scalar ✅ · tensor ✅ · psi-nano ✅ · framework ✅ · **GPU kernels ✅ — tuned Metal
  matmul at parity-class with MLX (bit-exact; 99% of MLX on the key shape) + a novel **ternary-weight GEMM**
  (16× smaller weights at full fp32-GEMM speed — capability-per-bit, which MLX has no path for). Full
  journey + findings: [docs/GPU_KERNELS.md](docs/GPU_KERNELS.md)** ← integrating ternary into the backend → model zoo
- [ ] First trained model + eval against size-matched baselines

## Build

Language: **C++17** (host + autograd + kernel dispatch). Compute kernels will be Metal
(Apple Silicon) and CUDA (Hopper/GH200) — same C++ family, no FFI boundary. See
[docs/DESIGN.md](docs/DESIGN.md) §8.

**Step 0** — scalar reverse-mode autograd + XOR MLP + finite-difference gradient check:

```sh
# direct (no extra tooling needed):
clang++ -std=c++17 -O2 src/step0_scalar_autograd/main.cpp -o step0 && ./step0

# or via CMake (brew install cmake):
cmake -B build && cmake --build build && ./build/step0
```

**Step 1** — tensor autograd (CPU oracle): core ops, per-op grad checks, tensor XOR:

```sh
clang++ -std=c++17 -O2 src/step1_tensor_autograd/main.cpp -o step1 && ./step1
```

**Step 2** — psi-nano: a tiny char-level GPT trained end-to-end on the custom stack:

```sh
# transformer-op grad checks (all PASS):
clang++ -std=c++17 -O2 src/step2_psi_nano/gradcheck.cpp -o step2_gradcheck && ./step2_gradcheck
# train psi-nano (~20s with the fast flags) then sample — or chat with it interactively:
# the float build links the Metal GPU matmul backend (CPU fallback if no Metal):
clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \
  src/step2_psi_nano/main.cpp src/step3_metal/metal_backend.mm \
  -framework Metal -framework Foundation -o psi_nano
./psi_nano train                      # train on the embedded corpus -> saves psi_model.bin (+ train/val loss)
./psi_nano train mytext.txt 3000      # or train on your own text file for N steps
./psi_nano gen  psi_model.bin "the "  # load the checkpoint and generate (no retrain)
./psi_nano chat psi_model.bin         # load the checkpoint and prompt it live
```

_Result (MacBook CPU, ~3 min / 2500 steps): cross-entropy loss `ln(27) ≈ 3.30 → ~0.16`; generated
text goes from noise to fluent corpus English, e.g._

> "the model is built from scratch with a custom autograd engine. every operation knows how to compute its own gradient."

**Step 3** — first GPU kernel: a Metal matmul on Apple Silicon, validated bit-for-bit vs the CPU:

```sh
clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 src/step3_metal/matmul_metal.mm \
  -framework Metal -framework Foundation -o matmul_metal && ./matmul_metal 2048   # autotune at size N
# fused single-dispatch attention, bit-close vs CPU:
clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 src/step3_metal/attention_metal.mm \
  -framework Metal -framework Foundation -o attention_metal && ./attention_metal
```

_Result (Apple M1, quiet machine, every config bit-exact vs CPU): an **autotuner** sweeps several
kernels/configs and picks the best. The winner is a **multi-simdgroup tiled-MMA kernel** —
threadgroup-staged reuse + 4 simdgroups for occupancy + the hardware matrix units — hitting **883
GFLOP/s at 2048³ (34% of the M1's ~2.6 TFLOP peak, 6× the naive kernel)**, % of peak climbing with size
as the roofline predicts. The journey is the lesson: three "smarter" kernels lost along the way (the big
128² tile starves the cores; naive MMA has no reuse; the 1-simdgroup MMA is starved) — only measurement
found that **reuse + occupancy + matrix units must all be present at once**. Unified-memory zero-copy buffers._

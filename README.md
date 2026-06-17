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
- [~] Custom stack — scalar ✅ · tensor ✅ · psi-nano GPT ✅ · framework-ize ✅ · **GPU kernels: Metal matmul ✅ (124 GFLOP/s, ~4× CPU) ← building** → model zoo
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
clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float src/step2_psi_nano/main.cpp -o psi_nano
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
  -framework Metal -framework Foundation -o matmul_metal && ./matmul_metal
```

_Result (Apple M1): correct (`max|diff|=0`) and **124 GFLOP/s** with a naive kernel — ~4× the CPU,
with ~20× more headroom (tiling / `simdgroup_matrix` next). Uses unified-memory shared buffers (zero copy)._

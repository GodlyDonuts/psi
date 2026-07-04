# CUDA backend: Psi on NVIDIA (UCF Newton H100 / V100)

My custom stack (own autograd, own kernels, no PyTorch) now runs on NVIDIA GPUs, not just Apple Metal. Same model, same training loop, same numbers. Only the GPU matmul backend changes.

## The abstraction

The autograd's entire GPU coupling is four functions ([`src/gpu_backend.h`](../src/gpu_backend.h)), dispatched from `tensor.hpp` only for float matmuls above a work threshold:

```
bool gpu_available();
void gpu_matmul   (A,B,C, M,K,N);            // fwd:  C = A @ B           (writes C)
void gpu_matmul_nt(P,Q,R, rows,cols,contract); // bwd:  R += P @ Qᵀ  (dA += dC @ Bᵀ)
void gpu_matmul_tn(P,Q,R, rows,cols,contract); // bwd:  R += Pᵀ @ Q  (dB += Aᵀ @ dC)
```

Exactly one implementation links per machine:
- macOS uses [`step3_metal/metal_backend.mm`](../src/step3_metal/metal_backend.mm) (Metal)
- Linux/NVIDIA uses [`step4_cuda/cuda_backend.cu`](../src/step4_cuda/cuda_backend.cu) (CUDA)

Porting to CUDA meant implementing those four functions. `tensor.hpp`, the model, and the training loop are unchanged.

## Two engines, verified against each other and the CPU oracle

`cuda_backend.cu` carries both a cuBLAS path and a hand-written tiled kernel, chosen at runtime:

| env | effect |
|---|---|
| `PSI_CUDA_KERNEL=cublas` (default) | cuBLAS SGEMM, the correctness/perf oracle |
| `PSI_CUDA_KERNEL=custom` | hand-written 16×16 tiled kernel, the number to beat |
| `PSI_CUDA_CHECK=1` | run both every call, report max\|cublas−custom\| |
| `PSI_CUDA_DISABLE=1` | force the CPU path (for A/B timing) |

Row-major to cuBLAS (column-major) is handled by computing `Cᵀ = op(B)ᵀ·op(A)ᵀ` (swap the operands); the backward ops accumulate via cuBLAS `beta=1` after uploading `R`. All three ops validate to ~1e-7 vs a double-precision CPU reference across non-power-of-2 shapes, and cuBLAS-vs-custom agree to ~3e-7 ([`cuda_backend_test.cu`](../src/step4_cuda/cuda_backend_test.cu)).

## Results (574K config, 100 steps, TinyStories)

Loss is bit-identical across CPU / V100 / H100 (`step 0 = 6.9673`, `step 100 = 6.0374`), so the GPU backend is numerically equivalent to the CPU autograd in the real model, not just microbenchmarks.

| matmul (end-to-end, incl. H2D/D2H) | 1024³ | 2048³ |
|---|---|---|
| V100 cuBLAS | 737 GFLOP/s | 1487 |
| H100 cuBLAS | **2054** | **3796** |
| H100 custom kernel | 1593 | 2468 (~65% of cuBLAS) |

| psi_stories 100 steps | wall |
|---|---|
| 16-core CPU | 150.7 s |
| V100 | 17.4 s (~8.7×) |
| H100 | 14.1 s (~10.7×) |

## The honest bottleneck

For these sub-1M models, H100 and V100 come out about even in training even though the H100 is 2.5× faster on raw matmul. The autograd holds tensors in host memory, so every dispatched matmul pays an H2D→GEMM→D2H round trip over PCIe. On Apple's unified memory those copies are free; on a discrete GPU they aren't. Add in the many small ops that stay on the CPU and training is copy/CPU-bound, not GPU-FLOP-bound, so a faster GPU barely moves the needle at this size.

The next steps that would actually exploit the H100:

1. Keep the forward/backward tensors GPU-resident across a step, eliminating per-op copies. This is the big one.
2. Bigger models and batches, so more GPU-bound work amortizes each copy.
3. Tune the custom kernel with float4 loads and larger register-blocked tiles to close the 65%→100% gap vs cuBLAS (the same journey as the Metal kernel, 28%→54% of peak).

## Build & run (UCF Newton)

```sh
module load cuda/cuda-12.6.0
bash src/step4_cuda/build_cuda.sh        # -> psi_stories_cuda, cuda_backend_test  (run on the GPU node)
sbatch src/step4_cuda/psi_cuda.sbatch [STEPS]
```

Accessible GPUs (account `arcc_pi_skattel`): `normal`/`preemptable` = H100 PCIe + V100. `highgpu` (8×H100-80GB) and `short`/`ucfit` (H200) need a separate account grant. Some H100 PCIe nodes are left in a "CUDA device busy/unavailable" state by prior jobs (persistence mode off). The sbatch health-gates the GPU and bails so you can resubmit; V100 nodes have been reliable.

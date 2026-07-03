// gpu_backend.h — the backend-generic GPU matmul interface the autograd dispatches to.
//
// tensor.hpp calls these four functions; exactly ONE implementation is linked per machine:
//   • src/step3_metal/metal_backend.mm  — Apple Metal   (macOS)
//   • src/step4_cuda/cuda_backend.cu    — NVIDIA CUDA    (cuBLAS + a hand-written kernel)
// Both fall back to a CPU reference if no GPU is present, so callers stay correct anywhere.
//
// Semantics (row-major throughout, matching the CPU oracle in tensor.hpp):
//   gpu_matmul     writes    C[M,N]      = A[M,K] @ B[K,N]
//   gpu_matmul_nt  ACCUMULATES R[rows,cols] += P[rows,·] @ Q[cols,·]^T   (dA += dC @ B^T)
//   gpu_matmul_tn  ACCUMULATES R[rows,cols] += P[·,rows]^T @ Q[·,cols]   (dB += A^T @ dC)
// The backward pair accumulate into R's existing contents (they read R in first).

#pragma once

namespace psi {

bool gpu_available();

// Forward:  C[M,N] = A[M,K] @ B[K,N].  (writes C)
void gpu_matmul(const float* A, const float* B, float* C, int M, int K, int N);

// Backward helpers — these ACCUMULATE into R (R += ...), matching how the autograd sums gradients:
//   nt:  R[rows,cols] += sum_c P[rows,c] * Q[cols,c]   (dA = dC @ B^T : P=dC, Q=B, contract over N)
//   tn:  R[rows,cols] += sum_c P[c,rows] * Q[c,cols]   (dB = A^T @ dC : P=A,  Q=dC, contract over M)
void gpu_matmul_nt(const float* P, const float* Q, float* R, int rows, int cols, int contract);
void gpu_matmul_tn(const float* P, const float* Q, float* R, int rows, int cols, int contract);

}  // namespace psi

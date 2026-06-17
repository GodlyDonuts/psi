// metal_backend.h — C++ interface to the Metal GPU backend (implemented in metal_backend.mm).
//
// The bridge that lets the pure-C++ autograd dispatch float matmuls to the GPU. All calls fall back
// to CPU if Metal is unavailable, so callers stay correct on any machine. Three flavors cover a
// matmul op's forward and both backward passes:

#pragma once

namespace psi {

bool metal_available();

// Forward:  C[M,N] = A[M,K] @ B[K,N].  (writes C)
void metal_matmul(const float* A, const float* B, float* C, int M, int K, int N);

// Backward helpers — these ACCUMULATE into R (R += ...), matching how the autograd sums gradients:
//   nt:  R[rows,cols] += sum_c P[rows,c] * Q[cols,c]   (dA = dC @ Bᵀ : P=dC, Q=B, contract over N)
//   tn:  R[rows,cols] += sum_c P[c,rows] * Q[c,cols]   (dB = Aᵀ @ dC : P=A,  Q=dC, contract over M)
void metal_matmul_nt(const float* P, const float* Q, float* R, int rows, int cols, int contract);
void metal_matmul_tn(const float* P, const float* Q, float* R, int rows, int cols, int contract);

}  // namespace psi

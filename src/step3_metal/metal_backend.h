// metal_backend.h — C++ interface to the Metal GPU backend (implemented in metal_backend.mm).
//
// This is the bridge that lets the pure-C++ autograd (tensor.hpp) dispatch float matmuls to the GPU.
// If Metal is unavailable, every call transparently falls back to a CPU implementation, so code that
// uses it stays correct on any machine.

#pragma once

namespace psi {

// True if a Metal GPU + pipeline was successfully initialized.
bool metal_available();

// Row-major float matmul on the GPU:  C[M x N] = A[M x K] @ B[K x N].
// Handles arbitrary (non-divisible) shapes. Falls back to CPU if Metal is unavailable.
void metal_matmul(const float* A, const float* B, float* C, int M, int K, int N);

}  // namespace psi

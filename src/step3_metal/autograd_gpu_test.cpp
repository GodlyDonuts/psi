// autograd_gpu_test.cpp — validates the Metal-backed autograd matmul (forward + both backward passes)
// bit-close vs a CPU float reference. Built float (-DPSI_REAL=float) so the GPU path is active.
// Uses non-divisible dims (200x150x176) so the tiled kernels' bounds handling is exercised in all
// three flavors (NN forward, NT for dA, TN for dB). Work = 200*150*176 = 5.28M >= 2^20 -> GPU.
//
// Build: clang++ -std=c++17 -O2 -DPSI_REAL=float autograd_gpu_test.cpp \
//                ../step3_metal/metal_backend.mm -framework Metal -framework Foundation -o agtest

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../step1_tensor_autograd/tensor.hpp"

using namespace psi;

int main() {
    std::printf("metal_available: %s\n", metal_available() ? "yes" : "no");
    std::mt19937 rng(0);
    const int M = 200, K = 150, N = 176;            // non-divisible -> stresses tiled-kernel bounds
    Tensor A = Tensor::randn({M, K}, rng, 1.0f), B = Tensor::randn({K, N}, rng, 1.0f);
    A.zero_grad(); B.zero_grad();
    Tensor loss = mean(matmul(A, B));               // forward on GPU
    loss.backward();                                // backward on GPU (dA via NT, dB via TN)

    // CPU float reference: dC = 1/(M*N) everywhere; dA = dC@B^T, dB = A^T@dC.
    real g = 1.0f / (real)(M * N);
    std::vector<real> rdA(M * K, 0), rdB(K * N, 0);
    for (int m = 0; m < M; ++m) for (int kk = 0; kk < K; ++kk) { real s = 0; for (int n = 0; n < N; ++n) s += g * B.data()[kk * N + n]; rdA[m * K + kk] = s; }
    for (int kk = 0; kk < K; ++kk) for (int n = 0; n < N; ++n) { real s = 0; for (int m = 0; m < M; ++m) s += A.data()[m * K + kk] * g; rdB[kk * N + n] = s; }

    auto rel = [](const std::vector<real>& ref, const std::vector<real>& got, int n) {
        double e = 0, r = 0;
        for (int i = 0; i < n; ++i) { e = std::fmax(e, std::fabs((double)got[i] - ref[i])); r = std::fmax(r, std::fabs((double)ref[i])); }
        return e / r;
    };
    double ra = rel(rdA, A.grad(), M * K), rb = rel(rdB, B.grad(), K * N);
    std::printf("GPU-backed autograd (%dx%dx%d, ragged)  dA rel=%.2e (%s)   dB rel=%.2e (%s)\n",
                M, K, N, ra, ra < 1e-3 ? "PASS" : "FAIL", rb, rb < 1e-3 ? "PASS" : "FAIL");
    return 0;
}

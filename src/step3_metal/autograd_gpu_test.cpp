// autograd_gpu_test.cpp — validates the Metal-backed autograd matmul (forward + both backward passes)
// bit-close vs a CPU float reference. Built float (-DPSI_REAL=float) so the GPU path is active; the
// 256^3 matmul exceeds the size gate and routes to Metal.
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
    const int S = 256;                              // 256^3 = 16.7M >= 2^20 -> GPU path
    Tensor A = Tensor::randn({S, S}, rng, 1.0f), B = Tensor::randn({S, S}, rng, 1.0f);
    A.zero_grad(); B.zero_grad();
    Tensor loss = mean(matmul(A, B));               // forward on GPU
    loss.backward();                                // backward on GPU (dA via NT, dB via TN)

    // CPU float reference: mean's grad makes dC = 1/(S*S) everywhere; dA = dC@B^T, dB = A^T@dC.
    real g = 1.0f / (real)(S * S);
    std::vector<real> rdA(S * S, 0), rdB(S * S, 0);
    for (int m = 0; m < S; ++m) for (int kk = 0; kk < S; ++kk) { real s = 0; for (int n = 0; n < S; ++n) s += g * B.data()[kk * S + n]; rdA[m * S + kk] = s; }
    for (int kk = 0; kk < S; ++kk) for (int n = 0; n < S; ++n) { real s = 0; for (int m = 0; m < S; ++m) s += A.data()[m * S + kk] * g; rdB[kk * S + n] = s; }

    auto rel = [&](const std::vector<real>& ref, const std::vector<real>& got) {
        double e = 0, r = 0;
        for (int i = 0; i < S * S; ++i) { e = std::fmax(e, std::fabs((double)got[i] - ref[i])); r = std::fmax(r, std::fabs((double)ref[i])); }
        return e / r;
    };
    double ra = rel(rdA, A.grad()), rb = rel(rdB, B.grad());
    std::printf("GPU-backed autograd  dA rel=%.2e (%s)   dB rel=%.2e (%s)\n",
                ra, ra < 1e-3 ? "PASS" : "FAIL", rb, rb < 1e-3 ? "PASS" : "FAIL");
    return 0;
}

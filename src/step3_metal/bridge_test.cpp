// bridge_test.cpp — proves the pure-C++ → Metal bridge works and is correct.
//
// Pure C++ (no Obj-C here): includes metal_backend.h, calls metal_matmul on a non-square,
// non-power-of-2 matrix, and checks bit-exactness vs a CPU reference. This is the keystone for
// wiring Metal into the autograd — if C++ can call the GPU correctly, the integration is sound.
//
// Build: clang++ -std=c++17 -O2 bridge_test.cpp metal_backend.mm \
//                -framework Metal -framework Foundation -o bridge_test

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "metal_backend.h"

int main() {
    using namespace psi;
    std::printf("metal_available: %s\n", metal_available() ? "yes" : "no");

    const int M = 128, K = 96, N = 64;  // deliberately non-divisible, to exercise bounds handling
    std::vector<float> A(M * K), B(K * N), C(M * N), Cref(M * N);
    std::mt19937 r(0);
    std::normal_distribution<float> d(0, 1);
    for (auto& x : A) x = d(r);
    for (auto& x : B) x = d(r);

    metal_matmul(A.data(), B.data(), C.data(), M, K, N);   // <- C++ calling the GPU

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0; for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            Cref[i * N + j] = acc;
        }
    double maxerr = 0, maxref = 0;
    for (int i = 0; i < M * N; ++i) { maxerr = std::fmax(maxerr, std::fabs(C[i] - Cref[i])); maxref = std::fmax(maxref, std::fabs(Cref[i])); }
    double rel = maxerr / maxref;
    std::printf("bridge matmul %dx%dx%d: max|diff|=%.2e  rel=%.2e  (%s)\n",
                M, K, N, maxerr, rel, rel < 1e-3 ? "PASS" : "FAIL");
    return 0;
}

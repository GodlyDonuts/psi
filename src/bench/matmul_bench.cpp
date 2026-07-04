// matmul_bench.cpp - the optimization scoreboard.
//
// matmul dominates the cost of the whole stack, so we measure it directly: time
// forward + backward of a sizable matmul and report GFLOP/s. Every speed optimization
// is judged by the number this prints (and must keep the grad-checks passing).
//
// Usage:  matmul_bench [iters] [size]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "../step1_tensor_autograd/tensor.hpp"

using namespace psi;

int main(int argc, char** argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 20;
    int S     = (argc > 2) ? std::atoi(argv[2]) : 256;   // square M=K=N=S
    int M = S, K = S, N = S;

    std::mt19937 rng(0);
    Tensor A = Tensor::randn({M, K}, rng, 1.0), B = Tensor::randn({K, N}, rng, 1.0);

    // One untimed warmup pass.
    { A.zero_grad(); B.zero_grad(); mean(matmul(A, B)).backward(); }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        A.zero_grad(); B.zero_grad();
        mean(matmul(A, B)).backward();          // forward + both backward passes
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    // FLOPs per iter: forward 2MKN, dA 2MKN, dB 2MKN  ->  6MKN.
    double flops = 6.0 * M * K * N * (double)iters;
    std::printf("matmul %dx%dx%d  fwd+bwd  %d iters  %.3fs  ->  %.2f GFLOP/s  (%.2f ms/iter)\n",
                M, K, N, iters, sec, flops / sec / 1e9, sec / iters * 1000.0);
    return 0;
}

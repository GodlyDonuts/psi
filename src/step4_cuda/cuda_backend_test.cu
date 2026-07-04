// cuda_backend_test.cu - correctness + benchmark harness for the CUDA GPU backend.
//
// Validates all three ops (forward write, both backward ACCUMULATES) against a double-precision CPU
// reference on deliberately non-power-of-2 shapes, then benchmarks end-to-end throughput. The engine
// (cublas/custom) is chosen by PSI_CUDA_KERNEL; PSI_CUDA_CHECK=1 additionally diffs the two internally.
// Run it under both engines to cover each vs the CPU oracle.
//
// Build (on the cluster):
//   nvcc -O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_90,code=sm_90 \
//        cuda_backend_test.cu cuda_backend.cu -lcublas -o cuda_backend_test

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include "../gpu_backend.h"

using clk = std::chrono::high_resolution_clock;
static double now_s() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

static std::vector<float> randv(int n, std::mt19937& r) {
    std::normal_distribution<float> d(0, 1);
    std::vector<float> v(n);
    for (auto& x : v) x = d(r);
    return v;
}

// double-precision references (tight oracle)
static double ref_nn(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int K, int N) {
    C.assign((size_t)M * N, 0);
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) { double a = 0; for (int k = 0; k < K; ++k) a += (double)A[i*K+k]*B[k*N+j]; C[i*N+j] = (float)a; }
    return 0;
}
static void ref_nt(const std::vector<float>& P, const std::vector<float>& Q, std::vector<float>& R, int rows, int cols, int contract) {
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) { double a = R[r*cols+c]; for (int i = 0; i < contract; ++i) a += (double)P[r*contract+i]*Q[c*contract+i]; R[r*cols+c] = (float)a; }
}
static void ref_tn(const std::vector<float>& P, const std::vector<float>& Q, std::vector<float>& R, int rows, int cols, int contract) {
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) { double a = R[r*cols+c]; for (int i = 0; i < contract; ++i) a += (double)P[i*rows+r]*Q[i*cols+c]; R[r*cols+c] = (float)a; }
}

static double relerr(const std::vector<float>& x, const std::vector<float>& y) {
    double me = 0, mr = 0;
    for (size_t i = 0; i < x.size(); ++i) { me = fmax(me, fabs((double)x[i]-y[i])); mr = fmax(mr, fabs((double)y[i])); }
    return mr > 0 ? me / mr : me;
}

int main() {
    using namespace psi;
    const char* eng = std::getenv("PSI_CUDA_KERNEL"); if (!eng) eng = "cublas";
    std::printf("=== CUDA backend test  (engine=%s, gpu_available=%s) ===\n", eng, gpu_available() ? "yes" : "no");
    if (!gpu_available()) { std::printf("no GPU visible - aborting\n"); return 2; }

    std::mt19937 rng(1234);
    int shapes[][3] = {{128,96,64},{200,200,200},{512,384,1024},{37,101,53},{1,512,768},{256,256,257}};
    int fails = 0;
    const double TOL = 2e-3;   // fp32 tiled reduction; cublas ~1e-6, custom-kernel ~1e-4

    for (auto& s : shapes) {
        int M = s[0], K = s[1], N = s[2];
        // forward: C = A@B
        auto A = randv(M*K, rng), B = randv(K*N, rng);
        std::vector<float> C(M*N), Cref; ref_nn(A, B, Cref, M, K, N);
        gpu_matmul(A.data(), B.data(), C.data(), M, K, N);
        double e_nn = relerr(C, Cref);
        // backward nt: R += P@Qᵀ   (P[M,N], Q[K,N])  -> R[M,K]      (uses m=M,k=K,n=N convention: dA += dC@Bᵀ)
        auto P1 = randv(M*N, rng), Q1 = randv(K*N, rng);
        std::vector<float> R1 = randv(M*K, rng), R1ref = R1;
        ref_nt(P1, Q1, R1ref, M, K, N);
        gpu_matmul_nt(P1.data(), Q1.data(), R1.data(), M, K, N);
        double e_nt = relerr(R1, R1ref);
        // backward tn: R += Pᵀ@Q   (P[M,K], Q[M,N]) -> R[K,N]      (dB += Aᵀ@dC)
        auto P2 = randv(M*K, rng), Q2 = randv(M*N, rng);
        std::vector<float> R2 = randv(K*N, rng), R2ref = R2;
        ref_tn(P2, Q2, R2ref, K, N, M);
        gpu_matmul_tn(P2.data(), Q2.data(), R2.data(), K, N, M);
        double e_tn = relerr(R2, R2ref);

        bool ok = e_nn < TOL && e_nt < TOL && e_tn < TOL;
        fails += !ok;
        std::printf("  %4dx%4dx%4d  nn=%.2e nt=%.2e tn=%.2e  %s\n", M, K, N, e_nn, e_nt, e_tn, ok ? "PASS" : "FAIL");
    }

    // benchmark: training-representative and compute-heavy shapes (end-to-end incl. H2D/D2H).
    std::printf("--- benchmark (end-to-end per call incl. transfers) ---\n");
    int bshapes[][3] = {{512,384,1024},{1024,1024,1024},{2048,2048,2048}};
    for (auto& s : bshapes) {
        int M = s[0], K = s[1], N = s[2];
        auto A = randv(M*K, rng), B = randv(K*N, rng);
        std::vector<float> C(M*N);
        for (int i = 0; i < 5; ++i) gpu_matmul(A.data(), B.data(), C.data(), M, K, N);   // warmup
        int iters = 30;
        double t0 = now_s();
        for (int i = 0; i < iters; ++i) gpu_matmul(A.data(), B.data(), C.data(), M, K, N);
        double dt = (now_s() - t0) / iters;
        double gf = 2.0 * M * K * N / dt / 1e9;
        std::printf("  %4dx%4dx%4d  %.3f ms/call  %.0f GFLOP/s\n", M, K, N, dt*1e3, gf);
    }
    std::printf("=== %s ===\n", fails ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return fails ? 1 : 0;
}

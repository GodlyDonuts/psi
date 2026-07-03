// cuda_backend.cu — NVIDIA implementation of the gpu_backend.h interface (peer to metal_backend.mm).
//
// Same four functions the autograd dispatches to (row-major, forward writes / backward accumulates).
// Two GEMM engines behind one code path, chosen at runtime:
//   PSI_CUDA_KERNEL=cublas   (default)  — cuBLAS SGEMM, the correctness/perf oracle
//   PSI_CUDA_KERNEL=custom               — a hand-written tiled kernel (the number to beat)
//   PSI_CUDA_CHECK=1                     — run BOTH every call and report max|cublas-custom|
//
// Row-major ↔ cuBLAS (column-major): compute Cᵀ = op(B)ᵀ·op(A)ᵀ by swapping the operands, i.e.
//   cublasSgemm(opB, opA, N, M, K, B, ldb, A, lda, C, N).   (derived + checked against the CPU oracle.)
//
// The autograd stores tensors in host memory, so each call is H2D→GEMM→D2H (like Metal, but Metal's
// unified memory makes its copies free — on a discrete GPU these transfers are real and cap the speedup
// of tiny matmuls; the win grows with matmul size, and a resident-activation path is the next step).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../gpu_backend.h"

#define TILE 16

// Non-fatal error reporting — a backend must never kill a training run; it warns and carries on.
#define CK(x) do{ cudaError_t e_=(x); if(e_) std::fprintf(stderr,"[cuda] %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); }while(0)
#define CB(x) do{ cublasStatus_t s_=(x); if(s_) std::fprintf(stderr,"[cublas] %s @ %d: status %d\n",#x,__LINE__,(int)s_); }while(0)

namespace {

// -------- general row-major tiled SGEMM: C[M,N] = op(A)[M,K] @ op(B)[K,N], then C = acc + beta*C --------
// tA/tB select whether the stored operand is the transpose of its logical [M,K]/[K,N] shape:
//   tA=0: A stored [M,K], op(A)[i,k]=A[i*K+k]      tA=1: A stored [K,M], op(A)[i,k]=A[k*M+i]
//   tB=0: B stored [K,N], op(B)[k,j]=B[k*N+j]      tB=1: B stored [N,K], op(B)[k,j]=B[j*K+k]
__global__ void sgemm_tiled(int M, int N, int K, const float* __restrict__ A,
                            const float* __restrict__ B, float* __restrict__ C,
                            int tA, int tB, float beta) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < K; t += TILE) {
        int aK = t + threadIdx.x;            // k-index this thread loads for A
        int bK = t + threadIdx.y;            // k-index this thread loads for B
        As[threadIdx.y][threadIdx.x] = (row < M && aK < K) ? (tA ? A[aK * M + row] : A[row * K + aK]) : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bK < K && col < N) ? (tB ? B[col * K + bK] : B[bK * N + col]) : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = (beta == 0.f) ? acc : acc + beta * C[idx];   // fwd writes; bwd accumulates onto uploaded R
    }
}

enum Mode { M_CUBLAS, M_CUSTOM };

std::once_flag  g_once;
bool            g_ok = false;
cublasHandle_t  g_h = nullptr;
std::mutex      g_mu;

Mode g_mode  = [] { const char* e = std::getenv("PSI_CUDA_KERNEL"); return (e && !std::strcmp(e, "custom")) ? M_CUSTOM : M_CUBLAS; }();
bool g_check = [] { const char* e = std::getenv("PSI_CUDA_CHECK");  return e && std::strcmp(e, "0") != 0; }();
bool g_off   = [] { const char* e = std::getenv("PSI_CUDA_DISABLE"); return e && std::strcmp(e, "0") != 0; }();  // force CPU path (for A/B timing)

// grow-only device scratch (the interface hands us host pointers each call; we reuse device buffers)
float* g_dA = nullptr; size_t g_cA = 0;
float* g_dB = nullptr; size_t g_cB = 0;
float* g_dC = nullptr; size_t g_cC = 0;
float* g_dX = nullptr; size_t g_cX = 0;   // check-mode second output

double g_maxrel = 0; long g_checks = 0;

void init_once() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return;
    if (cudaSetDevice(0) != cudaSuccess) return;
    if (cublasCreate(&g_h) != CUBLAS_STATUS_SUCCESS) return;
    g_ok = true;
}
bool ready() { if (g_off) return false; std::call_once(g_once, init_once); return g_ok; }

void ensure(float*& p, size_t& cap, size_t need) {
    if (cap >= need) return;
    if (p) cudaFree(p);
    CK(cudaMalloc(&p, need * sizeof(float)));
    cap = need;
}

void run(Mode m, int M, int N, int K, const float* dA, const float* dB, float* dC, int tA, int tB, float beta) {
    if (m == M_CUBLAS) {
        float alpha = 1.f;
        cublasOperation_t oa = tA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t ob = tB ? CUBLAS_OP_T : CUBLAS_OP_N;
        // row-major C = op(A)op(B)  ==  col-major Cᵀ = op(B)ᵀop(A)ᵀ : swap operands, sizes N,M,K.
        CB(cublasSgemm(g_h, ob, oa, N, M, K, &alpha, dB, tB ? K : N, dA, tA ? M : K, &beta, dC, N));
    } else {
        dim3 blk(TILE, TILE), grd((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        sgemm_tiled<<<grd, blk>>>(M, N, K, dA, dB, dC, tA, tB, beta);
    }
}

// C[M,N] = op(A)[M,K] @ op(B)[K,N] (+ beta*C).  Host C is uploaded first iff beta != 0 (accumulate).
void gemm(const float* A, const float* B, float* C, int M, int N, int K, int tA, int tB, float beta) {
    std::lock_guard<std::mutex> lk(g_mu);
    size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    ensure(g_dA, g_cA, nA); ensure(g_dB, g_cB, nB); ensure(g_dC, g_cC, nC);
    CK(cudaMemcpy(g_dA, A, nA * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_dB, B, nB * 4, cudaMemcpyHostToDevice));
    if (beta != 0.f) CK(cudaMemcpy(g_dC, C, nC * 4, cudaMemcpyHostToDevice));

    if (!g_check) {
        run(g_mode, M, N, K, g_dA, g_dB, g_dC, tA, tB, beta);
        CK(cudaMemcpy(C, g_dC, nC * 4, cudaMemcpyDeviceToHost));   // sync: waits for the GEMM
        return;
    }
    // self-diff: cuBLAS -> dC, custom -> dX, both from the same inputs, compare on host.
    ensure(g_dX, g_cX, nC);
    if (beta != 0.f) CK(cudaMemcpy(g_dX, C, nC * 4, cudaMemcpyHostToDevice));
    run(M_CUBLAS, M, N, K, g_dA, g_dB, g_dC, tA, tB, beta);
    run(M_CUSTOM, M, N, K, g_dA, g_dB, g_dX, tA, tB, beta);
    CK(cudaDeviceSynchronize());
    std::vector<float> ref(nC), cand(nC);
    CK(cudaMemcpy(ref.data(),  g_dC, nC * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cand.data(), g_dX, nC * 4, cudaMemcpyDeviceToHost));
    double me = 0, mr = 0;
    for (size_t i = 0; i < nC; ++i) { me = fmax(me, fabs((double)ref[i] - cand[i])); mr = fmax(mr, fabs((double)ref[i])); }
    double rel = mr > 0 ? me / mr : me;
    if (rel > g_maxrel) g_maxrel = rel;
    if (++g_checks <= 3 || rel > 1e-3)
        std::fprintf(stderr, "[cuda-check] %dx%dx%d tA%d tB%d  max|cublas-custom|=%.2e rel=%.2e  (running max rel %.2e)%s\n",
                     M, K, N, tA, tB, me, rel, g_maxrel, rel > 1e-3 ? "  <-- MISMATCH" : "");
    const std::vector<float>& sel = (g_mode == M_CUSTOM) ? cand : ref;
    std::memcpy(C, sel.data(), nC * 4);
}

// --- CPU reference (used only if no GPU is visible; mirrors metal_backend.mm's fallbacks exactly) ---
void cpu_nn(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) { float a = 0; for (int k = 0; k < K; ++k) a += A[i*K+k]*B[k*N+j]; C[i*N+j] = a; }
}
void cpu_nt(const float* P, const float* Q, float* R, int rows, int cols, int contract) {
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) { float a = 0; for (int i = 0; i < contract; ++i) a += P[r*contract+i]*Q[c*contract+i]; R[r*cols+c] += a; }
}
void cpu_tn(const float* P, const float* Q, float* R, int rows, int cols, int contract) {
    for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) { float a = 0; for (int i = 0; i < contract; ++i) a += P[i*rows+r]*Q[i*cols+c]; R[r*cols+c] += a; }
}

}  // namespace

namespace psi {

bool gpu_available() { return ready(); }

void gpu_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    if (!ready()) { cpu_nn(A, B, C, M, K, N); return; }
    gemm(A, B, C, M, N, K, /*tA=*/0, /*tB=*/0, /*beta=*/0.f);   // C = A @ B
}
void gpu_matmul_nt(const float* P, const float* Q, float* R, int rows, int cols, int contract) {
    if (!ready()) { cpu_nt(P, Q, R, rows, cols, contract); return; }
    gemm(P, Q, R, rows, cols, contract, /*tA=*/0, /*tB=*/1, /*beta=*/1.f);   // R += P @ Qᵀ
}
void gpu_matmul_tn(const float* P, const float* Q, float* R, int rows, int cols, int contract) {
    if (!ready()) { cpu_tn(P, Q, R, rows, cols, contract); return; }
    gemm(P, Q, R, rows, cols, contract, /*tA=*/1, /*tB=*/0, /*beta=*/1.f);   // R += Pᵀ @ Q
}

}  // namespace psi

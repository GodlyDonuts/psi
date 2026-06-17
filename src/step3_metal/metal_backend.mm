// metal_backend.mm — the C++ ↔ Metal bridge. Implements metal_backend.h.
//
// Lazily-initialized singleton (device + queue + three compiled pipelines: NN forward, NT/TN
// backward). Each call copies inputs into shared (unified-memory) buffers, dispatches, copies the
// result back. Kernels are threadgroup-tiled (16x16) and bounds-checked, so they handle arbitrary
// shapes. (Register-tiled / simdgroup_matrix with an autotuned config is a perf-round upgrade behind
// this same C++ API.)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>

#include "metal_backend.h"

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
#define T 16

// NN forward:  C[M,N] = A[M,K] @ B[K,N].   C = (write)
kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint2 lid [[thread_position_in_threadgroup]], uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[T][T], Bs[T][T];
    uint row = bid.y * T + lid.y, col = bid.x * T + lid.x;
    float acc = 0;
    for (uint t = 0; t < K; t += T) {
        As[lid.y][lid.x] = (row < M && t + lid.x < K) ? A[row * K + (t + lid.x)] : 0.0f;
        Bs[lid.y][lid.x] = (t + lid.y < K && col < N) ? B[(t + lid.y) * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < T; ++k) acc += As[lid.y][k] * Bs[k][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// NT backward:  R[rows,cols] += sum_c P[rows,c] * Q[cols,c]   (dA = dC @ B^T)
kernel void mm_nt(device const float* P [[buffer(0)]], device const float* Q [[buffer(1)]],
                  device float* R [[buffer(2)]], constant uint& rows [[buffer(3)]],
                  constant uint& cols [[buffer(4)]], constant uint& contract [[buffer(5)]],
                  uint2 lid [[thread_position_in_threadgroup]], uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float Ps[T][T], Qs[T][T];   // Ps[localRow][k], Qs[localCol][k]
    uint r = bid.y * T + lid.y, c = bid.x * T + lid.x;
    float acc = 0;
    for (uint t = 0; t < contract; t += T) {
        Ps[lid.y][lid.x] = (r < rows && t + lid.x < contract) ? P[r * contract + (t + lid.x)] : 0.0f;
        uint qcol = bid.x * T + lid.y;
        Qs[lid.y][lid.x] = (qcol < cols && t + lid.x < contract) ? Q[qcol * contract + (t + lid.x)] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < T; ++k) acc += Ps[lid.y][k] * Qs[lid.x][k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r < rows && c < cols) R[r * cols + c] += acc;
}

// TN backward:  R[rows,cols] += sum_c P[c,rows] * Q[c,cols]   (dB = A^T @ dC)
kernel void mm_tn(device const float* P [[buffer(0)]], device const float* Q [[buffer(1)]],
                  device float* R [[buffer(2)]], constant uint& rows [[buffer(3)]],
                  constant uint& cols [[buffer(4)]], constant uint& contract [[buffer(5)]],
                  uint2 lid [[thread_position_in_threadgroup]], uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float Ps[T][T], Qs[T][T];   // Ps[k][localRow], Qs[k][localCol]
    uint r = bid.y * T + lid.y, c = bid.x * T + lid.x;
    float acc = 0;
    for (uint t = 0; t < contract; t += T) {
        uint prow = bid.y * T + lid.x;
        Ps[lid.y][lid.x] = (t + lid.y < contract && prow < rows) ? P[(t + lid.y) * rows + prow] : 0.0f;
        uint qcol = bid.x * T + lid.x;
        Qs[lid.y][lid.x] = (t + lid.y < contract && qcol < cols) ? Q[(t + lid.y) * cols + qcol] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < T; ++k) acc += Ps[k][lid.y] * Qs[k][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r < rows && c < cols) R[r * cols + c] += acc;
}
)";

static id<MTLDevice> gDev;
static id<MTLCommandQueue> gQueue;
static id<MTLComputePipelineState> gNN, gNT, gTN;
static bool gInit = false, gOk = false;

static id<MTLComputePipelineState> makePSO(id<MTLLibrary> lib, const char* name) {
    NSError* e = nil;
    return [gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(name)] error:&e];
}

static void ensure_init() {
    if (gInit) return;
    gInit = true;
    gDev = MTLCreateSystemDefaultDevice();
    if (!gDev) return;
    NSError* e = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:@(kSrc) options:nil error:&e];
    if (!lib) return;
    gNN = makePSO(lib, "mm"); gNT = makePSO(lib, "mm_nt"); gTN = makePSO(lib, "mm_tn");
    gQueue = [gDev newCommandQueue];
    gOk = (gNN && gNT && gTN && gQueue);
}

// dispatch a tiled kernel over an outRows x outCols output, 16x16 threadgroups.
static void dispatch3(id<MTLComputePipelineState> pso, id<MTLBuffer> b0, id<MTLBuffer> b1,
                      id<MTLBuffer> b2, uint u0, uint u1, uint u2, int outRows, int outCols) {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:b0 offset:0 atIndex:0]; [enc setBuffer:b1 offset:0 atIndex:1]; [enc setBuffer:b2 offset:0 atIndex:2];
    [enc setBytes:&u0 length:4 atIndex:3]; [enc setBytes:&u1 length:4 atIndex:4]; [enc setBytes:&u2 length:4 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((outCols + 15) / 16, (outRows + 15) / 16, 1)
           threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
}

static id<MTLBuffer> bufIn(const float* p, long n) { return [gDev newBufferWithBytes:p length:(NSUInteger)n * sizeof(float) options:MTLResourceStorageModeShared]; }

namespace psi {

bool metal_available() { ensure_init(); return gOk; }

void metal_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    ensure_init();
    if (!gOk) {
        for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) { float a = 0; for (int k = 0; k < K; ++k) a += A[i * K + k] * B[k * N + j]; C[i * N + j] = a; }
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> bA = bufIn(A, (long)M * K), bB = bufIn(B, (long)K * N);
        id<MTLBuffer> bC = [gDev newBufferWithLength:(NSUInteger)M * N * sizeof(float) options:MTLResourceStorageModeShared];
        dispatch3(gNN, bA, bB, bC, M, K, N, M, N);
        std::memcpy(C, [bC contents], (size_t)M * N * sizeof(float));
    }
}

void metal_matmul_nt(const float* P, const float* Q, float* R, int rows, int cols, int contract) {
    ensure_init();
    if (!gOk) {
        for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) { float a = 0; for (int i = 0; i < contract; ++i) a += P[r * contract + i] * Q[c * contract + i]; R[r * cols + c] += a; }
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> bP = bufIn(P, (long)rows * contract), bQ = bufIn(Q, (long)cols * contract), bR = bufIn(R, (long)rows * cols);
        dispatch3(gNT, bP, bQ, bR, rows, cols, contract, rows, cols);
        std::memcpy(R, [bR contents], (size_t)rows * cols * sizeof(float));
    }
}

void metal_matmul_tn(const float* P, const float* Q, float* R, int rows, int cols, int contract) {
    ensure_init();
    if (!gOk) {
        for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) { float a = 0; for (int i = 0; i < contract; ++i) a += P[i * rows + r] * Q[i * cols + c]; R[r * cols + c] += a; }
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> bP = bufIn(P, (long)contract * rows), bQ = bufIn(Q, (long)contract * cols), bR = bufIn(R, (long)rows * cols);
        dispatch3(gTN, bP, bQ, bR, rows, cols, contract, rows, cols);
        std::memcpy(R, [bR contents], (size_t)rows * cols * sizeof(float));
    }
}

}  // namespace psi

// metal_backend.mm — the C++ ↔ Metal bridge. Implements metal_backend.h.
//
// Lazily-initialized singleton (device + queue + three compiled pipelines: NN forward, NT/TN
// backward). Each call copies inputs into shared (unified-memory) buffers, dispatches, copies the
// result back. Naive kernels for now (correctness first); the autotuned fast kernels slot in behind
// the same C++ API later, so the autograd never changes.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>

#include "metal_backend.h"

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= M || gid.x >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) acc += A[gid.y * K + k] * B[k * N + gid.x];
    C[gid.y * N + gid.x] = acc;                       // forward: write
}

kernel void mm_nt(device const float* P [[buffer(0)]], device const float* Q [[buffer(1)]],
                  device float* R [[buffer(2)]], constant uint& rows [[buffer(3)]],
                  constant uint& cols [[buffer(4)]], constant uint& contract [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= rows || gid.x >= cols) return;
    float acc = 0.0f;
    for (uint c = 0; c < contract; ++c) acc += P[gid.y * contract + c] * Q[gid.x * contract + c];
    R[gid.y * cols + gid.x] += acc;                   // backward: accumulate
}

kernel void mm_tn(device const float* P [[buffer(0)]], device const float* Q [[buffer(1)]],
                  device float* R [[buffer(2)]], constant uint& rows [[buffer(3)]],
                  constant uint& cols [[buffer(4)]], constant uint& contract [[buffer(5)]],
                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= rows || gid.x >= cols) return;
    float acc = 0.0f;
    for (uint c = 0; c < contract; ++c) acc += P[c * rows + gid.y] * Q[c * cols + gid.x];
    R[gid.y * cols + gid.x] += acc;                   // backward: accumulate
}
)";

static id<MTLDevice> gDev;
static id<MTLCommandQueue> gQueue;
static id<MTLComputePipelineState> gNN, gNT, gTN;
static bool gInit = false, gOk = false;

static id<MTLComputePipelineState> makePSO(id<MTLLibrary> lib, const char* name) {
    NSError* e = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
    return [gDev newComputePipelineStateWithFunction:fn error:&e];
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

static void dispatch3(id<MTLComputePipelineState> pso, id<MTLBuffer> b0, id<MTLBuffer> b1,
                      id<MTLBuffer> b2, uint u0, uint u1, uint u2, int gx, int gy) {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:b0 offset:0 atIndex:0]; [enc setBuffer:b1 offset:0 atIndex:1]; [enc setBuffer:b2 offset:0 atIndex:2];
    [enc setBytes:&u0 length:4 atIndex:3]; [enc setBytes:&u1 length:4 atIndex:4]; [enc setBytes:&u2 length:4 atIndex:5];
    [enc dispatchThreads:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
}

static id<MTLBuffer> bufIn(const float* p, long n)  { return [gDev newBufferWithBytes:p length:(NSUInteger)n * sizeof(float) options:MTLResourceStorageModeShared]; }

namespace psi {

bool metal_available() { ensure_init(); return gOk; }

void metal_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    ensure_init();
    if (!gOk) {
        for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
            float acc = 0; for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j]; C[i * N + j] = acc;
        }
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> bA = bufIn(A, (long)M * K), bB = bufIn(B, (long)K * N);
        id<MTLBuffer> bC = [gDev newBufferWithLength:(NSUInteger)M * N * sizeof(float) options:MTLResourceStorageModeShared];
        dispatch3(gNN, bA, bB, bC, M, K, N, N, M);
        std::memcpy(C, [bC contents], (size_t)M * N * sizeof(float));
    }
}

void metal_matmul_nt(const float* P, const float* Q, float* R, int rows, int cols, int contract) {
    ensure_init();
    if (!gOk) {
        for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) {
            float acc = 0; for (int i = 0; i < contract; ++i) acc += P[r * contract + i] * Q[c * contract + i]; R[r * cols + c] += acc;
        }
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> bP = bufIn(P, (long)rows * contract), bQ = bufIn(Q, (long)cols * contract), bR = bufIn(R, (long)rows * cols);
        dispatch3(gNT, bP, bQ, bR, rows, cols, contract, cols, rows);
        std::memcpy(R, [bR contents], (size_t)rows * cols * sizeof(float));
    }
}

void metal_matmul_tn(const float* P, const float* Q, float* R, int rows, int cols, int contract) {
    ensure_init();
    if (!gOk) {
        for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) {
            float acc = 0; for (int i = 0; i < contract; ++i) acc += P[i * rows + r] * Q[i * cols + c]; R[r * cols + c] += acc;
        }
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> bP = bufIn(P, (long)contract * rows), bQ = bufIn(Q, (long)contract * cols), bR = bufIn(R, (long)rows * cols);
        dispatch3(gTN, bP, bQ, bR, rows, cols, contract, cols, rows);
        std::memcpy(R, [bR contents], (size_t)rows * cols * sizeof(float));
    }
}

}  // namespace psi

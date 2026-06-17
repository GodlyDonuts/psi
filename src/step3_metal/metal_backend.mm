// metal_backend.mm — the C++ ↔ Metal bridge. Implements metal_backend.h.
//
// A lazily-initialized singleton (device + queue + compiled pipeline). matmul copies inputs into
// shared (unified-memory) buffers, dispatches the kernel, and copies the result back. v1 uses a
// simple naive kernel for correctness; the autotuned/fast kernel slots in here later behind the
// same C++ API, so the autograd never has to change.

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
    C[gid.y * N + gid.x] = acc;
}
)";

static id<MTLDevice> gDev;
static id<MTLCommandQueue> gQueue;
static id<MTLComputePipelineState> gPSO;
static bool gInit = false, gOk = false;

static void ensure_init() {
    if (gInit) return;
    gInit = true;
    gDev = MTLCreateSystemDefaultDevice();
    if (!gDev) return;
    NSError* e = nil;
    id<MTLLibrary> lib = [gDev newLibraryWithSource:@(kSrc) options:nil error:&e];
    if (!lib) return;
    id<MTLFunction> fn = [lib newFunctionWithName:@"mm"];
    gPSO = [gDev newComputePipelineStateWithFunction:fn error:&e];
    if (!gPSO) return;
    gQueue = [gDev newCommandQueue];
    gOk = (gQueue != nil);
}

namespace psi {

bool metal_available() { ensure_init(); return gOk; }

void metal_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    ensure_init();
    if (!gOk) {  // CPU fallback — keeps callers correct without a GPU
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float acc = 0; for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
                C[i * N + j] = acc;
            }
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> bA = [gDev newBufferWithBytes:A length:(NSUInteger)M * K * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bB = [gDev newBufferWithBytes:B length:(NSUInteger)K * N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bC = [gDev newBufferWithLength:(NSUInteger)M * N * sizeof(float) options:MTLResourceStorageModeShared];
        uint Mu = M, Ku = K, Nu = N;
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO];
        [enc setBuffer:bA offset:0 atIndex:0]; [enc setBuffer:bB offset:0 atIndex:1]; [enc setBuffer:bC offset:0 atIndex:2];
        [enc setBytes:&Mu length:4 atIndex:3]; [enc setBytes:&Ku length:4 atIndex:4]; [enc setBytes:&Nu length:4 atIndex:5];
        [enc dispatchThreads:MTLSizeMake(N, M, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        std::memcpy(C, [bC contents], (size_t)M * N * sizeof(float));
    }
}

}  // namespace psi

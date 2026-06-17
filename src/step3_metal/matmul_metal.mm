// matmul_metal.mm — Metal GPU matmul, naive vs threadgroup-tiled. C = A @ B.
//
// Step 3. Same O(n^3) FLOPs in every kernel — the speedup is pure hardware utilization (climbing
// toward the ~2.6 TFLOP/s M1 peak), not a lower exponent. We keep the bit-exact-vs-CPU gate.
//   * naive : one thread per output; reads A,B from global memory K times each (memory-bound).
//   * tiled : threadgroups stage 16x16 tiles of A,B in on-chip memory -> ~16x fewer global reads.
// Shaders compiled at runtime; unified-memory shared buffers (zero copy).
//
// Build: clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 matmul_metal.mm \
//                -framework Metal -framework Foundation -o matmul_metal

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static const char* kShaderSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void matmul_naive(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float*       C [[buffer(2)]],
                         constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]],
                         constant uint& N [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= M || gid.x >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) acc += A[gid.y * K + k] * B[k * N + gid.x];
    C[gid.y * N + gid.x] = acc;
}

#define TILE 16
kernel void matmul_tiled(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float*       C [[buffer(2)]],
                         constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]],
                         constant uint& N [[buffer(5)]],
                         uint2 tid [[thread_position_in_threadgroup]],
                         uint2 gid [[thread_position_in_grid]]) {
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];
    uint row = gid.y, col = gid.x;
    float acc = 0.0f;
    uint nTiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < nTiles; ++t) {
        uint aCol = t * TILE + tid.x;
        uint bRow = t * TILE + tid.y;
        As[tid.y][tid.x] = (row < M  && aCol < K) ? A[row * K + aCol]  : 0.0f;
        Bs[tid.y][tid.x] = (bRow < K && col  < N) ? B[bRow * N + col]  : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; ++k) acc += As[tid.y][k] * Bs[k][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
)";

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        std::printf("device: %s\n", dev.name.UTF8String);
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kShaderSrc) options:nil error:&err];
        if (!lib) { std::printf("shader compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        auto makePSO = [&](const char* name) {
            id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
            NSError* e = nil;
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&e];
            if (!p) std::printf("pipeline %s failed: %s\n", name, e.localizedDescription.UTF8String);
            return p;
        };
        id<MTLComputePipelineState> psoNaive = makePSO("matmul_naive");
        id<MTLComputePipelineState> psoTiled = makePSO("matmul_tiled");
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        const int M = 512, K = 512, N = 512;
        std::mt19937 rng(0);
        std::normal_distribution<float> dist(0, 1);
        std::vector<float> A(M * K), B(K * N);
        for (auto& x : A) x = dist(rng);
        for (auto& x : B) x = dist(rng);

        id<MTLBuffer> bA = [dev newBufferWithBytes:A.data() length:A.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bB = [dev newBufferWithBytes:B.data() length:B.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bC = [dev newBufferWithLength:M * N * sizeof(float) options:MTLResourceStorageModeShared];
        uint Mu = M, Ku = K, Nu = N;

        auto run = [&](id<MTLComputePipelineState> pso, bool tiled) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bA offset:0 atIndex:0];
            [enc setBuffer:bB offset:0 atIndex:1];
            [enc setBuffer:bC offset:0 atIndex:2];
            [enc setBytes:&Mu length:sizeof(uint) atIndex:3];
            [enc setBytes:&Ku length:sizeof(uint) atIndex:4];
            [enc setBytes:&Nu length:sizeof(uint) atIndex:5];
            if (tiled)
                [enc dispatchThreadgroups:MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1)
                       threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            else
                [enc dispatchThreads:MTLSizeMake(N, M, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        };

        std::vector<float> Cref(M * N);  // CPU reference (computed once)
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float acc = 0;
                for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
                Cref[i * N + j] = acc;
            }
        double maxref = 0;
        for (float v : Cref) maxref = std::fmax(maxref, std::fabs(v));

        auto bench = [&](const char* name, id<MTLComputePipelineState> pso, bool tiled) {
            run(pso, tiled);
            const float* Cg = static_cast<const float*>([bC contents]);
            double maxerr = 0;
            for (int i = 0; i < M * N; ++i) maxerr = std::fmax(maxerr, std::fabs(Cg[i] - Cref[i]));
            double rel = maxerr / maxref;
            const int iters = 50;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) run(pso, tiled);
            double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            double gflops = 2.0 * M * K * N * iters / sec / 1e9;
            std::printf("  %-7s rel=%.1e (%s)   %7.1f GFLOP/s   %4.1f%% of ~2.6 TFLOP peak\n",
                        name, rel, rel < 1e-3 ? "PASS" : "FAIL", gflops, 100.0 * gflops / 2600.0);
        };

        std::printf("matmul %dx%dx%d (same O(n^3) FLOPs; speedup = utilization):\n", M, K, N);
        bench("naive", psoNaive, false);
        bench("tiled", psoTiled, true);
    }
    return 0;
}

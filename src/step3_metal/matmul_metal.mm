// matmul_metal.mm — Psi's first GPU kernel: C = A @ B on the Apple GPU via Metal.
//
// Step 3, rung 1. Goals: (1) prove we can run a hand-written Metal compute kernel, (2) validate it
// bit-close against a CPU reference, (3) get a GFLOP/s number next to the CPU's ~22-40.
//
// Notes:
//   * Shaders are compiled AT RUNTIME from the source string below (no offline `metal` compiler
//     needed — only the Command Line Tools are installed).
//   * Buffers use MTLResourceStorageModeShared = UNIFIED MEMORY: the GPU reads our host arrays with
//     ZERO copy, and we read results straight from .contents(). This is the Apple-Silicon advantage.
//   * This naive one-thread-per-output kernel is the *baseline*; tiling / simdgroup_matrix come next.
//
// Build:  clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 matmul_metal.mm \
//                 -framework Metal -framework Foundation -o matmul_metal

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
                         constant uint& M [[buffer(3)]],
                         constant uint& K [[buffer(4)]],
                         constant uint& N [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= M || gid.x >= N) return;          // one thread per output element C[row,col]
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) acc += A[gid.y * K + k] * B[k * N + gid.x];
    C[gid.y * N + gid.x] = acc;
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
        id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_naive"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { std::printf("pipeline failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        const int M = 512, K = 512, N = 512;
        std::mt19937 rng(0);
        std::normal_distribution<float> dist(0, 1);
        std::vector<float> A(M * K), B(K * N);
        for (auto& x : A) x = dist(rng);
        for (auto& x : B) x = dist(rng);

        // Unified memory: GPU reads these host arrays directly, no copy.
        id<MTLBuffer> bA = [dev newBufferWithBytes:A.data() length:A.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bB = [dev newBufferWithBytes:B.data() length:B.size() * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bC = [dev newBufferWithLength:M * N * sizeof(float) options:MTLResourceStorageModeShared];
        uint Mu = M, Ku = K, Nu = N;

        auto run = [&] {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bA offset:0 atIndex:0];
            [enc setBuffer:bB offset:0 atIndex:1];
            [enc setBuffer:bC offset:0 atIndex:2];
            [enc setBytes:&Mu length:sizeof(uint) atIndex:3];
            [enc setBytes:&Ku length:sizeof(uint) atIndex:4];
            [enc setBytes:&Nu length:sizeof(uint) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(N, M, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        };

        run();  // warmup + the result we validate

        std::vector<float> Cref(M * N);  // CPU reference
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float acc = 0;
                for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
                Cref[i * N + j] = acc;
            }
        const float* Cg = static_cast<const float*>([bC contents]);  // read shared buffer directly
        double maxerr = 0, maxref = 0;
        for (int i = 0; i < M * N; ++i) {
            maxerr = std::fmax(maxerr, std::fabs(Cg[i] - Cref[i]));
            maxref = std::fmax(maxref, std::fabs(Cref[i]));
        }
        double rel = maxerr / maxref;
        std::printf("correctness vs CPU: max|diff|=%.3e  rel=%.3e  (%s)\n",
                    maxerr, rel, rel < 1e-3 ? "PASS" : "FAIL");

        const int iters = 50;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) run();
        double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        double gflops = 2.0 * M * K * N * iters / sec / 1e9;
        std::printf("matmul %dx%dx%d  fwd  %d iters  %.3fs  ->  %.1f GFLOP/s  (CPU baseline ~22-40)\n",
                    M, K, N, iters, sec, gflops);
    }
    return 0;
}

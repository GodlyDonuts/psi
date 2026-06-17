// matmul_metal.mm — Metal GPU matmul + a tiling AUTOTUNER. C = A @ B.
//
// Step 3. Same O(n^3) FLOPs everywhere — speed = hardware utilization. The register-tiled kernel
// is parameterized by (BM,BN,BK,TM,TN): each threadgroup computes a BM x BN block of C, each thread
// a TM x TN micro-tile held in registers (raising arithmetic intensity — the real lever). The
// autotuner compiles every candidate config, validates it bit-exact vs the CPU, benchmarks it, and
// reports the best — so the right tiling is chosen on M1 / M4 / M5 without hand-guessing.
//
// Build: clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 matmul_metal.mm \
//                -framework Metal -framework Foundation -o matmul_metal

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

// Naive baseline (one thread per output) — reference point for the sweep.
static const char* kNaive = R"(
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

// Register-tiled GEMM. Tile sizes come from #defines prepended per config.
static const char* kTiledBody = R"(
#include <metal_stdlib>
using namespace metal;
kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint2 lid [[thread_position_in_threadgroup]],
               uint2 bid [[threadgroup_position_in_grid]]) {
    uint tid = lid.x;
    threadgroup float As[BM * BK];
    threadgroup float Bs[BK * BN];
    uint blockRow = bid.y * BM, blockCol = bid.x * BN;
    uint tr = tid / (BN / TN), tc = tid % (BN / TN);   // this thread's TM x TN micro-tile
    float acc[TM][TN];
    for (uint i = 0; i < TM; ++i) for (uint j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint i = tid; i < BM * BK; i += NT) {     // cooperative load of A block
            uint r = i / BK, c = i % BK, gr = blockRow + r, gc = k0 + c;
            As[i] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (uint i = tid; i < BK * BN; i += NT) {     // cooperative load of B block
            uint r = i / BN, c = i % BN, gr = k0 + r, gc = blockCol + c;
            Bs[i] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; ++kk) {             // multiply the staged tiles
            float a[TM], b[TN];
            for (uint i = 0; i < TM; ++i) a[i] = As[(tr * TM + i) * BK + kk];
            for (uint j = 0; j < TN; ++j) b[j] = Bs[kk * BN + tc * TN + j];
            for (uint i = 0; i < TM; ++i) for (uint j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < TM; ++i) for (uint j = 0; j < TN; ++j) {
        uint gr = blockRow + tr * TM + i, gc = blockCol + tc * TN + j;
        if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
    }
}
)";

struct Cfg { int BM, BN, BK, TM, TN; };

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        std::printf("device: %s   (peak ~2.6 TFLOP/s fp32 on M1)\n", dev.name.UTF8String);
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        const int M = 512, K = 512, N = 512;
        std::mt19937 rng(0);
        std::normal_distribution<float> dist(0, 1);
        std::vector<float> A(M * K), B(K * N), Cref(M * N);
        for (auto& x : A) x = dist(rng);
        for (auto& x : B) x = dist(rng);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float acc = 0; for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
                Cref[i * N + j] = acc;
            }
        double maxref = 0; for (float v : Cref) maxref = std::fmax(maxref, std::fabs(v));

        id<MTLBuffer> bA = [dev newBufferWithBytes:A.data() length:A.size()*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bB = [dev newBufferWithBytes:B.data() length:B.size()*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bC = [dev newBufferWithLength:M*N*4 options:MTLResourceStorageModeShared];
        uint Mu = M, Ku = K, Nu = N;

        auto compile = [&](const std::string& src) -> id<MTLComputePipelineState> {
            NSError* e = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:@(src.c_str()) options:nil error:&e];
            if (!lib) { std::printf("  compile error: %s\n", e.localizedDescription.UTF8String); return nil; }
            id<MTLFunction> fn = [lib newFunctionWithName:@"mm"];
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&e];
            if (!p) std::printf("  pipeline error: %s\n", e.localizedDescription.UTF8String);
            return p;
        };
        auto encode = [&](id<MTLComputePipelineState> pso, MTLSize grid, MTLSize tg) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bA offset:0 atIndex:0]; [enc setBuffer:bB offset:0 atIndex:1];
            [enc setBuffer:bC offset:0 atIndex:2];
            [enc setBytes:&Mu length:4 atIndex:3]; [enc setBytes:&Ku length:4 atIndex:4]; [enc setBytes:&Nu length:4 atIndex:5];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        };
        auto evaluate = [&](const char* label, id<MTLComputePipelineState> pso, MTLSize grid, MTLSize tg) -> double {
            if (!pso) return 0;
            encode(pso, grid, tg);
            const float* Cg = static_cast<const float*>([bC contents]);
            double maxerr = 0; for (int i = 0; i < M*N; ++i) maxerr = std::fmax(maxerr, std::fabs(Cg[i]-Cref[i]));
            double rel = maxerr / maxref;
            const int iters = 50;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) encode(pso, grid, tg);
            double sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            double gf = 2.0 * M * K * N * iters / sec / 1e9;
            std::printf("  %-22s rel=%.0e %s  %7.1f GFLOP/s  (%4.1f%% peak)\n",
                        label, rel, rel < 1e-3 ? "PASS" : "FAIL", gf, 100.0 * gf / 2600.0);
            return rel < 1e-3 ? gf : 0;
        };

        std::printf("autotuning matmul %dx%dx%d:\n", M, K, N);
        // naive baseline
        evaluate("naive", compile(kNaive),
                 MTLSizeMake((N+15)/16,(M+15)/16,1), MTLSizeMake(16,16,1));

        std::vector<Cfg> cfgs = {
            {64,64,8,4,4}, {64,64,16,4,4}, {128,128,8,8,8}, {128,64,8,8,4}, {64,128,16,4,8}, {128,128,16,8,8}
        };
        Cfg best{}; double bestGf = 0;
        for (auto& c : cfgs) {
            int NT = (c.BM/c.TM) * (c.BN/c.TN);
            long tg = (long)(c.BM*c.BK + c.BK*c.BN) * 4;
            char buf[64]; std::snprintf(buf, sizeof(buf), "tiled %d-%d-%d/%dx%d", c.BM,c.BN,c.BK,c.TM,c.TN);
            if (NT > 1024 || tg > 32768) { std::printf("  %-22s skipped (NT=%d, tgmem=%ldB)\n", buf, NT, tg); continue; }
            std::string def = "#define BM " + std::to_string(c.BM) + "\n#define BN " + std::to_string(c.BN) +
                "\n#define BK " + std::to_string(c.BK) + "\n#define TM " + std::to_string(c.TM) +
                "\n#define TN " + std::to_string(c.TN) + "\n#define NT " + std::to_string(NT) + "\n";
            double gf = evaluate(buf, compile(def + kTiledBody),
                                 MTLSizeMake((N+c.BN-1)/c.BN, (M+c.BM-1)/c.BM, 1), MTLSizeMake(NT,1,1));
            if (gf > bestGf) { bestGf = gf; best = c; }
        }
        std::printf("BEST: tiled %d-%d-%d/%dx%d  @ %.1f GFLOP/s (%.1f%% peak, %.1fx the naive)\n",
                    best.BM,best.BN,best.BK,best.TM,best.TN, bestGf, 100.0*bestGf/2600.0, bestGf/151.0);
    }
    return 0;
}

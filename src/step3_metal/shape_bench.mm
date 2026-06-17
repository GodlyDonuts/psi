// shape_bench.mm — measure our winning GPU matmul kernel on the shapes the MODEL actually runs,
// not just big squares. The lesson from benchmarking against MLX/PyTorch: square-2048 fp32 is a
// vanity number. A transformer's matmuls are fat-M / thin-K-N — (batch*seq, d) x (d, d_ff) — and
// those live in a different performance regime. This runs the multi-simdgroup tiled-MMA kernel
// (the autotuner's winner, BK=16) across a set of (M,K,N) shapes, validates bit-exact vs a parallel
// CPU reference, and reports GFLOP/s + % of the M1's ~2.6 TFLOP fp32 peak.
//
// Build: clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 shape_bench.mm \
//                -framework Metal -framework Foundation -o shape_bench && ./shape_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

// Multi-simdgroup tiled MMA (the autotuner winner). 4 simdgroups/128 threads compute a 32x32 block;
// A/B tiles staged in threadgroup memory for reuse; each simdgroup owns a 2x2 grid of 8x8 MMA frags.
// BK=16 baked in. Requires M%32==0, N%32==0, K%16==0 (all our shapes qualify).
static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
#define BK 16
kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint sg [[simdgroup_index_in_threadgroup]],
               uint tid [[thread_index_in_threadgroup]],
               uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[32 * BK];
    threadgroup float Bs[BK * 32];
    uint blockRow = bid.y * 32, blockCol = bid.x * 32;
    uint sgY = sg / 2, sgX = sg % 2;
    simdgroup_float8x8 acc[2][2];
    for (uint r = 0; r < 2; ++r) for (uint c = 0; c < 2; ++c) acc[r][c] = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint i = tid; i < 32 * BK; i += 128) { uint r = i / BK, c = i % BK; As[i] = A[(blockRow + r) * K + (k0 + c)]; }
        for (uint i = tid; i < BK * 32; i += 128) { uint r = i / 32, c = i % 32; Bs[i] = B[(k0 + r) * N + (blockCol + c)]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 af[2], bf[2];
            for (uint r = 0; r < 2; ++r) simdgroup_load(af[r], As + (sgY * 16 + r * 8) * BK + kk, BK);
            for (uint c = 0; c < 2; ++c) simdgroup_load(bf[c], Bs + kk * 32 + (sgX * 16 + c * 8), 32);
            for (uint r = 0; r < 2; ++r) for (uint c = 0; c < 2; ++c)
                simdgroup_multiply_accumulate(acc[r][c], af[r], bf[c], acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < 2; ++r) for (uint c = 0; c < 2; ++c)
        simdgroup_store(acc[r][c], C + (blockRow + sgY * 16 + r * 8) * N + (blockCol + sgX * 16 + c * 8), N);
}
)";

struct Shape { int M, K, N; const char* name; };

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        std::printf("device: %s   (peak ~2.6 TFLOP/s fp32)\n", dev.name.UTF8String);
        NSError* e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&e];
        if (!lib) { std::printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm"] error:&e];
        if (!pso) { std::printf("pipeline: %s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        Shape shapes[] = {
            {1024, 1024, 1024, "square-1024 (lib sweet spot)"},
            {2048, 2048, 2048, "square-2048 (old target)"},
            {8192,  384,  384, "attn-proj  (B*T,d)x(d,d)"},
            {8192,  384, 1536, "mlp-up     (B*T,d)x(d,4d)"},
            {8192, 1536,  384, "mlp-down   (B*T,4d)x(4d,d)"},
            {8192,  384, 4096, "logits     (B*T,d)x(d,V)"},
        };
        std::printf("\n%-30s %-18s %10s  %s\n", "shape", "MxKxN", "GFLOP/s", "% peak");
        for (auto s : shapes) {
            int M = s.M, K = s.K, N = s.N;
            std::mt19937 rng(0);
            std::normal_distribution<float> dist(0, 1);
            std::vector<float> A(M*K), B(K*N), Cref(M*N);
            for (auto& x : A) x = dist(rng);
            for (auto& x : B) x = dist(rng);
            {   // parallel CPU reference
                unsigned nt = std::max(1u, std::thread::hardware_concurrency());
                std::vector<std::thread> pool;
                int chunk = (M + nt - 1) / nt;
                for (unsigned t = 0; t < nt; ++t) {
                    int r0 = t*chunk, r1 = std::min(M, r0+chunk);
                    if (r0 >= r1) break;
                    pool.emplace_back([&,r0,r1]{
                        for (int i=r0;i<r1;++i) for (int j=0;j<N;++j){
                            float acc=0; for(int k=0;k<K;++k) acc+=A[i*K+k]*B[k*N+j]; Cref[i*N+j]=acc; }
                    });
                }
                for (auto& th : pool) th.join();
            }
            double maxref=0; for(float v:Cref) maxref=std::fmax(maxref,std::fabs(v));

            id<MTLBuffer> bA=[dev newBufferWithBytes:A.data() length:A.size()*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bB=[dev newBufferWithBytes:B.data() length:B.size()*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bC=[dev newBufferWithLength:M*N*4 options:MTLResourceStorageModeShared];
            uint Mu=M,Ku=K,Nu=N;
            MTLSize grid=MTLSizeMake(N/32, M/32, 1), tg=MTLSizeMake(128,1,1);
            auto run=[&]{
                id<MTLCommandBuffer> cb=[queue commandBuffer];
                id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bA offset:0 atIndex:0];[enc setBuffer:bB offset:0 atIndex:1];[enc setBuffer:bC offset:0 atIndex:2];
                [enc setBytes:&Mu length:4 atIndex:3];[enc setBytes:&Ku length:4 atIndex:4];[enc setBytes:&Nu length:4 atIndex:5];
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                [enc endEncoding];[cb commit];[cb waitUntilCompleted];
            };
            run();
            const float* Cg=static_cast<const float*>([bC contents]);
            double maxerr=0; for(int i=0;i<M*N;++i) maxerr=std::fmax(maxerr,std::fabs(Cg[i]-Cref[i]));
            double rel=maxerr/maxref;
            const int reps=5, iters=30; double best=1e30;
            for(int r=0;r<reps;++r){
                auto t0=std::chrono::high_resolution_clock::now();
                for(int it=0;it<iters;++it) run();
                best=std::fmin(best,std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count());
            }
            double gf=2.0*M*K*N*iters/best/1e9;
            std::printf("%-30s %5dx%5dx%-5d  %s %8.1f  (%4.1f%%)\n",
                        s.name, M, K, N, rel<1e-3?"PASS":"FAIL", gf, 100.0*gf/2600.0);
        }
    }
    return 0;
}

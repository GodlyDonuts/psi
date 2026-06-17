// db_dev.mm — kernel-design dev harness for the multi-simdgroup MMA matmul. Compares variants
// head-to-head, all validated bit-exact vs a parallel CPU ref, timed best-of-N across the shapes
// the model actually runs. Goal: close the ~2x gap to MLX/PyTorch (which hit ~57% of M1 peak).
//
// Findings log (measured on Apple M1, this harness):
//  - DOUBLE-BUFFERING refuted: 0.69-0.85x (SLOWER). Doubling threadgroup mem halves per-core
//    threadgroup residency; on Apple GPUs that occupancy IS the latency-hiding, so manual pipelining
//    loses. => kernel is occupancy / memory-throughput bound, not barrier-bound.
//  - Next levers (don't cost occupancy): float4 vectorized loads; higher arithmetic intensity via a
//    64x64 block (each simdgroup computes a 4x4 grid of 8x8 MMA frags -> more reuse per staged tile).
//
// Build: clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 db_dev.mm \
//                -framework Metal -framework Foundation -o db_dev && ./db_dev

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ---- baseline: single-buffer, 32x32 block, 4 simdgroups, acc[2][2], scalar loads (autotuner winner)
static const char* kBase = R"(
#include <metal_stdlib>
using namespace metal;
#define BK 16
kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]],
               uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[32 * BK]; threadgroup float Bs[BK * 32];
    uint blockRow = bid.y * 32, blockCol = bid.x * 32, sgY = sg / 2, sgX = sg % 2;
    simdgroup_float8x8 acc[2][2];
    for (uint r=0;r<2;++r) for (uint c=0;c<2;++c) acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint i=tid;i<32*BK;i+=128){ uint r=i/BK,c=i%BK; As[i]=A[(blockRow+r)*K+(k0+c)]; }
        for (uint i=tid;i<BK*32;i+=128){ uint r=i/32,c=i%32; Bs[i]=B[(k0+r)*N+(blockCol+c)]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk=0;kk<BK;kk+=8){
            simdgroup_float8x8 af[2], bf[2];
            for (uint r=0;r<2;++r) simdgroup_load(af[r], As+(sgY*16+r*8)*BK+kk, BK);
            for (uint c=0;c<2;++c) simdgroup_load(bf[c], Bs+kk*32+(sgX*16+c*8), 32);
            for (uint r=0;r<2;++r) for (uint c=0;c<2;++c) simdgroup_multiply_accumulate(acc[r][c],af[r],bf[c],acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r=0;r<2;++r) for (uint c=0;c<2;++c)
        simdgroup_store(acc[r][c], C+(blockRow+sgY*16+r*8)*N+(blockCol+sgX*16+c*8), N);
}
)";

// ---- f4: baseline + float4 vectorized loads (same block/occupancy, fewer/wider load instructions) ----
static const char* kF4 = R"(
#include <metal_stdlib>
using namespace metal;
#define BK 16
kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]],
               uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[32 * BK]; threadgroup float Bs[BK * 32];
    uint blockRow = bid.y * 32, blockCol = bid.x * 32, sgY = sg / 2, sgX = sg % 2;
    simdgroup_float8x8 acc[2][2];
    for (uint r=0;r<2;++r) for (uint c=0;c<2;++c) acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        { uint r=tid/4, c4=(tid%4)*4;   // As 32x16 = 128 float4, one per thread
          *(threadgroup float4*)(As+r*BK+c4) = *(device const float4*)(A+(blockRow+r)*K+k0+c4); }
        { uint r=tid/8, c4=(tid%8)*4;   // Bs 16x32 = 128 float4, one per thread
          *(threadgroup float4*)(Bs+r*32+c4) = *(device const float4*)(B+(k0+r)*N+blockCol+c4); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk=0;kk<BK;kk+=8){
            simdgroup_float8x8 af[2], bf[2];
            for (uint r=0;r<2;++r) simdgroup_load(af[r], As+(sgY*16+r*8)*BK+kk, BK);
            for (uint c=0;c<2;++c) simdgroup_load(bf[c], Bs+kk*32+(sgX*16+c*8), 32);
            for (uint r=0;r<2;++r) for (uint c=0;c<2;++c) simdgroup_multiply_accumulate(acc[r][c],af[r],bf[c],acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r=0;r<2;++r) for (uint c=0;c<2;++c)
        simdgroup_store(acc[r][c], C+(blockRow+sgY*16+r*8)*N+(blockCol+sgX*16+c*8), N);
}
)";

// ---- f4_3264: 32x64 block, 4 simdgroups (2x2), each owns 16x32 = acc[2][4] (8 frags), float4 loads.
//      Bs staged tile is 2x wider -> reused across 4 column-frags. 8 frags should fit registers. ----
static const char* k3264 = R"(
#include <metal_stdlib>
using namespace metal;
#define BK 16
kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]],
               uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[32 * BK]; threadgroup float Bs[BK * 64];
    uint blockRow = bid.y * 32, blockCol = bid.x * 64, sgY = sg / 2, sgX = sg % 2;
    simdgroup_float8x8 acc[2][4];
    for (uint r=0;r<2;++r) for (uint c=0;c<4;++c) acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        { uint r=tid/4, c4=(tid%4)*4;                              // As 32x16 = 128 float4
          *(threadgroup float4*)(As+r*BK+c4) = *(device const float4*)(A+(blockRow+r)*K+k0+c4); }
        for (uint t=tid;t<256;t+=128){ uint r=t/16, c4=(t%16)*4;   // Bs 16x64 = 256 float4
          *(threadgroup float4*)(Bs+r*64+c4) = *(device const float4*)(B+(k0+r)*N+blockCol+c4); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk=0;kk<BK;kk+=8){
            simdgroup_float8x8 af[2], bf[4];
            for (uint r=0;r<2;++r) simdgroup_load(af[r], As+(sgY*16+r*8)*BK+kk, BK);
            for (uint c=0;c<4;++c) simdgroup_load(bf[c], Bs+kk*64+(sgX*32+c*8), 64);
            for (uint r=0;r<2;++r) for (uint c=0;c<4;++c) simdgroup_multiply_accumulate(acc[r][c],af[r],bf[c],acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r=0;r<2;++r) for (uint c=0;c<4;++c)
        simdgroup_store(acc[r][c], C+(blockRow+sgY*16+r*8)*N+(blockCol+sgX*32+c*8), N);
}
)";

// ---- f4_6432: 64x32 block, 4 simdgroups (2x2), each owns 32x16 = acc[4][2] (8 frags), float4 loads. ----
static const char* k6432 = R"(
#include <metal_stdlib>
using namespace metal;
#define BK 16
kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]],
               uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[64 * BK]; threadgroup float Bs[BK * 32];
    uint blockRow = bid.y * 64, blockCol = bid.x * 32, sgY = sg / 2, sgX = sg % 2;
    simdgroup_float8x8 acc[4][2];
    for (uint r=0;r<4;++r) for (uint c=0;c<2;++c) acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint t=tid;t<256;t+=128){ uint r=t/4, c4=(t%4)*4;     // As 64x16 = 256 float4
          *(threadgroup float4*)(As+r*BK+c4) = *(device const float4*)(A+(blockRow+r)*K+k0+c4); }
        { uint r=tid/8, c4=(tid%8)*4;                              // Bs 16x32 = 128 float4
          *(threadgroup float4*)(Bs+r*32+c4) = *(device const float4*)(B+(k0+r)*N+blockCol+c4); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk=0;kk<BK;kk+=8){
            simdgroup_float8x8 af[4], bf[2];
            for (uint r=0;r<4;++r) simdgroup_load(af[r], As+(sgY*32+r*8)*BK+kk, BK);
            for (uint c=0;c<2;++c) simdgroup_load(bf[c], Bs+kk*32+(sgX*16+c*8), 32);
            for (uint r=0;r<4;++r) for (uint c=0;c<2;++c) simdgroup_multiply_accumulate(acc[r][c],af[r],bf[c],acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r=0;r<4;++r) for (uint c=0;c<2;++c)
        simdgroup_store(acc[r][c], C+(blockRow+sgY*32+r*8)*N+(blockCol+sgX*16+c*8), N);
}
)";

struct Kern { const char* name; const char* src; int bM; int bN; id<MTLComputePipelineState> pso; };
struct Shape { int M, K, N; const char* name; };

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        std::printf("device: %s   (peak ~2.6 TFLOP/s fp32)\n", dev.name.UTF8String);
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        auto compile = [&](const char* src) -> id<MTLComputePipelineState> {
            NSError* e = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:@(src) options:nil error:&e];
            if (!lib) { std::printf("compile: %s\n", e.localizedDescription.UTF8String); return nil; }
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm"] error:&e];
            if (!p) std::printf("pipeline: %s\n", e.localizedDescription.UTF8String);
            return p;
        };
        Kern kerns[] = { {"base32", kBase, 32, 32, nil}, {"f4_32x32", kF4, 32, 32, nil},
                         {"f4_32x64", k3264, 32, 64, nil}, {"f4_64x32", k6432, 64, 32, nil} };
        for (auto& k : kerns) { k.pso = compile(k.src); if (!k.pso) return 1; }

        Shape shapes[] = {
            {1024,1024,1024,"square-1024"}, {2048,2048,2048,"square-2048"},
            {8192,384,1536,"mlp-up"}, {8192,1536,384,"mlp-down"},
        };
        for (auto s : shapes) {
            int M=s.M,K=s.K,N=s.N;
            std::mt19937 rng(0); std::normal_distribution<float> dist(0,1);
            std::vector<float> A(M*K), B(K*N), Cref(M*N);
            for (auto& x:A) x=dist(rng); for (auto& x:B) x=dist(rng);
            { unsigned nt=std::max(1u,std::thread::hardware_concurrency()); std::vector<std::thread> pool; int chunk=(M+nt-1)/nt;
              for (unsigned t=0;t<nt;++t){ int r0=t*chunk,r1=std::min(M,r0+chunk); if(r0>=r1)break;
                pool.emplace_back([&,r0,r1]{ for(int i=r0;i<r1;++i) for(int j=0;j<N;++j){ float a=0; for(int k=0;k<K;++k) a+=A[i*K+k]*B[k*N+j]; Cref[i*N+j]=a; } }); }
              for (auto& th:pool) th.join(); }
            double maxref=0; for(float v:Cref) maxref=std::fmax(maxref,std::fabs(v));
            id<MTLBuffer> bA=[dev newBufferWithBytes:A.data() length:A.size()*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bB=[dev newBufferWithBytes:B.data() length:B.size()*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bC=[dev newBufferWithLength:M*N*4 options:MTLResourceStorageModeShared];
            uint Mu=M,Ku=K,Nu=N;
            std::printf("\n%-12s %dx%dx%d\n", s.name, M, K, N);
            for (auto& k : kerns) {
                MTLSize grid=MTLSizeMake(N/k.bN, M/k.bM, 1), tg=MTLSizeMake(128,1,1);
                auto run=[&]{ id<MTLCommandBuffer> cb=[queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:k.pso];
                    [enc setBuffer:bA offset:0 atIndex:0];[enc setBuffer:bB offset:0 atIndex:1];[enc setBuffer:bC offset:0 atIndex:2];
                    [enc setBytes:&Mu length:4 atIndex:3];[enc setBytes:&Ku length:4 atIndex:4];[enc setBytes:&Nu length:4 atIndex:5];
                    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];[enc endEncoding];[cb commit];[cb waitUntilCompleted]; };
                run();
                const float* Cg=static_cast<const float*>([bC contents]);
                double maxerr=0; for(int i=0;i<M*N;++i) maxerr=std::fmax(maxerr,std::fabs(Cg[i]-Cref[i]));
                double rel=maxerr/maxref;
                const int reps=5, iters=30; double best=1e30;
                for(int r=0;r<reps;++r){ auto t0=std::chrono::high_resolution_clock::now();
                    for(int it=0;it<iters;++it) run();
                    best=std::fmin(best,std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count()); }
                double gf=2.0*M*K*N*iters/best/1e9;
                std::printf("  %-10s %s  %7.1f GFLOP/s  (%4.1f%% peak)\n", k.name, rel<1e-3?"OK":"XX", gf, 100*gf/2600.0);
            }
        }
    }
    return 0;
}

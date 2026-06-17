// db_dev.mm — kernel-design dev harness for the multi-simdgroup MMA matmul. Compares variants
// head-to-head, all validated bit-exact vs a parallel CPU ref, timed best-of-N across the shapes
// the model actually runs. Goal: MATCH OR BEAT MLX/PyTorch (which hit ~55-74% of M1 peak) on the
// real transformer shapes.
//
// Findings log (measured on Apple M1, this harness):
//  - float4 vectorized loads: BIG WIN (~+50%), bit-exact. Real shapes ~28-30% -> 40-46% peak. BANKED.
//  - DOUBLE-BUFFERING refuted: 0.69-0.85x. Doubling threadgroup mem halves per-core residency; on
//    Apple GPUs that occupancy IS the latency-hiding, so manual pipelining loses. => occupancy is king.
//  - big64 acc[4][4] (16 frags): catastrophic spill (2.4%). Register pressure kills occupancy.
//  - moderate tiles 32x64/64x32 (8 frags + f4): shape-dependent (32x32 best on real shapes).
//  This round: BK re-sweep WITH float4 (old BK=16 tuned for scalar loads), + no-staging "direct" kernel
//  (simdgroup_load fragments straight from device memory -> zero threadgroup mem -> max occupancy).
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

// f4 32x32 / 4 simdgroups / acc[2][2] / float4 loads — BK injected via #define prefix.
static const char* kF4Body = R"(
#include <metal_stdlib>
using namespace metal;
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
        for (uint t=tid;t<8*BK;t+=128){ uint lin=t*4, r=lin/BK, c4=lin%BK;
            *(threadgroup float4*)(As+r*BK+c4) = *(device const float4*)(A+(blockRow+r)*K+(k0+c4)); }
        for (uint t=tid;t<8*BK;t+=128){ uint lin=t*4, r=lin/32, c4=lin%32;
            *(threadgroup float4*)(Bs+r*32+c4) = *(device const float4*)(B+(k0+r)*N+(blockCol+c4)); }
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

// direct: NO threadgroup staging. Each simdgroup loads its A/B 8x8 fragments straight from device
// memory and MMAs. Zero threadgroup mem -> maximum per-core threadgroup residency (occupancy). Reuse
// comes only from the device cache + the 2x2 register-blocked acc. Tests "occupancy beats staging".
static const char* kDirect = R"(
#include <metal_stdlib>
using namespace metal;
kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint sg [[simdgroup_index_in_threadgroup]],
               uint2 bid [[threadgroup_position_in_grid]]) {
    uint blockRow = bid.y * 32, blockCol = bid.x * 32, sgY = sg / 2, sgX = sg % 2;
    uint row = blockRow + sgY * 16, col = blockCol + sgX * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint r=0;r<2;++r) for (uint c=0;c<2;++c) acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint k = 0; k < K; k += 8) {
        simdgroup_float8x8 af[2], bf[2];
        for (uint r=0;r<2;++r) simdgroup_load(af[r], A + (row + r*8)*K + k, K);
        for (uint c=0;c<2;++c) simdgroup_load(bf[c], B + k*N + (col + c*8), N);
        for (uint r=0;r<2;++r) for (uint c=0;c<2;++c) simdgroup_multiply_accumulate(acc[r][c],af[r],bf[c],acc[r][c]);
    }
    for (uint r=0;r<2;++r) for (uint c=0;c<2;++c)
        simdgroup_store(acc[r][c], C + (row + r*8)*N + (col + c*8), N);
}
)";

struct Kern { std::string name; std::string src; int bM; int bN; id<MTLComputePipelineState> pso; };
struct Shape { int M, K, N; const char* name; };

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        std::printf("device: %s   (peak ~2.6 TFLOP/s fp32)\n", dev.name.UTF8String);
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        auto compile = [&](const std::string& src) -> id<MTLComputePipelineState> {
            NSError* e = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:@(src.c_str()) options:nil error:&e];
            if (!lib) { std::printf("compile: %s\n", e.localizedDescription.UTF8String); return nil; }
            id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm"] error:&e];
            if (!p) std::printf("pipeline: %s\n", e.localizedDescription.UTF8String);
            return p;
        };
        std::vector<Kern> kerns;
        for (int bk : {8, 16, 32}) kerns.push_back({"f4_bk" + std::to_string(bk),
            "#define BK " + std::to_string(bk) + "\n" + kF4Body, 32, 32, nil});
        kerns.push_back({"direct", kDirect, 32, 32, nil});
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
                std::printf("  %-10s %s  %7.1f GFLOP/s  (%4.1f%% peak)\n", k.name.c_str(), rel<1e-3?"OK":"XX", gf, 100*gf/2600.0);
            }
        }
    }
    return 0;
}

// db_dev.mm — kernel-design dev harness for the multi-simdgroup MMA matmul. Head-to-head, all
// validated bit-exact vs a parallel CPU ref, best-of-N, on the shapes the model runs. Chasing MLX
// parity (~55-74% of M1 peak) on the laggards (mlp-up, square-1024).
//
// Findings log (Apple M1):
//  ✅ float4 loads (+50%), bigger block 64x64/8sg/FM2xFN4, bank-conflict padding PAD=4 (+12%) -> champion
//     64x64/8sg/bk16/2x4f/p4: mlp-down 99% of MLX, square-2048 90%, but mlp-up/square-1024 ~70%.
//  ❌ threadgroup double-buffer, no-staging direct loads, acc[4][4] spill, wide-N blocks -- occupancy is king.
//  THIS round: REGISTER PREFETCH. Single threadgroup buffer (occupancy intact) but prefetch the next
//  k-tile from global into private registers DURING compute -> hides global latency at ~8 regs/thread
//  instead of 2x threadgroup mem. The double-buffer benefit without the occupancy cost.
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

// champion: 64x64 block, 8 simdgroups (4x2), BK16, FM2xFN4 frags, PAD4, float4 loads, single-buffer.
static const char* kChamp = R"(
#include <metal_stdlib>
using namespace metal;
#define BK 16
#define PAD 4
#define AW (BK+PAD)
#define BW (64+PAD)
kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]],
               uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[64*AW]; threadgroup float Bs[BK*BW];
    uint blockRow=bid.y*64, blockCol=bid.x*64, sgY=sg/2, sgX=sg%2, rowBase=sgY*16, colBase=sgX*32;
    simdgroup_float8x8 acc[2][4];
    for (uint r=0;r<2;++r) for (uint c=0;c<4;++c) acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint k0=0;k0<K;k0+=BK){
        for (uint t=tid;t<(64*BK)/4;t+=256){ uint lin=t*4,r=lin/BK,c4=lin%BK;
            *(threadgroup float4*)(As+r*AW+c4)=*(device const float4*)(A+(blockRow+r)*K+(k0+c4)); }
        for (uint t=tid;t<(BK*64)/4;t+=256){ uint lin=t*4,r=lin/64,c4=lin%64;
            *(threadgroup float4*)(Bs+r*BW+c4)=*(device const float4*)(B+(k0+r)*N+(blockCol+c4)); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk=0;kk<BK;kk+=8){
            simdgroup_float8x8 af[2], bf[4];
            for (uint r=0;r<2;++r) simdgroup_load(af[r], As+(rowBase+r*8)*AW+kk, AW);
            for (uint c=0;c<4;++c) simdgroup_load(bf[c], Bs+kk*BW+(colBase+c*8), BW);
            for (uint r=0;r<2;++r) for (uint c=0;c<4;++c) simdgroup_multiply_accumulate(acc[r][c],af[r],bf[c],acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r=0;r<2;++r) for (uint c=0;c<4;++c)
        simdgroup_store(acc[r][c], C+(blockRow+rowBase+r*8)*N+(blockCol+colBase+c*8), N);
}
)";

// mlx16: MLX's fp32 GEMM shape -- 64x64 block, only 4 simdgroups (128 threads), 2x2 grid, each
// simdgroup owns a 32x32 = acc[4][4] = 16 fragments. The Volkov bet: fewer threads, more per-thread
// ILP. WITH padding this time (earlier unpadded acc[4][4] spilled to 2.4%). float4 loads, single-buffer.
static const char* kMlx16 = R"(
#include <metal_stdlib>
using namespace metal;
#define BK 16
#define PAD 4
#define AW (BK+PAD)
#define BW (64+PAD)
kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]],
               uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[64*AW]; threadgroup float Bs[BK*BW];
    uint blockRow=bid.y*64, blockCol=bid.x*64, sgY=sg/2, sgX=sg%2, rowBase=sgY*32, colBase=sgX*32;
    simdgroup_float8x8 acc[4][4];
    for (uint r=0;r<4;++r) for (uint c=0;c<4;++c) acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint k0=0;k0<K;k0+=BK){
        for (uint t=tid;t<(64*BK)/4;t+=128){ uint lin=t*4,r=lin/BK,c4=lin%BK;
            *(threadgroup float4*)(As+r*AW+c4)=*(device const float4*)(A+(blockRow+r)*K+(k0+c4)); }
        for (uint t=tid;t<(BK*64)/4;t+=128){ uint lin=t*4,r=lin/64,c4=lin%64;
            *(threadgroup float4*)(Bs+r*BW+c4)=*(device const float4*)(B+(k0+r)*N+(blockCol+c4)); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk=0;kk<BK;kk+=8){
            simdgroup_float8x8 af[4], bf[4];
            for (uint r=0;r<4;++r) simdgroup_load(af[r], As+(rowBase+r*8)*AW+kk, AW);
            for (uint c=0;c<4;++c) simdgroup_load(bf[c], Bs+kk*BW+(colBase+c*8), BW);
            for (uint r=0;r<4;++r) for (uint c=0;c<4;++c) simdgroup_multiply_accumulate(acc[r][c],af[r],bf[c],acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r=0;r<4;++r) for (uint c=0;c<4;++c)
        simdgroup_store(acc[r][c], C+(blockRow+rowBase+r*8)*N+(blockCol+colBase+c*8), N);
}
)";

struct Kern { const char* name; const char* src; id<MTLComputePipelineState> pso; };
struct Shape { int M, K, N; const char* name; double mlx; };

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        std::printf("device: %s\n", dev.name.UTF8String);
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        auto compile = [&](const char* src) -> id<MTLComputePipelineState> {
            NSError* e=nil; id<MTLLibrary> lib=[dev newLibraryWithSource:@(src) options:nil error:&e];
            if (!lib){ std::printf("compile: %s\n", e.localizedDescription.UTF8String); return nil; }
            return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm"] error:&e];
        };
        Kern kerns[]={{"champ",kChamp,nil},{"mlx16",kMlx16,nil}};
        for (auto& k:kerns){ k.pso=compile(k.src); if(!k.pso) return 1; }
        Shape shapes[]={   // small shapes first; mlp-up last (if mlx16 spills it's slow there)
            {1024,1024,1024,"square-1024",1678}, {8192,1536,384,"mlp-down",1414},
            {2048,2048,2048,"square-2048",1489}, {8192,384,1536,"mlp-up",1916},
        };
        for (auto s:shapes){
            int M=s.M,K=s.K,N=s.N;
            std::mt19937 rng(0); std::normal_distribution<float> dist(0,1);
            std::vector<float> A(M*K),B(K*N),Cref(M*N);
            for(auto&x:A)x=dist(rng); for(auto&x:B)x=dist(rng);
            { unsigned nt=std::max(1u,std::thread::hardware_concurrency()); std::vector<std::thread> pool; int chunk=(M+nt-1)/nt;
              for(unsigned t=0;t<nt;++t){ int r0=t*chunk,r1=std::min(M,r0+chunk); if(r0>=r1)break;
                pool.emplace_back([&,r0,r1]{ for(int i=r0;i<r1;++i) for(int j=0;j<N;++j){ float a=0; for(int k=0;k<K;++k) a+=A[i*K+k]*B[k*N+j]; Cref[i*N+j]=a; } }); }
              for(auto&th:pool) th.join(); }
            double maxref=0; for(float v:Cref) maxref=std::fmax(maxref,std::fabs(v));
            id<MTLBuffer> bA=[dev newBufferWithBytes:A.data() length:A.size()*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bB=[dev newBufferWithBytes:B.data() length:B.size()*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bC=[dev newBufferWithLength:M*N*4 options:MTLResourceStorageModeShared];
            uint Mu=M,Ku=K,Nu=N; MTLSize grid=MTLSizeMake(N/64,M/64,1), tg=MTLSizeMake(256,1,1);
            std::printf("\n%-12s %dx%dx%d  (MLX %.0f = %.1f%% peak)\n", s.name,M,K,N,s.mlx,100*s.mlx/2600.0);
            for (auto& k:kerns){
                auto run=[&]{ id<MTLCommandBuffer> cb=[queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:k.pso];
                    [enc setBuffer:bA offset:0 atIndex:0];[enc setBuffer:bB offset:0 atIndex:1];[enc setBuffer:bC offset:0 atIndex:2];
                    [enc setBytes:&Mu length:4 atIndex:3];[enc setBytes:&Ku length:4 atIndex:4];[enc setBytes:&Nu length:4 atIndex:5];
                    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];[enc endEncoding];[cb commit];[cb waitUntilCompleted]; };
                run();
                const float* Cg=static_cast<const float*>([bC contents]);
                double maxerr=0; for(int i=0;i<M*N;++i) maxerr=std::fmax(maxerr,std::fabs(Cg[i]-Cref[i]));
                double rel=maxerr/maxref;
                const int reps=3,iters=10; double best=1e30;   // reduced: a spilling kernel can't hang
                for(int r=0;r<reps;++r){ auto t0=std::chrono::high_resolution_clock::now();
                    for(int it=0;it<iters;++it) run();
                    best=std::fmin(best,std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count()); }
                double gf=2.0*M*K*N*iters/best/1e9;
                std::printf("  %-10s %s %7.1f GFLOP/s (%4.1f%% peak, %3.0f%% of MLX)\n",
                    k.name, rel<1e-3?"OK":"XX", gf, 100*gf/2600.0, 100*gf/s.mlx);
            }
        }
    }
    return 0;
}

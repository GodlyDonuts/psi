// mma_autotune.mm — parametrized tiled-MMA matmul autotuner. Searches the real GEMM design space to
// match/beat MLX (~55-74% of M1 peak) on the shapes the model runs. One kernel, parametrized by
// (BM,BN,BK,SGY,SGX) injected as #defines; the host compiles every config, validates it bit-exact vs
// a parallel CPU ref, times best-of-N, and reports the winner per shape.
//
//   block          = BM x BN output tile per threadgroup (staged As[BM*BK], Bs[BK*BN], float4 loads)
//   simdgroup grid = SGY x SGX simdgroups (SGY*SGX*32 threads); each owns a (BM/SGY) x (BN/SGX) sub-tile
//   register tile  = FM x FN  8x8 fragments per simdgroup, FM=BM/SGY/8, FN=BN/SGX/8 (keep FM*FN<=8: spill)
//
// Lessons already banked (see db_dev.mm): float4 loads ~+50%; double-buffering & no-staging both LOSE
// (occupancy is king); acc[4][4]=16 frags spills. So we sweep block/simdgroup shape at FM*FN<=8.
//
// Build: clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 mma_autotune.mm \
//                -framework Metal -framework Foundation -o mma_autotune && ./mma_autotune

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

// Parametrized kernel. All tile params arrive as #defines. Requires K%BK==0, N%4==0, BK%4==0,
// M%BM==0, N%BN==0 (the autotuner only dispatches configs that divide the shape).
// AW = As row stride (BK+PAD), BW = Bs row stride (BN+PAD). PAD pads the staging arrays to push
// successive rows onto different threadgroup-memory banks (kills simdgroup_load bank conflicts).
static const char* kBody = R"(
#include <metal_stdlib>
using namespace metal;
#define AW (BK + PAD)
#define BW (BN + PAD)
kernel void mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]],
               uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[BM * AW];
    threadgroup float Bs[BK * BW];
    uint blockRow = bid.y * BM, blockCol = bid.x * BN;
    uint sgY = sg / SGX, sgX = sg % SGX;
    uint rowBase = sgY * (BM / SGY), colBase = sgX * (BN / SGX);
    simdgroup_float8x8 acc[FM][FN];
    for (uint r=0;r<FM;++r) for (uint c=0;c<FN;++c) acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint t=tid; t < (BM*BK)/4; t += NT) { uint lin=t*4, r=lin/BK, c4=lin%BK;   // As float4 (padded rows)
            *(threadgroup float4*)(As + r*AW + c4) = *(device const float4*)(A + (blockRow+r)*K + (k0+c4)); }
        for (uint t=tid; t < (BK*BN)/4; t += NT) { uint lin=t*4, r=lin/BN, c4=lin%BN;   // Bs float4 (padded rows)
            *(threadgroup float4*)(Bs + r*BW + c4) = *(device const float4*)(B + (k0+r)*N + (blockCol+c4)); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk=0; kk<BK; kk+=8) {
            simdgroup_float8x8 af[FM], bf[FN];
            for (uint r=0;r<FM;++r) simdgroup_load(af[r], As + (rowBase + r*8)*AW + kk, AW);
            for (uint c=0;c<FN;++c) simdgroup_load(bf[c], Bs + kk*BW + (colBase + c*8), BW);
            for (uint r=0;r<FM;++r) for (uint c=0;c<FN;++c) simdgroup_multiply_accumulate(acc[r][c],af[r],bf[c],acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r=0;r<FM;++r) for (uint c=0;c<FN;++c)
        simdgroup_store(acc[r][c], C + (blockRow + rowBase + r*8)*N + (blockCol + colBase + c*8), N);
}
)";

struct Cfg { int BM, BN, BK, SGY, SGX, PAD; };
struct Shape { int M, K, N; const char* name; double mlx; };  // mlx = measured MLX GFLOP/s baseline

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        std::printf("device: %s   (peak ~2.6 TFLOP/s fp32)\n", dev.name.UTF8String);
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        auto compile = [&](const std::string& src) -> id<MTLComputePipelineState> {
            NSError* e = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:@(src.c_str()) options:nil error:&e];
            if (!lib) return nil;
            return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm"] error:&e];
        };

        // Candidate configs (FM*FN<=8 to avoid register spill). The autotuner skips any that don't
        // divide a given shape or that blow threadgroup memory.
        // Top performers from the unpadded sweep, now x bank-conflict padding PAD in {0,4,8}.
        std::vector<Cfg> cfgs;
        for (int p : {0, 4, 8}) {
            cfgs.push_back({64,64,16,2,4,p});   // champ: 8 sg, 4x2 frags
            cfgs.push_back({32,32,16,2,2,p});   // 4 sg, 2x2 frags (best on attn-proj)
            cfgs.push_back({128,32,16,4,2,p});  // 8 sg, 4x2 frags
            cfgs.push_back({64,64,16,4,2,p});   // 8 sg, 2x4 frags
        }
        Shape shapes[] = {
            {1024,1024,1024,"square-1024", 1678},
            {2048,2048,2048,"square-2048", 1489},
            {8192,384,1536,"mlp-up",       1916},
            {8192,1536,384,"mlp-down",     1414},
            {8192,384,384, "attn-proj",    1435},
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
            std::printf("\n=== %s %dx%dx%d  (MLX %.0f GFLOP/s = %.1f%% peak) ===\n",
                        s.name, M, K, N, s.mlx, 100*s.mlx/2600.0);
            double bestGf=0; std::string bestLabel;
            for (auto& c : cfgs) {
                int FM=c.BM/c.SGY/8, FN=c.BN/c.SGX/8, NSG=c.SGY*c.SGX, NT=NSG*32;
                long tgmem=(long)(c.BM*(c.BK+c.PAD) + c.BK*(c.BN+c.PAD))*4;
                char label[56]; std::snprintf(label,sizeof(label),"%dx%d/%dsg/bk%d/%dx%df/p%d",c.BM,c.BN,NSG,c.BK,FM,FN,c.PAD);
                // validity for this shape
                if (M%c.BM||N%c.BN||K%c.BK||c.BM%(c.SGY*8)||c.BN%(c.SGX*8)||(c.BM*c.BK)%4||(c.BK*c.BN)%4||c.PAD%4) continue;
                if (FM*FN>8 || tgmem>32768 || NT>1024) continue;
                std::string def="#define BM "+std::to_string(c.BM)+"\n#define BN "+std::to_string(c.BN)+
                    "\n#define BK "+std::to_string(c.BK)+"\n#define SGY "+std::to_string(c.SGY)+
                    "\n#define SGX "+std::to_string(c.SGX)+"\n#define FM "+std::to_string(FM)+
                    "\n#define FN "+std::to_string(FN)+"\n#define NT "+std::to_string(NT)+
                    "\n#define PAD "+std::to_string(c.PAD)+"\n";
                id<MTLComputePipelineState> pso=compile(def+kBody);
                if (!pso) { std::printf("  %-22s compile/pipeline failed\n", label); continue; }
                MTLSize grid=MTLSizeMake(N/c.BN, M/c.BM, 1), tg=MTLSizeMake(NT,1,1);
                auto run=[&]{ id<MTLCommandBuffer> cb=[queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                    [enc setComputePipelineState:pso];
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
                std::printf("  %-22s %s %7.1f GFLOP/s (%4.1f%% peak, %3.0f%% of MLX)\n",
                            label, rel<1e-3?"OK":"XX", gf, 100*gf/2600.0, 100*gf/s.mlx);
                if (rel<1e-3 && gf>bestGf) { bestGf=gf; bestLabel=label; }
            }
            std::printf("  >>> BEST %s @ %.1f GFLOP/s (%.1f%% peak, %.0f%% of MLX)\n",
                        bestLabel.c_str(), bestGf, 100*bestGf/2600.0, 100*bestGf/s.mlx);
        }
    }
    return 0;
}

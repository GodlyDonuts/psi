// fused_ternary.mm — DID activation fusion (matmul-epilogue act) pay off on M1? FINDING: NO, both ways.
// Computes C = act(A @ W_ternary). We tried to fuse the activation into the matmul's epilogue two ways,
// measured vs the UNFUSED two-pass (ternary matmul -> global, then a standalone elementwise act):
//   (a) threadgroup scratch [64x64] then act+write  -> 0.80-0.84x (the 16KB scratch crushes the matmul's
//       occupancy; the matmul is the expensive part, so we lose more there than we save on the act pass).
//   (b) act in-register via acc.thread_elements()    -> 0.17-0.21x (CATASTROPHIC). Confirmed NOT the
//       transcendental: ReLU is equally slow. thread_elements() forces acc out of the special MMA
//       registers, spilling it for the WHOLE matmul so every simdgroup_multiply_accumulate goes slow.
// LESSON (occupancy is king, again): on Apple's simdgroup-MMA path, epilogue fusion doesn't pay -- the
// accumulator lives in matrix registers that resist element access, and any scratch big enough to hold
// the output tanks occupancy. Standalone high-occupancy activation passes are the efficient design; MLX's
// separate dispatches are near-optimal here. (On GH200 the tradeoff differs.) All variants bit-exact (PASS).
// Set ACT (0 ReLU / 1 GELU) and the FUSE path below to reproduce.
//
// Build: clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 fused_ternary.mm \
//                -framework Metal -framework Foundation -o fused_ternary && ./fused_ternary

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

// Shared: ternary champion tiling + GELU. FUSE toggles whether GELU is applied before the store.
static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
#define BK 16
#define PAD 4
#define AW (BK+PAD)
#define BW (64+PAD)
// ACT=0 ReLU (cheap), ACT=1 GELU (transcendental). Used to isolate the cost of fusing an expensive act.
inline float act(float x){
#if ACT
    return 0.5f*x*(1.0f+precise::tanh(0.7978845608f*(x+0.044715f*x*x*x)));
#else
    return fmax(0.0f, x);
#endif
}

// ternary A@W, optionally GELU-fused. FUSE=1 -> store acc to threadgroup scratch, GELU, write once.
kernel void mm(device const float* A  [[buffer(0)]], device const uint* Wp [[buffer(1)]],
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
        { uint tileIdx=tid*4, r=tileIdx>>6, c=tileIdx&63; uint f=(k0+r)*N+blockCol+c;
          uint w=Wp[f>>4]; uint sh=(f&15)*2;
          for (uint i=0;i<4;++i){ uint tr=(w>>(sh+2*i))&3u; Bs[r*BW+c+i]=(tr==1u)?1.0f:(tr==2u)?-1.0f:0.0f; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk=0;kk<BK;kk+=8){
            simdgroup_float8x8 af[2], bf[4];
            for (uint r=0;r<2;++r) simdgroup_load(af[r], As+(rowBase+r*8)*AW+kk, AW);
            for (uint c=0;c<4;++c) simdgroup_load(bf[c], Bs+kk*BW+(colBase+c*8), BW);
            for (uint r=0;r<2;++r) for (uint c=0;c<4;++c) simdgroup_multiply_accumulate(acc[r][c],af[r],bf[c],acc[r][c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#if FUSE
    // GELU in-register on the 2 elements each thread owns per fragment -- ZERO extra threadgroup memory,
    // so the matmul's occupancy is untouched and the fused GELU is genuinely free (no intermediate roundtrip).
    for (uint r=0;r<2;++r) for (uint c=0;c<4;++c){
        thread auto& te = acc[r][c].thread_elements();
        te[0]=act(te[0]); te[1]=act(te[1]);
    }
#endif
    for (uint r=0;r<2;++r) for (uint c=0;c<4;++c)
        simdgroup_store(acc[r][c], C+(blockRow+rowBase+r*8)*N+(blockCol+colBase+c*8), N);
}

// elementwise GELU pass (the unfused second dispatch).
kernel void gelu_pass(device float* C [[buffer(0)]], constant uint& n [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) { if (i<n) C[i]=act(C[i]); }
)";

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        std::printf("device: %s\n", dev.name.UTF8String);
        id<MTLCommandQueue> queue=[dev newCommandQueue];
        const int ACT = 0;   // 0 = ReLU (cheap), 1 = GELU (transcendental)
        std::string hdr = "#define ACT "+std::to_string(ACT)+"\n";
        std::printf("activation: %s\n", ACT ? "GELU (transcendental)" : "ReLU (cheap)");
        auto build=[&](int fuse)->id<MTLComputePipelineState>{
            NSError* e=nil; std::string src=hdr+"#define FUSE "+std::to_string(fuse)+"\n"+kSrc;
            id<MTLLibrary> lib=[dev newLibraryWithSource:@(src.c_str()) options:nil error:&e];
            if(!lib){ std::printf("compile: %s\n", e.localizedDescription.UTF8String); return nil; }
            return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm"] error:&e];
        };
        id<MTLComputePipelineState> psoFused=build(1), psoRaw=build(0);
        NSError* e=nil; id<MTLLibrary> lib=[dev newLibraryWithSource:@((hdr+"#define FUSE 0\n"+std::string(kSrc)).c_str()) options:nil error:&e];
        id<MTLComputePipelineState> psoGelu=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gelu_pass"] error:&e];
        if(!psoFused||!psoRaw||!psoGelu){ std::printf("pipeline build failed\n"); return 1; }

        struct Shape{int M,K,N;const char*name;};
        Shape shapes[]={ {8192,384,1536,"mlp-up"}, {8192,1536,384,"mlp-down"}, {2048,2048,2048,"square-2048"} };
        auto gelu=[&](float x){ return ACT ? 0.5f*x*(1.0f+std::tanh(0.7978845608f*(x+0.044715f*x*x*x))) : std::fmax(0.0f,x); };
        std::printf("\n%-12s %-16s %12s %12s   %s\n","shape","MxKxN","unfused ms","fused ms","speedup / valid");
        for (auto s:shapes){
            int M=s.M,K=s.K,N=s.N;
            std::mt19937 rng(0); std::normal_distribution<float> da(0,1); std::uniform_int_distribution<int> dw(0,2);
            std::vector<float> A(M*K), Cref(M*N); std::vector<int8_t> W(K*N); std::vector<uint32_t> Wp((K*N+15)/16,0);
            for (auto&x:A) x=da(rng);
            for (int f=0; f<K*N; ++f){ int t=dw(rng); W[f]=(t==1)?1:(t==2)?-1:0; Wp[f>>4]|=((uint32_t)t)<<(2*(f&15)); }
            { unsigned nt=std::max(1u,std::thread::hardware_concurrency()); std::vector<std::thread> pool; int chunk=(M+nt-1)/nt;
              for (unsigned tt=0;tt<nt;++tt){ int r0=tt*chunk,r1=std::min(M,r0+chunk); if(r0>=r1)break;
                pool.emplace_back([&,r0,r1]{ for(int i=r0;i<r1;++i) for(int j=0;j<N;++j){ float a=0;
                    for(int k=0;k<K;++k){ int8_t w=W[k*N+j]; if(w) a+=w>0?A[i*K+k]:-A[i*K+k]; } Cref[i*N+j]=gelu(a); } }); }
              for (auto& th:pool) th.join(); }
            double maxref=0; for(float v:Cref) maxref=std::fmax(maxref,std::fabs(v));
            id<MTLBuffer> bA=[dev newBufferWithBytes:A.data() length:A.size()*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bW=[dev newBufferWithBytes:Wp.data() length:Wp.size()*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bC=[dev newBufferWithLength:M*N*4 options:MTLResourceStorageModeShared];
            uint Mu=M,Ku=K,Nu=N,n=M*N; MTLSize grid=MTLSizeMake(N/64,M/64,1), tg=MTLSizeMake(256,1,1);
            auto encMM=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso){
                id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bA offset:0 atIndex:0];[enc setBuffer:bW offset:0 atIndex:1];[enc setBuffer:bC offset:0 atIndex:2];
                [enc setBytes:&Mu length:4 atIndex:3];[enc setBytes:&Ku length:4 atIndex:4];[enc setBytes:&Nu length:4 atIndex:5];
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];[enc endEncoding]; };
            auto fused=[&]{ id<MTLCommandBuffer> cb=[queue commandBuffer]; encMM(cb,psoFused); [cb commit];[cb waitUntilCompleted]; };
            auto unfused=[&]{ id<MTLCommandBuffer> cb=[queue commandBuffer]; encMM(cb,psoRaw);
                id<MTLComputeCommandEncoder> g=[cb computeCommandEncoder]; [g setComputePipelineState:psoGelu];
                [g setBuffer:bC offset:0 atIndex:0]; [g setBytes:&n length:4 atIndex:1];
                [g dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [g endEncoding];
                [cb commit];[cb waitUntilCompleted]; };
            // validate fused
            fused(); const float* Cg=static_cast<const float*>([bC contents]);
            double maxerr=0; for(int i=0;i<M*N;++i) maxerr=std::fmax(maxerr,std::fabs(Cg[i]-Cref[i]));
            double rel=maxerr/maxref;
            auto timeit=[&](void(^fn)()){ const int reps=5,iters=20; double best=1e30;
                for(int r=0;r<reps;++r){ auto t0=std::chrono::high_resolution_clock::now();
                    for(int it=0;it<iters;++it) fn();
                    best=std::fmin(best,std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count()); }
                return best/iters*1e3; };
            double msU=timeit(^{unfused();}), msF=timeit(^{fused();});
            std::printf("%-12s %5dx%5dx%-5d %12.3f %12.3f   %.2fx  %s\n",
                s.name,M,K,N,msU,msF,msU/msF, rel<1e-3?"PASS":"FAIL");
        }
        std::printf("\n(fused = GELU(A@W) in one dispatch; unfused = matmul then separate GELU read+write.)\n");
    }
    return 0;
}

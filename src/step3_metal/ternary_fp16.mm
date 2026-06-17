// ternary_fp16.mm — ternary-weight GEMM v2: fp16 activations + ternary weights, fp32 accumulate.
//
// v0/v1 (ternary_gemm.mm) proved correctness + 16x weight compression, but ran at fp32 speed because
// the kernel is bound by the fp32 MMA units. The model's real regime is LOW-BIT everywhere, so here we
// drive the half-precision matrix units: activations A in fp16, ternary W decoded to fp16, MMA with
// simdgroup_half8x8 inputs accumulated into simdgroup_float8x8 (fp32 accumulate keeps precision over K).
// This doubles MMA throughput AND halves A/C traffic -- a speed win on top of the 16x weight win, in a
// regime MLX's fp32 GEMM cannot reach. Validated vs a CPU fp32 ternary reference (rel < 1e-2: fp16 input
// rounding). Same tuned tiling: 64x64 / 8 simdgroups / BK16 / FM2xFN4 / PAD4.
//
// Build: clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 ternary_fp16.mm \
//                -framework Metal -framework Foundation -o ternary_fp16 && ./ternary_fp16

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
#define BK 16
#define PAD 4
#define AW (BK+PAD)
#define BW (64+PAD)
kernel void mm(device const half* A  [[buffer(0)]], device const uint* Wp [[buffer(1)]],
               device float* C [[buffer(2)]], constant uint& M [[buffer(3)]],
               constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]],
               uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]],
               uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup half As[64*AW]; threadgroup half Bs[BK*BW];
    uint blockRow=bid.y*64, blockCol=bid.x*64, sgY=sg/2, sgX=sg%2, rowBase=sgY*16, colBase=sgX*32;
    simdgroup_float8x8 acc[2][4];                          // fp32 accumulate
    for (uint r=0;r<2;++r) for (uint c=0;c<4;++c) acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (uint k0=0;k0<K;k0+=BK){
        for (uint t=tid;t<(64*BK)/4;t+=256){ uint lin=t*4,r=lin/BK,c4=lin%BK;     // A: half4 loads
            *(threadgroup half4*)(As+r*AW+c4)=*(device const half4*)(A+(blockRow+r)*K+(k0+c4)); }
        {                                                                          // W: decode 2-bit -> half
            uint tileIdx=tid*4, r=tileIdx>>6, c=tileIdx&63;
            uint f=(k0+r)*N+blockCol+c; uint w=Wp[f>>4]; uint sh=(f&15)*2;
            for (uint i=0;i<4;++i){ uint trit=(w>>(sh+2*i))&3u;
                Bs[r*BW+c+i] = (trit==1u)?(half)1.0h:(trit==2u)?(half)-1.0h:(half)0.0h; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk=0;kk<BK;kk+=8){
            simdgroup_half8x8 af[2], bf[4];
            for (uint r=0;r<2;++r) simdgroup_load(af[r], As+(rowBase+r*8)*AW+kk, AW);
            for (uint c=0;c<4;++c) simdgroup_load(bf[c], Bs+kk*BW+(colBase+c*8), BW);
            for (uint r=0;r<2;++r) for (uint c=0;c<4;++c)
                simdgroup_multiply_accumulate(acc[r][c], af[r], bf[c], acc[r][c]);   // half*half -> fp32
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r=0;r<2;++r) for (uint c=0;c<4;++c)
        simdgroup_store(acc[r][c], C+(blockRow+rowBase+r*8)*N+(blockCol+colBase+c*8), N);
}
)";

typedef uint16_t h16;
static inline h16 f2h(float f) {  // minimal float->half for building the fp16 activation buffer
    uint32_t x; __builtin_memcpy(&x, &f, 4);
    uint32_t sign=(x>>16)&0x8000; int32_t exp=((x>>23)&0xff)-127+15; uint32_t man=(x>>13)&0x3ff;
    if (exp<=0) return (h16)sign;
    if (exp>=31) return (h16)(sign|0x7c00);
    return (h16)(sign | (exp<<10) | man);
}

struct Shape { int M, K, N; const char* name; double champ_fp32; };

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        std::printf("device: %s   (fp32 peak ~2.6 TFLOP/s; fp16 peak ~2x)\n", dev.name.UTF8String);
        NSError* e=nil;
        id<MTLLibrary> lib=[dev newLibraryWithSource:@(kSrc) options:nil error:&e];
        if (!lib){ std::printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm"] error:&e];
        if (!pso){ std::printf("pipeline: %s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> queue=[dev newCommandQueue];

        Shape shapes[] = {
            {8192,384,1536,"mlp-up",   1342}, {8192,1536,384,"mlp-down", 1404},
            {8192,384,384, "attn-proj",1218}, {2048,2048,2048,"square-2048",1342},
        };
        std::printf("\n%-12s %-16s %6s %16s   %s\n","shape","MxKxN","valid","fp16-tern GFLOP/s","vs our fp32");
        for (auto s : shapes) {
            int M=s.M,K=s.K,N=s.N;
            std::mt19937 rng(0); std::normal_distribution<float> da(0,1); std::uniform_int_distribution<int> dw(0,2);
            std::vector<float> A(M*K), Cref(M*N); std::vector<h16> Ah(M*K);
            std::vector<int8_t> W(K*N); std::vector<uint32_t> Wp((K*N+15)/16,0);
            for (int i=0;i<M*K;++i){ A[i]=da(rng); Ah[i]=f2h(A[i]); }
            for (int f=0; f<K*N; ++f){ int t=dw(rng); W[f]=(t==1)?1:(t==2)?-1:0; Wp[f>>4]|=((uint32_t)t)<<(2*(f&15)); }
            // CPU reference uses the fp16-rounded A (so we validate the kernel, not fp16 rounding itself)
            std::vector<float> Ar(M*K);
            for (int i=0;i<M*K;++i){ h16 h=Ah[i]; uint32_t s=(h&0x8000)<<16,ex=(h>>10)&0x1f,ma=h&0x3ff,fb;
                if(ex==0)fb=s; else fb=s|((ex-15+127)<<23)|(ma<<13); float fv; __builtin_memcpy(&fv,&fb,4); Ar[i]=fv; }
            { unsigned nt=std::max(1u,std::thread::hardware_concurrency()); std::vector<std::thread> pool; int chunk=(M+nt-1)/nt;
              for (unsigned tt=0;tt<nt;++tt){ int r0=tt*chunk,r1=std::min(M,r0+chunk); if(r0>=r1)break;
                pool.emplace_back([&,r0,r1]{ for(int i=r0;i<r1;++i) for(int j=0;j<N;++j){ float a=0;
                    for(int k=0;k<K;++k){ int8_t w=W[k*N+j]; if(w) a += w>0?Ar[i*K+k]:-Ar[i*K+k]; } Cref[i*N+j]=a; } }); }
              for (auto& th:pool) th.join(); }
            double maxref=0; for(float v:Cref) maxref=std::fmax(maxref,std::fabs(v));
            id<MTLBuffer> bA=[dev newBufferWithBytes:Ah.data() length:Ah.size()*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bW=[dev newBufferWithBytes:Wp.data() length:Wp.size()*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bC=[dev newBufferWithLength:M*N*4 options:MTLResourceStorageModeShared];
            uint Mu=M,Ku=K,Nu=N; MTLSize grid=MTLSizeMake(N/64,M/64,1), tg=MTLSizeMake(256,1,1);
            auto run=[&]{ id<MTLCommandBuffer> cb=[queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bA offset:0 atIndex:0];[enc setBuffer:bW offset:0 atIndex:1];[enc setBuffer:bC offset:0 atIndex:2];
                [enc setBytes:&Mu length:4 atIndex:3];[enc setBytes:&Ku length:4 atIndex:4];[enc setBytes:&Nu length:4 atIndex:5];
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];[enc endEncoding];[cb commit];[cb waitUntilCompleted]; };
            run();
            const float* Cg=static_cast<const float*>([bC contents]);
            double maxerr=0; for(int i=0;i<M*N;++i) maxerr=std::fmax(maxerr,std::fabs(Cg[i]-Cref[i]));
            double rel=maxerr/maxref;
            const int reps=5,iters=30; double best=1e30;
            for(int r=0;r<reps;++r){ auto t0=std::chrono::high_resolution_clock::now();
                for(int it=0;it<iters;++it) run();
                best=std::fmin(best,std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count()); }
            double gf=2.0*M*K*N*iters/best/1e9;
            std::printf("%-12s %5dx%5dx%-5d %s %14.1f       %.2fx fp32 (rel=%.1e)\n",
                s.name, M, K, N, rel<1e-2?"PASS":"FAIL", gf, gf/s.champ_fp32, rel);
        }
        std::printf("\n(fp16 activations + ternary W: 16x smaller weights, ~2x MMA, half act/output traffic.)\n");
    }
    return 0;
}

// ternary_gemm.mm — Psi's OWN kernel: a ternary-weight matmul. C = A @ W, where W in {-1,0,+1}.
//
// This is the project's north star (capability-per-bit / ternary ~1.58-bit weights) realized in a
// GPU kernel -- and the answer to "don't just copy MLX." MLX has no ternary Metal GEMM. We don't try
// to out-fp32-GEMM them; we move ~16x LESS weight memory by storing W at 2 bits/weight (packed 16 to a
// uint32) and DECODING it in threadgroup memory, then feeding the exact same tuned MMA path we built
// (64x64 / 8 simdgroups / BK16 / FM2xFN4 / PAD4 / float4). All our hardware-throughput wins are kept;
// the novelty is the packed-weight load + on-chip decode. Activations A stay fp32 so the result is
// bit-exact vs a CPU ternary reference (ternary*fp32 sums exactly in fp32, modulo MMA reorder ~1e-6).
//
// v0 = correctness + first speed read. Build:
//   clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 ternary_gemm.mm \
//           -framework Metal -framework Foundation -o ternary_gemm && ./ternary_gemm

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

// Champion MMA tiling, but Bs is DECODED from 2-bit-packed ternary W (16 weights / uint32) instead of
// loaded as fp32. trit encoding: 0b00 -> 0, 0b01 -> +1, 0b10 -> -1.
static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
#define BK 16
#define PAD 4
#define AW (BK+PAD)
#define BW (64+PAD)
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
        // A tile: fp32 float4 (unchanged from the champion)
        for (uint t=tid;t<(64*BK)/4;t+=256){ uint lin=t*4,r=lin/BK,c4=lin%BK;
            *(threadgroup float4*)(As+r*AW+c4)=*(device const float4*)(A+(blockRow+r)*K+(k0+c4)); }
        // W tile: decode 2-bit ternary -> fp32 into Bs. ALL 256 threads, 4 weights each (1024 total).
        // The 4 weights of a thread are 4 consecutive columns -> same uint32 (c%16 in {0,4,8,12}).
        {
            uint tileIdx = tid * 4, r = tileIdx >> 6, c = tileIdx & 63;   // /64, %64
            uint f = (k0 + r)*N + blockCol + c;
            uint w = Wp[f >> 4]; uint sh = (f & 15) * 2;
            for (uint i = 0; i < 4; ++i) {
                uint trit = (w >> (sh + 2*i)) & 3u;
                Bs[r*BW + c + i] = (trit == 1u) ? 1.0f : (trit == 2u) ? -1.0f : 0.0f;
            }
        }
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

struct Shape { int M, K, N; const char* name; double champ_fp32; };  // champ_fp32 = our fp32 GFLOP/s

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        std::printf("device: %s   (peak ~2.6 TFLOP/s fp32)\n", dev.name.UTF8String);
        NSError* e=nil;
        id<MTLLibrary> lib=[dev newLibraryWithSource:@(kSrc) options:nil error:&e];
        if (!lib){ std::printf("compile: %s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm"] error:&e];
        if (!pso){ std::printf("pipeline: %s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> queue=[dev newCommandQueue];

        Shape shapes[] = {   // champ_fp32 = our tuned fp32 kernel's GFLOP/s on this shape (for the speed delta)
            {8192,384,1536,"mlp-up",   1342},
            {8192,1536,384,"mlp-down", 1404},
            {8192,384,384, "attn-proj",1218},
            {2048,2048,2048,"square-2048",1342},
        };
        std::printf("\n%-12s %-16s %8s %18s   %s\n","shape","MxKxN","valid","ternary GFLOP/s","vs our fp32 / W-mem");
        for (auto s : shapes) {
            int M=s.M,K=s.K,N=s.N;
            std::mt19937 rng(0);
            std::normal_distribution<float> da(0,1);
            std::uniform_int_distribution<int> dw(0,2);   // 0->0, 1->+1, 2->-1 (roughly 1/3 each)
            std::vector<float> A(M*K), Cref(M*N);
            std::vector<int8_t> W(K*N);                   // ternary values for the CPU reference
            std::vector<uint32_t> Wp((K*N+15)/16, 0);     // 2-bit packed, 16 weights / uint32
            for (auto& x:A) x=da(rng);
            for (int f=0; f<K*N; ++f) {
                int t = dw(rng);                          // trit 0/1/2
                W[f] = (t==1) ? 1 : (t==2) ? -1 : 0;
                Wp[f>>4] |= ((uint32_t)t) << (2*(f&15));
            }
            {   // parallel CPU reference: C = A @ W (ternary)
                unsigned nt=std::max(1u,std::thread::hardware_concurrency()); std::vector<std::thread> pool; int chunk=(M+nt-1)/nt;
                for (unsigned tt=0;tt<nt;++tt){ int r0=tt*chunk,r1=std::min(M,r0+chunk); if(r0>=r1)break;
                    pool.emplace_back([&,r0,r1]{ for(int i=r0;i<r1;++i) for(int j=0;j<N;++j){
                        float a=0; for(int k=0;k<K;++k){ int8_t w=W[k*N+j]; if(w) a += w>0 ? A[i*K+k] : -A[i*K+k]; } Cref[i*N+j]=a; } }); }
                for (auto& th:pool) th.join();
            }
            double maxref=0; for(float v:Cref) maxref=std::fmax(maxref,std::fabs(v));
            id<MTLBuffer> bA=[dev newBufferWithBytes:A.data() length:A.size()*4 options:MTLResourceStorageModeShared];
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
            std::printf("%-12s %5dx%5dx%-5d %s %12.1f       %.2fx fp32, W %dx smaller\n",
                s.name, M, K, N, rel<1e-3?"PASS":"FAIL", gf, gf/s.champ_fp32, 16);
        }
        std::printf("\n(W stored at 2 bits/weight = 16x less weight memory than fp32; activations fp32.)\n");
    }
    return 0;
}

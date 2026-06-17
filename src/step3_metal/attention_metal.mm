// attention_metal.mm — fused single-head causal attention on the GPU (one Metal dispatch).
//
// Step 3, kernel #2. Computes O = softmax(QKᵀ·scale + causal_mask) @ V without ever writing the
// T×T score matrix to global memory: each thread owns one query row and runs the FlashAttention
// online-softmax recurrence, keeping its running max / denominator / output accumulator in
// registers. That on-chip fusion is where the big Apple-Silicon wins live (per arXiv:2604.03585).
// Validated bit-close vs a CPU reference. (Speed is measured on a quiet machine; correctness here.)
//
// Build: clang++ -x objective-c++ -fobjc-arc -O2 -std=c++17 attention_metal.mm \
//                -framework Metal -framework Foundation -o attention_metal

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void flash_attn(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]],
                       device const float* V [[buffer(2)]], device float* O [[buffer(3)]],
                       constant uint& T [[buffer(4)]], constant uint& D [[buffer(5)]],
                       constant float& scale [[buffer(6)]],
                       uint i [[thread_position_in_grid]]) {       // one thread per query row i
    if (i >= T) return;
    float m = -1e30f, lsum = 0.0f, acc[128];
    for (uint c = 0; c < D; ++c) acc[c] = 0.0f;
    for (uint j = 0; j <= i; ++j) {                                // causal: attend to keys j <= i
        float s = 0.0f;
        for (uint c = 0; c < D; ++c) s += Q[i * D + c] * K[j * D + c];
        s *= scale;
        float mnew = max(m, s);                                    // online softmax update
        float corr = exp(m - mnew), p = exp(s - mnew);
        lsum = lsum * corr + p;
        for (uint c = 0; c < D; ++c) acc[c] = acc[c] * corr + p * V[j * D + c];
        m = mnew;
    }
    float inv = 1.0f / lsum;
    for (uint c = 0; c < D; ++c) O[i * D + c] = acc[c] * inv;
}
)";

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::printf("no Metal device\n"); return 1; }
        std::printf("device: %s\n", dev.name.UTF8String);
        NSError* e = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSrc) options:nil error:&e];
        if (!lib) { std::printf("shader compile failed: %s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"flash_attn"] error:&e];
        if (!pso) { std::printf("pipeline failed: %s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        const int T = 32, D = 64;                  // psi-nano's attention shape (single head)
        float scale = 1.0f / std::sqrt((float)D);
        std::mt19937 rng(0);
        std::normal_distribution<float> dist(0, 1);
        std::vector<float> Q(T * D), K(T * D), V(T * D), O(T * D), Oref(T * D);
        for (auto& x : Q) x = dist(rng);
        for (auto& x : K) x = dist(rng);
        for (auto& x : V) x = dist(rng);

        id<MTLBuffer> bQ = [dev newBufferWithBytes:Q.data() length:T*D*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bK = [dev newBufferWithBytes:K.data() length:T*D*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bV = [dev newBufferWithBytes:V.data() length:T*D*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bO = [dev newBufferWithLength:T*D*4 options:MTLResourceStorageModeShared];
        uint Tu = T, Du = D;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bQ offset:0 atIndex:0]; [enc setBuffer:bK offset:0 atIndex:1];
        [enc setBuffer:bV offset:0 atIndex:2]; [enc setBuffer:bO offset:0 atIndex:3];
        [enc setBytes:&Tu length:4 atIndex:4]; [enc setBytes:&Du length:4 atIndex:5]; [enc setBytes:&scale length:4 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(T, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        std::memcpy(O.data(), [bO contents], T * D * 4);

        // CPU reference: standard causal softmax attention.
        for (int i = 0; i < T; ++i) {
            std::vector<float> s(i + 1);
            float mx = -1e30f;
            for (int j = 0; j <= i; ++j) { float d = 0; for (int c = 0; c < D; ++c) d += Q[i*D+c]*K[j*D+c]; s[j] = d*scale; mx = std::fmax(mx, s[j]); }
            float sum = 0; for (int j = 0; j <= i; ++j) { s[j] = std::exp(s[j]-mx); sum += s[j]; }
            for (int c = 0; c < D; ++c) { float o = 0; for (int j = 0; j <= i; ++j) o += (s[j]/sum) * V[j*D+c]; Oref[i*D+c] = o; }
        }
        double maxerr = 0, maxref = 0;
        for (int i = 0; i < T*D; ++i) { maxerr = std::fmax(maxerr, std::fabs(O[i]-Oref[i])); maxref = std::fmax(maxref, std::fabs(Oref[i])); }
        double rel = maxerr / maxref;
        std::printf("fused attention T=%d D=%d: max|diff|=%.2e rel=%.2e (%s)\n",
                    T, D, maxerr, rel, rel < 1e-3 ? "PASS" : "FAIL");
    }
    return 0;
}

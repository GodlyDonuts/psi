// model_stories.hpp — ModernGPT: the psi-stories architecture, every capability-per-param technique
// from docs/RESEARCH.md stacked, all on grad-checked ops:
//   · multi-head attention with GROUPED-QUERY ATTENTION (GQA)   — fewer K/V params      [MobileLLM/DeepSeek]
//   · ROTARY POSITION EMBEDDING (RoPE)                          — no pos-emb params      [RoFormer]
//   · SwiGLU MLP                                                — better capability/param[GLU variants/MobileLLM]
//   · BLOCK-WISE WEIGHT SHARING                                 — depth at ~0 param cost [MobileLLM]
//   · RMSNorm (pre-norm) + tied input/output embeddings
// Kept separate from model.hpp's GPT so psi-nano (the char demo) is unchanged. Reuses Config/AdamW.

#pragma once

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "model.hpp"   // Config, AdamW, GPT::causal_mask, generate(GPT&,…)
#include "nn.hpp"

namespace psi {

// Config fields used: vocab, d_model, n_layers, block, hidden, n_heads, n_kv_heads, n_unique.
//   n_kv_heads (0 ⇒ = n_heads, i.e. plain MHA);  n_unique (0 ⇒ = n_layers, i.e. no sharing).
struct SBlock {
    Tensor wq, wk, wv, wo;     // wq/wo: [d,d];  wk/wv: [d, n_kv*dh]  (GQA — smaller)
    Tensor attn_g, mlp_g;      // [d] RMSNorm gains
    Tensor w1, w3, w2;         // SwiGLU: w1,w3 [d,hidden]; w2 [hidden,d]
};

struct ModernGPT {
    Config cfg;
    int n_kv, n_uniq, dh, qpg;     // resolved: kv-heads, unique blocks, head dim, query-heads-per-kv
    Tensor tok_emb;                // [V,d] — tied with output (no pos-emb: RoPE)
    Tensor final_g;                // [d]
    std::vector<SBlock> blocks;    // n_uniq unique blocks, cycled across n_layers

    ModernGPT(Config c, std::mt19937& rng) : cfg(c) {
        n_kv   = (c.n_kv_heads > 0) ? c.n_kv_heads : c.n_heads;
        n_uniq = (c.n_unique  > 0) ? c.n_unique  : c.n_layers;
        dh = c.d_model / c.n_heads;
        qpg = c.n_heads / n_kv;
        int kvdim = n_kv * dh;
        auto W    = [&](std::vector<int> s) { return Tensor::randn(s, rng, 0.02); };
        auto ones = [&](int n) { Tensor t({n}); for (auto& v : t.data()) v = 1.0; return t; };
        tok_emb = W({c.vocab, c.d_model});
        final_g = ones(c.d_model);
        for (int l = 0; l < n_uniq; ++l) {
            SBlock b;
            b.wq = W({c.d_model, c.d_model}); b.wo = W({c.d_model, c.d_model});
            b.wk = W({c.d_model, kvdim});     b.wv = W({c.d_model, kvdim});
            b.attn_g = ones(c.d_model); b.mlp_g = ones(c.d_model);
            b.w1 = W({c.d_model, c.hidden}); b.w3 = W({c.d_model, c.hidden}); b.w2 = W({c.hidden, c.d_model});
            blocks.push_back(b);
        }
    }

    std::vector<Tensor> params() {
        std::vector<Tensor> p = {tok_emb, final_g};
        for (auto& b : blocks) {
            p.push_back(b.wq); p.push_back(b.wk); p.push_back(b.wv); p.push_back(b.wo);
            p.push_back(b.attn_g); p.push_back(b.mlp_g);
            p.push_back(b.w1); p.push_back(b.w3); p.push_back(b.w2);
        }
        return p;
    }

    // ids (length T) -> logits [T, vocab].
    Tensor forward(const std::vector<int>& ids) {
        int T = (int)ids.size(), d = cfg.d_model, H = cfg.n_heads;
        Tensor x    = embedding(tok_emb, ids);     // [T,d]  (position comes from RoPE, not an embedding)
        Tensor mask = GPT::causal_mask(T);
        real   scale = 1.0 / std::sqrt((real)dh);

        for (int layer = 0; layer < cfg.n_layers; ++layer) {
            SBlock& b = blocks[layer % n_uniq];    // block-wise weight sharing

            // --- GQA multi-head attention with RoPE (pre-norm, residual) ---
            Tensor h = rmsnorm(x, b.attn_g);
            Tensor Q = matmul(h, b.wq), K = matmul(h, b.wk), V = matmul(h, b.wv);  // Q:[T,d] K,V:[T,n_kv*dh]
            std::vector<Tensor> heads;
            for (int hh = 0; hh < H; ++hh) {
                int kvh = hh / qpg;                                          // which K/V head this query shares
                Tensor Qh = rope(slice_cols(Q, hh * dh, dh));
                Tensor Kh = rope(slice_cols(K, kvh * dh, dh));
                Tensor Vh = slice_cols(V, kvh * dh, dh);
                Tensor sc = add(scalar_mul(matmul(Qh, transpose(Kh)), scale), mask);   // [T,T]
                heads.push_back(matmul(softmax_rows(sc), Vh));                          // [T,dh]
            }
            Tensor ctx = (H == 1) ? heads[0] : concat_cols(heads);          // [T,d]
            x = add(x, matmul(ctx, b.wo));

            // --- SwiGLU MLP (pre-norm, residual):  (silu(h·w1) ⊙ (h·w3)) · w2 ---
            Tensor h2 = rmsnorm(x, b.mlp_g);
            Tensor gated = mul(silu(matmul(h2, b.w1)), matmul(h2, b.w3));    // [T,hidden]
            x = add(x, matmul(gated, b.w2));
        }
        x = rmsnorm(x, final_g);
        return matmul(x, transpose(tok_emb));      // tied output projection -> [T,V]
    }
};

// Autoregressive sampler for ModernGPT, decoding via a string (BPE) vocab.
inline std::string generate(ModernGPT& model, std::vector<int> ctx, int n_new,
                            real temp, std::mt19937& rng, const std::vector<std::string>& id2str) {
    int V = model.cfg.vocab, block = model.cfg.block;
    std::string out;
    for (int s = 0; s < n_new; ++s) {
        int start = std::max(0, (int)ctx.size() - block);
        std::vector<int> window(ctx.begin() + start, ctx.end());
        Tensor logits = model.forward(window);
        int L = (int)window.size();
        std::vector<real> probs(V);
        real mx = -1e30;
        for (int j = 0; j < V; ++j) mx = std::max(mx, logits.data()[(L - 1) * V + j] / temp);
        real Z = 0;
        for (int j = 0; j < V; ++j) { real e = std::exp(logits.data()[(L - 1) * V + j] / temp - mx); probs[j] = e; Z += e; }
        for (int j = 0; j < V; ++j) probs[j] /= Z;
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int next = dist(rng);
        ctx.push_back(next);
        out += id2str[next];
    }
    return out;
}

}  // namespace psi

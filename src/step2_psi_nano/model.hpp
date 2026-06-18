// model.hpp — psi-nano: a tiny GPT, plus an AdamW optimizer and a sampler.
//
// Single-head causal self-attention, pre-norm RMSNorm, GELU MLP, tied input/output
// embeddings. Built entirely from the Step 1 tensor autograd + Step 2 ops — no PyTorch.
// Correctness-first CPU reference (the Mac prototype); kernels/precision come later.

#pragma once

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "nn.hpp"

namespace psi {

struct Config {
    int vocab, d_model, n_layers, block, hidden;
};

struct Block {
    Tensor wq, wk, wv, wo;   // [d,d] attention projections
    Tensor attn_g, mlp_g;    // [d] RMSNorm gains
    Tensor w1, w2;           // [d,hidden], [hidden,d] MLP
};

struct GPT {
    Config cfg;
    Tensor tok_emb;          // [V,d] — tied with the output projection
    Tensor pos_emb;          // [block,d]
    Tensor final_g;          // [d]
    std::vector<Block> blocks;

    GPT(Config c, std::mt19937& rng) : cfg(c) {
        auto W    = [&](std::vector<int> s) { return Tensor::randn(s, rng, 0.02); };
        auto ones = [&](int n) { Tensor t({n}); for (auto& v : t.data()) v = 1.0; return t; };
        tok_emb = W({c.vocab, c.d_model});
        pos_emb = W({c.block, c.d_model});
        final_g = ones(c.d_model);
        for (int l = 0; l < c.n_layers; ++l) {
            Block b;
            b.wq = W({c.d_model, c.d_model}); b.wk = W({c.d_model, c.d_model});
            b.wv = W({c.d_model, c.d_model}); b.wo = W({c.d_model, c.d_model});
            b.attn_g = ones(c.d_model); b.mlp_g = ones(c.d_model);
            b.w1 = W({c.d_model, c.hidden}); b.w2 = W({c.hidden, c.d_model});
            blocks.push_back(b);
        }
    }

    std::vector<Tensor> params() {
        std::vector<Tensor> p = {tok_emb, pos_emb, final_g};
        for (auto& b : blocks) {
            p.push_back(b.wq); p.push_back(b.wk); p.push_back(b.wv); p.push_back(b.wo);
            p.push_back(b.attn_g); p.push_back(b.mlp_g); p.push_back(b.w1); p.push_back(b.w2);
        }
        return p;
    }

    // Additive causal mask [T,T]: 0 on/below the diagonal, -1e9 above (no peeking ahead).
    static Tensor causal_mask(int T) {
        Tensor m({T, T});
        for (int i = 0; i < T; ++i)
            for (int j = 0; j < T; ++j) m.data()[i * T + j] = (j <= i) ? 0.0 : -1e9;
        return m;
    }

    // ids (length T) -> logits [T, vocab].
    Tensor forward(const std::vector<int>& ids) {
        int T = (int)ids.size(), d = cfg.d_model;
        std::vector<int> pos(T);
        for (int i = 0; i < T; ++i) pos[i] = i;

        Tensor x    = add(embedding(tok_emb, ids), embedding(pos_emb, pos));  // [T,d]
        Tensor mask = causal_mask(T);
        real   scale = 1.0 / std::sqrt((real)d);

        for (auto& b : blocks) {
            // --- attention (pre-norm, residual) ---
            Tensor h      = rmsnorm(x, b.attn_g);
            Tensor Q      = matmul(h, b.wq), K = matmul(h, b.wk), V = matmul(h, b.wv);
            Tensor scores = add(scalar_mul(matmul(Q, transpose(K)), scale), mask);  // [T,T]
            Tensor attn   = softmax_rows(scores);
            Tensor ctx    = matmul(attn, V);          // [T,d]
            x = add(x, matmul(ctx, b.wo));            // residual

            // --- MLP (pre-norm, residual) ---
            Tensor h2 = rmsnorm(x, b.mlp_g);
            x = add(x, matmul(gelu(matmul(h2, b.w1)), b.w2));  // residual
        }
        x = rmsnorm(x, final_g);
        return matmul(x, transpose(tok_emb));        // tied output projection -> [T,V]
    }
};

// AdamW. Decoupled weight decay, skipped for 1-D params (RMSNorm gains).
struct AdamW {
    std::vector<Tensor> p;
    std::vector<std::vector<real>> m, v;
    real b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.01;
    int t = 0;

    explicit AdamW(std::vector<Tensor> params) : p(std::move(params)) {
        for (auto& pi : p) { m.emplace_back(pi.numel(), 0.0); v.emplace_back(pi.numel(), 0.0); }
    }
    void zero_grad() { for (auto& pi : p) pi.zero_grad(); }
    void step(real lr) {
        ++t;
        real bc1 = 1.0 - std::pow(b1, t), bc2 = 1.0 - std::pow(b2, t);
        for (size_t k = 0; k < p.size(); ++k) {
            real decay = (p[k].shape().size() == 1) ? 0.0 : wd;
            int n = p[k].numel();
            for (int i = 0; i < n; ++i) {
                real g = p[k].grad()[i];
                m[k][i] = b1 * m[k][i] + (1 - b1) * g;
                v[k][i] = b2 * v[k][i] + (1 - b2) * g * g;
                real mhat = m[k][i] / bc1, vhat = v[k][i] / bc2;
                p[k].data()[i] -= lr * (mhat / (std::sqrt(vhat) + eps) + decay * p[k].data()[i]);
            }
        }
    }
};

// Autoregressive sampling. Feeds the last `block` tokens, samples the next from a
// temperature-scaled softmax of the final-position logits.
inline std::string generate(GPT& model, std::vector<int> ctx, int n_new,
                            real temp, std::mt19937& rng,
                            const std::vector<char>& id2ch) {
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
        out.push_back(id2ch[next]);
    }
    return out;
}

// Same sampler, but decodes via a string vocab (multi-char tokens) — for the BPE psi-stories model.
inline std::string generate(GPT& model, std::vector<int> ctx, int n_new,
                            real temp, std::mt19937& rng,
                            const std::vector<std::string>& id2str) {
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

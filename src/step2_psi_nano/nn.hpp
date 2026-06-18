// nn.hpp — the transformer op set, built on the Step 1 tensor autograd engine.
//
// Step 2 of the Psi stack. These are the ops a GPT needs beyond Step 1's
// matmul/add_bias/mul/sub/tanh/mean. Every op records its local backward and is
// finite-difference grad-checked in gradcheck.cpp before psi-nano depends on it.
//
// Quality bar (per docs/QUALITY.md intent): softmax and cross_entropy use the
// log-sum-exp / max-subtraction trick for numerical stability; cross_entropy is the
// principled classification loss (not MSE); gelu uses the exact erf form.

#pragma once

#include <vector>

#include "../step1_tensor_autograd/tensor.hpp"

namespace psi {

// Elementwise add (same shape) — residual connections and additive masks.
inline Tensor add(const Tensor& A, const Tensor& B) {
    assert(A.shape() == B.shape());
    Tensor out = make_out(A.shape(), "add", {A.node, B.node});
    int n = A.numel();
    for (int i = 0; i < n; ++i) out.data()[i] = A.data()[i] + B.data()[i];
    TensorNode *Ap = A.node.get(), *Bp = B.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Bp, Op, n] {
        for (int i = 0; i < n; ++i) { Ap->grad[i] += Op->grad[i]; Bp->grad[i] += Op->grad[i]; }
    };
    return out;
}

// Multiply by a host-side scalar constant (e.g. 1/sqrt(d) attention scaling).
inline Tensor scalar_mul(const Tensor& A, real c) {
    Tensor out = make_out(A.shape(), "scalar_mul", {A.node});
    int n = A.numel();
    for (int i = 0; i < n; ++i) out.data()[i] = A.data()[i] * c;
    TensorNode *Ap = A.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Op, n, c] {
        for (int i = 0; i < n; ++i) Ap->grad[i] += c * Op->grad[i];
    };
    return out;
}

// 2D transpose:  out[j,i] = A[i,j].
inline Tensor transpose(const Tensor& A) {
    assert(A.shape().size() == 2);
    int r = A.dim(0), c = A.dim(1);
    Tensor out = make_out({c, r}, "transpose", {A.node});
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) out.data()[j * r + i] = A.data()[i * c + j];
    TensorNode *Ap = A.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Op, r, c] {
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j) Ap->grad[i * c + j] += Op->grad[j * r + i];
    };
    return out;
}

// Row-wise softmax over [r,c] (stable: subtract per-row max).
inline Tensor softmax_rows(const Tensor& A) {
    assert(A.shape().size() == 2);
    int r = A.dim(0), c = A.dim(1);
    Tensor out = make_out({r, c}, "softmax_rows", {A.node});
    for (int i = 0; i < r; ++i) {
        real m = A.data()[i * c];
        for (int j = 1; j < c; ++j) m = std::max(m, A.data()[i * c + j]);
        real Z = 0;
        for (int j = 0; j < c; ++j) { real e = std::exp(A.data()[i * c + j] - m); out.data()[i * c + j] = e; Z += e; }
        for (int j = 0; j < c; ++j) out.data()[i * c + j] /= Z;
    }
    TensorNode *Ap = A.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Op, r, c] {
        for (int i = 0; i < r; ++i) {
            real dot = 0;                                   // dot = sum_j (dL/dp_ij) p_ij
            for (int j = 0; j < c; ++j) dot += Op->grad[i * c + j] * Op->data[i * c + j];
            for (int j = 0; j < c; ++j) {
                real p = Op->data[i * c + j];
                Ap->grad[i * c + j] += p * (Op->grad[i * c + j] - dot);
            }
        }
    };
    return out;
}

// Embedding lookup: select rows of `table` [V,d] by integer ids (length n) -> [n,d].
// Gradient scatters back into the selected rows (scatter-add).
inline Tensor embedding(const Tensor& table, const std::vector<int>& ids) {
    assert(table.shape().size() == 2);
    int d = table.dim(1), n = (int)ids.size();
    Tensor out = make_out({n, d}, "embedding", {table.node});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j) out.data()[i * d + j] = table.data()[ids[i] * d + j];
    TensorNode *Tp = table.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Tp, Op, ids, d, n] {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j) Tp->grad[ids[i] * d + j] += Op->grad[i * d + j];
    };
    return out;
}

// RMSNorm over the last dim:  y[i,j] = x[i,j] / rms_i * gamma[j],
//   rms_i = sqrt(mean_j x[i,j]^2 + eps).
inline Tensor rmsnorm(const Tensor& x, const Tensor& gamma, real eps = 1e-5) {
    assert(x.shape().size() == 2 && gamma.shape().size() == 1 && gamma.dim(0) == x.dim(1));
    int n = x.dim(0), d = x.dim(1);
    Tensor out = make_out({n, d}, "rmsnorm", {x.node, gamma.node});
    std::vector<real> rinv(n);                              // 1/rms per row (reused in backward)
    for (int i = 0; i < n; ++i) {
        real ss = 0;
        for (int j = 0; j < d; ++j) { real v = x.data()[i * d + j]; ss += v * v; }
        rinv[i] = 1.0 / std::sqrt(ss / d + eps);
        for (int j = 0; j < d; ++j) out.data()[i * d + j] = x.data()[i * d + j] * rinv[i] * gamma.data()[j];
    }
    TensorNode *Xp = x.node.get(), *Gp = gamma.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Xp, Gp, Op, n, d, rinv] {
        for (int i = 0; i < n; ++i) {
            real ri = rinv[i];
            real s = 0;                                     // s_i = sum_j (dL/dy_ij) g_j x_ij
            for (int j = 0; j < d; ++j) s += Op->grad[i * d + j] * Gp->data[j] * Xp->data[i * d + j];
            for (int j = 0; j < d; ++j) {
                real a = Op->grad[i * d + j];
                Xp->grad[i * d + j] += Gp->data[j] * a * ri - Xp->data[i * d + j] * s * ri * ri * ri / d;
                Gp->grad[j]         += a * Xp->data[i * d + j] * ri;
            }
        }
    };
    return out;
}

// Exact GELU:  y = x * 0.5 * (1 + erf(x/sqrt2)).  dy/dx = Phi(x) + x*phi(x).
inline Tensor gelu(const Tensor& A) {
    Tensor out = make_out(A.shape(), "gelu", {A.node});
    int n = A.numel();
    const real inv_sqrt2 = 0.70710678118654752;
    for (int i = 0; i < n; ++i) { real x = A.data()[i]; out.data()[i] = x * 0.5 * (1.0 + std::erf(x * inv_sqrt2)); }
    TensorNode *Ap = A.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Op, n] {
        const real inv_sqrt2 = 0.70710678118654752, inv_sqrt2pi = 0.39894228040143268;
        for (int i = 0; i < n; ++i) {
            real x = Ap->data[i];
            real cdf = 0.5 * (1.0 + std::erf(x * inv_sqrt2));
            real pdf = inv_sqrt2pi * std::exp(-0.5 * x * x);
            Ap->grad[i] += (cdf + x * pdf) * Op->grad[i];
        }
    };
    return out;
}

// Column slice: out[i,j] = A[i, c0+j], for [r,c] -> [r,w]. Splits Q/K/V into per-head sub-tiles.
inline Tensor slice_cols(const Tensor& A, int c0, int w) {
    assert(A.shape().size() == 2);
    int r = A.dim(0), c = A.dim(1);
    assert(c0 >= 0 && c0 + w <= c);
    Tensor out = make_out({r, w}, "slice_cols", {A.node});
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < w; ++j) out.data()[i * w + j] = A.data()[i * c + c0 + j];
    TensorNode *Ap = A.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Op, r, c, c0, w] {
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < w; ++j) Ap->grad[i * c + c0 + j] += Op->grad[i * w + j];
    };
    return out;
}

// Column concat: join [r,w0],[r,w1],… -> [r, sum w]. Re-joins the per-head context tiles.
inline Tensor concat_cols(const std::vector<Tensor>& parts) {
    assert(!parts.empty());
    int r = parts[0].dim(0), total = 0;
    std::vector<NodePtr> deps;
    for (auto& p : parts) { assert(p.dim(0) == r); total += p.dim(1); deps.push_back(p.node); }
    Tensor out = make_out({r, total}, "concat_cols", deps);
    std::vector<TensorNode*> pps; std::vector<int> ws, offs;
    int off = 0;
    for (auto& p : parts) {
        int w = p.dim(1);
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < w; ++j) out.data()[i * total + off + j] = p.data()[i * w + j];
        pps.push_back(p.node.get()); ws.push_back(w); offs.push_back(off); off += w;
    }
    TensorNode* Op = out.node.get();
    out.node->backward_fn = [pps, ws, offs, Op, r, total] {
        for (size_t k = 0; k < pps.size(); ++k) {
            int w = ws[k], o = offs[k]; TensorNode* P = pps[k];
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < w; ++j) P->grad[i * w + j] += Op->grad[i * total + o + j];
        }
    };
    return out;
}

// SiLU / Swish:  y = x * sigmoid(x).  (For SwiGLU MLPs; pairs with the existing elementwise mul.)
inline Tensor silu(const Tensor& A) {
    Tensor out = make_out(A.shape(), "silu", {A.node});
    int n = A.numel();
    for (int i = 0; i < n; ++i) { real x = A.data()[i]; out.data()[i] = x / (1.0 + std::exp(-x)); }
    TensorNode *Ap = A.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Op, n] {
        for (int i = 0; i < n; ++i) {
            real x = Ap->data[i], s = 1.0 / (1.0 + std::exp(-x));
            Ap->grad[i] += (s + x * s * (1.0 - s)) * Op->grad[i];   // d/dx [x·σ(x)] = σ + x·σ(1-σ)
        }
    };
    return out;
}

// Rotary Position Embedding on a [T, dh] tensor (row r is at position r). Rotates dim-pairs (2i,2i+1)
// by angle r·base^(-2i/dh). No parameters (replaces the learned pos-emb, saving block·d params), and
// better relative-position behavior. Applied to Q and K per head. dh must be even.
inline Tensor rope(const Tensor& X, real base = 10000.0) {
    assert(X.shape().size() == 2 && X.dim(1) % 2 == 0);
    int T = X.dim(0), dh = X.dim(1), half = dh / 2;
    Tensor out = make_out({T, dh}, "rope", {X.node});
    for (int r = 0; r < T; ++r)
        for (int i = 0; i < half; ++i) {
            real ang = r * std::pow(base, -2.0 * i / dh), c = std::cos(ang), s = std::sin(ang);
            real x0 = X.data()[r * dh + 2 * i], x1 = X.data()[r * dh + 2 * i + 1];
            out.data()[r * dh + 2 * i]     = x0 * c - x1 * s;       // [[c,-s],[s,c]] · x
            out.data()[r * dh + 2 * i + 1] = x0 * s + x1 * c;
        }
    TensorNode *Xp = X.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Xp, Op, T, dh, half, base] {
        for (int r = 0; r < T; ++r)
            for (int i = 0; i < half; ++i) {
                real ang = r * std::pow(base, -2.0 * i / dh), c = std::cos(ang), s = std::sin(ang);
                real g0 = Op->grad[r * dh + 2 * i], g1 = Op->grad[r * dh + 2 * i + 1];
                Xp->grad[r * dh + 2 * i]     += c * g0 + s * g1;    // Rᵀ · dout
                Xp->grad[r * dh + 2 * i + 1] += -s * g0 + c * g1;
            }
    };
    return out;
}

// Fused softmax + cross-entropy (numerically stable). logits [n,V], integer targets [n].
// loss = mean_i -log softmax(logits_i)[target_i].  dlogits_i = (softmax_i - onehot_i)/n.
inline Tensor cross_entropy(const Tensor& logits, const std::vector<int>& targets) {
    assert(logits.shape().size() == 2);
    int n = logits.dim(0), V = logits.dim(1);
    assert((int)targets.size() == n);
    Tensor out = make_out({1}, "cross_entropy", {logits.node});
    auto probs = std::make_shared<std::vector<real>>(n * V);   // kept for backward
    real total = 0;
    for (int i = 0; i < n; ++i) {
        real m = logits.data()[i * V];
        for (int j = 1; j < V; ++j) m = std::max(m, logits.data()[i * V + j]);
        real Z = 0;
        for (int j = 0; j < V; ++j) { real e = std::exp(logits.data()[i * V + j] - m); (*probs)[i * V + j] = e; Z += e; }
        for (int j = 0; j < V; ++j) (*probs)[i * V + j] /= Z;
        total += -std::log((*probs)[i * V + targets[i]] + 1e-12);
    }
    out.data()[0] = total / n;
    TensorNode *Lp = logits.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Lp, Op, probs, targets, n, V] {
        real g = Op->grad[0] / n;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < V; ++j)
                Lp->grad[i * V + j] += g * ((*probs)[i * V + j] - (j == targets[i] ? 1.0 : 0.0));
    };
    return out;
}

}  // namespace psi

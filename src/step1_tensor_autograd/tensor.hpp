// tensor.hpp — tensor-level reverse-mode autograd (CPU reference oracle).
//
// Step 1 of the Psi stack. Same algorithm as Step 0 (record a local backward per op,
// then walk the graph in reverse), but the *unit* is now an N-D array instead of a
// scalar. That single change is what makes a real model trainable: one `matmul` node
// stands in for millions of scalar multiply-adds, so the per-node bookkeeping
// (allocation, pointer-chasing) is amortized to nothing, and the numbers live in
// contiguous buffers that vectorize.
//
// Deliberate non-goals here (correct-first, per RADICAL.md):
//   * We keep the shared_ptr node + std::function closure design from Step 0. At tensor
//     granularity the node count is tiny (a handful per layer), so the allocation/
//     refcount overhead that mattered for scalars is now negligible. The arena/tape/
//     index optimization is therefore DEFERRED until a profile says otherwise — and the
//     real cost will move into the kernels (Step 3), not the graph plumbing.
//   * Data type is `double` so this stays a high-precision correctness oracle we can
//     trust the tensor engine against. Low precision (bf16/fp8/ternary) is a kernel-layer
//     concern later, not the reference's job.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "../step3_metal/metal_backend.h"   // GPU matmul backend (used only in the float build)

namespace psi {

// `real` is the scalar type. Default `double` for the grad-check oracle (tight finite-diff);
// build the training path with `-DPSI_REAL=float` for ~2× bandwidth/SIMD width. Kernels go lower
// (bf16/fp8/ternary) later — this is just the CPU dtype knob.
#ifndef PSI_REAL
#define PSI_REAL double
#endif
using real = PSI_REAL;

struct TensorNode {
    std::vector<int>  shape;                              // row-major
    std::vector<real> data;                              // values
    std::vector<real> grad;                              // d(loss)/d(this), same size
    std::vector<std::shared_ptr<TensorNode>> parents;    // inputs that produced this
    std::function<void()> backward_fn;                   // local chain rule
    const char* op = "";

    explicit TensorNode(std::vector<int> shp) : shape(std::move(shp)) {
        int n = numel();
        data.assign(n, real(0));
        grad.assign(n, real(0));
    }
    int numel() const {
        int p = 1;
        for (int s : shape) p *= s;
        return p;
    }
};

using NodePtr = std::shared_ptr<TensorNode>;

class Tensor {
public:
    NodePtr node;

    Tensor() = default;
    explicit Tensor(std::vector<int> shape)
        : node(std::make_shared<TensorNode>(std::move(shape))) {}

    static Tensor from(std::vector<real> data, std::vector<int> shape) {
        Tensor t(std::move(shape));
        assert((int)data.size() == t.numel());
        t.node->data = std::move(data);
        return t;
    }
    // Gaussian init with an explicit std — we pass a fan-in-scaled std at the call site
    // (Xavier/He), the lesson from the Step-0 math review.
    static Tensor randn(std::vector<int> shape, std::mt19937& rng, real stddev) {
        Tensor t(std::move(shape));
        std::normal_distribution<real> d(0.0, stddev);
        for (auto& v : t.node->data) v = d(rng);
        return t;
    }

    const std::vector<int>& shape() const { return node->shape; }
    int   numel() const { return node->numel(); }
    int   dim(int i) const { return node->shape[i]; }
    std::vector<real>& data() const { return node->data; }
    std::vector<real>& grad() const { return node->grad; }
    void  zero_grad() const { std::fill(node->grad.begin(), node->grad.end(), real(0)); }

    void backward() const;
};

// Build an output tensor with its op label and parents wired up.
inline Tensor make_out(std::vector<int> shape, const char* op, std::vector<NodePtr> parents) {
    Tensor t(std::move(shape));
    t.node->op = op;
    t.node->parents = std::move(parents);
    return t;
}

// Split a row range [0,rows) across CPU threads when the work is large enough to amortize
// thread spawn (small matmuls — e.g. psi-nano's — stay serial). f(r0,r1) must write only rows
// in [r0,r1), so partitions are race-free and the result is identical to the serial version.
inline int psi_threads() {
    static int n = [] { unsigned h = std::thread::hardware_concurrency(); return h ? (int)h : 1; }();
    return n;
}
template <class F>
inline void parallel_rows(int rows, long work, F f) {
    int T = psi_threads();
    if (T <= 1 || rows < 2 || work < (1L << 20)) { f(0, rows); return; }  // serial for small work
    T = std::min(T, rows);
    int chunk = (rows + T - 1) / T;
    std::vector<std::thread> pool;
    for (int t = 0; t < T; ++t) {
        int b = t * chunk, e = std::min(rows, (t + 1) * chunk);
        if (b >= e) break;
        pool.emplace_back([&f, b, e] { f(b, e); });
    }
    for (auto& th : pool) th.join();
}

// ---------------------------------------------------------------------------
// Ops. Each computes the forward, then records the local backward.
// ---------------------------------------------------------------------------

// GPU dispatch helpers — templated so `if constexpr` genuinely discards the Metal calls in the
// double (oracle) build (a non-template `if constexpr` would still type-check the dead branch).
// Return true if the GPU handled the op.
template <class T>
inline bool gpu_matmul_fwd(const T* a, const T* b, T* c, int m, int k, int n, long work) {
    if constexpr (std::is_same<T, float>::value) {
        if (metal_available() && work >= (1L << 20)) { metal_matmul(a, b, c, m, k, n); return true; }
    }
    return false;
}
template <class T>
inline bool gpu_matmul_bwd(const T* dc, const T* a, const T* b, T* da, T* db, int m, int k, int n, long work) {
    if constexpr (std::is_same<T, float>::value) {
        if (metal_available() && work >= (1L << 20)) {
            metal_matmul_nt(dc, b, da, m, k, n);   // dA += dC @ B^T
            metal_matmul_tn(a, dc, db, k, n, m);   // dB += A^T @ dC
            return true;
        }
    }
    return false;
}

// C[m,n] = A[m,k] @ B[k,n].  dA = dC @ B^T,  dB = A^T @ dC.
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    assert(A.shape().size() == 2 && B.shape().size() == 2);
    int m = A.dim(0), k = A.dim(1), n = B.dim(1);
    assert(B.dim(0) == k);
    Tensor out = make_out({m, n}, "matmul", {A.node, B.node});
    const auto& a = A.data(); const auto& b = B.data(); auto& c = out.data();
    long work = (long)m * k * n;
    bool did_gpu = gpu_matmul_fwd(a.data(), b.data(), c.data(), m, k, n, work);  // GPU when float+big
    if (!did_gpu) {
        // forward: i-l-j (contiguous inner -> vectorizes), parallel over disjoint output rows i.
        parallel_rows(m, work, [&](int i0, int i1) {
            for (int i = i0; i < i1; ++i)
                for (int l = 0; l < k; ++l) {
                    real ail = a[i * k + l];
                    const real* brow = &b[l * n];
                    real* crow = &c[i * n];
                    for (int j = 0; j < n; ++j) crow[j] += ail * brow[j];
                }
        });
    }
    TensorNode *Ap = A.node.get(), *Bp = B.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Bp, Op, m, k, n] {
        const auto& dc = Op->grad; const auto& a = Ap->data; const auto& b = Bp->data;
        auto& da = Ap->grad; auto& db = Bp->grad;
        long work = (long)m * k * n;
        bool did_gpu = gpu_matmul_bwd(dc.data(), a.data(), b.data(), da.data(), db.data(), m, k, n, work);
        if (!did_gpu) {
            // dA = dC @ B^T, parallel over disjoint rows i of dA.
            parallel_rows(m, work, [&](int i0, int i1) {
                for (int i = i0; i < i1; ++i)
                    for (int l = 0; l < k; ++l) {
                        real s = 0;
                        const real* dcrow = &dc[i * n];
                        const real* brow = &b[l * n];
                        for (int j = 0; j < n; ++j) s += dcrow[j] * brow[j];
                        da[i * k + l] += s;
                    }
            });
            // dB = A^T @ dC, l-outer so threads own disjoint rows l of dB (contiguous inner j).
            parallel_rows(k, work, [&](int l0, int l1) {
                for (int l = l0; l < l1; ++l)
                    for (int i = 0; i < m; ++i) {
                        real ail = a[i * k + l];
                        const real* dcrow = &dc[i * n];
                        real* dbrow = &db[l * n];
                        for (int j = 0; j < n; ++j) dbrow[j] += ail * dcrow[j];
                    }
            });
        }
    };
    return out;
}

// C[m,n] = A[m,n] + bias[n]  (bias broadcast across rows).
inline Tensor add_bias(const Tensor& A, const Tensor& bias) {
    assert(A.shape().size() == 2 && bias.shape().size() == 1 && bias.dim(0) == A.dim(1));
    int m = A.dim(0), n = A.dim(1);
    Tensor out = make_out({m, n}, "add_bias", {A.node, bias.node});
    const auto& a = A.data(); const auto& bb = bias.data(); auto& c = out.data();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) c[i * n + j] = a[i * n + j] + bb[j];
    TensorNode *Ap = A.node.get(), *Bp = bias.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Bp, Op, m, n] {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                Ap->grad[i * n + j] += Op->grad[i * n + j];   // dA = dC
                Bp->grad[j]         += Op->grad[i * n + j];   // dbias = sum over rows
            }
    };
    return out;
}

// Elementwise (same-shape) subtract:  C = A - B.
inline Tensor sub(const Tensor& A, const Tensor& B) {
    assert(A.shape() == B.shape());
    Tensor out = make_out(A.shape(), "sub", {A.node, B.node});
    int n = A.numel();
    for (int i = 0; i < n; ++i) out.data()[i] = A.data()[i] - B.data()[i];
    TensorNode *Ap = A.node.get(), *Bp = B.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Bp, Op, n] {
        for (int i = 0; i < n; ++i) { Ap->grad[i] += Op->grad[i]; Bp->grad[i] -= Op->grad[i]; }
    };
    return out;
}

// Elementwise (same-shape) multiply (Hadamard):  C = A * B.
inline Tensor mul(const Tensor& A, const Tensor& B) {
    assert(A.shape() == B.shape());
    Tensor out = make_out(A.shape(), "mul", {A.node, B.node});
    int n = A.numel();
    for (int i = 0; i < n; ++i) out.data()[i] = A.data()[i] * B.data()[i];
    TensorNode *Ap = A.node.get(), *Bp = B.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Bp, Op, n] {
        for (int i = 0; i < n; ++i) {
            Ap->grad[i] += Bp->data[i] * Op->grad[i];
            Bp->grad[i] += Ap->data[i] * Op->grad[i];
        }
    };
    return out;
}

inline Tensor tanh(const Tensor& A) {
    Tensor out = make_out(A.shape(), "tanh", {A.node});
    int n = A.numel();
    for (int i = 0; i < n; ++i) out.data()[i] = std::tanh(A.data()[i]);
    TensorNode *Ap = A.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Op, n] {
        for (int i = 0; i < n; ++i) {
            real t = Op->data[i];
            Ap->grad[i] += (1.0 - t * t) * Op->grad[i];   // 1 - tanh^2
        }
    };
    return out;
}

// Mean of all elements -> scalar [1].
inline Tensor mean(const Tensor& A) {
    Tensor out = make_out({1}, "mean", {A.node});
    int n = A.numel();
    real s = 0;
    for (int i = 0; i < n; ++i) s += A.data()[i];
    out.data()[0] = s / n;
    TensorNode *Ap = A.node.get(), *Op = out.node.get();
    out.node->backward_fn = [Ap, Op, n] {
        real g = Op->grad[0] / n;
        for (int i = 0; i < n; ++i) Ap->grad[i] += g;
    };
    return out;
}

// ---------------------------------------------------------------------------
// Reverse-mode autodiff (identical structure to Step 0, now over tensor nodes).
// ---------------------------------------------------------------------------
inline void Tensor::backward() const {
    assert(numel() == 1 && "backward() must start from a scalar");
    std::vector<TensorNode*> topo;
    std::unordered_set<TensorNode*> seen;
    std::function<void(TensorNode*)> build = [&](TensorNode* v) {
        if (seen.count(v)) return;
        seen.insert(v);
        for (auto& p : v->parents) build(p.get());
        topo.push_back(v);
    };
    build(node.get());

    node->grad[0] = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        if ((*it)->backward_fn) (*it)->backward_fn();
}

}  // namespace psi

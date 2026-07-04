// value.hpp - scalar reverse-mode automatic differentiation, from scratch.
//
// This is Step 0 of the Psi custom stack. The whole point is to internalize how
// backprop actually works before we ever touch tensors or kernels. A `Value` is a
// single scalar in a computation graph. Every arithmetic op builds a new node and
// records a tiny closure ("backward_fn") that knows how to push gradient from the
// op's output back to its inputs - the local chain rule. Calling `.backward()` on a
// final scalar walks the graph in reverse and fills in d(output)/d(node) for every
// node. That's all reverse-mode autodiff is.
//
// Design note on memory: the graph is owned top-down via shared_ptr (each node holds
// shared_ptr to its parents). The backward closures capture *raw* pointers, never the
// shared_ptr of their own output node - capturing the output's shared_ptr would form a
// cycle (node -> closure -> node) and leak. While backward() runs, the root Value keeps
// the entire DAG alive, so the raw pointers are valid for the whole pass.

#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

namespace psi {

// One node in the autograd DAG.
struct Node {
    double data;                                   // forward value
    double grad = 0.0;                             // d(final output) / d(this node)
    std::vector<std::shared_ptr<Node>> parents;    // inputs that produced this node
    std::function<void()> backward_fn;             // distributes `grad` into parents
    const char* op;                                // label, for debugging

    explicit Node(double d, std::vector<std::shared_ptr<Node>> p = {}, const char* o = "")
        : data(d), parents(std::move(p)), op(o) {}
};

using NodePtr = std::shared_ptr<Node>;

// A value-semantics handle around a Node. Copying a Value shares the same underlying
// node (the graph is shared, not duplicated) - so parameters reused across the network
// all refer to one node, and updating one updates all references.
class Value {
public:
    NodePtr node;

    Value(double d) : node(std::make_shared<Node>(d)) {}   // leaf from a number
    Value(NodePtr n) : node(std::move(n)) {}               // wrap an existing node

    double data() const { return node->data; }
    double grad() const { return node->grad; }
    void   set_data(double d) { node->data = d; }
    void   zero_grad() { node->grad = 0.0; }

    void backward();   // reverse-mode autodiff seeded from this node
};

// ---------------------------------------------------------------------------
// Primitive ops. Each computes the forward value, links parents, and records the
// local backward (the partial derivatives of this op w.r.t. its inputs).
// ---------------------------------------------------------------------------

inline Value operator+(const Value& a, const Value& b) {
    auto out = std::make_shared<Node>(a.data() + b.data(),
                                      std::vector<NodePtr>{a.node, b.node}, "+");
    Node *ap = a.node.get(), *bp = b.node.get(), *op = out.get();
    out->backward_fn = [ap, bp, op] {
        ap->grad += op->grad;   // d(a+b)/da = 1
        bp->grad += op->grad;   // d(a+b)/db = 1
    };
    return Value(out);
}

inline Value operator*(const Value& a, const Value& b) {
    auto out = std::make_shared<Node>(a.data() * b.data(),
                                      std::vector<NodePtr>{a.node, b.node}, "*");
    Node *ap = a.node.get(), *bp = b.node.get(), *op = out.get();
    out->backward_fn = [ap, bp, op] {
        ap->grad += bp->data * op->grad;   // d(a*b)/da = b
        bp->grad += ap->data * op->grad;   // d(a*b)/db = a
    };
    return Value(out);
}

inline Value operator-(const Value& a) {   // unary negation
    auto out = std::make_shared<Node>(-a.data(), std::vector<NodePtr>{a.node}, "neg");
    Node *ap = a.node.get(), *op = out.get();
    out->backward_fn = [ap, op] { ap->grad += -op->grad; };
    return Value(out);
}

inline Value operator-(const Value& a, const Value& b) { return a + (-b); }

inline Value vpow(const Value& a, double e) {   // a^e for a constant exponent e
    auto out = std::make_shared<Node>(std::pow(a.data(), e),
                                      std::vector<NodePtr>{a.node}, "pow");
    Node *ap = a.node.get(), *op = out.get();
    out->backward_fn = [ap, op, e] {
        ap->grad += e * std::pow(ap->data, e - 1.0) * op->grad;   // d(a^e)/da = e*a^(e-1)
    };
    return Value(out);
}

inline Value vtanh(const Value& a) {
    double t = std::tanh(a.data());
    auto out = std::make_shared<Node>(t, std::vector<NodePtr>{a.node}, "tanh");
    Node *ap = a.node.get(), *op = out.get();
    out->backward_fn = [ap, op, t] {
        ap->grad += (1.0 - t * t) * op->grad;   // d(tanh)/da = 1 - tanh^2
    };
    return Value(out);
}

inline Value vrelu(const Value& a) {
    double r = a.data() > 0.0 ? a.data() : 0.0;
    auto out = std::make_shared<Node>(r, std::vector<NodePtr>{a.node}, "relu");
    Node *ap = a.node.get(), *op = out.get();
    out->backward_fn = [ap, op] {
        ap->grad += (op->data > 0.0 ? 1.0 : 0.0) * op->grad;
    };
    return Value(out);
}

// Mixed Value/double conveniences so we can write `2.0 * x + 1.0` naturally.
inline Value operator+(const Value& a, double b) { return a + Value(b); }
inline Value operator+(double a, const Value& b) { return Value(a) + b; }
inline Value operator*(const Value& a, double b) { return a * Value(b); }
inline Value operator*(double a, const Value& b) { return Value(a) * b; }
inline Value operator-(const Value& a, double b) { return a - Value(b); }
inline Value operator-(double a, const Value& b) { return Value(a) - b; }

// ---------------------------------------------------------------------------
// Reverse-mode autodiff.
// ---------------------------------------------------------------------------
inline void Value::backward() {
    // 1) Reverse topological order: a node appears only after all of its parents.
    //    Processing in reverse then guarantees a node's grad is fully accumulated
    //    (every consumer has contributed) before we use it.
    std::vector<Node*> topo;
    std::unordered_set<Node*> visited;
    std::function<void(Node*)> build = [&](Node* v) {
        if (visited.count(v)) return;
        visited.insert(v);
        for (auto& p : v->parents) build(p.get());
        topo.push_back(v);   // parents already pushed -> this node comes after them
    };
    build(node.get());

    // 2) Seed the output: d(output)/d(output) = 1.
    node->grad = 1.0;

    // 3) Walk output -> leaves, applying each local backward.
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) (*it)->backward_fn();
    }
    // (Recursive topo build is fine for tiny Step-0 graphs; we'll make it iterative
    //  when graphs get deep in the tensor-autograd step.)
}

}  // namespace psi

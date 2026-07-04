// main.cpp - Step 0 driver.
//
// Two things prove the autograd engine works:
//   1) grad_check(): compare analytic gradients (from backward()) against numerical
//      finite-difference gradients. This is our permanent correctness net - every
//      future op gets one of these before we trust it.
//   2) train_xor(): build a small MLP out of Value scalars and train it on XOR with
//      plain SGD. If loss falls and predictions match, the forward+backward+update
//      loop is sound end to end.

#include <cstdio>
#include <random>
#include <vector>

#include "value.hpp"

using psi::Value;
using psi::vtanh;
using psi::vpow;

// ---------------------------------------------------------------------------
// 1) Finite-difference gradient check.
// ---------------------------------------------------------------------------
static void grad_check() {
    // f(a,b,c) = (a*b + tanh(c))^2  - exercises *, +, tanh, and pow together.
    auto f = [](const Value& a, const Value& b, const Value& c) {
        return vpow(a * b + vtanh(c), 2.0);
    };

    const double A = 1.3, B = -2.0, C = 0.7;

    // Analytic gradients via backprop.
    Value a(A), b(B), c(C);
    Value out = f(a, b, c);
    out.backward();

    // Numerical gradients via central differences: df/dx ~ (f(x+h) - f(x-h)) / 2h.
    const double h = 1e-6;
    auto fval = [&](double x, double y, double z) {
        return f(Value(x), Value(y), Value(z)).data();
    };
    double na = (fval(A + h, B, C) - fval(A - h, B, C)) / (2 * h);
    double nb = (fval(A, B + h, C) - fval(A, B - h, C)) / (2 * h);
    double nc = (fval(A, B, C + h) - fval(A, B, C - h)) / (2 * h);

    std::printf("grad check (analytic vs numerical):\n");
    std::printf("  da: %.8f vs %.8f   |err|=%.2e\n", a.grad(), na, std::fabs(a.grad() - na));
    std::printf("  db: %.8f vs %.8f   |err|=%.2e\n", b.grad(), nb, std::fabs(b.grad() - nb));
    std::printf("  dc: %.8f vs %.8f   |err|=%.2e\n", c.grad(), nc, std::fabs(c.grad() - nc));

    double max_err = std::fmax(std::fabs(a.grad() - na),
                     std::fmax(std::fabs(b.grad() - nb), std::fabs(c.grad() - nc)));
    std::printf("  -> max error %.2e  (%s)\n\n", max_err, max_err < 1e-4 ? "PASS" : "FAIL");
}

// ---------------------------------------------------------------------------
// A tiny MLP built from Value scalars.
// ---------------------------------------------------------------------------
struct Neuron {
    std::vector<Value> w;
    Value b;
    bool nonlin;

    Neuron(int nin, bool nl, std::mt19937& rng) : b(0.0), nonlin(nl) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int i = 0; i < nin; ++i) w.push_back(Value(dist(rng)));
    }

    Value operator()(const std::vector<Value>& x) const {
        Value act = b;                                  // start from the bias
        for (size_t i = 0; i < w.size(); ++i) act = act + w[i] * x[i];
        return nonlin ? vtanh(act) : act;
    }

    void collect(std::vector<Value>& out) {
        for (auto& wi : w) out.push_back(wi);
        out.push_back(b);
    }
};

struct Layer {
    std::vector<Neuron> neurons;
    Layer(int nin, int nout, bool nl, std::mt19937& rng) {
        for (int i = 0; i < nout; ++i) neurons.emplace_back(nin, nl, rng);
    }
    std::vector<Value> operator()(const std::vector<Value>& x) const {
        std::vector<Value> out;
        for (auto& n : neurons) out.push_back(n(x));
        return out;
    }
    void collect(std::vector<Value>& out) {
        for (auto& n : neurons) n.collect(out);
    }
};

struct MLP {
    std::vector<Layer> layers;
    // sizes = {nin, h1, h2, ..., nout}. Hidden layers use tanh; output is linear.
    MLP(std::vector<int> sizes, std::mt19937& rng) {
        for (size_t i = 0; i + 1 < sizes.size(); ++i) {
            bool nonlin = (i + 2 < sizes.size());   // all but the last layer
            layers.emplace_back(sizes[i], sizes[i + 1], nonlin, rng);
        }
    }
    std::vector<Value> operator()(std::vector<Value> x) const {
        for (auto& l : layers) x = l(x);
        return x;
    }
    std::vector<Value> parameters() {
        std::vector<Value> ps;
        for (auto& l : layers) l.collect(ps);
        return ps;
    }
};

// ---------------------------------------------------------------------------
// 2) Train XOR.
// ---------------------------------------------------------------------------
static void train_xor() {
    std::mt19937 rng(42);
    MLP net({2, 16, 1}, rng);            // 2 inputs -> 16 tanh hidden -> 1 linear out
    auto params = net.parameters();

    const std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<double>              Y = {0, 1, 1, 0};

    const double lr = 0.1;
    const int epochs = 1000;

    std::printf("training XOR (%zu params):\n", params.size());
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        // Full-batch MSE loss over the 4 examples, as one scalar graph.
        Value loss(0.0);
        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<Value> x = {Value(X[i][0]), Value(X[i][1])};
            Value pred = net(x)[0];
            loss = loss + vpow(pred - Y[i], 2.0);
        }
        loss = (1.0 / X.size()) * loss;

        // Reset parameter grads, then backprop (fresh graph nodes start at grad 0).
        for (auto& p : params) p.zero_grad();
        loss.backward();

        // SGD step.
        for (auto& p : params) p.set_data(p.data() - lr * p.grad());

        if (epoch % 100 == 0)
            std::printf("  epoch %4d   loss %.6f\n", epoch, loss.data());
    }

    std::printf("predictions:\n");
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<Value> x = {Value(X[i][0]), Value(X[i][1])};
        std::printf("  (%g, %g) -> %.4f   (target %g)\n",
                    X[i][0], X[i][1], net(x)[0].data(), Y[i]);
    }
}

int main() {
    grad_check();
    train_xor();
    return 0;
}

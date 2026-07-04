// main.cpp - Step 1 driver.
//
//   1) grad_checks(): finite-difference check on every op, individually and composed.
//      Same correctness net as Step 0, now for tensor ops. This is what lets us trust
//      the engine before any model depends on it.
//   2) train_xor_tensor(): rebuild the XOR MLP at TENSOR granularity (one matmul per
//      layer over the whole 4-example batch) and train it. Matching Step 0's result is a
//      cross-check of the new engine against the scalar oracle. We also apply the
//      fan-in-scaled (Xavier) init lesson from the Step-0 math review.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "tensor.hpp"

using namespace psi;

// Finite-difference check: perturb each element of each input, compare numeric
// d(scalar)/d(element) to the analytic gradient from backward().
static void check(const char* name, std::vector<Tensor> inputs,
                  const std::function<Tensor()>& forward) {
    for (auto& t : inputs) t.zero_grad();
    Tensor out = forward();
    out.backward();   // fills analytic grads into the input nodes

    const real h = 1e-5;
    real max_err = 0;
    for (auto& t : inputs)
        for (int i = 0; i < t.numel(); ++i) {
            real orig = t.data()[i];
            t.data()[i] = orig + h; real fp = forward().data()[0];
            t.data()[i] = orig - h; real fm = forward().data()[0];
            t.data()[i] = orig;
            real numeric = (fp - fm) / (2 * h);
            max_err = std::max(max_err, std::fabs(numeric - t.grad()[i]));
        }
    std::printf("  %-22s max|analytic-numeric| = %.2e  (%s)\n",
                name, max_err, max_err < 1e-6 ? "PASS" : "FAIL");
}

static void grad_checks() {
    std::mt19937 rng(0);
    std::printf("grad checks (tensor ops):\n");

    {   // matmul
        Tensor A = Tensor::randn({2, 3}, rng, 1.0), B = Tensor::randn({3, 4}, rng, 1.0);
        check("matmul", {A, B}, [&] { return mean(matmul(A, B)); });
    }
    {   // add_bias
        Tensor A = Tensor::randn({4, 3}, rng, 1.0), b = Tensor::randn({3}, rng, 1.0);
        check("add_bias", {A, b}, [&] { return mean(add_bias(A, b)); });
    }
    {   // tanh
        Tensor A = Tensor::randn({3, 5}, rng, 1.0);
        check("tanh", {A}, [&] { return mean(tanh(A)); });
    }
    {   // mul / sub
        Tensor A = Tensor::randn({3, 3}, rng, 1.0), B = Tensor::randn({3, 3}, rng, 1.0);
        check("mul", {A, B}, [&] { return mean(mul(A, B)); });
        check("sub", {A, B}, [&] { return mean(sub(A, B)); });
    }
    {   // full MLP + MSE, end to end (the real test): grads of all params at once
        Tensor X  = Tensor::from({0,0, 0,1, 1,0, 1,1}, {4, 2});
        Tensor Y  = Tensor::from({0, 1, 1, 0}, {4, 1});
        Tensor W1 = Tensor::randn({2, 5}, rng, 0.5), b1 = Tensor({5});
        Tensor W2 = Tensor::randn({5, 1}, rng, 0.5), b2 = Tensor({1});
        auto fwd = [&] {
            Tensor h    = tanh(add_bias(matmul(X, W1), b1));
            Tensor pred = add_bias(matmul(h, W2), b2);
            Tensor diff = sub(pred, Y);
            return mean(mul(diff, diff));
        };
        check("mlp+mse (W1,b1,W2,b2)", {W1, b1, W2, b2}, fwd);
    }
    std::printf("\n");
}

static void train_xor_tensor() {
    std::mt19937 rng(42);
    // Fan-in-scaled init (Xavier ~ 1/sqrt(fan_in)) - the Step-0 math lesson, applied.
    Tensor W1 = Tensor::randn({2, 16}, rng, std::sqrt(1.0 / 2));
    Tensor b1 = Tensor({16});                         // zeros
    Tensor W2 = Tensor::randn({16, 1}, rng, std::sqrt(1.0 / 16));
    Tensor b2 = Tensor({1});
    std::vector<Tensor> params = {W1, b1, W2, b2};

    Tensor X = Tensor::from({0,0, 0,1, 1,0, 1,1}, {4, 2});
    Tensor Y = Tensor::from({0, 1, 1, 0}, {4, 1});

    const real lr = 0.1;
    const int  epochs = 2000;

    std::printf("training XOR (tensor engine):\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        Tensor h    = tanh(add_bias(matmul(X, W1), b1));   // [4,16]
        Tensor pred = add_bias(matmul(h, W2), b2);          // [4,1]
        Tensor diff = sub(pred, Y);
        Tensor loss = mean(mul(diff, diff));

        for (auto& p : params) p.zero_grad();
        loss.backward();
        for (auto& p : params)
            for (int i = 0; i < p.numel(); ++i) p.data()[i] -= lr * p.grad()[i];

        if (epoch % 200 == 0)
            std::printf("  epoch %4d   loss %.6f\n", epoch, loss.data()[0]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    Tensor h    = tanh(add_bias(matmul(X, W1), b1));
    Tensor pred = add_bias(matmul(h, W2), b2);
    std::printf("predictions:\n");
    for (int i = 0; i < 4; ++i)
        std::printf("  (%g, %g) -> %.4f   (target %g)\n",
                    X.data()[i * 2], X.data()[i * 2 + 1], pred.data()[i], Y.data()[i]);

    // Seed of the benchmark harness (real tokens/sec + MFU arrive with psi-nano).
    std::printf("timing: %d epochs in %.1f ms  (%.0f steps/s)\n",
                epochs, ms, epochs / (ms / 1000.0));
}

int main() {
    grad_checks();
    train_xor_tensor();
    return 0;
}

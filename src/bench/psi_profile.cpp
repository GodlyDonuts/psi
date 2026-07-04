// psi_profile.cpp - phase breakdown of a psi-nano training step.
//
// Iters 4 & 6 showed psi-nano's time doesn't move with matmul size, so the cost is per-op
// overhead, not GEMM. This splits a step into forward+loss / backward / optimizer to locate
// it precisely. Synthetic random tokens (timing is data-independent).
//
// Build like the training path:  -O3 -march=native -pthread -ffast-math -DPSI_REAL=float

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../step2_psi_nano/model.hpp"

using namespace psi;
using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

int main(int argc, char** argv) {
    int steps = (argc > 1) ? std::atoi(argv[1]) : 300;
    const int V = 27, T = 32, B = 8, d = 64;

    std::mt19937 rng(1234);
    std::vector<int> data(2000);
    std::uniform_int_distribution<int> ch(0, V - 1);
    for (auto& x : data) x = ch(rng);

    Config cfg{V, d, 2, T, 256};
    GPT model(cfg, rng);
    AdamW opt(model.params());
    std::uniform_int_distribution<int> sp(0, (int)data.size() - T - 2);

    double tf = 0, tb = 0, to = 0;
    for (int s = 0; s < steps; ++s) {
        auto a = clk::now();
        std::vector<Tensor> losses;
        for (int b = 0; b < B; ++b) {
            int i = sp(rng);
            std::vector<int> ids(data.begin() + i, data.begin() + i + T);
            std::vector<int> tgt(data.begin() + i + 1, data.begin() + i + 1 + T);
            losses.push_back(cross_entropy(model.forward(ids), tgt));
        }
        Tensor loss = losses[0];
        for (size_t k = 1; k < losses.size(); ++k) loss = add(loss, losses[k]);
        loss = scalar_mul(loss, 1.0 / B);
        auto b1 = clk::now();
        opt.zero_grad();
        loss.backward();
        auto b2 = clk::now();
        opt.step(1e-3);
        auto b3 = clk::now();
        tf += secs(a, b1); tb += secs(b1, b2); to += secs(b2, b3);
    }
    double tot = tf + tb + to;
    std::printf("psi-nano phase profile, %d steps  (%.2fs total, %.2f ms/step):\n",
                steps, tot, tot / steps * 1000);
    std::printf("  forward+loss : %.2fs  (%.0f%%)\n", tf, 100 * tf / tot);
    std::printf("  backward     : %.2fs  (%.0f%%)\n", tb, 100 * tb / tot);
    std::printf("  optimizer    : %.2fs  (%.0f%%)\n", to, 100 * to / tot);
    return 0;
}

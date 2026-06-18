// gradcheck.cpp — finite-difference validation of every transformer op in nn.hpp.
// Each op must PASS here before psi-nano is allowed to use it.

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "nn.hpp"

using namespace psi;

// Scalarize a tensor by a FIXED random weighting (created once, reused across forward
// calls) so the grad-check exercises a non-trivial gradient. Returns mean(t * W).
struct Scalarizer {
    Tensor W;
    Scalarizer(std::vector<int> shape, std::mt19937& rng) : W(shape) {
        std::normal_distribution<real> d(0, 1);
        for (auto& v : W.data()) v = d(rng);
    }
    Tensor operator()(const Tensor& t) const { return mean(mul(t, W)); }
};

static void check(const char* name, std::vector<Tensor> inputs,
                  const std::function<Tensor()>& forward) {
    for (auto& t : inputs) t.zero_grad();
    forward().backward();
    const real h = 1e-5;
    real max_err = 0;
    for (auto& t : inputs)
        for (int i = 0; i < t.numel(); ++i) {
            real orig = t.data()[i];
            t.data()[i] = orig + h; real fp = forward().data()[0];
            t.data()[i] = orig - h; real fm = forward().data()[0];
            t.data()[i] = orig;
            max_err = std::max(max_err, std::fabs((fp - fm) / (2 * h) - t.grad()[i]));
        }
    std::printf("  %-22s max|analytic-numeric| = %.2e  (%s)\n",
                name, max_err, max_err < 1e-6 ? "PASS" : "FAIL");
}

int main() {
    std::mt19937 rng(0);
    std::printf("grad checks (transformer ops):\n");

    { Tensor A = Tensor::randn({3,3}, rng, 1), B = Tensor::randn({3,3}, rng, 1);
      Scalarizer s({3,3}, rng);
      check("add",        {A,B}, [&]{ return s(add(A,B)); }); }

    { Tensor A = Tensor::randn({3,3}, rng, 1);
      Scalarizer s({3,3}, rng);
      check("scalar_mul", {A}, [&]{ return s(scalar_mul(A, 2.5)); }); }

    { Tensor A = Tensor::randn({2,4}, rng, 1);
      Scalarizer s({4,2}, rng);
      check("transpose",  {A}, [&]{ return s(transpose(A)); }); }

    { Tensor A = Tensor::randn({3,5}, rng, 1);
      Scalarizer s({3,5}, rng);
      check("softmax_rows",{A}, [&]{ return s(softmax_rows(A)); }); }

    { Tensor table = Tensor::randn({5,4}, rng, 1);
      std::vector<int> ids = {0,2,4,1};
      Scalarizer s({4,4}, rng);
      check("embedding",  {table}, [&]{ return s(embedding(table, ids)); }); }

    { Tensor x = Tensor::randn({3,4}, rng, 1), g = Tensor::randn({4}, rng, 1);
      Scalarizer s({3,4}, rng);
      check("rmsnorm",    {x,g}, [&]{ return s(rmsnorm(x,g)); }); }

    { Tensor A = Tensor::randn({3,4}, rng, 1);
      Scalarizer s({3,4}, rng);
      check("gelu",       {A}, [&]{ return s(gelu(A)); }); }

    { Tensor logits = Tensor::randn({3,5}, rng, 1);
      std::vector<int> tgt = {1,3,0};
      check("cross_entropy",{logits}, [&]{ return cross_entropy(logits, tgt); }); }

    // --- modern-architecture ops (multi-head attention + SwiGLU) ---
    { Tensor A = Tensor::randn({4,6}, rng, 1);
      Scalarizer s({4,3}, rng);
      check("slice_cols",  {A}, [&]{ return s(slice_cols(A, 2, 3)); }); }

    { Tensor A = Tensor::randn({3,2}, rng, 1), B = Tensor::randn({3,4}, rng, 1);
      Scalarizer s({3,6}, rng);
      check("concat_cols", {A,B}, [&]{ return s(concat_cols({A,B})); }); }

    { Tensor A = Tensor::randn({3,4}, rng, 1);
      Scalarizer s({3,4}, rng);
      check("silu",        {A}, [&]{ return s(silu(A)); }); }

    { Tensor A = Tensor::randn({5,8}, rng, 1);   // [T=5, dh=8], RoPE rotates dim-pairs by position
      Scalarizer s({5,8}, rng);
      check("rope",        {A}, [&]{ return s(rope(A)); }); }

    return 0;
}

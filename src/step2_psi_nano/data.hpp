// data.hpp — data pipeline: read a corpus, hold out a validation split, measure val loss.
//
// The held-out eval is the point: it's what tells us the model is *generalizing* rather than
// *memorizing* — psi-nano's current blind spot. A real generalization win needs scale (kernels),
// but the machinery to measure it lives here.

#pragma once

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "model.hpp"   // GPT, forward, cross_entropy

namespace psi {

inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Contiguous train/val split: the last `val_frac` of the token stream is held out.
struct Dataset {
    std::vector<int> train, val;
    Dataset(const std::vector<int>& tokens, double val_frac = 0.1) {
        int n = (int)tokens.size();
        int n_val = std::max(1, (int)(n * val_frac));
        int n_train = std::max(0, n - n_val);
        train.assign(tokens.begin(), tokens.begin() + n_train);
        val.assign(tokens.begin() + n_train, tokens.end());
    }
};

// Mean cross-entropy over strided windows of `data` (no backward — pure evaluation).
// Templated on the model type so both GPT (psi-nano) and ModernGPT (psi-stories) use it.
template <class M>
inline double eval_loss(M& model, const std::vector<int>& data, int block, int max_windows = 64) {
    int n = (int)data.size();
    if (n < block + 2) return 0.0;
    double total = 0;
    int count = 0;
    int stride = std::max(1, (n - block - 1) / max_windows);
    for (int i = 0; i + block + 1 <= n && count < max_windows; i += stride) {
        std::vector<int> ids(data.begin() + i, data.begin() + i + block);
        std::vector<int> tgt(data.begin() + i + 1, data.begin() + i + 1 + block);
        total += cross_entropy(model.forward(ids), tgt).data()[0];
        ++count;
    }
    return count ? total / count : 0.0;
}

}  // namespace psi

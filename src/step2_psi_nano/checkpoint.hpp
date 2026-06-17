// checkpoint.hpp — save/load a trained model as a self-contained artifact.
//
// So each zoo model (psi-stories, psi-chess, …) is a file you can ship and run without
// retraining. Format: magic "PSI1", Config (5 ints), tokenizer vocab, then every parameter's
// raw `real` data in GPT::params() order (deterministic, so save/load line up).

#pragma once

#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

#include "model.hpp"
#include "tokenizer.hpp"

namespace psi {

inline void save_checkpoint(const std::string& path, GPT& model, const CharTokenizer& tok) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write checkpoint: " + path);
    f.write("PSI1", 4);
    int cfg[5] = {model.cfg.vocab, model.cfg.d_model, model.cfg.n_layers, model.cfg.block, model.cfg.hidden};
    f.write(reinterpret_cast<char*>(cfg), sizeof(cfg));
    int vsz = tok.vocab();
    f.write(reinterpret_cast<char*>(&vsz), sizeof(int));
    f.write(tok.id2ch.data(), vsz);
    for (auto& p : model.params()) {
        auto& d = p.data();
        f.write(reinterpret_cast<char*>(d.data()), (std::streamsize)(d.size() * sizeof(real)));
    }
}

// Reconstructs the exact model + tokenizer from a checkpoint.
inline GPT load_checkpoint(const std::string& path, CharTokenizer& tok, std::mt19937& rng) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open checkpoint: " + path);
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "PSI1") throw std::runtime_error("bad checkpoint magic: " + path);
    int cfg[5];
    f.read(reinterpret_cast<char*>(cfg), sizeof(cfg));
    Config c{cfg[0], cfg[1], cfg[2], cfg[3], cfg[4]};
    int vsz;
    f.read(reinterpret_cast<char*>(&vsz), sizeof(int));
    tok.id2ch.resize(vsz);
    f.read(tok.id2ch.data(), vsz);
    tok.ch2id.clear();
    for (int i = 0; i < vsz; ++i) tok.ch2id[tok.id2ch[i]] = i;

    GPT model(c, rng);                       // random init, then overwrite with saved params
    for (auto& p : model.params()) {
        auto& d = p.data();
        f.read(reinterpret_cast<char*>(d.data()), (std::streamsize)(d.size() * sizeof(real)));
    }
    return model;
}

}  // namespace psi

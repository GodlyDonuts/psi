// stories.cpp — psi-stories: the sub-1M TinyStories model. Same autograd / ops / GPT as psi-nano,
// but with the small-BPE tokenizer (bpe.hpp) so the tiny parameter budget goes to the transformer,
// not to spelling. The goal: the smallest model (in params, then bits) that clears the TinyStories
// capability bar (docs/EVAL.md).
//
//   stories train <data> <steps> [vocab] [d] [layers] [block] [hidden]   train + save psi_stories.bin
//   stories eval  <model.bin> [prompts] [temp]                           completions for grading
//   stories gen   <model.bin> [prompt]                                   sample from a prompt
//
// Build (float + Metal GPU matmul):
//   clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \
//     src/step2_psi_nano/stories.cpp src/step3_metal/metal_backend.mm \
//     -framework Metal -framework Foundation -o psi_stories

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "bpe.hpp"
#include "data.hpp"
#include "model.hpp"

using namespace psi;

static const char* MODEL_PATH = "psi_stories.bin";

// --- BPE-aware checkpoint: magic, Config, the tokenizer (base+merges), then params in order. ---
static void save_stories(const std::string& path, GPT& model, const BPETokenizer& tok) {
    std::ofstream f(path, std::ios::binary);
    f.write("PSST", 4);
    int cfg[5] = {model.cfg.vocab, model.cfg.d_model, model.cfg.n_layers, model.cfg.block, model.cfg.hidden};
    f.write(reinterpret_cast<char*>(cfg), sizeof(cfg));
    tok.save(f);
    for (auto& p : model.params()) { auto& d = p.data(); f.write(reinterpret_cast<char*>(d.data()), (std::streamsize)(d.size() * sizeof(real))); }
}
static GPT load_stories(const std::string& path, BPETokenizer& tok, std::mt19937& rng) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4]; f.read(magic, 4);
    if (std::string(magic, 4) != "PSST") throw std::runtime_error("bad magic in " + path);
    int cfg[5]; f.read(reinterpret_cast<char*>(cfg), sizeof(cfg));
    Config c{cfg[0], cfg[1], cfg[2], cfg[3], cfg[4]};
    tok.load(f);
    GPT model(c, rng);
    for (auto& p : model.params()) { auto& d = p.data(); f.read(reinterpret_cast<char*>(d.data()), (std::streamsize)(d.size() * sizeof(real))); }
    return model;
}

static int cmd_train(const std::string& datafile, int steps, int vocab, int d, int layers, int block, int hidden) {
    std::string text = read_file(datafile);
    if (text.empty()) { std::fprintf(stderr, "error: empty/unreadable data (%s)\n", datafile.c_str()); return 1; }

    std::printf("fitting BPE (vocab=%d) ...\n", vocab); std::fflush(stdout);
    BPETokenizer tok;
    tok.fit(text, vocab);
    std::vector<int> ids = tok.encode(text);
    Dataset ds(ids, 0.1);

    Config cfg{tok.vocab(), d, layers, block, hidden};
    std::mt19937 rng(1234);
    GPT model(cfg, rng);
    AdamW opt(model.params());
    const int batch = 8;
    const real lr = 1e-3;

    int nparams = 0; for (auto& p : model.params()) nparams += p.numel();
    std::printf("psi-stories | chars=%zu  vocab=%d  tokens=%zu  (%.2f chars/tok)  train=%zu val=%zu  params=%d\n",
                text.size(), tok.vocab(), ids.size(), (double)text.size() / ids.size(),
                ds.train.size(), ds.val.size(), nparams);
    std::fflush(stdout);
    if ((int)ds.train.size() < cfg.block + 2) { std::fprintf(stderr, "error: corpus too small\n"); return 1; }

    std::uniform_int_distribution<int> pick(0, (int)ds.train.size() - cfg.block - 2);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step <= steps; ++step) {
        std::vector<Tensor> losses;
        for (int b = 0; b < batch; ++b) {
            int i = pick(rng);
            std::vector<int> in(ds.train.begin() + i, ds.train.begin() + i + cfg.block);
            std::vector<int> tg(ds.train.begin() + i + 1, ds.train.begin() + i + 1 + cfg.block);
            losses.push_back(cross_entropy(model.forward(in), tg));
        }
        Tensor loss = losses[0];
        for (size_t k = 1; k < losses.size(); ++k) loss = add(loss, losses[k]);
        loss = scalar_mul(loss, 1.0 / batch);
        opt.zero_grad(); loss.backward(); opt.step(lr);

        if (step % 100 == 0) {
            double vl = eval_loss(model, ds.val, cfg.block, 32);
            double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::printf("step %5d   train %.4f   val %.4f   (%.1fs)\n", step, loss.data()[0], vl, el);
            std::fflush(stdout);
        }
    }
    save_stories(MODEL_PATH, model, tok);
    std::printf("saved -> %s\n\nsample:\n", MODEL_PATH);
    std::vector<int> seed = tok.encode("Once upon a time");
    std::mt19937 grng(7);
    std::printf("  Once upon a time%s\n", generate(model, seed, 220, 0.8, grng, tok.id2str).c_str());
    return 0;
}

static int cmd_eval(const std::string& path, const std::string& promptsfile, real temp) {
    std::mt19937 rng(0);
    BPETokenizer tok;
    GPT model = load_stories(path, tok, rng);
    std::string content = read_file(promptsfile);
    if (content.empty()) { std::fprintf(stderr, "error: no prompts (%s)\n", promptsfile.c_str()); return 1; }
    std::istringstream iss(content);
    std::string line; int idx = 1;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        std::vector<int> ctx = tok.encode(line);
        if (ctx.empty()) ctx.push_back(0);
        std::string comp = generate(model, ctx, 200, temp, rng, tok.id2str);
        std::printf("=== prompt %d ===\n%s  ┃>>>┃  %s\n\n", idx++, line.c_str(), comp.c_str());
    }
    return 0;
}

static int cmd_gen(const std::string& path, const std::string& prompt) {
    std::mt19937 rng(0);
    BPETokenizer tok;
    GPT model = load_stories(path, tok, rng);
    std::vector<int> ctx = tok.encode(prompt.empty() ? std::string("Once upon a time") : prompt);
    if (ctx.empty()) ctx.push_back(0);
    std::printf("%s%s\n", prompt.c_str(), generate(model, ctx, 300, 0.8, rng, tok.id2str).c_str());
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "train";
    if (mode == "train") {
        if (argc < 3) { std::fprintf(stderr, "usage: psi_stories train <data> <steps> [vocab] [d] [layers] [block] [hidden]\n"); return 1; }
        std::string data = argv[2];
        int steps = (argc > 3) ? std::atoi(argv[3]) : 2000;
        int vocab = (argc > 4) ? std::atoi(argv[4]) : 1024;
        int d = (argc > 5) ? std::atoi(argv[5]) : 128;
        int layers = (argc > 6) ? std::atoi(argv[6]) : 5;
        int block = (argc > 7) ? std::atoi(argv[7]) : 128;
        int hidden = (argc > 8) ? std::atoi(argv[8]) : 384;
        return cmd_train(data, steps, vocab, d, layers, block, hidden);
    }
    if (mode == "eval") {
        if (argc < 3) { std::fprintf(stderr, "usage: psi_stories eval <model.bin> [prompts] [temp]\n"); return 1; }
        std::string pf = (argc > 3) ? argv[3] : "eval/tinystories_prompts.txt";
        real temp = (argc > 4) ? (real)std::atof(argv[4]) : 0.8;
        return cmd_eval(argv[2], pf, temp);
    }
    if (mode == "gen") {
        if (argc < 3) { std::fprintf(stderr, "usage: psi_stories gen <model.bin> [prompt]\n"); return 1; }
        return cmd_gen(argv[2], (argc > 3) ? argv[3] : "");
    }
    std::fprintf(stderr, "usage: psi_stories (train <data> <steps> [vocab d layers block hidden] | eval <model> [prompts] [temp] | gen <model> [prompt])\n");
    return 1;
}

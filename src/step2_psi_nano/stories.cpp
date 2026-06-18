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
#include "model_stories.hpp"   // ModernGPT: GQA + RoPE + SwiGLU + block-weight-sharing

using namespace psi;

static const char* MODEL_PATH = "psi_stories.bin";

// --- BPE-aware checkpoint: magic, Config (8 ints), the tokenizer (base+merges), then params in order. ---
static void save_stories(const std::string& path, ModernGPT& model, const BPETokenizer& tok) {
    std::ofstream f(path, std::ios::binary);
    f.write("PSST", 4);
    auto& c = model.cfg;
    int cfg[8] = {c.vocab, c.d_model, c.n_layers, c.block, c.hidden, c.n_heads, c.n_kv_heads, c.n_unique};
    f.write(reinterpret_cast<char*>(cfg), sizeof(cfg));
    tok.save(f);
    for (auto& p : model.params()) { auto& d = p.data(); f.write(reinterpret_cast<char*>(d.data()), (std::streamsize)(d.size() * sizeof(real))); }
}
static ModernGPT load_stories(const std::string& path, BPETokenizer& tok, std::mt19937& rng) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4]; f.read(magic, 4);
    if (std::string(magic, 4) != "PSST") throw std::runtime_error("bad magic in " + path);
    int cfg[8]; f.read(reinterpret_cast<char*>(cfg), sizeof(cfg));
    Config c{cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7]};
    tok.load(f);
    ModernGPT model(c, rng);
    for (auto& p : model.params()) { auto& d = p.data(); f.read(reinterpret_cast<char*>(d.data()), (std::streamsize)(d.size() * sizeof(real))); }
    return model;
}

// Warmup-Stable-Decay LR schedule (MiniCPM): linear warmup -> constant -> linear decay to 0.1·lr in
// the last 20%. Better than flat/cosine for SLMs and enables stable-phase checkpoint reuse.
static real wsd_lr(int step, int steps, real lr) {
    int warm = std::max(50, steps / 33);          // ~3% warmup
    int decay_start = (steps * 4) / 5;            // last 20% decays
    if (step < warm) return lr * (real)step / warm;
    if (step < decay_start) return lr;
    real frac = (real)(steps - step) / std::max(1, steps - decay_start);   // 1 -> 0
    return lr * (0.1 + 0.9 * frac);               // decay to 0.1·lr
}

static int cmd_train(const std::string& datafile, int steps, int vocab, int d, int layers,
                     int block, int hidden, int heads, int nkv, int nuniq) {
    std::string text = read_file(datafile);
    if (text.empty()) { std::fprintf(stderr, "error: empty/unreadable data (%s)\n", datafile.c_str()); return 1; }
    if (d % heads != 0) { std::fprintf(stderr, "error: d_model %d not divisible by n_heads %d\n", d, heads); return 1; }
    if (nkv > 0 && heads % nkv != 0) { std::fprintf(stderr, "error: n_heads %d not divisible by n_kv %d\n", heads, nkv); return 1; }

    std::printf("fitting BPE (vocab=%d) ...\n", vocab); std::fflush(stdout);
    BPETokenizer tok;
    tok.fit(text, vocab);
    std::vector<int> ids = tok.encode(text);
    Dataset ds(ids, 0.1);

    Config cfg{tok.vocab(), d, layers, block, hidden, heads, nkv, nuniq};
    std::mt19937 rng(1234);
    ModernGPT model(cfg, rng);
    AdamW opt(model.params());
    const int batch = 8;
    const real lr = 1e-3;

    int nparams = 0; for (auto& p : model.params()) nparams += p.numel();
    std::printf("psi-stories | vocab=%d d=%d layers=%d(uniq=%d) heads=%d/kv%d ctx=%d hid=%d(SwiGLU) RoPE  "
                "tokens=%zu (%.2f c/tok)  train=%zu val=%zu  params=%d\n",
                tok.vocab(), d, layers, model.n_uniq, heads, model.n_kv, block, hidden,
                ids.size(), (double)text.size() / ids.size(), ds.train.size(), ds.val.size(), nparams);
    std::fflush(stdout);
    if ((int)ds.train.size() < cfg.block + 2) { std::fprintf(stderr, "error: corpus too small\n"); return 1; }

    std::uniform_int_distribution<int> pick(0, (int)ds.train.size() - cfg.block - 2);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step <= steps; ++step) {
        // Gradient accumulation: forward+backward ONE sequence at a time, accumulating grads into the
        // params, then step. Holds only one forward graph in memory at a time (~batch× less peak than
        // building all batch graphs then one backward) — essential on the 8GB M1. Same effective batch.
        opt.zero_grad();
        real lsum = 0;
        for (int b = 0; b < batch; ++b) {
            int i = pick(rng);
            std::vector<int> in(ds.train.begin() + i, ds.train.begin() + i + cfg.block);
            std::vector<int> tg(ds.train.begin() + i + 1, ds.train.begin() + i + 1 + cfg.block);
            Tensor l = scalar_mul(cross_entropy(model.forward(in), tg), 1.0 / batch);
            l.backward();                          // accumulates into param grads; graph freed at scope end
            lsum += l.data()[0];
        }
        opt.step(wsd_lr(step, steps, lr));

        if (step % 100 == 0) {
            double vl = (step % 1000 == 0) ? eval_loss(model, ds.val, cfg.block, 16) : 0.0;  // eval rarely (memory)
            double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::printf("step %5d   train %.4f   val %.4f   (%.1fs)\n", step, lsum, vl, el);
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
    ModernGPT model = load_stories(path, tok, rng);
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
    ModernGPT model = load_stories(path, tok, rng);
    std::vector<int> ctx = tok.encode(prompt.empty() ? std::string("Once upon a time") : prompt);
    if (ctx.empty()) ctx.push_back(0);
    std::printf("%s%s\n", prompt.c_str(), generate(model, ctx, 300, 0.8, rng, tok.id2str).c_str());
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "train";
    if (mode == "train") {
        if (argc < 3) { std::fprintf(stderr, "usage: psi_stories train <data> <steps> [vocab] [d] [layers] [block] [hidden] [heads] [n_kv] [n_unique]\n"); return 1; }
        std::string data = argv[2];
        int steps  = (argc > 3)  ? std::atoi(argv[3])  : 2000;
        int vocab  = (argc > 4)  ? std::atoi(argv[4])  : 1024;
        int d      = (argc > 5)  ? std::atoi(argv[5])  : 128;
        int layers = (argc > 6)  ? std::atoi(argv[6])  : 5;
        int block  = (argc > 7)  ? std::atoi(argv[7])  : 128;
        int hidden = (argc > 8)  ? std::atoi(argv[8])  : 384;
        int heads  = (argc > 9)  ? std::atoi(argv[9])  : 4;
        int nkv    = (argc > 10) ? std::atoi(argv[10]) : 0;   // 0 = MHA (n_kv = n_heads)
        int nuniq  = (argc > 11) ? std::atoi(argv[11]) : 0;   // 0 = no weight sharing
        return cmd_train(data, steps, vocab, d, layers, block, hidden, heads, nkv, nuniq);
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

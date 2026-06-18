// main.cpp — Psi training-framework CLI (psi-nano, the first zoo model).
//
//   psi_nano train [datafile] [steps]   train (file or embedded corpus), report train+val loss,
//                                        save psi_model.bin, print samples
//   psi_nano gen   <model.bin> [prompt]  load a checkpoint and generate from a prompt
//   psi_nano chat  [model.bin]           interactive prompt (loads a model, or trains embedded first)
//
// The split into config / tokenizer / data / checkpoint is what turns the old hardcoded script
// into a reusable framework: same code trains psi-stories, psi-chess, … — just different data.

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "checkpoint.hpp"
#include "data.hpp"
#include "model.hpp"
#include "tokenizer.hpp"

using namespace psi;

// Embedded fallback corpus (used when no data file is given) — keeps `train` working anywhere.
static const std::string CORPUS =
    "psi is a small language model. it learns to predict the next character in a "
    "sequence. the model is built from scratch with a custom autograd engine. every "
    "operation knows how to compute its own gradient. we train the model on a tiny "
    "corpus of text. the loss goes down as the model learns. when the loss is low, the "
    "model can generate text that looks like the data. this is only a prototype. later "
    "we will make it fast with custom kernels and run it on a powerful gpu. for now it "
    "runs slowly on a laptop. the goal is to understand every part of the system. small "
    "models can still be smart. we measure quality per bit. the smallest model that is "
    "still clever wins. one character at a time, the model learns the shape of language. ";

static const char* MODEL_PATH = "psi_model.bin";

static bool is_number(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (!std::isdigit((unsigned char)c)) return false;
    return true;
}

// Run training and save a checkpoint.
static int cmd_train(const std::string& datafile, int steps) {
    std::string text = datafile.empty() ? CORPUS : read_file(datafile);
    if (text.empty()) { std::fprintf(stderr, "error: empty/unreadable data (%s)\n", datafile.c_str()); return 1; }

    CharTokenizer tok;
    tok.fit(text);
    Dataset ds(tok.encode(text), 0.1);

    Config cfg{tok.vocab(), /*d*/ 64, /*layers*/ 2, /*block*/ 32, /*hidden*/ 256};
    std::mt19937 rng(1234);
    GPT model(cfg, rng);
    AdamW opt(model.params());
    const int batch = 8;
    const real lr = 1e-3;

    int nparams = 0;
    for (auto& p : model.params()) nparams += p.numel();
    std::printf("psi-nano | source=%s  chars=%zu  vocab=%d  train=%zu val=%zu tokens  params=%d\n",
                datafile.empty() ? "embedded" : datafile.c_str(), text.size(), tok.vocab(),
                ds.train.size(), ds.val.size(), nparams);
    if ((int)ds.train.size() < cfg.block + 2) { std::fprintf(stderr, "error: corpus too small\n"); return 1; }

    std::uniform_int_distribution<int> pick(0, (int)ds.train.size() - cfg.block - 2);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step <= steps; ++step) {
        std::vector<Tensor> losses;
        for (int b = 0; b < batch; ++b) {
            int i = pick(rng);
            std::vector<int> ids(ds.train.begin() + i, ds.train.begin() + i + cfg.block);
            std::vector<int> tgt(ds.train.begin() + i + 1, ds.train.begin() + i + 1 + cfg.block);
            losses.push_back(cross_entropy(model.forward(ids), tgt));
        }
        Tensor loss = losses[0];
        for (size_t k = 1; k < losses.size(); ++k) loss = add(loss, losses[k]);
        loss = scalar_mul(loss, 1.0 / batch);

        opt.zero_grad();
        loss.backward();
        opt.step(lr);

        if (step % 100 == 0) {
            double vl = eval_loss(model, ds.val, cfg.block);
            double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::printf("step %5d   train %.4f   val %.4f   (%.1fs)\n", step, loss.data()[0], vl, el);
        }
    }

    save_checkpoint(MODEL_PATH, model, tok);
    std::printf("saved -> %s\n\nsamples:\n", MODEL_PATH);
    std::vector<int> seed = tok.encode("the ");
    for (int k = 0; k < 3; ++k)
        std::printf("  \"the %s\"\n", generate(model, seed, 160, 0.7, rng, tok.id2ch).c_str());
    return 0;
}

static int cmd_gen(const std::string& path, const std::string& prompt) {
    std::mt19937 rng(0);
    CharTokenizer tok;
    GPT model = load_checkpoint(path, tok, rng);
    std::vector<int> ctx = tok.encode(prompt.empty() ? std::string(" ") : prompt);
    if (ctx.empty()) ctx.push_back(0);
    std::printf("%s%s\n", prompt.c_str(), generate(model, ctx, 300, 0.7, rng, tok.id2ch).c_str());
    return 0;
}

// Capability eval: generate a completion for each story-opening prompt (see docs/EVAL.md).
// The completions are graded by a strong model (Claude) against the rubric — this is the bar that
// the capability-per-bit search shrinks against.
static int cmd_eval(const std::string& path, const std::string& promptsfile, real temp) {
    std::mt19937 rng(0);
    CharTokenizer tok;
    GPT model = load_checkpoint(path, tok, rng);
    std::string content = read_file(promptsfile);
    if (content.empty()) { std::fprintf(stderr, "error: no prompts (%s)\n", promptsfile.c_str()); return 1; }
    std::istringstream iss(content);
    std::string line;
    int idx = 1;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        std::vector<int> ctx = tok.encode(line);
        if (ctx.empty()) ctx.push_back(0);
        std::string comp = generate(model, ctx, 220, temp, rng, tok.id2ch);
        std::printf("=== prompt %d ===\n%s  ┃>>>┃  %s\n\n", idx++, line.c_str(), comp.c_str());
    }
    return 0;
}

static int cmd_chat(GPT& model, CharTokenizer& tok, std::mt19937& rng) {
    std::printf("\n[interactive] type a seed (chars from the training vocab); 'quit' to exit.\n> ");
    std::fflush(stdout);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit" || line == "exit") break;
        std::vector<int> ctx = tok.encode(line);
        if (ctx.empty()) ctx.push_back(0);
        std::printf("%s%s\n> ", line.c_str(), generate(model, ctx, 200, 0.7, rng, tok.id2ch).c_str());
        std::fflush(stdout);
    }
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "train";

    if (mode == "train") {
        std::string datafile;
        int steps = 2000;
        if (argc > 2) {
            std::string a2 = argv[2];
            if (is_number(a2)) steps = std::atoi(a2.c_str());
            else { datafile = a2; if (argc > 3) steps = std::atoi(argv[3]); }
        }
        return cmd_train(datafile, steps);
    }
    if (mode == "gen") {
        if (argc < 3) { std::fprintf(stderr, "usage: psi_nano gen <model.bin> [prompt]\n"); return 1; }
        std::string prompt = (argc > 3) ? argv[3] : "";
        return cmd_gen(argv[2], prompt);
    }
    if (mode == "eval") {
        if (argc < 3) { std::fprintf(stderr, "usage: psi_nano eval <model.bin> [prompts.txt] [temp]\n"); return 1; }
        std::string pf = (argc > 3) ? argv[3] : "eval/tinystories_prompts.txt";
        real temp = (argc > 4) ? (real)std::atof(argv[4]) : 0.8;
        return cmd_eval(argv[2], pf, temp);
    }
    if (mode == "chat") {
        std::mt19937 rng(0);
        CharTokenizer tok;
        if (argc > 2) {                          // load a saved model
            GPT model = load_checkpoint(argv[2], tok, rng);
            return cmd_chat(model, tok, rng);
        }
        tok.fit(CORPUS);                         // else quick-train on the embedded corpus
        Config cfg{tok.vocab(), 64, 2, 32, 256};
        GPT model(cfg, rng);
        AdamW opt(model.params());
        Dataset ds(tok.encode(CORPUS), 0.1);
        std::uniform_int_distribution<int> pick(0, (int)ds.train.size() - cfg.block - 2);
        std::printf("training on embedded corpus (~20s)...\n");
        for (int step = 0; step <= 2000; ++step) {
            std::vector<Tensor> losses;
            for (int b = 0; b < 8; ++b) {
                int i = pick(rng);
                std::vector<int> ids(ds.train.begin() + i, ds.train.begin() + i + cfg.block);
                std::vector<int> tgt(ds.train.begin() + i + 1, ds.train.begin() + i + 1 + cfg.block);
                losses.push_back(cross_entropy(model.forward(ids), tgt));
            }
            Tensor loss = losses[0];
            for (size_t k = 1; k < losses.size(); ++k) loss = add(loss, losses[k]);
            loss = scalar_mul(loss, 1.0 / 8);
            opt.zero_grad(); loss.backward(); opt.step(1e-3);
        }
        return cmd_chat(model, tok, rng);
    }

    std::fprintf(stderr, "usage: psi_nano (train [data] [steps] | gen <model> [prompt] | eval <model> [prompts] [temp] | chat [model])\n");
    return 1;
}

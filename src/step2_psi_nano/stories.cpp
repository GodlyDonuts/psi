// stories.cpp — psi-stories: the sub-1M TinyStories model. Same autograd / ops / GPT as psi-nano,
// but with the small-BPE tokenizer (bpe.hpp) so the tiny parameter budget goes to the transformer,
// not to spelling. Goal: the smallest model (params, then bits) that clears the TinyStories bar
// (docs/EVAL.md) — smartest-per-param. This is the record-campaign build (docs/RECORD_CAMPAIGN.md).
//
//   stories tokenize <data> <vocab> <out_prefix>              fit BPE once -> out_prefix.tok + out_prefix.ids
//   stories train    <data> <steps> [vocab d layers block hidden heads n_kv n_unique] [flags]
//   stories eval     <model.bin> [prompts] [temp] [flags]     completions for grading
//   stories gen      <model.bin> [prompt]
//
// train flags:  --batch N  --lr X  --clip X  --out PATH  --save-every N  --resume PATH
//               --ids FILE --tok FILE (use a prebuilt tokenize cache; skips fit/encode)  --val FILE
// eval  flags:  --k N (completions/prompt)  --seed N  --nnew N
//
// Build (CUDA):   bash src/step4_cuda/build_cuda.sh
// Build (Metal):  clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \
//                   src/step2_psi_nano/stories.cpp src/step3_metal/metal_backend.mm \
//                   -framework Metal -framework Foundation -o psi_stories

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <sys/resource.h>   // getrusage — peak RSS, the long-run memory-growth canary

#include "bpe.hpp"
#include "data.hpp"
#include "model_stories.hpp"   // ModernGPT: GQA + RoPE + SwiGLU + block-weight-sharing

using namespace psi;

// Peak resident-set size in MB (getrusage high-water mark). Linux reports KB, macOS bytes.
static double peak_rss_mb() {
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return ru.ru_maxrss / (1024.0 * 1024.0);
#else
    return ru.ru_maxrss / 1024.0;
#endif
}

// ---------------------------------------------------------------------------------------------------
// Checkpoint (PSS2): magic, Config(8), planned_steps, step, opt.t, rng(len-prefixed), tokenizer,
// params, AdamW m, AdamW v.  Full training state, so a preempted run resumes bit-identically.
// ---------------------------------------------------------------------------------------------------
static void save_ckpt(const std::string& path, ModernGPT& model, const BPETokenizer& tok,
                      AdamW& opt, int step, int planned, std::mt19937& rng) {
    std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) { std::fprintf(stderr, "warn: cannot open %s for write\n", tmp.c_str()); return; }
    f.write("PSS2", 4);
    auto& c = model.cfg;
    int cfg[8] = {c.vocab, c.d_model, c.n_layers, c.block, c.hidden, c.n_heads, c.n_kv_heads, c.n_unique};
    f.write(reinterpret_cast<char*>(cfg), sizeof(cfg));
    f.write(reinterpret_cast<char*>(&planned), 4);
    f.write(reinterpret_cast<char*>(&step), 4);
    f.write(reinterpret_cast<char*>(&opt.t), 4);
    std::ostringstream ss; ss << rng; std::string rs = ss.str();    // mt19937 state as a length-prefixed text blob
    int rlen = (int)rs.size(); f.write(reinterpret_cast<char*>(&rlen), 4); f.write(rs.data(), rlen);
    tok.save(f);
    for (auto& p : model.params()) { auto& d = p.data(); f.write(reinterpret_cast<char*>(d.data()), (std::streamsize)(d.size() * sizeof(real))); }
    for (auto& mk : opt.m) f.write(reinterpret_cast<char*>(mk.data()), (std::streamsize)(mk.size() * sizeof(real)));
    for (auto& vk : opt.v) f.write(reinterpret_cast<char*>(vk.data()), (std::streamsize)(vk.size() * sizeof(real)));
    f.flush();
    if (!f.good()) { std::fprintf(stderr, "warn: write error on %s (checkpoint NOT committed)\n", tmp.c_str()); return; }
    f.close();
    std::rename(tmp.c_str(), path.c_str());                          // atomic replace: readers never see a half-file
}

struct Ckpt { ModernGPT model; int planned, step, opt_t; std::vector<std::vector<real>> m, v; };

static Ckpt load_ckpt(const std::string& path, BPETokenizer& tok, std::mt19937& rng) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4]; f.read(magic, 4);
    if (std::string(magic, 4) != "PSS2") throw std::runtime_error("bad magic in " + path + " (need PSS2)");
    int cfg[8]; f.read(reinterpret_cast<char*>(cfg), sizeof(cfg));
    Config c{cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7]};
    int planned, step, t; f.read(reinterpret_cast<char*>(&planned), 4); f.read(reinterpret_cast<char*>(&step), 4); f.read(reinterpret_cast<char*>(&t), 4);
    int rlen; f.read(reinterpret_cast<char*>(&rlen), 4); std::string rs(rlen, '\0'); f.read(&rs[0], rlen);
    { std::istringstream ss(rs); ss >> rng; }                        // restore the TRAINING rng
    tok.load(f);
    std::mt19937 init_rng(0);                                        // throwaway: params are overwritten below
    ModernGPT model(c, init_rng);
    for (auto& p : model.params()) { auto& d = p.data(); f.read(reinterpret_cast<char*>(d.data()), (std::streamsize)(d.size() * sizeof(real))); }
    std::vector<std::vector<real>> m, v;
    for (auto& p : model.params()) { std::vector<real> b(p.numel()); f.read(reinterpret_cast<char*>(b.data()), (std::streamsize)(b.size() * sizeof(real))); m.push_back(std::move(b)); }
    for (auto& p : model.params()) { std::vector<real> b(p.numel()); f.read(reinterpret_cast<char*>(b.data()), (std::streamsize)(b.size() * sizeof(real))); v.push_back(std::move(b)); }
    if (!f) throw std::runtime_error("truncated/corrupt checkpoint: " + path);
    return Ckpt{std::move(model), planned, step, t, std::move(m), std::move(v)};
}

// Weights + tokenizer only (eval/gen): rng is a throwaway.
static ModernGPT load_weights(const std::string& path, BPETokenizer& tok) {
    std::mt19937 junk(0);
    Ckpt ck = load_ckpt(path, tok, junk);
    return std::move(ck.model);
}

// Warmup-Stable-Decay LR (MiniCPM): ~3% linear warmup -> constant -> last 20% linear decay to 0.1·lr.
// A function of the PLANNED step count (stored in the checkpoint) so the schedule is stable across resume.
static real wsd_lr(int step, int steps, real lr) {
    int warm = std::max(50, steps / 33);
    int decay_start = (steps * 4) / 5;
    if (step < warm) return lr * (real)(step + 1) / warm;           // +1: step 0 already trains (no wasted lr=0 step)
    if (step < decay_start) return lr;
    real frac = (real)(steps - step) / std::max(1, steps - decay_start);
    return lr * (0.1 + 0.9 * frac);
}

// int32 id-stream cache (from `tokenize`): count then the ids.
static void write_ids(const std::string& path, const std::vector<int>& ids) {
    std::ofstream f(path, std::ios::binary);
    int n = (int)ids.size(); f.write(reinterpret_cast<char*>(&n), 4);
    f.write(reinterpret_cast<const char*>(ids.data()), (std::streamsize)((size_t)n * sizeof(int)));
}
static std::vector<int> read_ids(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open ids cache " + path);
    int n = 0; f.read(reinterpret_cast<char*>(&n), 4);
    std::vector<int> ids(n); f.read(reinterpret_cast<char*>(ids.data()), (std::streamsize)((size_t)n * sizeof(int)));
    if (!f) throw std::runtime_error("truncated ids cache " + path);
    return ids;
}
static void save_tok(const std::string& path, const BPETokenizer& tok) { std::ofstream f(path, std::ios::binary); tok.save(f); }
static void load_tok(const std::string& path, BPETokenizer& tok) { std::ifstream f(path, std::ios::binary); if (!f) throw std::runtime_error("cannot open tok " + path); tok.load(f); }

// ---------------------------------------------------------------------------------------------------
static int cmd_tokenize(const std::string& datafile, int vocab, const std::string& out_prefix) {
    std::string text = read_file(datafile);
    if (text.empty()) { std::fprintf(stderr, "error: empty/unreadable data (%s)\n", datafile.c_str()); return 1; }
    std::printf("fitting BPE (vocab=%d) on %zu bytes ...\n", vocab, text.size()); std::fflush(stdout);
    auto t0 = std::chrono::high_resolution_clock::now();
    BPETokenizer tok; tok.fit(text, vocab);
    std::vector<int> ids = tok.encode_stream(text);
    double dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    long neos = 0; for (int x : ids) if (x == tok.eos_id()) ++neos;
    save_tok(out_prefix + ".tok", tok);
    write_ids(out_prefix + ".ids", ids);
    std::printf("tokenized: vocab=%d eos_id=%d  tokens=%zu (%.3f chars/tok)  stories=%ld  %.1fs\n"
                "  -> %s.tok  %s.ids\n",
                tok.vocab(), tok.eos_id(), ids.size(), (double)text.size() / ids.size(), neos + 1, dt,
                out_prefix.c_str(), out_prefix.c_str());
    return 0;
}

struct TrainOpts {
    int batch = 32; real lr = 2e-3; real clip = 1.0; int save_every = 1000; int stop_at = -1; int log_every = 100;
    std::string out = "psi_stories.bin", resume, ids_cache, tok_cache, val_file;
};

static int cmd_train(const std::string& datafile, int steps, int vocab, int d, int layers,
                     int block, int hidden, int heads, int nkv, int nuniq, const TrainOpts& o) {
    BPETokenizer tok;
    std::vector<int> ids;

    // --- tokens: from a prebuilt cache (fast, deterministic, shared across jobs) or fit fresh ---
    if (!o.ids_cache.empty() || !o.tok_cache.empty()) {
        if (o.ids_cache.empty() || o.tok_cache.empty()) { std::fprintf(stderr, "error: --ids and --tok must be given together\n"); return 1; }
        load_tok(o.tok_cache, tok);
        ids = read_ids(o.ids_cache);
        std::printf("loaded tokenize cache: vocab=%d eos=%d tokens=%zu\n", tok.vocab(), tok.eos_id(), ids.size());
    } else {
        std::string text = read_file(datafile);
        if (text.empty()) { std::fprintf(stderr, "error: empty/unreadable data (%s)\n", datafile.c_str()); return 1; }
        std::printf("fitting BPE (vocab=%d) ...\n", vocab); std::fflush(stdout);
        tok.fit(text, vocab);
        ids = tok.encode_stream(text);
    }

    if (d % heads != 0) { std::fprintf(stderr, "error: d_model %d not divisible by n_heads %d\n", d, heads); return 1; }
    if (nkv > 0 && heads % nkv != 0) { std::fprintf(stderr, "error: n_heads %d not divisible by n_kv %d\n", heads, nkv); return 1; }

    // --- train/val tokens ---
    Dataset ds(ids, o.val_file.empty() ? 0.1 : 0.0);
    std::vector<int> val = ds.val;
    if (!o.val_file.empty()) {                                       // dedicated held-out file (no split)
        std::string vt = read_file(o.val_file);
        val = tok.encode_stream(vt);
        ds.train = ids;                                             // train on all of it
    }
    if ((int)ds.train.size() < block + 2) { std::fprintf(stderr, "error: corpus too small\n"); return 1; }

    // --- model + optimizer, fresh or resumed ---
    std::mt19937 rng(1234);
    int planned = steps, start = 0;
    real lr = o.lr;
    std::unique_ptr<ModernGPT> model_holder;
    std::vector<std::vector<real>> resume_m, resume_v; int resume_t = 0;
    if (!o.resume.empty()) {
        Ckpt ck = load_ckpt(o.resume, tok, rng);                   // restores model, opt state, step, planned, rng
        planned = ck.planned; start = ck.step; resume_t = ck.opt_t;
        resume_m = std::move(ck.m); resume_v = std::move(ck.v);
        model_holder = std::make_unique<ModernGPT>(std::move(ck.model));
        std::printf("RESUMED from %s at step %d/%d\n", o.resume.c_str(), start, planned);
    } else {
        model_holder = std::make_unique<ModernGPT>(Config{tok.vocab(), d, layers, block, hidden, heads, nkv, nuniq}, rng);
    }
    ModernGPT& model = *model_holder;
    AdamW opt(model.params());
    if (!o.resume.empty()) { opt.m = std::move(resume_m); opt.v = std::move(resume_v); opt.t = resume_t; }

    // --- story-boundary-aware sampling: window starts snap to story beginnings (pos 0 or right after eos),
    //     so a training sequence is one story from position 0 — matching how generation runs. ---
    std::vector<int> starts;
    int eos = tok.eos_id();
    for (int i = 0; i + block + 1 <= (int)ds.train.size(); ++i)
        if (i == 0 || (eos >= 0 && ds.train[i - 1] == eos)) starts.push_back(i);
    bool boundary = starts.size() >= 32;                            // fall back to uniform if no eos structure
    std::uniform_int_distribution<int> pick_uniform(0, std::max(0, (int)ds.train.size() - block - 2));
    std::uniform_int_distribution<int> pick_start(0, std::max(0, (int)starts.size() - 1));
    auto sample_start = [&](std::mt19937& r) { return boundary ? starts[pick_start(r)] : pick_uniform(r); };

    int nparams = 0; for (auto& p : model.params()) nparams += p.numel();
    std::printf("psi-stories | vocab=%d d=%d layers=%d(uniq=%d) heads=%d/kv%d ctx=%d hid=%d(SwiGLU) RoPE  "
                "params=%d  batch=%d lr=%.1e clip=%.1f  tokens/step=%d  planned=%d\n"
                "  train=%zu val=%zu  stories(train)=%zu  sampling=%s  out=%s\n",
                tok.vocab(), d, layers, model.n_uniq, heads, model.n_kv, block, hidden, nparams,
                o.batch, lr, o.clip, o.batch * block, planned,
                ds.train.size(), val.size(), starts.size(), boundary ? "story-boundary" : "uniform", o.out.c_str());
    std::fflush(stdout);

    auto t0 = std::chrono::high_resolution_clock::now();
    int done = start;
    for (int step = start; step < planned && (o.stop_at < 0 || step < o.stop_at); ++step) {
        opt.zero_grad();
        real lsum = 0;
        for (int b = 0; b < o.batch; ++b) {
            int i = sample_start(rng);
            std::vector<int> in(ds.train.begin() + i, ds.train.begin() + i + block);
            std::vector<int> tg(ds.train.begin() + i + 1, ds.train.begin() + i + 1 + block);
            Tensor l = scalar_mul(cross_entropy(model.forward(in), tg), 1.0 / o.batch);
            l.backward();                                          // accumulates grads; graph freed at scope end
            lsum += l.data()[0];
        }
        if (!std::isfinite(lsum)) {                                 // NaN/Inf guard: stop rather than burn hours on garbage
            std::fprintf(stderr, "FATAL: non-finite loss %.4g at step %d — stopping (last checkpoint: %s)\n", (double)lsum, step, o.out.c_str());
            return 2;
        }
        real gnorm = clip_grad_global_norm(opt.p, o.clip);
        opt.step(wsd_lr(step, planned, lr));

        if (step % o.log_every == 0) {
            double vl = (step % 1000 == 0) ? eval_loss(model, val, block, 64) : -1.0;
            double el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            if (vl >= 0) std::printf("step %6d   train %.4f   val %.4f   |g|=%.2f  rss=%.0fMB  (%.1fs)\n", step, lsum, vl, gnorm, peak_rss_mb(), el);
            else         std::printf("step %6d   train %.4f   val   —    |g|=%.2f  rss=%.0fMB  (%.1fs)\n", step, lsum, gnorm, peak_rss_mb(), el);
            std::fflush(stdout);
        }
        done = step + 1;
        if (o.save_every > 0 && step > start && step % o.save_every == 0) save_ckpt(o.out, model, tok, opt, done, planned, rng);
    }
    save_ckpt(o.out, model, tok, opt, done, planned, rng);          // `done` = true reached step (may be < planned if --stop-at)
    if (done < planned) { std::printf("stopped early at step %d/%d -> %s (resume to continue)\n", done, planned, o.out.c_str()); return 0; }
    std::printf("saved -> %s\n\nsample:\n", o.out.c_str());
    std::vector<int> seed = tok.encode("Once upon a time");
    std::mt19937 grng(7);
    std::printf("  Once upon a time%s\n", generate(model, seed, 240, 0.8, grng, tok.id2str, tok.eos_id()).c_str());
    return 0;
}

static int cmd_eval(const std::string& path, const std::string& promptsfile, real temp, int k, int seed, int nnew) {
    BPETokenizer tok;
    ModernGPT model = load_weights(path, tok);
    std::string content = read_file(promptsfile);
    if (content.empty()) { std::fprintf(stderr, "error: no prompts (%s)\n", promptsfile.c_str()); return 1; }
    std::istringstream iss(content);
    std::string line; int idx = 1;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        std::vector<int> ctx = tok.encode(line);
        if (ctx.empty()) ctx.push_back(0);
        for (int c = 0; c < k; ++c) {
            std::mt19937 rng(seed + c);                            // fixed seed per completion — reproducible grading
            std::string comp = generate(model, ctx, nnew, temp, rng, tok.id2str, tok.eos_id());
            std::printf("=== prompt %d / completion %d (temp=%.2f seed=%d) ===\n%s  ┃>>>┃  %s\n\n", idx, c, (double)temp, seed + c, line.c_str(), comp.c_str());
        }
        ++idx;
    }
    return 0;
}

static int cmd_gen(const std::string& path, const std::string& prompt) {
    BPETokenizer tok;
    ModernGPT model = load_weights(path, tok);
    std::mt19937 rng(0);
    std::vector<int> ctx = tok.encode(prompt.empty() ? std::string("Once upon a time") : prompt);
    if (ctx.empty()) ctx.push_back(0);
    std::printf("%s%s\n", prompt.c_str(), generate(model, ctx, 300, 0.8, rng, tok.id2str, tok.eos_id()).c_str());
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "train";
    // parse argv[2..] into positionals + --flags (flag value = next non---token, else "1")
    std::vector<std::string> pos; std::map<std::string, std::string> fl;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
            std::string k = a.substr(2), v = "1";
            if (i + 1 < argc) { std::string nx = argv[i + 1]; if (!(nx.size() > 2 && nx[0] == '-' && nx[1] == '-')) v = argv[++i]; }
            fl[k] = v;
        } else pos.push_back(a);
    }
    auto P  = [&](size_t i, const char* d) { return i < pos.size() ? pos[i] : std::string(d); };
    auto Pi = [&](size_t i, int d) { return i < pos.size() ? std::atoi(pos[i].c_str()) : d; };
    auto F  = [&](const char* k, const char* d) { auto it = fl.find(k); return it != fl.end() ? it->second : std::string(d); };
    auto Fi = [&](const char* k, int d) { auto it = fl.find(k); return it != fl.end() ? std::atoi(it->second.c_str()) : d; };
    auto Fr = [&](const char* k, real d) { auto it = fl.find(k); return it != fl.end() ? (real)std::atof(it->second.c_str()) : d; };

    if (mode == "tokenize") {
        if (pos.size() < 3) { std::fprintf(stderr, "usage: psi_stories tokenize <data> <vocab> <out_prefix>\n"); return 1; }
        return cmd_tokenize(P(0, ""), Pi(1, 1024), P(2, "tok"));
    }
    if (mode == "train") {
        if (pos.empty()) { std::fprintf(stderr, "usage: psi_stories train <data> <steps> [vocab d layers block hidden heads n_kv n_unique] [--batch --lr --clip --out --save-every --resume --ids --tok --val]\n"); return 1; }
        TrainOpts o;
        o.batch = Fi("batch", 32); o.lr = Fr("lr", 2e-3); o.clip = Fr("clip", 1.0);
        o.save_every = Fi("save-every", 1000); o.stop_at = Fi("stop-at", -1); o.log_every = Fi("log-every", 100);
        o.out = F("out", "psi_stories.bin"); o.resume = F("resume", "");
        o.ids_cache = F("ids", ""); o.tok_cache = F("tok", ""); o.val_file = F("val", "");
        return cmd_train(P(0, ""), Pi(1, 2000), Pi(2, 1024), Pi(3, 128), Pi(4, 5), Pi(5, 128),
                         Pi(6, 384), Pi(7, 4), Pi(8, 0), Pi(9, 0), o);
    }
    if (mode == "eval") {
        if (pos.empty()) { std::fprintf(stderr, "usage: psi_stories eval <model.bin> [prompts] [temp] [--k --seed --nnew]\n"); return 1; }
        return cmd_eval(P(0, ""), P(1, "eval/tinystories_prompts.txt"), (real)std::atof(P(2, "0.7").c_str()),
                        Fi("k", 1), Fi("seed", 0), Fi("nnew", 256));
    }
    if (mode == "gen") {
        if (pos.empty()) { std::fprintf(stderr, "usage: psi_stories gen <model.bin> [prompt]\n"); return 1; }
        return cmd_gen(P(0, ""), P(1, ""));
    }
    std::fprintf(stderr, "usage: psi_stories (tokenize|train|eval|gen) ...\n");
    return 1;
}

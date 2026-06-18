// bpe_test.cpp — validate the small-BPE tokenizer: fit on a slice of TinyStories, report vocab +
// compression (chars/token) + a lossless round-trip, and show the longest learned word-pieces.
// Light CPU job (fits on a few-MB slice). Build:
//   clang++ -std=c++17 -O2 src/step2_psi_nano/bpe_test.cpp -o bpe_test && ./bpe_test

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include "bpe.hpp"

using namespace psi;

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "data/tinystories-valid.txt";
    int target = (argc > 2) ? std::atoi(argv[2]) : 1024;
    size_t slice = (argc > 3) ? (size_t)std::atoll(argv[3]) : (3u << 20);   // default 3 MB

    std::ifstream f(path, std::ios::binary);
    std::string text(slice, '\0');
    f.read(&text[0], slice);
    text.resize((size_t)f.gcount());
    if (text.empty()) { std::printf("could not read %s\n", path); return 1; }

    BPETokenizer tok;
    tok.fit(text, target);

    auto ids = tok.encode(text);
    int base = 0; for (auto& kv : tok.base_id) (void)kv, ++base;
    std::printf("BPE fit on %.1f MB slice of %s\n", text.size() / 1048576.0, path);
    std::printf("  base chars   : %d\n", base);
    std::printf("  final vocab  : %d  (target %d)\n", tok.vocab(), target);
    std::printf("  chars        : %zu\n", text.size());
    std::printf("  tokens       : %zu\n", ids.size());
    std::printf("  compression  : %.2f chars/token  (vs 1.0 for char-level)\n",
                (double)text.size() / ids.size());

    // round-trip on a sample
    std::string sample = "Once upon a time, there was a little girl named Lily who loved to play.";
    std::string rt = tok.decode(tok.encode(sample));
    std::printf("  round-trip   : %s\n", rt == sample ? "OK (lossless)" : "MISMATCH");
    std::printf("  sample tokens: ");
    for (int id : tok.encode(sample)) std::printf("[%s]", tok.id2str[id].c_str());
    std::printf("\n");

    // show the longest merged word-pieces (what BPE actually learned)
    std::vector<std::string> toks = tok.id2str;
    std::sort(toks.begin(), toks.end(), [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    std::printf("  longest pieces: ");
    for (int i = 0; i < 20 && i < (int)toks.size(); ++i) std::printf("[%s]", toks[i].c_str());
    std::printf("\n");
    return 0;
}

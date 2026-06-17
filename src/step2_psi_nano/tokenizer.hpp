// tokenizer.hpp — pluggable tokenizer (char-level for now).
//
// Part of framework-ization: the vocab is no longer baked into main. A byte/BPE tokenizer
// can implement the same fit/encode/decode interface later for the `psi-stories` model.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace psi {

struct CharTokenizer {
    std::vector<char> id2ch;                 // id -> char (also doubles as the decode table)
    std::unordered_map<char, int> ch2id;     // char -> id

    void fit(const std::string& text) {      // build the vocab from a corpus
        id2ch.clear();
        ch2id.clear();
        for (char c : text)
            if (!ch2id.count(c)) { ch2id[c] = (int)id2ch.size(); id2ch.push_back(c); }
    }
    int vocab() const { return (int)id2ch.size(); }

    std::vector<int> encode(const std::string& s) const {   // unknown chars are skipped
        std::vector<int> ids;
        ids.reserve(s.size());
        for (char c : s) { auto it = ch2id.find(c); if (it != ch2id.end()) ids.push_back(it->second); }
        return ids;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string s;
        s.reserve(ids.size());
        for (int i : ids) if (i >= 0 && i < vocab()) s.push_back(id2ch[i]);
        return s;
    }
};

}  // namespace psi

// bpe.hpp — a SMALL byte-pair-encoding tokenizer, for sub-1M models.
//
// At sub-1M params the embedding table (vocab x d) dominates the budget, so the usual ~32k vocab is
// impossible — it would *be* the whole model. This learns a small vocab (~512-1024) of word-pieces:
// enough that each token carries more than a character (capability-per-param), cheap enough to leave
// the parameter budget for actual transformer layers. Same fit/encode/decode/vocab interface as
// CharTokenizer, plus id2str for multi-char decode.
//
// Standard BPE: pre-tokenize into space-attached words (merges never cross word boundaries), then
// iteratively merge the most frequent adjacent pair until the target vocab is reached.

#pragma once

#include <climits>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace psi {

struct BPETokenizer {
    std::vector<std::string> id2str;                  // id -> token string (decode table)
    std::unordered_map<std::string, int> base_id;     // single-char -> base id
    std::unordered_map<int64_t, int> rank_;           // merge pair -> rank (lower = higher priority)
    std::unordered_map<int64_t, int> merged_;         // merge pair -> resulting id

    static int64_t key(int a, int b) { return ((int64_t)a << 21) | (int64_t)b; }
    int vocab() const { return (int)id2str.size(); }

    // Lossless pre-tokenization: each whitespace char begins a new word (and is part of it), so
    // concatenating all words rebuilds the text exactly, and merges stay within a word.
    static std::vector<std::string> pretokenize(const std::string& text) {
        std::vector<std::string> words;
        std::string cur;
        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t') { if (!cur.empty()) words.push_back(cur); cur = std::string(1, c); }
            else cur += c;
        }
        if (!cur.empty()) words.push_back(cur);
        return words;
    }

    std::vector<int> to_base(const std::string& w) const {       // chars -> base ids (unknown skipped)
        std::vector<int> v; v.reserve(w.size());
        for (char c : w) { auto it = base_id.find(std::string(1, c)); if (it != base_id.end()) v.push_back(it->second); }
        return v;
    }

    void fit(const std::string& text, int target_vocab) {
        id2str.clear(); base_id.clear(); rank_.clear(); merged_.clear();
        for (char c : text) { std::string s(1, c); if (!base_id.count(s)) { base_id[s] = (int)id2str.size(); id2str.push_back(s); } }

        std::unordered_map<std::string, int> wf;                 // unique words + counts
        for (auto& w : pretokenize(text)) wf[w]++;
        std::vector<std::vector<int>> words; std::vector<long> freq;
        words.reserve(wf.size()); freq.reserve(wf.size());
        for (auto& kv : wf) { words.push_back(to_base(kv.first)); freq.push_back(kv.second); }

        while ((int)id2str.size() < target_vocab) {
            std::unordered_map<int64_t, long> pc;                // pair counts (weighted by word freq)
            pc.reserve(1 << 16);
            for (size_t i = 0; i < words.size(); ++i) {
                auto& w = words[i]; long f = freq[i];
                for (size_t j = 0; j + 1 < w.size(); ++j) pc[key(w[j], w[j + 1])] += f;
            }
            if (pc.empty()) break;
            int64_t best = -1; long bc = 0;
            for (auto& kv : pc) if (kv.second > bc) { bc = kv.second; best = kv.first; }
            if (best < 0) break;
            int a = (int)(best >> 21), b = (int)(best & 0x1FFFFF);
            int nid = (int)id2str.size();
            id2str.push_back(id2str[a] + id2str[b]);
            rank_[best] = (int)rank_.size(); merged_[best] = nid;
            for (auto& w : words) {                              // apply the merge everywhere
                if (w.size() < 2) continue;
                std::vector<int> nw; nw.reserve(w.size());
                for (size_t j = 0; j < w.size();) {
                    if (j + 1 < w.size() && w[j] == a && w[j + 1] == b) { nw.push_back(nid); j += 2; }
                    else { nw.push_back(w[j]); ++j; }
                }
                w.swap(nw);
            }
        }
    }

    std::vector<int> encode_word(std::vector<int> w) const {     // greedily apply lowest-rank merge
        while (w.size() >= 2) {
            int br = INT_MAX; size_t bi = 0; bool found = false;
            for (size_t j = 0; j + 1 < w.size(); ++j) {
                auto it = rank_.find(key(w[j], w[j + 1]));
                if (it != rank_.end() && it->second < br) { br = it->second; bi = j; found = true; }
            }
            if (!found) break;
            w[bi] = merged_.at(key(w[bi], w[bi + 1]));
            w.erase(w.begin() + bi + 1);
        }
        return w;
    }

    std::vector<int> encode(const std::string& s) const {
        std::vector<int> out;
        for (auto& w : pretokenize(s)) { auto e = encode_word(to_base(w)); out.insert(out.end(), e.begin(), e.end()); }
        return out;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string s; for (int i : ids) if (i >= 0 && i < vocab()) s += id2str[i]; return s;
    }
};

}  // namespace psi

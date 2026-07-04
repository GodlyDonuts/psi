// bpe.hpp - a SMALL byte-pair-encoding tokenizer, for sub-1M models.
//
// At sub-1M params the embedding table (vocab x d) dominates the budget, so the usual ~32k vocab is
// impossible - it would *be* the whole model. This learns a small vocab (~512-1024) of word-pieces:
// enough that each token carries more than a character (capability-per-param), cheap enough to leave
// the parameter budget for actual transformer layers. Same fit/encode/decode/vocab interface as
// CharTokenizer, plus id2str for multi-char decode.
//
// Standard BPE: pre-tokenize into space-attached words (merges never cross word boundaries), then
// iteratively merge the most frequent adjacent pair until the target vocab is reached.

#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace psi {

struct BPETokenizer {
    std::vector<std::string> id2str;                  // id -> token string (decode table)
    std::unordered_map<std::string, int> base_id;     // single-char -> base id
    std::unordered_map<int64_t, int> rank_;           // merge pair -> rank (lower = higher priority)
    std::unordered_map<int64_t, int> merged_;         // merge pair -> resulting id
    int eos = -1;                                     // dedicated end-of-story id (never emitted by encode; injected at story boundaries)
    static constexpr const char* EOS_STR = "<|endoftext|>";

    static int64_t key(int a, int b) { return ((int64_t)a << 21) | (int64_t)b; }
    int vocab() const { return (int)id2str.size(); }
    int eos_id() const { return eos; }

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

    // Split text on the literal "<|endoftext|>" separator into per-story chunks (separators dropped).
    static std::vector<std::string> split_on_eos(const std::string& text) {
        std::vector<std::string> out;
        const std::string sep = EOS_STR;
        size_t pos = 0;
        while (true) {
            size_t hit = text.find(sep, pos);
            if (hit == std::string::npos) { out.push_back(text.substr(pos)); break; }
            out.push_back(text.substr(pos, hit - pos));
            pos = hit + sep.size();
        }
        return out;
    }

    std::vector<int> to_base(const std::string& w) const {       // chars -> base ids (unknown skipped)
        std::vector<int> v; v.reserve(w.size());
        for (char c : w) { auto it = base_id.find(std::string(1, c)); if (it != base_id.end()) v.push_back(it->second); }
        return v;
    }

    void fit(const std::string& text, int target_vocab) {
        id2str.clear(); base_id.clear(); rank_.clear(); merged_.clear();
        for (char c : text) { std::string s(1, c); if (!base_id.count(s)) { base_id[s] = (int)id2str.size(); id2str.push_back(s); } }
        // Reserve a dedicated EOS id right after the base chars (before any merge). It is NOT in base_id,
        // so encode() never produces it from text - it's injected only at <|endoftext|> story boundaries.
        eos = (int)id2str.size(); id2str.push_back(EOS_STR);

        // Count words WITHOUT letting the literal "<|endoftext|>" separator become BPE vocabulary:
        // strip it first so merges are learned from real story text only.
        std::unordered_map<std::string, int> wf;                 // unique words + counts
        for (auto& chunk : split_on_eos(text))
            for (auto& w : pretokenize(chunk)) wf[w]++;
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
            // Deterministic argmax: highest count, ties broken by SMALLEST key - so an identical corpus
            // yields an identical tokenizer regardless of hash-map iteration order (libstdc++ vs libc++).
            int64_t best = -1; long bc = 0;
            for (auto& kv : pc) if (kv.second > bc || (kv.second == bc && kv.first < best)) { bc = kv.second; best = kv.first; }
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
    // Encode a full corpus WITH in-band EOS: encode each story, join with the eos id between stories.
    // This is the training/eval tokenization - the model learns story boundaries as a real token.
    std::vector<int> encode_stream(const std::string& text) const {
        std::vector<int> out;
        auto chunks = split_on_eos(text);
        for (size_t c = 0; c < chunks.size(); ++c) {
            if (c > 0 && eos >= 0) out.push_back(eos);              // separator between stories
            for (auto& w : pretokenize(chunks[c])) { auto e = encode_word(to_base(w)); out.insert(out.end(), e.begin(), e.end()); }
        }
        return out;
    }
    std::string decode(const std::vector<int>& ids) const {
        std::string s; for (int i : ids) if (i >= 0 && i < vocab()) s += id2str[i]; return s;
    }
    std::string token(int id) const { return (id >= 0 && id < vocab()) ? id2str[id] : std::string(); }

    // Serialize: base chars (ids 0..nbase-1, all length-1), an EOS flag (its id is implicitly nbase),
    // then the merge pairs (a,b) in id order. Fully reconstructs id2str + merge tables + eos.
    void save(std::ostream& o) const {
        int nb = (int)base_id.size();
        o.write(reinterpret_cast<const char*>(&nb), 4);
        for (int i = 0; i < nb; ++i) o.write(id2str[i].data(), 1);   // base tokens are single chars
        int has_eos = (eos >= 0) ? 1 : 0;                            // eos sits at id == nb (right after base)
        o.write(reinterpret_cast<const char*>(&has_eos), 4);
        int nm = (int)merged_.size();
        o.write(reinterpret_cast<const char*>(&nm), 4);
        std::vector<std::pair<int, std::pair<int,int>>> mv;          // (nid, (a,b))
        mv.reserve(merged_.size());
        for (auto& kv : merged_) mv.push_back({kv.second, {(int)(kv.first >> 21), (int)(kv.first & 0x1FFFFF)}});
        std::sort(mv.begin(), mv.end());                            // by nid == merge order
        for (auto& m : mv) { int a = m.second.first, b = m.second.second; o.write((char*)&a, 4); o.write((char*)&b, 4); }
    }
    void load(std::istream& in) {
        id2str.clear(); base_id.clear(); rank_.clear(); merged_.clear(); eos = -1;
        int nb; in.read(reinterpret_cast<char*>(&nb), 4);
        for (int i = 0; i < nb; ++i) { char c; in.read(&c, 1); std::string s(1, c); base_id[s] = (int)id2str.size(); id2str.push_back(s); }
        int has_eos; in.read(reinterpret_cast<char*>(&has_eos), 4);
        if (has_eos) { eos = (int)id2str.size(); id2str.push_back(EOS_STR); }
        int nm; in.read(reinterpret_cast<char*>(&nm), 4);
        for (int i = 0; i < nm; ++i) {
            int a, b; in.read((char*)&a, 4); in.read((char*)&b, 4);
            int nid = (int)id2str.size();
            id2str.push_back(id2str[a] + id2str[b]);
            int64_t k = key(a, b); rank_[k] = (int)rank_.size(); merged_[k] = nid;
        }
    }
};

}  // namespace psi

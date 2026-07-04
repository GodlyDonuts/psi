#!/usr/bin/env python3
# make_slice.py - build the shared training slice for the TinyStories record campaign.
#
# Cuts a target-size slice from TinyStoriesV2-GPT4-train, on <|endoftext|> story boundaries, with:
#   - exact story-level dedup (drops repeated stories),
#   - exclusion of every story that appears in the held-out valid file (contamination guard),
#   - a SHA256 manifest for reproducibility.
# All campaign jobs train on this one file, so results are comparable and the tokenizer is fixed.
#
#   python3 tools/make_slice.py <train.txt> <valid.txt> <out.txt> [target_mb=500]
import sys, hashlib

SEP = "<|endoftext|>"

def stories(path):
    # stream the file in chunks, yielding one story (text between separators) at a time.
    # Never holds more than a chunk + one pending story in memory (the full file is >2GB).
    buf = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(1 << 20)   # 1 MB
            if not chunk:
                break
            buf += chunk
            while SEP in buf:
                s, buf = buf.split(SEP, 1)
                s2 = s.strip()
                if s2:
                    yield s2
    s2 = buf.strip()
    if s2:
        yield s2

def norm_hash(s):
    return hashlib.sha256(" ".join(s.split()).encode()).hexdigest()  # whitespace-normalized

def main():
    train, valid, out = sys.argv[1], sys.argv[2], sys.argv[3]
    target = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    target_bytes = target * 1024 * 1024

    sys.stderr.write(f"hashing valid stories from {valid} ...\n")
    valid_set = set(norm_hash(s) for s in stories(valid))
    sys.stderr.write(f"  {len(valid_set)} unique valid stories (excluded from slice)\n")

    seen = set()
    kept = 0
    dropped_dup = 0
    dropped_valid = 0
    size = 0
    with open(out, "w", encoding="utf-8") as o:
        for s in stories(train):
            h = norm_hash(s)
            if h in valid_set:
                dropped_valid += 1
                continue
            if h in seen:
                dropped_dup += 1
                continue
            seen.add(h)
            o.write(s)
            o.write(SEP)
            size += len(s.encode()) + len(SEP)
            kept += 1
            if size >= target_bytes:
                break

    manifest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    sys.stderr.write(
        f"wrote {out}: {size/1e6:.1f} MB, {kept} stories "
        f"(dropped {dropped_dup} dup, {dropped_valid} valid-overlap)\n"
        f"SHA256={manifest}\n")
    with open(out + ".manifest", "w") as m:
        m.write(f"file={out}\nbytes={size}\nstories={kept}\nsha256={manifest}\n"
                f"dropped_dup={dropped_dup}\ndropped_valid_overlap={dropped_valid}\n"
                f"source={train}\nexcluded={valid}\n")

if __name__ == "__main__":
    main()

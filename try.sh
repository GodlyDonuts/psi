#!/usr/bin/env bash
# try.sh - run one of the trained TinyStories models on a prompt.
#   bash try.sh                 list the models you can run
#   bash try.sh small           generate from a default opening
#   bash try.sh small "The dragon woke up and"
#
# Build the binary first (Apple Silicon):
#   clang++ -std=c++17 -O3 -march=native -ffast-math -DPSI_REAL=float \
#     src/step2_psi_nano/stories.cpp src/step3_metal/metal_backend.mm \
#     -framework Metal -framework Foundation -o psi_stories
set -uo pipefail
cd "$(dirname "$0")"

BIN=./psi_stories
[ -x "$BIN" ] || BIN=./psi_stories_cuda
[ -x "$BIN" ] || { echo "no psi_stories binary found. Build it with the clang++ command at the top of this file."; exit 1; }

if [ $# -eq 0 ]; then
  echo "models you can run:"
  for d in models/*/; do [ -f "$d/model.bin" ] && echo "  $(basename "$d")"; done
  echo "usage: bash try.sh <model> [prompt]"
  exit 0
fi

name="$1"
m="models/$name/model.bin"
if [ ! -f "$m" ]; then
  echo "no model called '$name'. Available: $(for d in models/*/; do [ -f "$d/model.bin" ] && basename "$d"; done | tr '\n' ' ')"
  exit 1
fi

prompt="${2:-Once upon a time, there was a little girl who}"
echo "$name: $prompt"
PSI_THREADS=1 "$BIN" gen "$m" "$prompt"

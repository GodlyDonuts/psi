#!/usr/bin/env bash
# try.sh — run a finished psi model interactively (CPU, no GPU needed — safe on the login node).
#   bash try.sh <model> [prompt]
#   bash try.sh chess_nano "e2e4 e7e5"       # chess: continue a game (UCI moves)
#   bash try.sh femto "Once upon a time"     # stories
#   bash try.sh math_small "128+457="        # math: answer prints LSB-first (read right-to-left)
#   bash try.sh nano_gen_1m "<qa> Q: what is the opposite of hot? A:"   # generalist
# List finished models:  bash try.sh
set -uo pipefail
cd "$(dirname "$0")"
BIN=./psi_stories_cuda
[ -x "$BIN" ] || { echo "binary missing — build it: bash src/step4_cuda/build_cuda.sh"; exit 1; }

if [ $# -eq 0 ]; then
  echo "finished models (have model.bin):"
  for d in models/*/; do [ -f "$d/model.bin" ] && echo "  $(basename "$d")"; done
  echo "usage: bash try.sh <model> [prompt]"; exit 0
fi

name="$1"; m="models/$name/model.bin"
[ -f "$m" ] || { echo "model '$name' not finished yet (no $m). Finished: $(for d in models/*/; do [ -f "$d/model.bin" ] && basename "$d"; done | tr '\n' ' ')"; exit 1; }

case "$name" in
  chess*)     def="e2e4 e7e5 g1f3 b8c6" ;;
  math*)      def="128+457=" ;;
  nano_gen*)  def="<story> Once upon a time" ;;
  *)          def="Once upon a time" ;;
esac
prompt="${2:-$def}"
echo "=== $name  |  prompt: $prompt ==="
PSI_CUDA_DISABLE=1 PSI_THREADS=1 "$BIN" gen "$m" "$prompt"   # CPU, single-thread: works on the login node

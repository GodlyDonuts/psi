#!/usr/bin/env bash
# submit_campaign.sh — launch the sub-1M TinyStories record sweep (bracket-and-extend).
# Step counts are finalized from preflight timing (each job kept safely < its walltime). Long jobs go on
# 'normal' (infinite walltime, no preemption). Record candidates + the insurance headline start first.
#   bash src/step4_cuda/submit_campaign.sh          # submit all
#   bash src/step4_cuda/submit_campaign.sh femto    # submit one by name
set -euo pipefail
cd "$HOME/Psi"
[ -s data/s512.ids ] && [ -s data/s1024.ids ] || { echo "tokenize caches missing — run preflight first"; exit 1; }

# name        arch (vocab d layers blk hid heads kv uniq)   steps   lr      ids    partition  walltime
# walltime caps = (steps × measured V100 s/step) + build/eval overhead + 15% margin; 'normal' is
# infinite-walltime so these are just safety caps. Measured s/step (preflight 677601): femto 2.67,
# nano 3.19, small 3.58, mid 2.79, flagship 3.96, insurance 4.35. RSS flat ~1.7GB (no leak on Linux).
CONFIGS=(
"insurance | 1024 160 8 256 384 4 2 4 | 26000 | 0.0015 | s1024 | normal | 42:00:00"
"flagship  | 1024 128 8 256 384 4 2 4 | 26000 | 0.002  | s1024 | normal | 38:00:00"
"small     | 512 96 9 256 256 4 2 3   | 20000 | 0.0025 | s512  | normal | 27:00:00"
"mid       | 1024 128 6 256 256 4 2 3 | 25000 | 0.002  | s1024 | normal | 26:00:00"
"nano      | 512 96 8 256 192 4 2 2   | 16000 | 0.0025 | s512  | normal | 19:00:00"
"femto     | 512 64 8 256 160 4 1 2   | 14000 | 0.003  | s512  | normal | 14:00:00"
)
only="${1:-}"
for row in "${CONFIGS[@]}"; do
  IFS='|' read -r name arch steps lr ids part wall <<< "$row"
  name=$(echo "$name" | xargs); arch=$(echo "$arch" | xargs); steps=$(echo "$steps" | xargs)
  lr=$(echo "$lr" | xargs); ids=$(echo "$ids" | xargs); part=$(echo "$part" | xargs); wall=$(echo "$wall" | xargs)
  [ -n "$only" ] && [ "$only" != "$name" ] && continue
  echo "submit $name  ($arch | $steps steps | lr $lr | $ids | $part $wall)"
  sbatch -J "psi_$name" -p "$part" -t "$wall" src/step4_cuda/train_config.sbatch "$name" "$arch" "$steps" "$lr" "$ids"
done
echo; squeue -u sa305415 -o "%.10i %.14j %.2t %.10M %R"

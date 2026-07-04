#!/usr/bin/env bash
# status.sh - one-glance campaign status (run on the cluster, or via: nssh 'bash ~/Psi/src/step4_cuda/status.sh')
cd "$HOME/Psi" 2>/dev/null || cd ~
echo "=== queue ($(date '+%H:%M:%S')) ==="
squeue -u sa305415 -o "%.10i %.14j %.2t %.10M %.12l %.20R" 2>/dev/null
echo "=== per-model progress (latest step / loss / grad-norm / rss) ==="
for d in models/*/; do
  [ -f "$d/train.txt" ] || continue
  n=$(basename "$d")
  params=$(grep -o 'params=[0-9]*' "$d/train.txt" | head -1 | cut -d= -f2)
  planned=$(grep -o 'planned=[0-9]*' "$d/train.txt" | head -1 | cut -d= -f2)
  last=$(grep '^step' "$d/train.txt" | tail -1)
  done_mark=""; grep -q "DONE" "$d/train.txt" 2>/dev/null && done_mark=" [saved]"
  grep -q "clears bar" "$d/MODEL.md" 2>/dev/null && :
  printf "  %-10s params=%-8s planned=%-6s  %s%s\n" "$n" "${params:-?}" "${planned:-?}" "$last" "$done_mark"
done
echo "=== recent NaN/errors ==="
grep -l -i "FATAL\|BUILD_FAIL\|non-finite" models/*/train.txt results/*.log 2>/dev/null | head || echo "  none"

#!/usr/bin/env bash
#SBATCH --job-name=psi_preflight
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=results/%x_%j.log
set -uo pipefail
cd ~/Psi; mkdir -p results; module load cuda/cuda-12.6.0 2>/dev/null
echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name --format=csv,noheader
# GPU health gate (some H100 nodes are left busy)
cat > /tmp/probe_$$.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main(){void*p;if(cudaMalloc(&p,1<<20)){printf("BUSY\n");return 1;}printf("OK\n");return 0;}
EOF
nvcc -o /tmp/probe_$$ /tmp/probe_$$.cu 2>/dev/null && /tmp/probe_$$ | grep -q OK || { echo "GPU BUSY - resubmit"; exit 1; }

echo; echo "########## 1. BUILD ##########"
bash src/step4_cuda/build_cuda.sh || { echo BUILD_FAIL; exit 1; }

echo; echo "########## 2. TOKENIZE (shared caches, timed) ##########"
/usr/bin/time -v ./psi_stories_cuda tokenize data/slice500.txt 512  data/s512  2>&1 | grep -E "tokenized|Maximum resident|Elapsed"
/usr/bin/time -v ./psi_stories_cuda tokenize data/slice500.txt 1024 data/s1024 2>&1 | grep -E "tokenized|Maximum resident|Elapsed"

echo; echo "########## 3. CONTAMINATION (12 eval prompts verbatim in slice?) ##########"
n=0; while IFS= read -r line; do [ -z "$line" ] && continue; c=$(grep -Fc -- "$line" data/slice500.txt || true); echo "  [$c] ${line:0:50}"; n=$((n+c)); done < eval/tinystories_prompts.txt
echo "  total verbatim prompt matches in slice: $n"

echo; echo "########## 4. RESUME-EQUIVALENCE on Linux (CPU path, bit-exact) ##########"
CFG="512 32 2 16 64 4 2 1 --ids data/s512.ids --tok data/s512.tok --batch 8 --lr 2e-3 --save-every 5000"
PSI_CUDA_DISABLE=1 ./psi_stories_cuda train data/slice500.txt 20 $CFG --out /tmp/a_$$.bin  >/dev/null 2>&1
PSI_CUDA_DISABLE=1 ./psi_stories_cuda train data/slice500.txt 20 $CFG --out /tmp/b_$$.bin  --stop-at 10 >/dev/null 2>&1
PSI_CUDA_DISABLE=1 ./psi_stories_cuda train data/slice500.txt 20 $CFG --out /tmp/b2_$$.bin --resume /tmp/b_$$.bin >/dev/null 2>&1
if cmp -s /tmp/a_$$.bin /tmp/b2_$$.bin; then echo "  RESUME BIT-IDENTICAL on Linux: PASS"; else echo "  RESUME MISMATCH: FAIL"; fi
rm -f /tmp/a_$$.bin /tmp/b_$$.bin /tmp/b2_$$.bin

echo; echo "########## 5. PER-CONFIG TIMING + MEMORY PROBES (60 steps each, GPU cuBLAS) ##########"
probe () { # name  ids  "vocab d layers blk hid heads kv uniq"  lr
  local name=$1 ids=$2 arch=$3 lr=$4
  echo "--- $name ---"
  PSI_CUDA_KERNEL=cublas ./psi_stories_cuda train data/slice500.txt 60 $arch \
     --ids data/$ids.ids --tok data/$ids.tok --batch 32 --lr $lr --save-every 100000 --log-every 20 --out /tmp/probe_$$_$name.bin 2>&1 \
     | grep -E "params=|^step"
  rm -f /tmp/probe_$$_$name.bin
}
probe femto     s512  "512 64 8 256 160 4 1 2"  0.003
probe nano      s512  "512 96 8 256 192 4 2 2"  0.0025
probe small     s512  "512 96 9 256 256 4 2 3"  0.0025
probe mid       s1024 "1024 128 6 256 256 4 2 3" 0.002
probe flagship  s1024 "1024 128 8 256 384 4 2 4" 0.002
probe insurance s1024 "1024 160 8 256 384 4 2 4" 0.0015
echo; echo "=== PREFLIGHT DONE ==="

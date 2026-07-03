#!/usr/bin/env bash
# psi_cuda.sbatch — build + train psi-stories on a UCF Newton GPU node (CUDA backend).
#   sbatch src/step4_cuda/psi_cuda.sbatch [STEPS]
#
# Accessible GPUs on the arcc_pi_skattel account: `normal`/`preemptable` = H100 PCIe (80GB) + V100.
# The `highgpu` (8xH100-80GB) and `short`/`ucfit` (H200) partitions need a separate account grant.
#
# NOTE: some H100 PCIe nodes get left in a "CUDA device busy/unavailable" state by prior jobs
# (persistence mode is off). This job smoke-tests the assigned GPU and bails early if it's bad —
# just resubmit (or pin a healthy node with `-w evcNN`; V100 nodes have been reliable).
#SBATCH --job-name=psi_cuda
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=results/%x_%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd ~/Psi
mkdir -p results
module load cuda/cuda-12.6.0 2>/dev/null

echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# health gate: a 1-line CUDA probe; if the GPU is in a bad state, exit so you can resubmit.
cat > /tmp/psi_probe_$$.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main(){ void* p; if(cudaMalloc(&p,1<<20)!=cudaSuccess){printf("GPU_BUSY\n");return 1;} printf("GPU_OK\n"); return 0; }
EOF
nvcc -o /tmp/psi_probe_$$ /tmp/psi_probe_$$.cu 2>/dev/null && /tmp/psi_probe_$$ | grep -q GPU_OK \
  || { echo "assigned GPU is busy/unavailable — resubmit or pin a healthy node with -w"; exit 1; }

bash src/step4_cuda/build_cuda.sh || { echo "BUILD FAILED"; exit 1; }

STEPS=${1:-4000}
DATA=data/tinystories-train-100mb.txt
# mid_570k config — one of the sizes that got memory-capped on the 8GB Mac; trains freely here.
CFG="train $DATA $STEPS 1024 128 6 64 256 4 2 3"
echo "=== train (H100/V100, cuBLAS): $CFG ==="
PSI_CUDA_KERNEL=cublas ./psi_stories_cuda $CFG
echo "=== eval ==="
./psi_stories_cuda eval psi_stories.bin eval/tinystories_prompts.txt 0.7 || true
echo "=== DONE ==="

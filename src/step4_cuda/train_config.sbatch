#!/usr/bin/env bash
# train_config.sbatch - train ONE sweep config into a reproducible models/<name>/ folder.
#   sbatch [-p normal] [-t HH:MM:SS] train_config.sbatch <name> "<vocab d layers blk hid heads kv uniq>" <steps> <lr> <s512|s1024>
# Collision-safe (per-config --out), self-building into a private dir (no .o races between concurrent
# jobs), auto-resumes its own checkpoint on requeue/restart, archives the exact binary (completions are
# binary-specific under -march=native/-ffast-math).
#SBATCH --job-name=psi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --output=results/%x_%j.log
set -uo pipefail
NAME=$1; ARCH=$2; STEPS=$3; LR=$4; IDS=$5
cd "${SLURM_SUBMIT_DIR:-$HOME/Psi}"
mkdir -p results "models/$NAME" build
module load cuda/cuda-12.6.0 2>/dev/null
echo "=== $NAME on $(hostname)  $(date)  job $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- GPU health gate: some H100 nodes are left busy; requeue onto another node if so ---
cat > /tmp/pb_$SLURM_JOB_ID.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main(){void*p;if(cudaMalloc(&p,1<<20)){return 1;}return 0;}
EOF
if ! ( nvcc -o /tmp/pb_$SLURM_JOB_ID /tmp/pb_$SLURM_JOB_ID.cu 2>/dev/null && /tmp/pb_$SLURM_JOB_ID ); then
  echo "GPU busy/unavailable on $(hostname) - requeuing"; scontrol requeue "$SLURM_JOB_ID"; sleep 30; exit 1
fi
rm -f /tmp/pb_$SLURM_JOB_ID /tmp/pb_$SLURM_JOB_ID.cu

# --- per-job build (unique object dir => no races with concurrent jobs; -march=native safe on-node) ---
BD="build/$SLURM_JOB_ID"; mkdir -p "$BD"
GEN="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_90,code=sm_90"
g++ -std=c++17 -O3 -march=native -ffast-math -pthread -DPSI_REAL=float -c src/step2_psi_nano/stories.cpp -o "$BD/stories.o" || { echo BUILD_FAIL; exit 1; }
nvcc -O3 $GEN -c src/step4_cuda/cuda_backend.cu -o "$BD/cuda_backend.o" || { echo BUILD_FAIL; exit 1; }
g++ "$BD/stories.o" "$BD/cuda_backend.o" -o "$BD/psi_stories" -L"$CUDA_HOME/lib64" -lcudart -lcublas -pthread || { echo BUILD_FAIL; exit 1; }

GIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
MODEL="models/$NAME/model.bin"
RESUME=""; [ -f "$MODEL" ] && RESUME="--resume $MODEL" && echo "resuming existing checkpoint $MODEL"

# --- train (cuBLAS backend; checkpoints every 1000 steps, atomic) ---
echo "config: $ARCH  steps=$STEPS lr=$LR ids=$IDS"
PSI_CUDA_KERNEL=cublas "./$BD/psi_stories" train data/slice500.txt "$STEPS" $ARCH \
    --ids "data/$IDS.ids" --tok "data/$IDS.tok" --batch 32 --lr "$LR" --clip 1.0 \
    --save-every 1000 --log-every 100 --out "$MODEL" $RESUME 2>&1 | tee "models/$NAME/train.txt"
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && echo "WARNING: train exited $rc"

# --- eval under the frozen protocol (temp 0.7, k=3 completions, fixed seeds) ---
PSI_CUDA_KERNEL=cublas "./$BD/psi_stories" eval "$MODEL" eval/tinystories_prompts.txt 0.7 \
    --k 3 --seed 100 --nnew 256 2>&1 | tee "models/$NAME/eval.txt"

cp "$BD/psi_stories" "models/$NAME/psi_stories.exe"   # exact binary that produced eval.txt (archival)

# --- reproducible MODEL.md ---
PARAMS=$(grep -o 'params=[0-9]*' "models/$NAME/train.txt" | head -1 | cut -d= -f2)
CPT=$(grep -o '([0-9.]* c/tok)\|[0-9.]* chars/tok' "models/$NAME/train.txt" | head -1)
LAST=$(grep '^step' "models/$NAME/train.txt" | tail -1)
TOKS=$(( STEPS * 32 * 256 ))
cat > "models/$NAME/MODEL.md" <<EOF
# psi-stories / $NAME

Sweep config in the sub-1M **capability-per-bit** TinyStories record campaign
(see [docs/RECORD_CAMPAIGN.md](../../docs/RECORD_CAMPAIGN.md), [docs/EVAL.md](../../docs/EVAL.md)).

| field | value |
|---|---|
| parameters | **$PARAMS** |
| architecture (args) | \`$ARCH\` = vocab d layers block hidden heads n_kv n_unique |
| training | $STEPS steps · batch 32 · ctx 256 · AdamW lr=$LR wd 0.01 · WSD · grad-clip 1.0 |
| tokens seen | ~$TOKS ($(python3 -c "print(f'{$TOKS/max(1,$PARAMS):.0f}')" 2>/dev/null || echo ?) tok/param) |
| data | data/slice500.txt (500MB dedup, valid-excluded - see data/slice500.txt.manifest) |
| tokenizer | $IDS (BPE, in-band EOS, boundary-aware sampling) |
| code version | git \`$GIT\` |
| final | $LAST |

## Reproduce
\`\`\`sh
module load cuda/cuda-12.6.0 && bash src/step4_cuda/build_cuda.sh
./psi_stories_cuda train data/slice500.txt $STEPS $ARCH --ids data/$IDS.ids --tok data/$IDS.tok \\
   --batch 32 --lr $LR --clip 1.0 --out model.bin
./psi_stories_cuda eval model.bin eval/tinystories_prompts.txt 0.7 --k 3 --seed 100
\`\`\`

## Capability-bar grade (docs/EVAL.md v2 rubric, graded off-cluster)
_Filled after grading:_  Grammar - · Coherence - · Consistency - · Plot - · **clears bar? -**
EOF

rm -rf "$BD"
echo "=== $NAME DONE  $(date) ==="

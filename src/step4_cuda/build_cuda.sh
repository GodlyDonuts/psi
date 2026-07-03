#!/usr/bin/env bash
# build_cuda.sh — build the NVIDIA/CUDA training path (peer to the Metal build in run_overnight.sh).
#
# Same custom autograd + model; only the GPU backend differs (cuda_backend.cu instead of
# metal_backend.mm). Produces psi_stories_cuda (train/eval/gen) and cuda_backend_test (correctness+bench).
#
# Prereqs on UCF Newton:  module load cuda/cuda-12.6.0     (nvcc + libcublas + CUDA_HOME)
# Run this ON the GPU node you'll train on (-march=native), e.g. inside an srun/sbatch allocation,
# so the host ISA matches; the CUDA fatbin below already covers V100 (sm_70) and H100 (sm_90).
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
: "${CUDA_HOME:?load a cuda module first: module load cuda/cuda-12.6.0}"

ARCH="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_90,code=sm_90"

echo "compiling model (host C++) ..."
g++ -std=c++17 -O3 -march=native -ffast-math -pthread -DPSI_REAL=float \
    -c src/step2_psi_nano/stories.cpp -o stories.o

echo "compiling CUDA backend (nvcc, V100+H100 fatbin) ..."
nvcc -O3 $ARCH -c src/step4_cuda/cuda_backend.cu -o cuda_backend.o

echo "linking psi_stories_cuda ..."
g++ stories.o cuda_backend.o -o psi_stories_cuda -L"$CUDA_HOME/lib64" -lcudart -lcublas -pthread

echo "building cuda_backend_test (correctness + benchmark) ..."
nvcc -O3 $ARCH src/step4_cuda/cuda_backend_test.cu src/step4_cuda/cuda_backend.cu -lcublas -o cuda_backend_test

echo "done: psi_stories_cuda, cuda_backend_test"
echo "engine toggles:  PSI_CUDA_KERNEL=cublas|custom   PSI_CUDA_CHECK=1 (diff both)   PSI_CUDA_DISABLE=1 (force CPU)"

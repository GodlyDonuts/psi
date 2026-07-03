// metal_backend.h — the Metal implementation lives in metal_backend.mm.
//
// The backend interface was hoisted to src/gpu_backend.h so Metal and CUDA can both implement it
// behind one set of names (gpu_available / gpu_matmul / gpu_matmul_nt / gpu_matmul_tn). This header
// stays as a thin include so existing Metal-side includers keep compiling.

#pragma once
#include "../gpu_backend.h"

cmake_minimum_required(VERSION 3.16)
project(psi LANGUAGES CXX OBJCXX)   # OBJCXX for the Metal backend (.mm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3 -march=native -Wall -Wextra -pthread)
add_link_options(-pthread)

# Step 0 — scalar reverse-mode autograd + XOR MLP + finite-difference grad check.
add_executable(step0 src/step0_scalar_autograd/main.cpp)

# Step 1 — tensor autograd (CPU oracle): core ops, grad checks, tensor XOR.
add_executable(step1 src/step1_tensor_autograd/main.cpp)

# Step 2 — psi-nano: a tiny char-level GPT on the custom stack.
add_executable(step2_gradcheck src/step2_psi_nano/gradcheck.cpp)  # transformer-op grad checks
add_executable(psi_nano        src/step2_psi_nano/main.cpp src/step3_metal/metal_backend.mm)  # train + sample
target_compile_definitions(psi_nano PRIVATE PSI_REAL=float)        # training path runs float32 (+GPU matmul)
target_compile_options(psi_nano PRIVATE -ffast-math)              # ~2x; grad-check oracle stays strict
target_link_libraries(psi_nano "-framework Metal" "-framework Foundation")

# Benchmarks — the optimization scoreboard.
add_executable(matmul_bench src/bench/matmul_bench.cpp)
add_executable(psi_profile  src/bench/psi_profile.cpp src/step3_metal/metal_backend.mm)  # training-step profile
target_compile_definitions(psi_profile PRIVATE PSI_REAL=float)
target_compile_options(psi_profile PRIVATE -ffast-math)
target_link_libraries(psi_profile "-framework Metal" "-framework Foundation")

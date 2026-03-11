#!/bin/bash
# Quick build script for CARC (run this before sbatch)
module purge
module load cuda/12.2.0 gcc/11.3.0 openmpi/4.1.4 cmake/3.24.2

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70;80" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native"

make -j$(nproc)
echo "Build complete. Binary: build/poker_cuda"
echo "GPU info: ./build/poker_cuda --mode info"

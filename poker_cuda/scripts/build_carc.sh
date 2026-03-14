#!/bin/bash
# Quick build script for CARC (run this before sbatch)
module purge
module load ver/2506 gcc/14.3.0 openmpi/5.0.8 cmake/3.31.7 cuda/12.9.1
export LD_LIBRARY_PATH=$(dirname $(which nvcc))/../lib64:$LD_LIBRARY_PATH

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70;80" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native"

make -j$(nproc)
echo "Build complete. Binary: build/poker_cuda"
echo "GPU info: ./build/poker_cuda --mode info"

#!/bin/bash
# =============================================================================
# carc_build_test.sh -- Interactive build + smoke test on CARC
#
# Run this in an interactive GPU job:
#   salloc --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G
#   bash scripts/carc_build_test.sh
# =============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "============================================================"
echo " Step 1: Load ver/2506 first to unlock hierarchical modules"
echo "============================================================"
module purge
module load ver/2506
echo "ver/2506 loaded."

echo ""
echo "--- CUDA versions now available ---"
module spider cuda 2>&1 | grep -E "cuda/[0-9]"
echo ""
echo "--- GCC versions now available ---"
module spider gcc 2>&1 | grep -E "gcc/[0-9]"
echo ""
echo "--- OpenMPI versions now available ---"
module spider openmpi 2>&1 | grep -E "openmpi/[0-9]"
echo ""
echo "--- CMake versions now available ---"
module spider cmake 2>&1 | grep -E "cmake/[0-9]"
echo ""

echo "============================================================"
echo " Step 2: Load required modules"
echo "============================================================"

# GCC -- load before CUDA so nvcc uses the right host compiler
for v in gcc/14.3.0 gcc/13.3.0 gcc/12.3.0 gcc/11.3.0; do
    if module load $v 2>/dev/null; then echo "Loaded $v"; break; fi
done

# CUDA -- try 12.x first (A100 supports sm_80 from CUDA 11.1+)
for v in cuda/12.9.1 cuda/12.6.0 cuda/12.3.0 cuda/12.2.0 cuda/12.1.0 \
          cuda/12.0.0 cuda/11.8.0 cuda/11.7.0 cuda/11.6.2 cuda/11.5.1; do
    if module load $v 2>/dev/null; then echo "Loaded $v"; break; fi
done

# OpenMPI
for v in openmpi/5.0.8 openmpi/5.0.5 openmpi/5.0.4 openmpi/4.1.6 openmpi/4.1.5; do
    if module load $v 2>/dev/null; then echo "Loaded $v"; break; fi
done

# CMake
for v in cmake/3.31.7 cmake/3.29.4 cmake/3.27.1 cmake/3.23.2; do
    if module load $v 2>/dev/null; then echo "Loaded $v"; break; fi
done

echo ""
echo "--- Loaded module state ---"
module list 2>&1
echo ""

# Verify tools
echo "nvcc:  $(which nvcc 2>/dev/null || echo NOT FOUND)"
nvcc --version 2>/dev/null | head -2 || true
echo "gcc:   $(which gcc) ($(gcc --version | head -1))"
echo "mpicc: $(which mpicc 2>/dev/null || echo NOT FOUND)"
echo "cmake: $(which cmake 2>/dev/null || echo NOT FOUND) $(cmake --version 2>/dev/null | head -1 || true)"
echo ""

if ! which nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. CUDA module did not load."
    echo "Run 'module spider cuda' to see all available versions."
    exit 1
fi

if ! which cmake &>/dev/null; then
    echo "ERROR: cmake not found."
    exit 1
fi

# Point CMake at the right compiler
export CUDAHOSTCXX=$(which g++)
export LD_LIBRARY_PATH=$(dirname $(which nvcc))/../lib64:$LD_LIBRARY_PATH

echo "============================================================"
echo " Step 3: Build"
echo "============================================================"
cd "$PROJECT_DIR"
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80" \
    -DCMAKE_CUDA_HOST_COMPILER="$(which g++)"

make -j${SLURM_CPUS_PER_TASK:-8}
cd ..

echo ""
echo "============================================================"
echo " Step 4: Smoke tests"
echo "============================================================"
echo ""
echo "--- GPU Info ---"
./build/poker_cuda --mode info
echo ""
echo "--- Benchmark (batch=4096, iters=10, requires handranks.dat) ---"
./build/poker_cuda \
    --mode benchmark \
    --players 2 \
    --iters 10 \
    --batch 4096 \
    --stack 1000 --sb 10 --bb 20 \
    --handranks data/handranks.dat

echo ""
echo "All checks passed. Ready to submit:"
echo "  sbatch scripts/carc_train.sh"

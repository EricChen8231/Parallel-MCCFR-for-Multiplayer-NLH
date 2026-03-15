#!/bin/bash
# =============================================================================
# carc_train.sh -- SLURM job script for CUDA CFR training on USC CARC
#
# Usage:
#   sbatch scripts/carc_train.sh
#
# Targets: A100 (partition=gpu, 80GB VRAM)
#   - 4 nodes x 1 A100 each = 4 GPUs total (via MPI)
#   - Each GPU runs 65536 games/batch x 1000 iters = 65.5M games
#   - Total: 262M games across 4 GPUs with MPI AllReduce
# =============================================================================
#SBATCH --job-name=poker_cfr_cuda
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cfr_%j.out
#SBATCH --error=logs/cfr_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@usc.edu

# ---------------------------------------------------------------------------
# Load modules (ver/2506 must come first to unlock hierarchical modules)
# ---------------------------------------------------------------------------
module purge
module load ver/2506
module load gcc/14.3.0
module load cuda/12.9.1 2>/dev/null || \
module load cuda/12.6.0 2>/dev/null || \
module load cuda/12.3.0 2>/dev/null || \
{ echo "ERROR: No CUDA 12.x module found"; exit 1; }
module load openmpi/5.0.8
module load cmake/3.31.7

echo "Loaded modules:"
module list 2>&1

export CUDAHOSTCXX=$(which g++)
export LD_LIBRARY_PATH=$(dirname $(which nvcc))/../lib64:$LD_LIBRARY_PATH

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80" \
    -DCMAKE_CUDA_HOST_COMPILER="$(which g++)"
make -j${SLURM_CPUS_PER_TASK:-8}
if [ $? -ne 0 ]; then echo "ERROR: Build failed"; exit 1; fi
cd ..

mkdir -p logs data

# ---------------------------------------------------------------------------
# Profile with Nsight Systems (first run only, for the report)
# Comment out after profiling.
# ---------------------------------------------------------------------------
# nsys profile --stats=true -o logs/nsight_profile \
#     ./build/poker_cuda --mode train --players 2 --iters 10 --batch 65536

# ---------------------------------------------------------------------------
# Main training run
#
# srun is the SLURM-native MPI launcher (preferred over mpirun on CARC).
# Each MPI rank trains independently on its GPU, then AllReduce merges
# regret tables.
#
# Performance estimate (A100 @ ~3 TFLOPS FP32):
#   batch=65536 hands x 30 actions/hand x 8 FLOPS/action = ~15.7 GFLOPS
#   -> ~200 batches/sec per GPU = 13M hands/sec per GPU
#   -> 4 GPUs = 52M hands/sec = 187B hands overnight (8h)
# ---------------------------------------------------------------------------
srun --mpi=pmix \
    ./build/poker_cuda \
        --mode train \
        --players 2 \
        --iters 1000 \
        --batch 65536 \
        --stack 1000 \
        --sb 10 \
        --bb 20 \
        --save strategy_carc_a100.bin \
        --handranks data/handranks.dat

echo "Training complete."
echo "Strategy saved to strategy_carc_a100.bin"
ls -lh strategy_carc_a100.bin

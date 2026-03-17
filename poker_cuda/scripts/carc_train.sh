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
# Performance (OS-MCCFR on A100, measured):
#   batch=65536  -> ~15M hands/sec per GPU
#   batch=262144 -> ~22M hands/sec per GPU
#   4 GPUs x 22M x 8h = ~2.5 trillion hands overnight
#
# Player scaling:
#   --players 2  : heads-up (fastest convergence, ~8h for good strategy)
#   --players 3-4: medium convergence time
#   --players 6  : full table (needs more iters; increase --iters accordingly)
#
# Iteration guidance (at 22M hands/sec per GPU, 4 GPUs):
#   --iters 50000000  (~1h wall time)  good for testing
#   --iters 200000000 (~4h wall time)  reasonable strategy quality
#   --iters 500000000 (~8h wall time)  full overnight run
# ---------------------------------------------------------------------------
srun --mpi=pmix \
    ./build/poker_cuda \
        --mode train \
        --players 6 \
        --iters 200000 \
        --batch 262144 \
        --stack 1000 \
        --sb 10 \
        --bb 20 \
        --save strategy_carc_a100_6p.bin \
        --handranks data/handranks.dat

echo "Training complete."
echo "Strategy saved to strategy_carc_a100_6p.bin"
ls -lh strategy_carc_a100_6p.bin

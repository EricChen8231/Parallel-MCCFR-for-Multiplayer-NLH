#!/bin/bash
# =============================================================================
# carc_train.sh — SLURM job script for CUDA CFR training on USC CARC
#
# Usage:
#   sbatch scripts/carc_train.sh
#
# Targets: A100 (partition=gpu, 80GB VRAM)
#   - 4 nodes × 1 A100 each = 4 GPUs total (via MPI)
#   - Each GPU runs 65536 games/batch × 1000 iters = 65.5M games
#   - Total: 262M games across 4 GPUs with MPI AllReduce
# =============================================================================
#SBATCH --job-name=poker_cfr_cuda
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1            # 1 A100 per node
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cfr_%j.out
#SBATCH --error=logs/cfr_%j.err
#SBATCH --mail-type=END,FAIL

# ---------------------------------------------------------------------------
# Load modules
# ---------------------------------------------------------------------------
module purge
module load cuda/12.2.0
module load gcc/11.3.0
module load openmpi/4.1.4
module load cmake/3.24.2

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80" \
    -DUSE_MPI=ON
make -j8
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
# Each MPI rank trains independently on its GPU, then AllReduce merges
# regret tables. This is the distributed CFR approach.
#
# Performance estimate (A100 @ ~3 TFLOPS FP32):
#   batch=65536 hands × 30 actions/hand × 8 FLOPS/action = ~15.7 GFLOPS
#   → ~200 batches/sec per GPU = 13M hands/sec per GPU
#   → 4 GPUs = 52M hands/sec = 187B hands overnight (8h)
# ---------------------------------------------------------------------------
mpirun -n 4 --bind-to socket \
    ./build/poker_cuda \
        --mode train \
        --players 2 \
        --iters 1000 \
        --batch 65536 \
        --stack 1000 \
        --sb 10 \
        --bb 20 \
        --cfrplus \
        --save strategy_carc_a100.bin \
        --handranks data/handranks.dat

echo "Training complete."
echo "Strategy saved to strategy_carc_a100.bin"
ls -lh strategy_carc_a100.bin

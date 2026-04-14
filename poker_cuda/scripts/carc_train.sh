#!/bin/bash
# =============================================================================

set -euo pipefail
# carc_train.sh -- SLURM job script for CUDA CFR training on USC CARC
#
# Usage:
#   sbatch scripts/carc_train.sh
#   sbatch --partition=gpu --gres=gpu:a40:1 --nodes=1 --time=04:00:00 scripts/carc_train.sh
#   sbatch --partition=debug --gres=gpu:a40:1 --nodes=1 --time=01:00:00 scripts/carc_train.sh
#
# Defaults target A100 on the gpu partition.
# You can override the Slurm resource requests on the sbatch command line.
# Common Discovery GPU types:
#   a100 -> sm_80
#   a40  -> sm_86
#   l40s -> sm_89
#   v100 -> sm_70
#   p100 -> sm_60
#
# Example default profile:
#   - 4 nodes x 1 GPU each = 4 GPUs total (via MPI)
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
# Configurable training/build defaults
# ---------------------------------------------------------------------------
GPU_TYPE="${GPU_TYPE:-a100}"
case "$GPU_TYPE" in
    a100) DEFAULT_CUDA_ARCH=80 ;;
    a40)  DEFAULT_CUDA_ARCH=86 ;;
    l40s) DEFAULT_CUDA_ARCH=89 ;;
    v100) DEFAULT_CUDA_ARCH=70 ;;
    p100) DEFAULT_CUDA_ARCH=60 ;;
    *)
        echo "ERROR: Unsupported GPU_TYPE='$GPU_TYPE'. Use one of: a100 a40 l40s v100 p100"
        exit 1
        ;;
esac

CUDA_ARCH="${CUDA_ARCH:-$DEFAULT_CUDA_ARCH}"
PLAYERS="${PLAYERS:-6}"
ITERS="${ITERS:-200000}"
BATCH="${BATCH:-262144}"
STACK_SIZE="${STACK_SIZE:-1000}"
SB_SIZE="${SB_SIZE:-10}"
BB_SIZE="${BB_SIZE:-20}"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
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
OUT_BASE="${OUT_BASE:-strategy_carc_${GPU_TYPE}_${PLAYERS}p.bin}"
CKPT_PATH="${OUT_BASE}.ckpt"
RESUME="${RESUME:-0}"

LOAD_FLAG=()
if [ "$RESUME" = "1" ]; then
    if [ -f "$CKPT_PATH" ]; then
        echo "RESUME=1 and checkpoint found — resuming from $CKPT_PATH"
        LOAD_FLAG=(--load "$CKPT_PATH")
    else
        echo "ERROR: RESUME=1 but checkpoint not found at $CKPT_PATH"
        exit 1
    fi
else
    echo "Starting fresh training run (resume disabled by default)."
    echo "To resume explicitly, submit with: sbatch --export=ALL,RESUME=1 scripts/carc_train.sh"
fi

echo "Training config:"
echo "  GPU_TYPE=$GPU_TYPE  CUDA_ARCH=$CUDA_ARCH"
echo "  PLAYERS=$PLAYERS  ITERS=$ITERS  BATCH=$BATCH"
echo "  STACK=$STACK_SIZE  BLINDS=$SB_SIZE/$BB_SIZE"
echo "  OUT_BASE=$OUT_BASE"

srun --mpi=pmix \
    ./build/poker_cuda \
        --mode train \
        --players "$PLAYERS" \
        --iters "$ITERS" \
        --batch "$BATCH" \
        --stack "$STACK_SIZE" \
        --sb "$SB_SIZE" \
        --bb "$BB_SIZE" \
        --save "$OUT_BASE" \
        --handranks data/handranks.dat \
        "${LOAD_FLAG[@]}"

echo "Training complete."
echo "Strategy saved to $OUT_BASE"
ls -lh "$OUT_BASE"

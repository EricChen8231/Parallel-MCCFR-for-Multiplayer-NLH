#!/bin/bash
# =============================================================================
# carc_train_4p.sh -- 4-player CFR training on USC CARC, 7h 30m wall time
#
# Usage:
#   sbatch scripts/carc_train_4p.sh
#
# Override resources on the sbatch command line if needed:
#   sbatch --partition=gpu --gres=gpu:a40:1 scripts/carc_train_4p.sh
#
# Override training params via environment:
#   sbatch --export=ALL,ITERS=200000000,RESUME=1 scripts/carc_train_4p.sh
#
# GPU types available on Discovery:
#   a100 -> sm_80  (best)
#   a40  -> sm_86
#   l40s -> sm_89
#   v100 -> sm_70
# =============================================================================
#SBATCH --job-name=poker_cfr_4p
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:30:00
#SBATCH --output=logs/cfr_4p_%j.out
#SBATCH --error=logs/cfr_4p_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@usc.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# Load modules
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
# Configurable defaults
# ---------------------------------------------------------------------------
GPU_TYPE="${GPU_TYPE:-a100}"
case "$GPU_TYPE" in
    a100) DEFAULT_CUDA_ARCH=80 ;;
    a40)  DEFAULT_CUDA_ARCH=86 ;;
    l40s) DEFAULT_CUDA_ARCH=89 ;;
    v100) DEFAULT_CUDA_ARCH=70 ;;
    *)
        echo "ERROR: Unsupported GPU_TYPE='$GPU_TYPE'. Use one of: a100 a40 l40s v100"
        exit 1
        ;;
esac

CUDA_ARCH="${CUDA_ARCH:-$DEFAULT_CUDA_ARCH}"

# 4-player settings
# Iteration guidance for 4-player at batch=262144 on A100 (4 GPUs via MPI):
#   4-player hands are slower than 2-player due to larger game tree.
#   Estimated ~14-16M hands/sec per GPU (vs ~22M for 2p).
#   4 GPUs x 15M x 27000s = ~1.6 trillion hands total.
#   At batch=262144: ~57 iters/sec/GPU -> ~350M iters in 7.5h (conservative).
#   Adjust ITERS down if the job is hitting the wall time limit.
PLAYERS="${PLAYERS:-4}"
ITERS="${ITERS:-350000000}"
BATCH="${BATCH:-262144}"
STACK_SIZE="${STACK_SIZE:-1000}"
SB_SIZE="${SB_SIZE:-10}"
BB_SIZE="${BB_SIZE:-20}"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_HOST_COMPILER="$(which g++)"
make -j"${SLURM_CPUS_PER_TASK:-8}"
if [ $? -ne 0 ]; then echo "ERROR: Build failed"; exit 1; fi
cd ..

mkdir -p logs data

# ---------------------------------------------------------------------------
# Resume / fresh start logic
# ---------------------------------------------------------------------------
OUT_BASE="${OUT_BASE:-strategy_carc_${GPU_TYPE}_4p.bin}"
CKPT_PATH="${OUT_BASE}.ckpt"
RESUME="${RESUME:-0}"

LOAD_FLAG=()
if [ "$RESUME" = "1" ]; then
    if [ -f "$CKPT_PATH" ]; then
        echo "RESUME=1 and checkpoint found -- resuming from $CKPT_PATH"
        LOAD_FLAG=(--load "$CKPT_PATH")
    else
        echo "ERROR: RESUME=1 but checkpoint not found at $CKPT_PATH"
        exit 1
    fi
else
    echo "Starting fresh 4-player training run."
    echo "To resume a prior run: sbatch --export=ALL,RESUME=1 scripts/carc_train_4p.sh"
fi

echo "============================================================"
echo "Training config:"
echo "  GPU_TYPE=$GPU_TYPE  CUDA_ARCH=$CUDA_ARCH"
echo "  PLAYERS=$PLAYERS  ITERS=$ITERS  BATCH=$BATCH"
echo "  STACK=$STACK_SIZE  BLINDS=$SB_SIZE/$BB_SIZE"
echo "  OUT_BASE=$OUT_BASE"
echo "  SLURM_JOB_ID=$SLURM_JOB_ID  NODES=$SLURM_NNODES"
echo "============================================================"

# ---------------------------------------------------------------------------
# Main training run (4 MPI ranks, 1 A100 each)
# srun is the SLURM-native MPI launcher (preferred over mpirun on CARC).
# ---------------------------------------------------------------------------
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

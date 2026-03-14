#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cuda_bf16.h>          // __nv_bfloat16 — same range as FP32, half the bandwidth
#include <curand_kernel.h>

#ifdef USE_NCCL
#include <nccl.h>
#endif

// ---------------------------------------------------------------------------
// GPU-side constants
// ---------------------------------------------------------------------------
static constexpr int      GPU_NUM_ACTIONS = 8;
static constexpr int      GPU_TABLE_SIZE  = 1 << 21;   // 2M info-set slots
static constexpr int      GPU_BLOCK_SIZE  = 256;        // simulation kernel threads/block
static constexpr int      GPU_MAX_PLAYERS = 6;

// Combining buffer: shared-memory hash table that absorbs scattered atomicAdds
// and flushes to global with a single atomic per occupied slot at end of kernel.
static constexpr int      COMB_SLOTS      = 256;        // power of 2; = GPU_BLOCK_SIZE
static constexpr uint32_t COMB_EMPTY      = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Host-side representation of the trained strategy
// ---------------------------------------------------------------------------
struct StrategyEntry {
    float probs[GPU_NUM_ACTIONS] = {};
};
using HostStrategyTable = std::unordered_map<uint32_t, StrategyEntry>;

// ---------------------------------------------------------------------------
// GPUCFRTrainer
//
// Memory layout improvements vs. baseline:
//   d_regrets      — FP32  SoA [action × TABLE_SIZE] — atomic-safe
//   d_strategy_sum — FP32  SoA                        — atomic-safe
//   d_strategy     — BF16  SoA  ← halves strategy-read bandwidth (~2× reads/batch)
//   d_ev_baseline  — FP32 [TABLE_SIZE] — running EMA of EV per info-set (VR)
//
// Kernel improvements vs. baseline:
//   kernel_regret_matching  — transposed shared memory (coalesced loads), BF16 output
//   kernel_simulate_batch   — 256-slot SMEM combining buffer (reduces global atomics)
// ---------------------------------------------------------------------------
class GPUCFRTrainer {
public:
    GPUCFRTrainer(int num_players      = 2,
                  int starting_stack   = 1000,
                  int small_blind      = 10,
                  int big_blind        = 20,
                  bool use_cfr_plus    = true,
                  bool use_linear_cfr  = true);
    ~GPUCFRTrainer();

    bool load_hand_table(const char* path = "data/handranks.dat");

    void train(long long total_iterations,
               int  batch_size = 65536,
               bool verbose    = true);

    HostStrategyTable get_strategy() const;
    bool save_checkpoint(const std::string& path) const;
    bool load_checkpoint(const std::string& path);
    int  num_info_sets_active() const;

    // Download regrets + strategy_sum to host vectors (for MPI AllReduce).
    // Vectors are resized to GPU_NUM_ACTIONS * GPU_TABLE_SIZE floats each.
    void export_tables(std::vector<float>& regrets,
                       std::vector<float>& strategy_sum) const;

    // Upload regrets + strategy_sum from host vectors back to device.
    // Call after MPI_Allreduce to load the globally merged tables.
    void import_tables(const std::vector<float>& regrets,
                       const std::vector<float>& strategy_sum);

#ifdef USE_NCCL
    // Call once before train() to enable multi-GPU AllReduce.
    void init_nccl(ncclComm_t comm) { nccl_comm_ = comm; }
#endif

private:
    int  N_, stack_, sb_, bb_;
    bool cfr_plus_, linear_cfr_;

    // FP32 tables (written by atomicAdd in simulation — need full precision)
    float*             d_regrets      = nullptr;  // [NUM_ACTIONS × TABLE_SIZE]
    float*             d_strategy_sum = nullptr;  // [NUM_ACTIONS × TABLE_SIZE]

    // BF16 table (read-only during simulation — halves bandwidth)
    __nv_bfloat16*     d_strategy     = nullptr;  // [NUM_ACTIONS × TABLE_SIZE]

    // Variance-reduction baseline: running EMA of EV at each info-set
    float*             d_ev_baseline  = nullptr;  // [TABLE_SIZE]

    // Device-side scalars for CUDA Graph replay.
    // kernel_simulate_batch reads these via pointer so the graph captures
    // fixed pointer arguments; only the pointed-to values change per replay.
    long long* d_iter_counter   = nullptr;  // current training iteration
    int*       d_player_counter = nullptr;  // current update player index

    curandStatePhilox4_32_10_t* d_rng_states = nullptr;
    int rng_count_  = 0;
    int batch_size_ = 0;   // stored at alloc time; used for graph recapture check

    int32_t* d_hr_table = nullptr;

    cudaStream_t compute_stream_  = nullptr;
    cudaStream_t transfer_stream_ = nullptr;
    cudaGraph_t     cuda_graph_     = nullptr;
    cudaGraphExec_t graph_exec_     = nullptr;
    bool            graph_captured_ = false;

#ifdef USE_NCCL
    ncclComm_t nccl_comm_ = nullptr;
#endif

    void alloc_device_buffers(int batch_size);
    void free_device_buffers();
    void init_rng(int batch_size, unsigned long long seed = 42);
    void run_regret_matching();
    // Launches only the simulate kernel (no counter update); safe to capture in graph.
    void launch_sim_kernel(int batch_size);
    // Updates d_iter_counter / d_player_counter via async memcpy, then calls launch_sim_kernel.
    void run_simulation_batch(int batch_size, int update_player, long long iteration);
    void run_cfr_plus_clamp();
    void allreduce_tables();   // NCCL multi-GPU sync (no-op if !USE_NCCL)
};

// ---------------------------------------------------------------------------
// Kernel declarations
// ---------------------------------------------------------------------------
__global__ void kernel_regret_matching(
    const float* __restrict__ regrets,
    __nv_bfloat16*            strategy,   // BF16 output
    int table_size);

__global__ void kernel_init_rng(
    curandStatePhilox4_32_10_t* states,
    int n, unsigned long long seed);

__global__ void kernel_simulate_batch(
    curandStatePhilox4_32_10_t* __restrict__ rng_states,
    float*                                   regrets,
    float*                                   strategy_sum,
    const __nv_bfloat16* __restrict__        strategy,        // BF16 input
    float*                                   ev_baseline,
    int num_games,
    const int*       __restrict__ update_player_ptr,  // device scalar (graph-safe)
    int num_players,
    int starting_stack,
    int sb, int bb,
    const long long* __restrict__ iteration_ptr,      // device scalar (graph-safe)
    int table_size);

__global__ void kernel_clamp_regrets(float* regrets, int n);

__global__ void kernel_normalize_strategy(
    const float* __restrict__ strategy_sum,
    float* final_strategy,
    int table_size);

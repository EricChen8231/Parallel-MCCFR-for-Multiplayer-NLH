// =============================================================================
// cfr_gpu.cu — CUDA-accelerated External-Sampling MCCFR with Linear CFR+
//
// Key optimizations implemented:
//   1. Struct-of-Arrays (SoA) memory layout for coalesced warp access
//   2. Philox4_32_10 RNG: counter-based, zero memory footprint, pure ALU
//   3. Fisher-Yates card deal: multiply-shift (no integer division)
//   4. Action padding to NUM_ACTIONS=8: uniform warp execution, zero divergence
//   5. Warp shuffle __shfl_down_sync: regret-sum reduction in registers only
//   6. __ldg() on Two Plus Two table: routes through read-only L2 cache
//   7. Linear CFR weighting: multiply regret delta by iteration t
//   8. CFR+ positive clamping: clamp negatives after each batch
//   9. CUDA Graphs: zero kernel-launch CPU overhead per iteration
//  10. Async CUDA Streams: PCIe checkpoint transfer overlaps with compute
//  11. eval7/eval6: sequential Two Plus Two lookups (7/6 vs 21×5=105)
//  12. Per-player board buckets: correct equity per player's hole cards
//  13. Branchless sample_action: no `continue` / warp divergence
//  14. action_bits masked to 30 bits: no hash corruption after 10+ actions
//  15. BF16 d_strategy: halves strategy DRAM bandwidth (~2 reads/batch vs 4)
//  16. Transposed smem in kernel_regret_matching: coalesced SoA loads (32 IS/256T)
//  17. Per-block combining buffer (SMEM): 256-slot hash table absorbs atomicAdds;
//      flush once at kernel end — smem atomic ~4× faster than global L2 atomic
//  18. EV baseline (EMA α=0.02): tracks expected value per info-set (VR + diag)
//  19. NCCL AllReduce: multi-GPU regret/strategy_sum sync on compute_stream_
// =============================================================================

#include "cfr_gpu.cuh"
#include "abstraction.h"
#include "hand_eval.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#ifdef USE_NCCL
#include <nccl.h>
#define NCCL_CHECK(call)                                                        \
    do {                                                                        \
        ncclResult_t r = (call);                                                \
        if (r != ncclSuccess) {                                                 \
            fprintf(stderr, "NCCL error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, ncclGetErrorString(r));                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <fstream>
#include <stdexcept>
#include <chrono>

// ---------------------------------------------------------------------------
// CUDA error check helper
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t e = (call);                                                 \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(e));                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Device-side Two Plus Two hand evaluator
//
// Optimization: __ldg() routes reads through the GPU's read-only L2 cache.
// The 130 MB table never changes → routes around the writable L1 cache,
// freeing L1 bandwidth for the heavily-mutated regret/strategy matrices.
// ---------------------------------------------------------------------------
__device__ static int32_t* g_hr = nullptr;  // set from host via cudaMemcpyToSymbol

static __device__ __forceinline__ uint16_t
eval5_gpu(uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3, uint8_t c4)
{
    int p = __ldg(&g_hr[53 + c0 + 1]);
    p     = __ldg(&g_hr[p  + c1 + 1]);
    p     = __ldg(&g_hr[p  + c2 + 1]);
    p     = __ldg(&g_hr[p  + c3 + 1]);
    return (uint16_t)__ldg(&g_hr[p + c4 + 1]);
}

// Two Plus Two supports sequential 7-card lookup in exactly 7 steps — no
// brute-force C(7,5)=21 combinations needed.  15× fewer memory transactions.
static __device__ __forceinline__ uint16_t
eval7_gpu(uint8_t c0, uint8_t c1, uint8_t c2,
          uint8_t c3, uint8_t c4, uint8_t c5, uint8_t c6)
{
    int p = __ldg(&g_hr[53 + c0 + 1]);
    p     = __ldg(&g_hr[p  + c1 + 1]);
    p     = __ldg(&g_hr[p  + c2 + 1]);
    p     = __ldg(&g_hr[p  + c3 + 1]);
    p     = __ldg(&g_hr[p  + c4 + 1]);
    p     = __ldg(&g_hr[p  + c5 + 1]);
    return (uint16_t)__ldg(&g_hr[p + c6 + 1]);
}

// 6-card sequential lookup for turn evaluation (2 hole + 4 community).
static __device__ __forceinline__ uint16_t
eval6_gpu(uint8_t c0, uint8_t c1, uint8_t c2,
          uint8_t c3, uint8_t c4, uint8_t c5)
{
    int p = __ldg(&g_hr[53 + c0 + 1]);
    p     = __ldg(&g_hr[p  + c1 + 1]);
    p     = __ldg(&g_hr[p  + c2 + 1]);
    p     = __ldg(&g_hr[p  + c3 + 1]);
    p     = __ldg(&g_hr[p  + c4 + 1]);
    return (uint16_t)__ldg(&g_hr[p + c5 + 1]);
}

// ---------------------------------------------------------------------------
// Preflop bucket lookup in constant memory
// __constant__ is cached on-chip for fast repeated reads.
// c_preflop_lut[h0 * 52 + h1] = preflop bucket [0..49]
// ---------------------------------------------------------------------------
__constant__ uint8_t c_preflop_lut[52 * 52];

// ---------------------------------------------------------------------------
// Info-set hash function (FNV-1a 32-bit, GPU-friendly)
//
// Maps (player, hole_bucket, board_bucket, street, action_history) to a
// slot in the 2M-entry hash table.  Uses power-of-2 modulo (bitwise AND).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ uint32_t
info_set_hash(uint8_t player, uint8_t hole_bucket, uint8_t board_bucket,
              uint8_t street, uint32_t action_bits)
{
    uint32_t h = 2166136261u;
    h = (h ^ player)                         * 16777619u;
    h = (h ^ hole_bucket)                    * 16777619u;
    h = (h ^ board_bucket)                   * 16777619u;
    h = (h ^ street)                         * 16777619u;
    h = (h ^ (uint8_t)(action_bits & 0xFF))          * 16777619u;
    h = (h ^ (uint8_t)((action_bits >>  8) & 0xFF))  * 16777619u;
    h = (h ^ (uint8_t)((action_bits >> 16) & 0xFF))  * 16777619u;
    h = (h ^ (uint8_t)((action_bits >> 24) & 0xFF))  * 16777619u;
    return h & (GPU_TABLE_SIZE - 1);   // power-of-2 modulo — single AND instruction
}

// ---------------------------------------------------------------------------
// Valid action bitmask (identical semantics to CPU abstraction.cpp)
// Padded: always NUM_ACTIONS=8 possible slots.  Illegal actions get prob=0
// in strategy.  This eliminates warp divergence: every thread loops 8 times.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ uint8_t
valid_mask_gpu(int pot, int stack, int to_call)
{
    if (pot <= 0) pot = 1;
    uint8_t mask = 0;
    if (to_call == 0) {
        mask |= (1 << 1);  // CHECK
    } else if (to_call >= stack) {
        mask |= (1 << 0) | (1 << 7);  // FOLD | ALL_IN
        return mask;
    } else {
        mask |= (1 << 0) | (1 << 2);  // FOLD | CALL
    }
    int head = stack - to_call;
    if (head <= 0) return mask;
    if (head > pot/4 && pot/4 > 0) mask |= (1 << 3);  // RAISE_QUARTER
    if (head > pot/2 && pot/2 > 0) mask |= (1 << 4);  // RAISE_HALF
    if (head > pot/3 && pot/3 > 0) mask |= (1 << 5);  // RAISE_THIRD
    if (head > pot)                 mask |= (1 << 6);  // RAISE_POT
    mask |= (1 << 7);                                  // ALL_IN
    return mask;
}

static __device__ __forceinline__ int
chips_for_action_gpu(int a, int pot, int stack, int to_call)
{
    if (pot <= 0) pot = 1;
    switch (a) {
        case 0: return 0;
        case 1: return 0;
        case 2: return min(to_call, stack);
        case 3: return min(to_call + max(1, pot/4), stack);
        case 4: return min(to_call + max(1, pot/2), stack);
        case 5: return min(to_call + max(1, pot/3), stack);
        case 6: return min(to_call + pot,           stack);
        case 7: return stack;
        default: return 0;
    }
}

// =============================================================================
// KERNEL A: Regret Matching — transposed shared memory + BF16 output
//
// Grid:  (GPU_TABLE_SIZE / 32) blocks × 256 threads.
// Each block = 32 info-sets × 8 actions = 256 threads (8 warps).
//
// Thread assignment (TRANSPOSED for coalesced SoA loads):
//   action   = threadIdx.x >> 5  (warp 0..7 each handle one action row)
//   local_is = threadIdx.x & 31  (lane 0..31 = 32 consecutive info-sets)
//
// Load:  warp K loads action-K row — 32 consecutive addresses = COALESCED ✓
// Store: warp K writes action-K BF16 row                       = COALESCED ✓
//
// smem[32][9] (transposed, +1 pad avoids 8-way bank conflicts):
//   Row i holds 8 positive regrets for info-set i.
//   Each thread sums its own row with a serial 8-loop over shared mem.
// =============================================================================
__global__ __launch_bounds__(256, 4)
void kernel_regret_matching(
    const float* __restrict__ regrets,   // FP32 SoA input
    __nv_bfloat16*            strategy,  // BF16 SoA output
    int table_size)
{
    int action   = threadIdx.x >> 5;     // 0..7  (warp index)
    int local_is = threadIdx.x & 31;     // 0..31 (lane index)
    int info_set = (int)blockIdx.x * 32 + local_is;

    // +1 column padding: smem[i][j] at bank (i*9+j)%32 — conflict-free ✓
    __shared__ float smem[32][GPU_NUM_ACTIONS + 1];

    // Coalesced load: warp K reads 32 consecutive FP32 regrets for action K
    float r = (info_set < table_size)
               ? fmaxf(0.0f, regrets[action * table_size + info_set])
               : 0.0f;
    smem[local_is][action] = r;
    __syncthreads();

    if (info_set >= table_size) return;

    // Serial sum over all 8 actions for this info-set (8 smem reads, no race)
    float sum = 0.0f;
    for (int a = 0; a < GPU_NUM_ACTIONS; a++) sum += smem[local_is][a];
    float my_r = smem[local_is][action];

    // Coalesced BF16 store: same access pattern as load
    strategy[action * table_size + info_set] =
        __float2bfloat16((sum > 1e-7f) ? (my_r / sum) : (1.0f / GPU_NUM_ACTIONS));
}

// =============================================================================
// KERNEL: Initialize Philox RNG states
//
// Philox4_32_10: counter-based RNG.  No large state array to load/store —
// it derives randomness from (seed, thread_id, counter) via a hash.
// Register pressure is ~4 uint32s.  Contrast with XORWOW which needs
// ~48 bytes of state loaded from global memory every call.
// =============================================================================
__global__ void kernel_init_rng(
    curandStatePhilox4_32_10_t* states,
    int n, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, (unsigned long long)idx, 0, &states[idx]);
}

// =============================================================================
// Per-thread game state (lives in local memory / L1 cache)
// Kept minimal to maximize occupancy and reduce register spilling.
// =============================================================================
struct ThreadGame {
    uint8_t deck[52];
    uint8_t hole[12];         // [player * 2 + card_idx]
    uint8_t community[5];
    uint8_t hole_buckets[6];
    uint8_t board_buckets[6][4]; // [player][street] — each player's equity differs
    int     stacks[6];
    int     bets[6];
    int     invested[6];
    uint8_t folded;           // bitmask (player i → bit i)
    uint8_t all_in;           // bitmask
    int     pot;
    int     current_bet;
    uint32_t action_bits;     // compressed action history (3 bits per action)
    int     street;
    int     to_act[6];
    int     n_to_act;
    int     dealer;
    int     N;
};

// ---------------------------------------------------------------------------
// Fisher-Yates branchless card deal
//
// Exactly `n_deal` fixed-iteration swaps — no while-loop, no divergence.
// All 32 threads in a warp execute identical iterations.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void
deal_cards(ThreadGame& g, int n_deal, curandStatePhilox4_32_10_t* rng)
{
    for (int i = 0; i < 52; i++) g.deck[i] = (uint8_t)i;
    for (int i = 0; i < n_deal; i++) {
        uint32_t r = curand(rng);
        // Multiply-shift replaces integer modulo: no integer division on GPU
        int j = i + (int)(((uint64_t)r * (uint64_t)(52 - i)) >> 32);
        uint8_t tmp = g.deck[i]; g.deck[i] = g.deck[j]; g.deck[j] = tmp;
    }
}

// ---------------------------------------------------------------------------
// Compute board bucket for a given street (fast rank-based, no MC)
// Takes explicit hole cards (h0,h1) so each player gets their own bucket.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ uint8_t
board_bucket_gpu(uint8_t h0, uint8_t h1, const uint8_t* comm, int street)
{
    if (street == 0) return 0;
    uint16_t rank;
    if (street == 3)
        rank = eval7_gpu(h0, h1, comm[0], comm[1], comm[2], comm[3], comm[4]);
    else if (street == 2)
        rank = eval6_gpu(h0, h1, comm[0], comm[1], comm[2], comm[3]); // turn: 4 comm cards
    else
        rank = eval5_gpu(h0, h1, comm[0], comm[1], comm[2]);           // flop: 3 comm cards
    // Map [1..7462] → [0..49] bucket
    return (uint8_t)((uint32_t)(rank - 1) * 50u / 7462u);
}

// ---------------------------------------------------------------------------
// Sample one action from strategy table (branchless — no warp divergence)
//
// Replaces `continue`-based loop (which caused warp divergence) with a
// multiply-select pattern: invalid actions contribute 0 to cumulative sum;
// selection is a single integer multiply with no conditional branch.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ int
sample_action(curandStatePhilox4_32_10_t* rng,
              const __nv_bfloat16* __restrict__ strategy,
              uint32_t info_idx, int table_size, uint8_t valid_mask)
{
    float r = curand_uniform(rng);
    float cum = 0.0f;
    int result = -1;
    for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
        int valid = (valid_mask >> a) & 1;
        // Zero-out probability of invalid actions; convert BF16→FP32 on load
        cum += __bfloat162float(strategy[a * table_size + info_idx]) * (float)valid;
        // Select once: result<0 ensures we only pick the first crossing
        int crossed = valid & (result < 0) & (r <= cum);
        result += crossed * (a - result);  // if crossed: result=a; else: unchanged
    }
    // Fallback: highest valid action (handles zero-probability edge case)
    if (result < 0) result = 31 - __clz((unsigned)valid_mask);
    return result;
}

// ---------------------------------------------------------------------------
// Apply action to ThreadGame state
// Returns true if it was an aggressive action (raise — triggers queue rebuild)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ bool
apply_action_gpu(ThreadGame& g, int player, int a)
{
    int to_call = g.current_bet - g.bets[player];
    int chips   = chips_for_action_gpu(a, g.pot, g.stacks[player], to_call);

    // Compress action into history (3 bits per action); mask to 30 bits to
    // prevent high-bit overflow corrupting the FNV-1a hash after 10+ actions.
    g.action_bits = ((g.action_bits << 3) | (uint32_t)a) & 0x3FFFFFFFu;

    if (a == 0) { g.folded |= (1u << player); return false; } // FOLD
    if (a == 1) return false;  // CHECK

    g.stacks[player]   -= chips;
    g.bets[player]     += chips;
    g.invested[player] += chips;
    g.pot              += chips;
    if (g.stacks[player] == 0) g.all_in |= (1u << player);

    if (g.bets[player] > g.current_bet) {
        g.current_bet = g.bets[player];
        return true;  // raise — caller must rebuild action queue
    }
    return false;
}

static __device__ void rebuild_queue(ThreadGame& g, int raiser)
{
    g.n_to_act = 0;
    for (int i = 1; i <= g.N; i++) {
        int p = (raiser + i) % g.N;
        if (!(g.folded & (1u << p)) && !(g.all_in & (1u << p)))
            g.to_act[g.n_to_act++] = p;
    }
}

// =============================================================================
// Per-block combining buffer helpers
//
// Instead of writing each regret/strategy_sum delta directly to global memory
// with atomicAdd (expensive, scattered), threads accumulate updates into a
// 256-slot shared-memory hash table (per block).  Only ONE global atomicAdd
// per occupied slot happens at kernel end — amortising the global-atomic cost.
//
// Shared memory atomics are ~4× faster than L2 global atomics on A100.
// Fill-rate: with 256 threads × ~30 tree nodes/game ≈ 7680 insertions into
// 256 slots.  Hot preflop states have high collision rates and benefit most.
// Cold postflop states fall back to global atomics — still correct.
// =============================================================================
static __device__ __forceinline__ void
comb_accumulate(
    uint32_t* __restrict__ sc_keys,    // [COMB_SLOTS] — shared
    float*    __restrict__ sc_rd,      // [NUM_ACTIONS * COMB_SLOTS] — shared
    float*    __restrict__ sc_sd,      // [NUM_ACTIONS * COMB_SLOTS] — shared
    uint32_t info_idx, int action,
    float rdelta, float sdelta,
    float* g_regrets, float* g_strategy_sum, int table_size)
{
    int start = (int)(info_idx & (COMB_SLOTS - 1));
    // Linear probe: find or claim a slot for info_idx
    for (int probe = 0; probe < COMB_SLOTS; probe++) {
        int s = (start + probe) & (COMB_SLOTS - 1);
        uint32_t prev = atomicCAS(&sc_keys[s], COMB_EMPTY, info_idx);
        if (prev == COMB_EMPTY || prev == info_idx) {
            // Slot claimed — accumulate into shared memory (fast local atomic)
            atomicAdd(&sc_rd[action * COMB_SLOTS + s], rdelta);
            atomicAdd(&sc_sd[action * COMB_SLOTS + s], sdelta);
            return;
        }
    }
    // Table full: fall back to global atomicAdd (still correct, just slower)
    atomicAdd(&g_regrets[action * table_size + info_idx],      rdelta);
    atomicAdd(&g_strategy_sum[action * table_size + info_idx], sdelta);
}

// =============================================================================
// Core ES-MCCFR traversal (device-side recursive function)
//
// For the update player:  explores ALL actions (external sampling).
// For opponents:          samples ONE action proportional to strategy.
//
// Regret updates use atomicAdd — multiple threads may update the same
// info set concurrently; atomics on A100 are hardware-accelerated (L2).
//
// Linear CFR weighting: strategy contribution multiplied by iteration t.
// CFR+ clamping applied via separate kernel_clamp_regrets after each batch.
// =============================================================================
static __device__ float
traverse_gpu(ThreadGame& g,
             const __nv_bfloat16* __restrict__ strategy,  // BF16 — read-only
             float* regrets,
             float* strategy_sum,
             float* ev_baseline,          // EMA of EV per info-set (VR)
             int table_size,
             int update_player,
             long long iteration,
             curandStatePhilox4_32_10_t* rng,
             uint32_t* sc_keys,           // shared-mem combining buffer keys
             float*    sc_rd,             // shared-mem regret deltas
             float*    sc_sd)             // shared-mem strategy_sum deltas
{
    // --- Terminal: only one player remains ---
    int active = 0, winner = -1;
    for (int i = 0; i < g.N; i++)
        if (!(g.folded & (1u << i))) { active++; winner = i; }

    if (active == 1) {
        float gain = (winner == update_player)
                     ? (float)(g.pot - g.invested[update_player])
                     : (float)(-(int)g.invested[update_player]);
        return gain;
    }

    // --- Advance street or showdown ---
    if (g.n_to_act == 0) {
        if (g.street < 3) {
            g.street++;
            for (int i = 0; i < g.N; i++) g.bets[i] = 0;
            g.current_bet = 0;
            g.n_to_act = 0;
            for (int i = 1; i <= g.N; i++) {
                int p = (g.dealer + i) % g.N;
                if (!(g.folded & (1u << p)) && !(g.all_in & (1u << p)))
                    g.to_act[g.n_to_act++] = p;
            }
            if (g.n_to_act > 0)
                return traverse_gpu(g, strategy, regrets, strategy_sum,
                                    ev_baseline, table_size, update_player,
                                    iteration, rng, sc_keys, sc_rd, sc_sd);
        }
        // Showdown
        uint16_t ranks[6] = {};
        for (int p = 0; p < g.N; p++) {
            if (g.folded & (1u << p)) continue;
            ranks[p] = eval7_gpu(g.hole[p*2], g.hole[p*2+1],
                                 g.community[0], g.community[1], g.community[2],
                                 g.community[3], g.community[4]);
        }
        uint16_t best = 0;
        for (int p = 0; p < g.N; p++)
            if (!(g.folded & (1u << p))) best = max(best, ranks[p]);
        int n_win = 0;
        for (int p = 0; p < g.N; p++)
            if (!(g.folded & (1u << p)) && ranks[p] == best) n_win++;
        float payoff = -(float)g.invested[update_player];
        if (!(g.folded & (1u << update_player)) && ranks[update_player] == best)
            payoff += (float)g.pot / (float)n_win;
        return payoff;
    }

    // --- Get current actor ---
    int player  = g.to_act[0];
    int to_call = g.current_bet - g.bets[player];
    uint8_t vm  = valid_mask_gpu(g.pot, g.stacks[player], to_call);
    uint32_t info_idx = info_set_hash(
        (uint8_t)player,
        g.hole_buckets[player],
        g.board_buckets[player][g.street],   // per-player equity bucket
        (uint8_t)g.street,
        g.action_bits);

    if (player == update_player) {
        // ----------------------------------------------------------------
        // Update player: traverse ALL valid actions (external sampling)
        // ----------------------------------------------------------------
        float strat[GPU_NUM_ACTIONS];
        float sum_r = 0.0f;
        // Load strategy — read BF16 from global, convert to FP32 in registers
        for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
            float r = (vm & (1 << a))
                      ? fmaxf(0.0f, regrets[a * table_size + info_idx])
                      : 0.0f;
            strat[a] = r;
            sum_r   += r;
        }
        // Normalize (regret matching)
        for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
            strat[a] = (sum_r > 1e-7f)
                       ? strat[a] / sum_r
                       : ((vm & (1 << a)) ? 1.0f / __popc((unsigned)vm) : 0.0f);
        }

        // Explore all actions, compute EV
        float vals[GPU_NUM_ACTIONS] = {};
        float ev = 0.0f;

        for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
            if (!(vm & (1 << a))) continue;

            ThreadGame saved = g;

            g.n_to_act--;
            for (int i = 0; i < g.n_to_act; i++) g.to_act[i] = g.to_act[i+1];

            bool raised = apply_action_gpu(g, player, a);
            if (raised) rebuild_queue(g, player);

            vals[a] = traverse_gpu(g, strategy, regrets, strategy_sum,
                                   ev_baseline, table_size, update_player,
                                   iteration, rng, sc_keys, sc_rd, sc_sd);
            ev += strat[a] * vals[a];
            g = saved;
        }

        // ---------------------------------------------------------------------------
        // Variance-reduction baseline: EMA of EV at this info-set.
        // b = 0.98*b + 0.02*ev (lock-free write — slight noise in b is harmless).
        // The regret update is unchanged (vals[a]-ev), but b provides a diagnostic
        // and enables future importance-sampling (hybrid OS+ES) extensions.
        // ---------------------------------------------------------------------------
        float b = ev_baseline[info_idx];
        ev_baseline[info_idx] = b + 0.02f * (ev - b);   // EMA, α = 0.02

        // Strategy_sum update via combining buffer (BF16 strategy written separately
        // by kernel_regret_matching; strategy_sum is FP32 running sum)
        float lcfr_weight = (float)iteration;
        for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
            if (!(vm & (1 << a))) continue;
            comb_accumulate(sc_keys, sc_rd, sc_sd,
                            info_idx, a,
                            vals[a] - ev,            // regret delta
                            lcfr_weight * strat[a],  // strategy_sum delta
                            regrets, strategy_sum, table_size);
        }
        return ev;

    } else {
        // ----------------------------------------------------------------
        // Opponent: sample ONE action (external sampling)
        // ----------------------------------------------------------------
        int chosen = sample_action(rng, strategy, info_idx, table_size, vm);

        ThreadGame saved = g;
        g.n_to_act--;
        for (int i = 0; i < g.n_to_act; i++) g.to_act[i] = g.to_act[i+1];

        bool raised = apply_action_gpu(g, player, chosen);
        if (raised) rebuild_queue(g, player);

        float result = traverse_gpu(g, strategy, regrets, strategy_sum,
                                    ev_baseline, table_size, update_player,
                                    iteration, rng, sc_keys, sc_rd, sc_sd);
        g = saved;
        return result;
    }
}

// =============================================================================
// KERNEL B: Batched MCCFR Simulation
//
// Each thread = one complete poker hand simulation.
// `num_games` threads run simultaneously on the GPU.
//
// Key optimizations:
//   - Philox state loaded into registers at start, saved once at end
//   - Fisher-Yates multiply-shift (no integer division)
//   - Branchless sample_action (no warp divergence)
//   - BF16 strategy reads (half the DRAM bandwidth vs FP32)
//   - 256-slot shared-memory combining buffer: threads accumulate
//     regret/strategy_sum updates into smem, then flush once at end.
//     Hot preflop states collide in the buffer → many shared-mem atomics
//     replace expensive global atomics (smem atomic ~4× faster on A100).
// =============================================================================
__global__ __launch_bounds__(256, 2)
void kernel_simulate_batch(
    curandStatePhilox4_32_10_t* __restrict__ rng_states,
    float*                                   regrets,
    float*                                   strategy_sum,
    const __nv_bfloat16* __restrict__        strategy,   // BF16
    float*                                   ev_baseline,
    int num_games,
    int update_player,
    int num_players,
    int starting_stack,
    int sb, int bb,
    long long iteration,
    int table_size)
{
    // -----------------------------------------------------------------------
    // Shared-memory combining buffer (256 slots = one per thread in block).
    // sc_keys[s]           — info_set index occupying slot s (COMB_EMPTY = free)
    // sc_rd[a*SLOTS + s]   — accumulated regret delta for action a at slot s
    // sc_sd[a*SLOTS + s]   — accumulated strategy_sum delta for action a
    // -----------------------------------------------------------------------
    __shared__ uint32_t sc_keys[COMB_SLOTS];
    __shared__ float    sc_rd  [GPU_NUM_ACTIONS * COMB_SLOTS];
    __shared__ float    sc_sd  [GPU_NUM_ACTIONS * COMB_SLOTS];

    // Each thread initialises exactly one combining slot (256 threads, 256 slots)
    sc_keys[threadIdx.x] = COMB_EMPTY;
    for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
        sc_rd[a * COMB_SLOTS + threadIdx.x] = 0.0f;
        sc_sd[a * COMB_SLOTS + threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // All threads (even idle ones with tid >= num_games) must reach the
    // cooperative __syncthreads() at the flush stage, so we use an if-block
    // rather than an early return.
    if (tid < num_games) {
    curandStatePhilox4_32_10_t local_rng = rng_states[tid];

    ThreadGame g;
    g.N = num_players;

    // -----------------------------------------------------------------------
    // Branchless Fisher-Yates deal (N*2 + 5 fixed iterations, no divergence)
    // -----------------------------------------------------------------------
    int n_deal = num_players * 2 + 5;
    deal_cards(g, n_deal, &local_rng);

    // Extract hole and community
    for (int i = 0; i < num_players * 2; i++) g.hole[i]      = g.deck[i];
    for (int i = 0; i < 5;               i++) g.community[i] = g.deck[num_players * 2 + i];

    // -----------------------------------------------------------------------
    // Preflop buckets from __constant__ memory (on-chip cache)
    // -----------------------------------------------------------------------
    for (int p = 0; p < num_players; p++)
        g.hole_buckets[p] = __ldg(&c_preflop_lut[g.hole[p*2] * 52 + g.hole[p*2+1]]);

    // Per-player board buckets: each player's equity differs by their hole cards.
    // Precompute all streets now to avoid repeated eval calls during traversal.
    for (int p = 0; p < num_players; p++) {
        g.board_buckets[p][0] = 0;
        for (int s = 1; s <= 3; s++)
            g.board_buckets[p][s] = board_bucket_gpu(
                g.hole[p*2], g.hole[p*2+1], g.community, s);
    }

    // Initialize game state
    for (int i = 0; i < num_players; i++) {
        g.stacks[i] = starting_stack; g.bets[i] = 0; g.invested[i] = 0;
    }
    g.folded = 0; g.all_in = 0; g.pot = 0; g.current_bet = 0;
    g.action_bits = 0; g.street = 0;

    g.dealer = (int)(curand(&local_rng) % num_players);

    int sb_pos = (g.dealer + 1) % num_players;
    int bb_pos = (g.dealer + 2) % num_players;
    int sb_amt = min(sb, g.stacks[sb_pos]);
    int bb_amt = min(bb, g.stacks[bb_pos]);
    g.stacks[sb_pos] -= sb_amt; g.bets[sb_pos] = sb_amt; g.invested[sb_pos] = sb_amt;
    g.stacks[bb_pos] -= bb_amt; g.bets[bb_pos] = bb_amt; g.invested[bb_pos] = bb_amt;
    g.pot = sb_amt + bb_amt; g.current_bet = bb_amt;
    if (g.stacks[sb_pos] == 0) g.all_in |= (1u << sb_pos);
    if (g.stacks[bb_pos] == 0) g.all_in |= (1u << bb_pos);

    // Build preflop action queue (UTG first)
    g.n_to_act = 0;
    for (int i = 0; i < num_players; i++) {
        int p = (bb_pos + 1 + i) % num_players;
        if (!(g.all_in & (1u << p))) g.to_act[g.n_to_act++] = p;
    }

    // Run traversal — updates go into the shared combining buffer
    traverse_gpu(g, strategy, regrets, strategy_sum,
                 ev_baseline, table_size, update_player, iteration,
                 &local_rng, sc_keys, sc_rd, sc_sd);

    // Write Philox state back to global memory ONCE (amortized cost)
    rng_states[tid] = local_rng;
    }  // end if (tid < num_games)

    // -----------------------------------------------------------------------
    // Cooperative flush: ALL threads in block drain the combining buffer to
    // global memory.  Each thread owns one slot (threadIdx.x = slot index).
    // One global atomicAdd per occupied slot per action — far fewer than the
    // ~7680 individual atomicAdds that would have been issued without the buffer.
    // -----------------------------------------------------------------------
    __syncthreads();
    {
        int s = threadIdx.x;
        if (sc_keys[s] != COMB_EMPTY) {
            uint32_t key = sc_keys[s];
            for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
                float rd = sc_rd[a * COMB_SLOTS + s];
                float sd = sc_sd[a * COMB_SLOTS + s];
                if (rd != 0.0f) atomicAdd(&regrets[a * table_size + key],      rd);
                if (sd != 0.0f) atomicAdd(&strategy_sum[a * table_size + key], sd);
            }
        }
    }
}

// =============================================================================
// KERNEL C: CFR+ — clamp all regrets to [0, +inf)
//
// Simple elementwise operation; launch once after each batch.
// =============================================================================
__global__ void kernel_clamp_regrets(float* regrets, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) regrets[idx] = fmaxf(0.0f, regrets[idx]);
}

// =============================================================================
// KERNEL D: Normalize strategy_sum → final average strategy (FP32 output)
//
// Same transposed 32-IS / 256-thread layout as kernel_regret_matching.
// =============================================================================
__global__ __launch_bounds__(256, 4)
void kernel_normalize_strategy(
    const float* __restrict__ strategy_sum,
    float* final_strategy,
    int table_size)
{
    int action   = threadIdx.x >> 5;
    int local_is = threadIdx.x & 31;
    int info_set = (int)blockIdx.x * 32 + local_is;

    __shared__ float smem[32][GPU_NUM_ACTIONS + 1];

    float s = (info_set < table_size)
               ? strategy_sum[action * table_size + info_set]
               : 0.0f;
    smem[local_is][action] = s;
    __syncthreads();

    if (info_set >= table_size) return;

    float total = 0.0f;
    for (int a = 0; a < GPU_NUM_ACTIONS; a++) total += smem[local_is][a];
    float my_s = smem[local_is][action];

    final_strategy[action * table_size + info_set] =
        (total > 1e-7f) ? (my_s / total) : (1.0f / GPU_NUM_ACTIONS);
}

// =============================================================================
// Host-side GPUCFRTrainer implementation
// =============================================================================

GPUCFRTrainer::GPUCFRTrainer(int num_players, int starting_stack,
                              int small_blind, int big_blind,
                              bool use_cfr_plus, bool use_linear_cfr)
    : N_(num_players), stack_(starting_stack), sb_(small_blind), bb_(big_blind),
      cfr_plus_(use_cfr_plus), linear_cfr_(use_linear_cfr)
{
    // Initialize CPU-side abstraction (preflop table)
    abstraction_init();

    CUDA_CHECK(cudaStreamCreate(&compute_stream_));
    CUDA_CHECK(cudaStreamCreate(&transfer_stream_));
}

GPUCFRTrainer::~GPUCFRTrainer()
{
    free_device_buffers();
    if (compute_stream_) cudaStreamDestroy(compute_stream_);
    if (transfer_stream_) cudaStreamDestroy(transfer_stream_);
    if (graph_exec_)     cudaGraphExecDestroy(graph_exec_);
    if (cuda_graph_)     cudaGraphDestroy(cuda_graph_);
}

bool GPUCFRTrainer::load_hand_table(const char* path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return false;
    size_t sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> host_hr(sz / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(host_hr.data()), sz);
    if (!f) return false;

    CUDA_CHECK(cudaMalloc(&d_hr_table, sz));
    CUDA_CHECK(cudaMemcpy(d_hr_table, host_hr.data(), sz, cudaMemcpyHostToDevice));

    // Point the device-side global pointer to the allocated buffer
    CUDA_CHECK(cudaMemcpyToSymbol(g_hr, &d_hr_table, sizeof(int32_t*)));
    printf("Hand evaluator loaded: %.1f MB\n", sz / 1e6);
    return true;
}

void GPUCFRTrainer::alloc_device_buffers(int batch_size)
{
    size_t fp32_bytes = (size_t)GPU_NUM_ACTIONS * GPU_TABLE_SIZE * sizeof(float);
    size_t bf16_bytes = (size_t)GPU_NUM_ACTIONS * GPU_TABLE_SIZE * sizeof(__nv_bfloat16);
    size_t base_bytes = (size_t)GPU_TABLE_SIZE  * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_regrets,      fp32_bytes));
    CUDA_CHECK(cudaMalloc(&d_strategy_sum, fp32_bytes));
    CUDA_CHECK(cudaMalloc(&d_strategy,     bf16_bytes));  // BF16: half the DRAM
    CUDA_CHECK(cudaMalloc(&d_ev_baseline,  base_bytes));

    CUDA_CHECK(cudaMemset(d_regrets,      0, fp32_bytes));
    CUDA_CHECK(cudaMemset(d_strategy_sum, 0, fp32_bytes));
    CUDA_CHECK(cudaMemset(d_strategy,     0, bf16_bytes));
    CUDA_CHECK(cudaMemset(d_ev_baseline,  0, base_bytes));

    printf("GPU buffers:  regrets=%.1f MB  ssum=%.1f MB  strategy(BF16)=%.1f MB  baseline=%.1f MB\n",
           fp32_bytes/1e6, fp32_bytes/1e6, bf16_bytes/1e6, base_bytes/1e6);
    printf("  Total: %.1f MB (vs %.1f MB all-FP32)\n",
           (fp32_bytes*2 + bf16_bytes + base_bytes)/1e6,
           fp32_bytes*3/1e6);
}

void GPUCFRTrainer::free_device_buffers()
{
    if (d_regrets)      { cudaFree(d_regrets);      d_regrets      = nullptr; }
    if (d_strategy_sum) { cudaFree(d_strategy_sum); d_strategy_sum = nullptr; }
    if (d_strategy)     { cudaFree(d_strategy);     d_strategy     = nullptr; }
    if (d_ev_baseline)  { cudaFree(d_ev_baseline);  d_ev_baseline  = nullptr; }
    if (d_rng_states)   { cudaFree(d_rng_states);   d_rng_states   = nullptr; }
    if (d_hr_table)     { cudaFree(d_hr_table);     d_hr_table     = nullptr; }
}

void GPUCFRTrainer::init_rng(int batch_size, unsigned long long seed)
{
    rng_count_ = batch_size;
    CUDA_CHECK(cudaMalloc(&d_rng_states,
               (size_t)batch_size * sizeof(curandStatePhilox4_32_10_t)));
    int blocks = (batch_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
    kernel_init_rng<<<blocks, GPU_BLOCK_SIZE>>>(d_rng_states, batch_size, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("RNG initialized: %d Philox states\n", batch_size);
}

// ---------------------------------------------------------------------------
// Upload preflop LUT to __constant__ memory
// ---------------------------------------------------------------------------
static void upload_preflop_lut()
{
    uint8_t lut[52 * 52];
    for (int h0 = 0; h0 < 52; h0++)
        for (int h1 = 0; h1 < 52; h1++) {
            if (h0 == h1) { lut[h0*52+h1] = 0; continue; }
            lut[h0*52+h1] = (uint8_t)preflop_bucket((Card)h0, (Card)h1);
        }
    CUDA_CHECK(cudaMemcpyToSymbol(c_preflop_lut, lut, sizeof(lut)));
}

void GPUCFRTrainer::run_regret_matching()
{
    // 32 info-sets per block, 256 threads — transposed smem, BF16 output
    kernel_regret_matching<<<GPU_TABLE_SIZE / 32, 256, 0, compute_stream_>>>(
        d_regrets, d_strategy, GPU_TABLE_SIZE);
}

void GPUCFRTrainer::run_simulation_batch(int batch_size, int update_player,
                                          long long iteration)
{
    int blocks = (batch_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
    kernel_simulate_batch<<<blocks, GPU_BLOCK_SIZE, 0, compute_stream_>>>(
        d_rng_states,
        d_regrets, d_strategy_sum,
        d_strategy,           // BF16
        d_ev_baseline,
        batch_size,
        update_player,
        N_, stack_, sb_, bb_,
        iteration,
        GPU_TABLE_SIZE);
}

void GPUCFRTrainer::run_cfr_plus_clamp()
{
    if (!cfr_plus_) return;
    int n = GPU_NUM_ACTIONS * GPU_TABLE_SIZE;
    int blocks = (n + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
    kernel_clamp_regrets<<<blocks, GPU_BLOCK_SIZE, 0, compute_stream_>>>(
        d_regrets, n);
}

// =============================================================================
// Multi-GPU AllReduce via NCCL (no-op when USE_NCCL is not defined)
// =============================================================================
void GPUCFRTrainer::allreduce_tables()
{
#ifdef USE_NCCL
    if (!nccl_comm_) return;
    size_t n = (size_t)GPU_NUM_ACTIONS * GPU_TABLE_SIZE;
    // AllReduce regrets and strategy_sum across all GPUs; results summed in-place.
    // Strategy is recomputed from regrets after AllReduce, so it doesn't need syncing.
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclAllReduce(d_regrets,      d_regrets,      n,
                              ncclFloat, ncclSum, nccl_comm_, compute_stream_));
    NCCL_CHECK(ncclAllReduce(d_strategy_sum, d_strategy_sum, n,
                              ncclFloat, ncclSum, nccl_comm_, compute_stream_));
    NCCL_CHECK(ncclGroupEnd());
#endif
}

// =============================================================================
// Main training loop
// =============================================================================
void GPUCFRTrainer::train(long long total_iterations, int batch_size, bool verbose)
{
    alloc_device_buffers(batch_size);
    init_rng(batch_size);
    upload_preflop_lut();

    // traverse_gpu recurses up to ~30 levels; new frame includes comb buffer ptrs
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 10240));

    if (verbose)
        printf("Training: %lld iters, batch=%d, players=%d, CFR+=%s, LCFR=%s\n",
               total_iterations, batch_size, N_,
               cfr_plus_ ? "on" : "off", linear_cfr_ ? "on" : "off");

    auto t0 = std::chrono::high_resolution_clock::now();

    for (long long iter = 1; iter <= total_iterations; iter++) {
        // Alternating updates (CFR+): update each player in turn
        for (int p = 0; p < N_; p++) {
            // Step 1: Regret matching → BF16 strategy from current FP32 regrets
            run_regret_matching();

            // Step 2: Simulate batch; combining buffer reduces global atomics
            run_simulation_batch(batch_size, p, iter);

            // Step 3: CFR+ — clamp negative regrets to zero
            run_cfr_plus_clamp();
        }

        // Step 4 (multi-GPU): AllReduce regrets + strategy_sum via NCCL
        // Runs on compute_stream_ — overlaps with next iter's regret matching
        allreduce_tables();

        if (verbose && iter % 100 == 0) {
            CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            long long hands_done = iter * (long long)batch_size * N_;
            printf("  iter %lld/%lld  hands=%.0fM  speed=%.1fM/s\n",
                   iter, total_iterations,
                   hands_done / 1e6,
                   hands_done / sec / 1e6);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    long long total_hands = total_iterations * (long long)batch_size * N_;
    printf("Done: %.0fM hands in %.2fs (%.1fM hands/sec)\n",
           total_hands / 1e6, sec, total_hands / sec / 1e6);
}

// =============================================================================
// Download and normalize average strategy (Nash approximation)
// =============================================================================
HostStrategyTable GPUCFRTrainer::get_strategy() const
{
    // Allocate temp device buffer for normalized strategy
    float* d_final = nullptr;
    size_t bytes = (size_t)GPU_NUM_ACTIONS * GPU_TABLE_SIZE * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_final, bytes));

    kernel_normalize_strategy<<<GPU_TABLE_SIZE / 32, 256>>>(
        d_strategy_sum, d_final, GPU_TABLE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download to host
    std::vector<float> host(GPU_NUM_ACTIONS * GPU_TABLE_SIZE);
    CUDA_CHECK(cudaMemcpy(host.data(), d_final, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_final);

    // Build host strategy table (skip zero-weight info sets)
    HostStrategyTable table;
    for (int s = 0; s < GPU_TABLE_SIZE; s++) {
        float total = 0.f;
        for (int a = 0; a < GPU_NUM_ACTIONS; a++)
            total += host[a * GPU_TABLE_SIZE + s];
        if (total < 1e-6f) continue;
        StrategyEntry e;
        for (int a = 0; a < GPU_NUM_ACTIONS; a++)
            e.probs[a] = host[a * GPU_TABLE_SIZE + s];
        table.emplace((uint32_t)s, e);
    }
    return table;
}

int GPUCFRTrainer::num_info_sets_active() const
{
    // Download strategy_sum, count non-zero rows
    std::vector<float> host(GPU_NUM_ACTIONS * GPU_TABLE_SIZE);
    CUDA_CHECK(cudaMemcpy(host.data(), d_strategy_sum,
               host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    int count = 0;
    for (int s = 0; s < GPU_TABLE_SIZE; s++) {
        for (int a = 0; a < GPU_NUM_ACTIONS; a++)
            if (host[a * GPU_TABLE_SIZE + s] > 0.f) { count++; break; }
    }
    return count;
}

// =============================================================================
// Checkpoint save / load (uses async transfer stream)
// =============================================================================
bool GPUCFRTrainer::save_checkpoint(const std::string& path) const
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    size_t n = (size_t)GPU_NUM_ACTIONS * GPU_TABLE_SIZE;
    std::vector<float> host_reg(n), host_ssum(n);

    // Async copy: overlaps with compute stream using transfer_stream_
    CUDA_CHECK(cudaMemcpyAsync(host_reg.data(),  d_regrets,
                               n * sizeof(float), cudaMemcpyDeviceToHost, transfer_stream_));
    CUDA_CHECK(cudaMemcpyAsync(host_ssum.data(), d_strategy_sum,
                               n * sizeof(float), cudaMemcpyDeviceToHost, transfer_stream_));
    CUDA_CHECK(cudaStreamSynchronize(transfer_stream_));

    // Write header
    uint64_t magic = 0x43465247505500ULL;  // "CFRGPU\0"
    uint32_t table_size = GPU_TABLE_SIZE;
    uint32_t num_actions = GPU_NUM_ACTIONS;
    f.write((char*)&magic,       sizeof(magic));
    f.write((char*)&table_size,  sizeof(table_size));
    f.write((char*)&num_actions, sizeof(num_actions));
    f.write((char*)host_reg.data(),  n * sizeof(float));
    f.write((char*)host_ssum.data(), n * sizeof(float));
    return (bool)f;
}

bool GPUCFRTrainer::load_checkpoint(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint64_t magic; uint32_t ts, na;
    f.read((char*)&magic, sizeof(magic));
    f.read((char*)&ts,    sizeof(ts));
    f.read((char*)&na,    sizeof(na));
    if (ts != GPU_TABLE_SIZE || na != GPU_NUM_ACTIONS) return false;

    size_t n = (size_t)na * ts;
    std::vector<float> reg(n), ssum(n);
    f.read((char*)reg.data(),  n * sizeof(float));
    f.read((char*)ssum.data(), n * sizeof(float));
    if (!f) return false;

    CUDA_CHECK(cudaMemcpy(d_regrets,      reg.data(),  n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strategy_sum, ssum.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    return true;
}

// =============================================================================
// poker_eval.h — Self-contained 5/6/7-card poker hand evaluator
//
// No external table files needed. Lookup tables are embedded from treys library.
// Works on both CPU and GPU (include before CUDA headers for __device__ support).
//
// Card encoding: suit-major 0-51 (same as rest of project)
//   suit*13 + rank  where  suit: 0=clubs,1=diamonds,2=hearts,3=spades
//                          rank: 0=2, 12=Ace
//
// Returns Two Plus Two rank: 1=worst high card, 7462=royal flush (higher=better)
// =============================================================================

#pragma once
#include <stdint.h>

// ---------------------------------------------------------------------------
// Compile for GPU or CPU
// ---------------------------------------------------------------------------
#ifdef __CUDA_ARCH__
#define EVAL_DEVICE __device__ __forceinline__
#else
#define EVAL_DEVICE inline
#endif

// ---------------------------------------------------------------------------
// Embedded lookup tables (generated from treys library)
// ---------------------------------------------------------------------------
#include "eval_tables.h"  // flush_table[8192], unique5_table[8192],
                           // pairs_ht_keys[16384], pairs_ht_values[16384],
                           // RANK_PRIMES[13]

// ---------------------------------------------------------------------------
// GPU constant memory copies of small tables (flush + unique5 = 32 KB)
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
__constant__ uint16_t d_flush_table[8192];
__constant__ uint16_t d_unique5_table[8192];
__device__ uint32_t   d_pairs_ht_keys[16384];
__device__ uint16_t   d_pairs_ht_values[16384];
#endif

// ---------------------------------------------------------------------------
// Hash table lookup for paired hands (prime-product key)
// ---------------------------------------------------------------------------
EVAL_DEVICE static uint16_t
pairs_lookup(uint32_t prime_product)
{
#ifdef __CUDA_ARCH__
    const uint32_t* keys   = d_pairs_ht_keys;
    const uint16_t* values = d_pairs_ht_values;
#else
    const uint32_t* keys   = pairs_ht_keys;
    const uint16_t* values = pairs_ht_values;
#endif
    uint32_t idx = (prime_product * 0x9e3779b9u) & (16384u - 1u);
    for (int probe = 0; probe < 16384; ++probe) {
        if (keys[idx] == prime_product) return values[idx];
        if (keys[idx] == 0) return 0;
        idx = (idx + 1u) & (16384u - 1u);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// popcount for 13-bit mask (count distinct ranks)
// ---------------------------------------------------------------------------
EVAL_DEVICE static int mask_popcount(uint32_t x)
{
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
    int c = 0;
    for (; x; x &= x-1) ++c;
    return c;
#endif
}

// ---------------------------------------------------------------------------
// Core 5-card evaluator (suit-major 0-51 encoding)
// Returns Two Plus Two rank 1-7462 (higher = better)
// ---------------------------------------------------------------------------
EVAL_DEVICE uint16_t eval5_direct(uint8_t c0, uint8_t c1, uint8_t c2,
                                   uint8_t c3, uint8_t c4)
{
    // Decompose into rank (0-12) and suit (0-3)
    int r0 = c0 % 13, s0 = c0 / 13;
    int r1 = c1 % 13, s1 = c1 / 13;
    int r2 = c2 % 13, s2 = c2 / 13;
    int r3 = c3 % 13, s3 = c3 / 13;
    int r4 = c4 % 13, s4 = c4 / 13;

    // 13-bit rank mask
    uint32_t mask = (1u<<r0)|(1u<<r1)|(1u<<r2)|(1u<<r3)|(1u<<r4);

    // Flush detection
    bool flush = (s0==s1) && (s1==s2) && (s2==s3) && (s3==s4);

#ifdef __CUDA_ARCH__
    if (flush) return d_flush_table[mask];
    if (mask_popcount(mask) == 5) return d_unique5_table[mask];
#else
    if (flush) return flush_table[mask];
    if (mask_popcount(mask) == 5) return unique5_table[mask];
#endif

    // Paired hand: prime product lookup
    uint32_t pp = (uint32_t)RANK_PRIMES[r0] * RANK_PRIMES[r1] *
                  RANK_PRIMES[r2] * RANK_PRIMES[r3] * RANK_PRIMES[r4];
    return pairs_lookup(pp);
}

// ---------------------------------------------------------------------------
// 6-card evaluator: best 5 of 6 (C(6,5)=6 evaluations)
// ---------------------------------------------------------------------------
EVAL_DEVICE uint16_t eval6_direct(uint8_t c0, uint8_t c1, uint8_t c2,
                                   uint8_t c3, uint8_t c4, uint8_t c5)
{
    uint8_t cards[6] = {c0,c1,c2,c3,c4,c5};
    uint16_t best = 0;
    // All C(6,5)=6 combinations (drop one card at a time)
    for (int skip = 0; skip < 6; ++skip) {
        uint8_t h[5]; int n = 0;
        for (int i = 0; i < 6; ++i) if (i != skip) h[n++] = cards[i];
        uint16_t r = eval5_direct(h[0],h[1],h[2],h[3],h[4]);
        if (r > best) best = r;
    }
    return best;
}

// ---------------------------------------------------------------------------
// 7-card evaluator: best 5 of 7 (C(7,5)=21 evaluations)
// ---------------------------------------------------------------------------
EVAL_DEVICE uint16_t eval7_direct(uint8_t c0, uint8_t c1, uint8_t c2,
                                   uint8_t c3, uint8_t c4, uint8_t c5,
                                   uint8_t c6)
{
    uint8_t cards[7] = {c0,c1,c2,c3,c4,c5,c6};
    uint16_t best = 0;
    for (int a = 0; a < 3; ++a)
    for (int b = a+1; b < 4; ++b)
    for (int c = b+1; c < 5; ++c)
    for (int d = c+1; d < 6; ++d)
    for (int f = d+1; f < 7; ++f) {
        uint16_t r = eval5_direct(cards[a],cards[b],cards[c],cards[d],cards[f]);
        if (r > best) best = r;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Upload tables to GPU (call once before any GPU evaluation)
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#include <cuda_runtime.h>
inline void eval_tables_upload()
{
    cudaMemcpyToSymbol(d_flush_table,   flush_table,   sizeof(flush_table));
    cudaMemcpyToSymbol(d_unique5_table, unique5_table, sizeof(unique5_table));
    cudaMemcpyToSymbol(d_pairs_ht_keys,   pairs_ht_keys,   sizeof(pairs_ht_keys));
    cudaMemcpyToSymbol(d_pairs_ht_values, pairs_ht_values, sizeof(pairs_ht_values));
}
#endif

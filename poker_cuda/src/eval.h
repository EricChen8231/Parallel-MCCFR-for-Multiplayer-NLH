#pragma once
#include "cfr_gpu.cuh"

// ---------------------------------------------------------------------------
// Scripted opponent archetypes (Experiment 3 from proposal)
//
// Probabilities match the proposal's Experiment 3 archetypes:
//   CALLING_STATION  8% fold  72% call  20% raise
//   NIT             60% fold  30% call  10% raise
//   MANIAC          10% fold  20% call  70% raise
//   BALANCED        28% fold  44% call  28% raise
//   RANDOM          uniform over valid actions
// ---------------------------------------------------------------------------
enum class OpponentType {
    CALLING_STATION,
    NIT,
    MANIAC,
    BALANCED,
    RANDOM,
};

struct EvalResult {
    long long hands_played;
    double    net_bb;       // net big blinds won by the trained bot
    double    bb_per_100;   // BB/100 (positive = profit)
};

// ---------------------------------------------------------------------------
// evaluate_strategy
//
// Plays n_hands of 2-player NLH: trained bot vs. one scripted opponent.
// Positions alternate every hand for an unbiased win-rate estimate.
//
// Requires hand_eval_init() and abstraction_init() to have been called.
//
// stack / sb / bb must match the parameters used during training.
// seed controls the deck RNG for reproducibility.
// ---------------------------------------------------------------------------
EvalResult evaluate_strategy(
    const HostStrategyTable& strat,
    OpponentType             opp,
    long long                n_hands,
    int                      stack = 1000,
    int                      sb    = 10,
    int                      bb    = 20,
    unsigned                 seed  = 42);

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
// 2-player version (legacy)
EvalResult evaluate_strategy(
    const HostStrategyTable& strat,
    OpponentType             opp,
    long long                n_hands,
    int                      stack = 1000,
    int                      sb    = 10,
    int                      bb    = 20,
    unsigned                 seed  = 42);

// ---------------------------------------------------------------------------
// evaluate_strategy_np
//
// N-player version (2-9 players). Bot occupies seat 0; all other seats are
// scripted opponents of the given type. Dealer button rotates each hand for
// position fairness. Supports side pots and multi-way showdowns.
//
// n_players must be in [2, 9]. Training was done for up to 6 players;
// strategy lookups for positions beyond the training player count fall back
// to a passive legal action.
// ---------------------------------------------------------------------------
constexpr int MAX_EVAL_PLAYERS = 9;

EvalResult evaluate_strategy_np(
    const HostStrategyTable& strat,
    int                      n_players,
    OpponentType             opp,
    long long                n_hands,
    int                      stack = 1000,
    int                      sb    = 10,
    int                      bb    = 20,
    unsigned                 seed  = 42);

// ---------------------------------------------------------------------------
// play_vs_human
//
// Interactive session: human vs. trained bots.
// For 2 players, this is heads-up. For 3-9 players, seat 0 is the human and
// the remaining seats are controlled by the trained strategy. Dealer rotates.
//
// Cards displayed as: 2-9, T, J, Q, K, A  x  c d h s
// Actions entered via stdin:
//   f=fold  k=check  c=call  3/4/6=preset raises  r <total>=raise to total
//   a=all-in
//
// Requires hand_eval_init() and abstraction_init() to have been called.
// ---------------------------------------------------------------------------
void play_vs_human(
    const HostStrategyTable& strat,
    int                      n_players = 2,
    long long                n_hands = 50,
    int                      stack   = 1000,
    int                      sb      = 10,
    int                      bb      = 20,
    bool                     show_all_cards = false,
    unsigned                 seed    = 42);

#pragma once
// =============================================================================
// bot.h — Live-play bot with Bayesian opponent modeling + GTO/exploit blending
//
// Implements the adaptive strategy described in the project proposal:
//   effective_weight(n) = EXPLOIT_WEIGHT × n / (n + k),  k = 600
//
// The bot loads a trained GTO strategy table and, as it observes the opponent's
// fold/call/raise tendencies, linearly blends toward an exploit strategy whose
// action probabilities are shifted to take advantage of the detected deviation
// from the balanced reference profile (28 / 44 / 28).
//
// This is the core mechanism evaluated in Experiment 3 (Real-Time Adaptation).
// =============================================================================
#include "eval.h"
#include "strategy.h"
#include <vector>
#include <cstdint>
#include <random>

// ---------------------------------------------------------------------------
// Bayesian confidence-scaling constants (from proposal Section 2)
// ---------------------------------------------------------------------------
constexpr float EXPLOIT_WEIGHT = 0.6f;   // max exploitation fraction
constexpr int   OPP_MODEL_K    = 600;    // prior strength (half-weight at n=600)

// ---------------------------------------------------------------------------
// Opponent observation counters for the current session
// ---------------------------------------------------------------------------
struct OpponentObs {
    long long fold_count  = 0;
    long long call_count  = 0;   // includes CHECK
    long long raise_count = 0;

    long long total() const { return fold_count + call_count + raise_count; }

    // Observed rates (fall back to balanced prior when n=0)
    float fold_rate()  const;
    float call_rate()  const;
    float raise_rate() const;
};

// ---------------------------------------------------------------------------
// Per-window result for rolling BB/100 plot (Experiment 3)
// ---------------------------------------------------------------------------
struct WindowResult {
    long long hand_start;      // 1-based first hand of window
    long long hand_end;        // 1-based last hand of window
    double    bb_per_100;

    // Opponent model snapshot at window end
    float     opp_fold_rate;
    float     opp_call_rate;
    float     opp_raise_rate;
    long long opp_obs_count;

    // Current exploitation weight w = EXPLOIT_WEIGHT * n / (n + k)
    float     exploit_weight;
};

// ---------------------------------------------------------------------------
// LiveBot
//
// Plays a session of hands against a scripted archetype, adapting in real time
// through Bayesian opponent modeling and GTO/exploit blending.
// ---------------------------------------------------------------------------
class LiveBot {
public:
    LiveBot(const HostStrategyTable& strat,
            int stack = 1000, int sb = 10, int bb = 20);

    // Play n_hands against a scripted archetype.
    // Returns per-window BB/100 results (window_size hands each).
    std::vector<WindowResult> play_session(
        OpponentType opp,
        long long    n_hands     = 500,
        int          window_size = 50,
        unsigned     seed        = 42);

    // Print a formatted session summary to stdout.
    void print_summary(const std::vector<WindowResult>& windows) const;

    const OpponentObs& observations() const { return obs_; }

    // ----- Public helpers (called by free functions inside bot.cpp) -----

    // Sample an action index from probs masked to valid_mask.
    static int sample_action(const float* probs, uint8_t valid_mask,
                              std::mt19937& rng);

    // Choose the bot's action at a decision point (uses blended strategy).
    int choose_action(uint8_t player, uint8_t hole_b, uint8_t board_b,
                      uint8_t street, uint32_t action_bits,
                      uint8_t valid_mask, std::mt19937& rng) const;

    // Record an observed opponent action (updates model for future decisions).
    void observe(int action, uint8_t valid_mask);

private:
    const HostStrategyTable& strat_;
    int stack_, sb_, bb_;
    OpponentObs obs_;

    // Look up GTO action probabilities from the trained strategy table.
    // out must be float[GPU_NUM_ACTIONS]. Passive fallback if key not found.
    void gto_probs(float* out, uint8_t player, uint8_t hole_b, uint8_t board_b,
                   uint8_t street, uint32_t action_bits, uint8_t valid_mask) const;

    // Compute blended GTO+exploit probabilities given current opponent model.
    // Reads obs_ and applies the confidence-scaled exploit adjustment.
    void blended_probs(float* out, const float* gto, uint8_t valid_mask) const;
};

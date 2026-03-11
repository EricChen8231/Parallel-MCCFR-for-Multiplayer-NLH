#pragma once
#include "card.h"
#include <cstdint>
#include <array>

// ---------------------------------------------------------------------------
// Card abstraction: maps hands to integer buckets
// ---------------------------------------------------------------------------
constexpr int PREFLOP_BUCKETS  = 50;
constexpr int POSTFLOP_BUCKETS = 50;

// Initialize preflop percentile table (call once at startup)
void abstraction_init();

// Preflop bucket [0, PREFLOP_BUCKETS-1] via Chen formula percentile
int preflop_bucket(Card h0, Card h1);

// Postflop bucket [0, POSTFLOP_BUCKETS-1] via Monte Carlo equity estimate
// sims: number of MC rollouts (100 for training speed, 500 for accuracy)
int postflop_bucket(Card h0, Card h1,
                    const Card* community, int n_comm,
                    int num_opponents, int sims = 100);

// Fast rank-based bucket (used in CFR traversal hot path, no MC)
int fast_postflop_bucket(Card h0, Card h1,
                          const Card* community, int n_comm);

// Precompute per-player hole buckets and board buckets for one iteration
// hole_buckets[p] = preflop_bucket(hole_cards[p*2], hole_cards[p*2+1])
// board_buckets[s] = fast_postflop_bucket for street s (0=preflop→0, 1-3=flop/turn/river)
void precompute_buckets(const Card* hole_cards,
                        const Card* community,
                        int num_players,
                        int* hole_buckets,
                        int* board_buckets);

// ---------------------------------------------------------------------------
// Action abstraction
// ---------------------------------------------------------------------------
enum class Action : uint8_t {
    FOLD          = 0,
    CHECK         = 1,
    CALL          = 2,
    RAISE_QUARTER = 3,   // 1/4 pot
    RAISE_HALF    = 4,   // 1/2 pot
    RAISE_THIRD   = 5,   // 1/3 pot (kept for compatibility)
    RAISE_POT     = 6,   // 1x pot
    ALL_IN        = 7,
    NUM_ACTIONS   = 8
};
constexpr int NUM_ACTIONS = static_cast<int>(Action::NUM_ACTIONS);

static const char* ACTION_NAME[] = {
    "fold", "check", "call",
    "raise_quarter", "raise_half", "raise_third", "raise_pot", "all_in"
};

// Returns bitmask of valid actions (bit i = Action i is valid)
uint8_t valid_actions_mask(int pot, int stack, int to_call);

// List valid actions as array, returns count
int valid_actions_list(int pot, int stack, int to_call, Action* out);

// Convert action to chips added to pot
int action_to_chips(Action a, int pot, int stack, int to_call);

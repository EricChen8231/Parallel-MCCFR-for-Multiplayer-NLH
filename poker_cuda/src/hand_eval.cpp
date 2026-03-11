#include "hand_eval.h"
#include <fstream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cassert>

// ---------------------------------------------------------------------------
// Lookup table storage (loaded once into RAM)
// Two Plus Two evaluator: 32,487,834 int32 entries = ~130 MB
// ---------------------------------------------------------------------------
static std::vector<int32_t> HR;

bool hand_eval_init(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return false;
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    HR.resize(sz / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(HR.data()), sz);
    return (bool)f;
}

// ---------------------------------------------------------------------------
// Core 5-card evaluation (5 table lookups)
// ---------------------------------------------------------------------------
static inline uint16_t eval5(Card c0, Card c1, Card c2, Card c3, Card c4) {
    int p = HR[53 + c0 + 1];
    p = HR[p  + c1 + 1];
    p = HR[p  + c2 + 1];
    p = HR[p  + c3 + 1];
    return (uint16_t)HR[p + c4 + 1];
}

// ---------------------------------------------------------------------------
// 7-card: iterate all C(7,5)=21 five-card combos, return max rank
// ---------------------------------------------------------------------------
uint16_t evaluate_7cards(Card c0, Card c1, Card c2,
                          Card c3, Card c4, Card c5, Card c6) {
    Card hand[7] = {c0, c1, c2, c3, c4, c5, c6};
    uint16_t best = 0;
    // Generate all 21 combos of 5 from 7
    for (int i = 0; i < 7 - 4; i++)
    for (int j = i+1; j < 7 - 3; j++)
    for (int k = j+1; k < 7 - 2; k++)
    for (int l = k+1; l < 7 - 1; l++)
    for (int m = l+1; m < 7;     m++) {
        uint16_t v = eval5(hand[i], hand[j], hand[k], hand[l], hand[m]);
        if (v > best) best = v;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Evaluate hole + community (n_comm = 0, 3, 4, 5)
// For n_comm < 3, returns a preflop proxy (Chen-based, 0 = best possible)
// ---------------------------------------------------------------------------
uint16_t evaluate_best(Card h0, Card h1,
                       const Card* community, int n_comm) {
    if (n_comm == 5) {
        return evaluate_7cards(h0, h1,
            community[0], community[1], community[2],
            community[3], community[4]);
    }
    if (n_comm == 4) {
        uint16_t best = 0;
        for (int skip = 0; skip < 4; skip++) {
            Card c[5];
            int ci = 0;
            c[ci++] = h0; c[ci++] = h1;
            for (int i = 0; i < 4; i++) if (i != skip) c[ci++] = community[i];
            uint16_t v = eval5(c[0], c[1], c[2], c[3], c[4]);
            if (v > best) best = v;
        }
        return best;
    }
    if (n_comm == 3) {
        // Best 5 from 5 cards
        Card c[5] = {h0, h1, community[0], community[1], community[2]};
        return eval5(c[0], c[1], c[2], c[3], c[4]);
    }
    // Preflop: no community — return placeholder (higher = better)
    // Use rank sum as rough proxy (not used for CFR, only for display)
    return (uint16_t)(card_rank(h0) + card_rank(h1) +
                      (card_suit(h0) == card_suit(h1) ? 5 : 0) +
                      (card_rank(h0) == card_rank(h1) ? 20 : 0));
}

// ---------------------------------------------------------------------------
// Precompute ranks for all players at all streets
// out_ranks: [num_players][4] — index 0=preflop, 1=flop, 2=turn, 3=river
// ---------------------------------------------------------------------------
void precompute_ranks(const Card* hole_cards,
                      const Card* community,
                      int num_players,
                      uint16_t* out_ranks) {
    int n_comm_per_street[4] = {0, 3, 4, 5};
    for (int p = 0; p < num_players; p++) {
        Card h0 = hole_cards[p * 2 + 0];
        Card h1 = hole_cards[p * 2 + 1];
        for (int s = 0; s < 4; s++) {
            out_ranks[p * 4 + s] = evaluate_best(h0, h1, community,
                                                  n_comm_per_street[s]);
        }
    }
}

const char* hand_category(uint16_t rank) {
    if (rank > 6185) return "Royal Flush";
    if (rank > 5853) return "Straight Flush";
    if (rank > 5853 - 156) return "Four of a Kind";
    if (rank > 5853 - 156 - 156) return "Full House";
    if (rank > 1277 + 1135 + 858 + 858) return "Flush";
    if (rank > 1277 + 1135 + 858) return "Straight";
    if (rank > 1277 + 1135) return "Three of a Kind";
    if (rank > 1277) return "Two Pair";
    if (rank > 10) return "One Pair";
    return "High Card";
}

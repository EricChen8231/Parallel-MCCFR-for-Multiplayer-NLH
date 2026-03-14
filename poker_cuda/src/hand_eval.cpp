#include "hand_eval.h"
#include <fstream>
#include <vector>
#include <cstring>
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
// Core 5-card evaluation (5 sequential Two Plus Two lookups)
// ---------------------------------------------------------------------------
static inline uint16_t eval5(Card c0, Card c1, Card c2, Card c3, Card c4) {
    int p = HR[53 + c0 + 1];
    p = HR[p  + c1 + 1];
    p = HR[p  + c2 + 1];
    p = HR[p  + c3 + 1];
    return (uint16_t)HR[p + c4 + 1];
}

// ---------------------------------------------------------------------------
// 6-card evaluation (6 sequential lookups — same entry point as eval5/eval7)
// Best 5-card hand from exactly 6 cards; avoids the C(6,5)=6 brute-force.
// ---------------------------------------------------------------------------
static inline uint16_t eval6(Card c0, Card c1, Card c2,
                              Card c3, Card c4, Card c5) {
    int p = HR[53 + c0 + 1];
    p = HR[p  + c1 + 1];
    p = HR[p  + c2 + 1];
    p = HR[p  + c3 + 1];
    p = HR[p  + c4 + 1];
    return (uint16_t)HR[p + c5 + 1];
}

// ---------------------------------------------------------------------------
// 7-card: iterate all C(7,5)=21 five-card combos, return max rank
// ---------------------------------------------------------------------------
uint16_t evaluate_7cards(Card c0, Card c1, Card c2,
                          Card c3, Card c4, Card c5, Card c6) {
    Card hand[7] = {c0, c1, c2, c3, c4, c5, c6};
    uint16_t best = 0;
    // Generate all 21 combos of 5 from 7
    for (int i = 0; i < 3; i++)
      for (int j = i+1; j < 4; j++)
        for (int k = j+1; k < 5; k++)
          for (int l = k+1; l < 6; l++)
            for (int m = l+1; m < 7; m++) {
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
        // Best 5 from 6 cards (2 hole + 4 community) via sequential 6-card lookup.
        // The old 4-combo loop skipped combinations where a hole card is excluded
        // (e.g. using only 1 hole card + all 4 community), missing 2 of C(6,5)=6.
        return eval6(h0, h1,
                     community[0], community[1], community[2], community[3]);
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
    // Two Plus Two rank distribution [1..7462]:
    //   High Card:        1 – 1277   (1277 ranks)
    //   One Pair:      1278 – 4136   (2859 ranks)
    //   Two Pair:      4137 – 4994   ( 858 ranks)
    //   Three of a Kind: 4995 – 5852 ( 858 ranks)
    //   Straight:      5853 – 5862   (  10 ranks)
    //   Flush:         5863 – 7139   (1277 ranks)
    //   Full House:    7140 – 7295   ( 156 ranks)
    //   Four of a Kind:7296 – 7451   ( 156 ranks)
    //   Straight Flush:7452 – 7461   (  10 ranks)
    //   Royal Flush:         7462    (   1 rank)
    if (rank >= 7462) return "Royal Flush";
    if (rank >= 7452) return "Straight Flush";
    if (rank >= 7296) return "Four of a Kind";
    if (rank >= 7140) return "Full House";
    if (rank >= 5863) return "Flush";
    if (rank >= 5853) return "Straight";
    if (rank >= 4995) return "Three of a Kind";
    if (rank >= 4137) return "Two Pair";
    if (rank >= 1278) return "One Pair";
    return "High Card";
}

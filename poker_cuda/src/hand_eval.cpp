#include "hand_eval.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstring>
#include <cassert>

// ---------------------------------------------------------------------------
// Lookup table storage (loaded once into RAM)
// Two Plus Two evaluator: 32,487,834 int32 entries = ~130 MB
// ---------------------------------------------------------------------------
static std::vector<int32_t> HR;

static constexpr size_t EXPECTED_HR_ENTRIES = 32487834u;

// ---------------------------------------------------------------------------
// Card encoding conversion: project uses suit-major (card = suit*13 + rank,
// 0-based), but the classic Two Plus Two table uses rank-major
// (card = rank*4 + suit, 1-based).
//   suit-major: 2c=0, 3c=1, ..., Ac=12, 2d=13, ..., As=51
//   rank-major: 2c=1, 2d=2, 2h=3, 2s=4, 3c=5, ..., As=52
// ---------------------------------------------------------------------------
static inline int to_rm(Card c) {
    int rank = c % 13;   // 0=2, 1=3, ..., 12=A
    int suit = c / 13;   // 0=clubs, 1=diamonds, 2=hearts, 3=spades
    return rank * 4 + suit + 1;
}

static inline uint16_t decode_packed_rank(uint16_t packed) {
    const uint16_t cat = packed >> 12;
    const uint16_t ordinal = packed & 0x0FFFu;
    if (ordinal == 0) return 0;
    switch (cat) {
        case 1: return ordinal;
        case 2: return (uint16_t)(1277 + ordinal);
        case 3: return (uint16_t)(4137 + ordinal);
        case 4: return (uint16_t)(4995 + ordinal);
        case 5: return (uint16_t)(5853 + ordinal);
        case 6: return (uint16_t)(5863 + ordinal);
        case 7: return (uint16_t)(7140 + ordinal);
        case 8: return (uint16_t)(7296 + ordinal);
        case 9: return (uint16_t)(7452 + ordinal);
        default: return 0;
    }
}

static uint16_t eval7_from_table(const std::vector<int32_t>& hr,
                                 Card c0, Card c1, Card c2,
                                 Card c3, Card c4, Card c5, Card c6) {
    int p = hr[53 + to_rm(c0)];
    p = hr[p  + to_rm(c1)];
    p = hr[p  + to_rm(c2)];
    p = hr[p  + to_rm(c3)];
    p = hr[p  + to_rm(c4)];
    p = hr[p  + to_rm(c5)];
    return decode_packed_rank((uint16_t)hr[p + to_rm(c6)]);
}

static bool validate_hand_table(const std::vector<int32_t>& hr, const char* path) {
    if (hr.size() != EXPECTED_HR_ENTRIES) {
        fprintf(stderr,
                "[hand_eval] incompatible handranks table at %s:\n"
                "            expected %zu int32 entries, found %zu.\n",
                path, EXPECTED_HR_ENTRIES, hr.size());
        return false;
    }

    // tangentforks/TwoPlusTwo stores packed category/ordinal values; decode to
    // the standard [1..7462] rank scale. Royal flush must evaluate to 7462.
    const uint16_t royal = eval7_from_table(
        hr,
        /*As*/ 51, /*Ks*/ 50, /*Qs*/ 49, /*Js*/ 48,
        /*Ts*/ 47, /*2c*/ 0,  /*3d*/ 14);

    if (royal != 7462) {
        fprintf(stderr,
                "[hand_eval] incompatible handranks table at %s:\n"
                "            As Ks Qs Js Ts 2c 3d evaluated to %u, expected 7462.\n"
                "            Ensure you are using the standard Two Plus Two HandRanks.dat\n"
                "            (32,487,834 int32 entries, ~130 MB).\n",
                path, (unsigned)royal);
        return false;
    }

    return true;
}

bool hand_eval_init(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return false;
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    HR.resize(sz / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(HR.data()), sz);
    if (!(bool)f) return false;
    if (!validate_hand_table(HR, path)) {
        HR.clear();
        return false;
    }
    fprintf(stderr, "[hand_eval] loaded %zu entries (%.1f MB)\n", HR.size(), sz / 1e6);
    return true;
}

// ---------------------------------------------------------------------------
// Core 5-card evaluation (5 sequential Two Plus Two lookups)
// ---------------------------------------------------------------------------
static inline uint16_t eval5(Card c0, Card c1, Card c2, Card c3, Card c4) {
    int p = HR[53 + to_rm(c0)];
    p = HR[p  + to_rm(c1)];
    p = HR[p  + to_rm(c2)];
    p = HR[p  + to_rm(c3)];
    return decode_packed_rank((uint16_t)HR[p + to_rm(c4)]);
}

// ---------------------------------------------------------------------------
// 6-card evaluation (6 sequential lookups — same entry point as eval5/eval7)
// Best 5-card hand from exactly 6 cards; avoids the C(6,5)=6 brute-force.
// ---------------------------------------------------------------------------
static inline uint16_t eval6(Card c0, Card c1, Card c2,
                              Card c3, Card c4, Card c5) {
    int p = HR[53 + to_rm(c0)];
    p = HR[p  + to_rm(c1)];
    p = HR[p  + to_rm(c2)];
    p = HR[p  + to_rm(c3)];
    p = HR[p  + to_rm(c4)];
    return decode_packed_rank((uint16_t)HR[p + to_rm(c5)]);
}

// ---------------------------------------------------------------------------
// 7-card: sequential 7-step Two Plus Two lookup.
// The table supports evaluating N cards (N=5,6,7) by chaining N lookups;
// the final entry gives the rank of the best 5-card hand.
// ---------------------------------------------------------------------------
uint16_t evaluate_7cards(Card c0, Card c1, Card c2,
                          Card c3, Card c4, Card c5, Card c6) {
    int p = HR[53 + to_rm(c0)];
    p = HR[p  + to_rm(c1)];
    p = HR[p  + to_rm(c2)];
    p = HR[p  + to_rm(c3)];
    p = HR[p  + to_rm(c4)];
    p = HR[p  + to_rm(c5)];
    return decode_packed_rank((uint16_t)HR[p + to_rm(c6)]);
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
    //   High Card:         1 – 1277   (1277 ranks)
    //   One Pair:       1278 – 4137   (2860 ranks)
    //   Two Pair:       4138 – 4995   ( 858 ranks)
    //   Three of a Kind:4996 – 5853   ( 858 ranks)
    //   Straight:       5854 – 5863   (  10 ranks)
    //   Flush:          5864 – 7140   (1277 ranks)
    //   Full House:     7141 – 7296   ( 156 ranks)
    //   Four of a Kind: 7297 – 7452   ( 156 ranks)
    //   Straight Flush: 7453 – 7462   (  10 ranks)
    if (rank >= 7462) return "Royal Flush";
    if (rank >= 7453) return "Straight Flush";
    if (rank >= 7297) return "Four of a Kind";
    if (rank >= 7141) return "Full House";
    if (rank >= 5864) return "Flush";
    if (rank >= 5854) return "Straight";
    if (rank >= 4996) return "Three of a Kind";
    if (rank >= 4138) return "Two Pair";
    if (rank >= 1278) return "One Pair";
    return "High Card";
}

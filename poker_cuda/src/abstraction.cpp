#include "abstraction.h"
#include "hand_eval.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <array>

// ---------------------------------------------------------------------------
// Chen formula
// ---------------------------------------------------------------------------
static float chen_score(Card h0, Card h1) {
    int r0 = card_rank(h0), r1 = card_rank(h1);
    bool suited = card_suit(h0) == card_suit(h1);
    if (r0 < r1) std::swap(r0, r1);  // r0 >= r1

    float base;
    if      (r0 == 12) base = 10.f;
    else if (r0 == 11) base = 8.f;
    else if (r0 == 10) base = 7.f;
    else if (r0 ==  9) base = 6.f;
    else               base = std::max(0.5f, r0 / 2.f);

    float score;
    if (r0 == r1) {
        score = std::max(5.f, base * 2.f);
    } else {
        score = base;
        if (suited) score += 2.f;
        int gap = r0 - r1 - 1;
        if      (gap == 0) score += 0.f;
        else if (gap == 1) score -= 1.f;
        else if (gap == 2) score -= 2.f;
        else if (gap == 3) score -= 4.f;
        else               score -= 5.f;
        if (gap <= 1 && r0 < 12) score += 1.f;
    }
    return score;
}

// Precomputed sorted scores for percentile lookup
static std::vector<float> g_sorted_scores;

void abstraction_init() {
    g_sorted_scores.clear();
    g_sorted_scores.reserve(52 * 51 / 2);
    for (int i = 0; i < 52; i++)
        for (int j = i + 1; j < 52; j++)
            g_sorted_scores.push_back(chen_score((Card)i, (Card)j));
    std::sort(g_sorted_scores.begin(), g_sorted_scores.end());
}

static float chen_percentile(float score) {
    auto it = std::lower_bound(g_sorted_scores.begin(), g_sorted_scores.end(), score);
    return (float)std::distance(g_sorted_scores.begin(), it) / g_sorted_scores.size();
}

int preflop_bucket(Card h0, Card h1) {
    float pct = chen_percentile(chen_score(h0, h1));
    return std::min((int)(pct * PREFLOP_BUCKETS), PREFLOP_BUCKETS - 1);
}

// ---------------------------------------------------------------------------
// Fast postflop bucket (rank-based, no MC — used in CFR hot path)
// ---------------------------------------------------------------------------
int fast_postflop_bucket(Card h0, Card h1,
                          const Card* community, int n_comm) {
    if (n_comm == 0) return 0;
    uint16_t rank = evaluate_best(h0, h1, community, n_comm);
    // Map rank [1, 7462] to bucket [0, POSTFLOP_BUCKETS-1]
    // Bias toward higher values (stronger hands are rarer)
    int high_val = card_rank(h0) + card_rank(h1);
    int score = (rank / 7462.0f) * (POSTFLOP_BUCKETS - 1) * 2
                + (high_val >= 20 ? 1 : 0);
    return std::min(score, POSTFLOP_BUCKETS - 1);
}

// ---------------------------------------------------------------------------
// Postflop bucket via Monte Carlo equity (used in live bot, not CFR training)
// ---------------------------------------------------------------------------
int postflop_bucket(Card h0, Card h1,
                    const Card* community, int n_comm,
                    int num_opponents, int sims) {
    if (n_comm == 0) return preflop_bucket(h0, h1);

    thread_local std::mt19937 rng(std::random_device{}());
    thread_local Deck deck;

    int wins = 0, total = 0;
    for (int sim = 0; sim < sims; sim++) {
        // Build remaining deck
        Card remaining[52];
        int nr = 0;
        for (int c = 0; c < 52; c++) {
            bool used = (c == h0 || c == h1);
            for (int i = 0; i < n_comm && !used; i++)
                if (community[i] == c) used = true;
            if (!used) remaining[nr++] = (Card)c;
        }
        std::shuffle(remaining, remaining + nr, rng);

        // Fill community to 5
        Card full_comm[5];
        std::copy(community, community + n_comm, full_comm);
        int ri = 0;
        for (int i = n_comm; i < 5; i++) full_comm[i] = remaining[ri++];

        // Deal opponents
        uint16_t my_rank = evaluate_7cards(h0, h1,
            full_comm[0], full_comm[1], full_comm[2], full_comm[3], full_comm[4]);
        bool win = true;
        for (int opp = 0; opp < num_opponents; opp++) {
            Card oh0 = remaining[ri++], oh1 = remaining[ri++];
            uint16_t opp_rank = evaluate_7cards(oh0, oh1,
                full_comm[0], full_comm[1], full_comm[2], full_comm[3], full_comm[4]);
            if (opp_rank >= my_rank) { win = false; break; }
        }
        wins += win ? 1 : 0;
        total++;
    }
    float equity = (float)wins / total;
    return std::min((int)(equity * POSTFLOP_BUCKETS), POSTFLOP_BUCKETS - 1);
}

void precompute_buckets(const Card* hole_cards,
                        const Card* community,
                        int num_players,
                        int* hole_buckets,
                        int* board_buckets) {
    for (int p = 0; p < num_players; p++)
        hole_buckets[p] = preflop_bucket(hole_cards[p*2], hole_cards[p*2+1]);

    board_buckets[0] = 0;  // preflop
    int n_comm_per_street[3] = {3, 4, 5};
    for (int s = 0; s < 3; s++) {
        // Use player 0's hole cards as reference for board bucket
        board_buckets[s + 1] = fast_postflop_bucket(
            hole_cards[0], hole_cards[1],
            community, n_comm_per_street[s]);
    }
}

// ---------------------------------------------------------------------------
// Action abstraction
// ---------------------------------------------------------------------------
uint8_t valid_actions_mask(int pot, int stack, int to_call) {
    if (pot <= 0) pot = 1;
    uint8_t mask = 0;

    if (to_call == 0) {
        mask |= (1 << (int)Action::CHECK);
    } else if (to_call >= stack) {
        mask |= (1 << (int)Action::FOLD);
        mask |= (1 << (int)Action::ALL_IN);
        return mask;
    } else {
        mask |= (1 << (int)Action::FOLD);
        mask |= (1 << (int)Action::CALL);
    }

    int headroom = stack - to_call;
    if (headroom <= 0) return mask;

    if (headroom > pot / 4 && pot / 4 > 0)
        mask |= (1 << (int)Action::RAISE_QUARTER);
    if (headroom > pot / 2 && pot / 2 > 0)
        mask |= (1 << (int)Action::RAISE_HALF);
    if (headroom > pot / 3 && pot / 3 > 0)
        mask |= (1 << (int)Action::RAISE_THIRD);
    if (headroom > pot)
        mask |= (1 << (int)Action::RAISE_POT);
    mask |= (1 << (int)Action::ALL_IN);
    return mask;
}

int valid_actions_list(int pot, int stack, int to_call, Action* out) {
    uint8_t mask = valid_actions_mask(pot, stack, to_call);
    int n = 0;
    for (int a = 0; a < NUM_ACTIONS; a++)
        if (mask & (1 << a)) out[n++] = (Action)a;
    return n;
}

int action_to_chips(Action a, int pot, int stack, int to_call) {
    if (pot <= 0) pot = 1;
    switch (a) {
        case Action::FOLD:          return 0;
        case Action::CHECK:         return 0;
        case Action::CALL:          return std::min(to_call, stack);
        case Action::RAISE_QUARTER: return std::min(to_call + std::max(1, pot/4), stack);
        case Action::RAISE_HALF:    return std::min(to_call + std::max(1, pot/2), stack);
        case Action::RAISE_THIRD:   return std::min(to_call + std::max(1, pot/3), stack);
        case Action::RAISE_POT:     return std::min(to_call + pot,               stack);
        case Action::ALL_IN:        return stack;
        default:                    return 0;
    }
}

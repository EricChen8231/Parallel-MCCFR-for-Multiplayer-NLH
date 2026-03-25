#include "abstraction.h"
#include "hand_eval.h"
#include <algorithm>
#include <vector>
#include <random>

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
    // Map rank [1, 7462] uniformly to bucket [0, POSTFLOP_BUCKETS-1].
    // Formula matches board_bucket_gpu() in cfr_gpu.cu exactly so that
    // CPU and GPU bucket assignments are consistent.
    return std::min((int)((uint32_t)(rank - 1) * POSTFLOP_BUCKETS / 7462u),
                    POSTFLOP_BUCKETS - 1);
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
            if (opp_rank > my_rank) { win = false; break; }
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
    static const int n_comm_per_street[4] = {0, 3, 4, 5};
    for (int p = 0; p < num_players; p++) {
        hole_buckets[p] = preflop_bucket(hole_cards[p*2], hole_cards[p*2+1]);
        // Each player gets their own board bucket since equity depends on hole cards.
        // Matches the per-player board_bucket_gpu() computation in cfr_gpu.cu.
        board_buckets[p * 4 + 0] = 0;  // preflop: no community cards yet
        for (int s = 1; s < 4; s++) {
            board_buckets[p * 4 + s] = fast_postflop_bucket(
                hole_cards[p*2], hole_cards[p*2+1],
                community, n_comm_per_street[s]);
        }
    }
}

// ---------------------------------------------------------------------------
// Action abstraction
// ---------------------------------------------------------------------------
static int min_raise_to_total(int current_bet, int last_full_raise, int bb_amt)
{
    if (current_bet == 0) return std::max(1, bb_amt);
    return current_bet + std::max(last_full_raise, bb_amt);
}

static int chips_from_raise_to(int raise_to, int stack, int to_call, int current_bet)
{
    return std::min(to_call + std::max(0, raise_to - current_bet), stack);
}

static int raise_target_for_action(Action a, int pot, int current_bet,
                                   int min_raise_to)
{
    switch (a) {
        case Action::RAISE_QUARTER: return std::max(min_raise_to, current_bet + std::max(1, pot / 4));
        case Action::RAISE_HALF:    return std::max(min_raise_to, current_bet + std::max(1, pot / 2));
        case Action::RAISE_THIRD:   return std::max(min_raise_to, current_bet + std::max(1, pot / 3));
        case Action::RAISE_POT:     return std::max(min_raise_to, current_bet + std::max(1, pot));
        default:                    return min_raise_to;
    }
}

uint8_t valid_actions_mask(int pot, int stack, int to_call,
                           int current_bet, int last_full_raise, int bb_amt) {
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

    const int max_raise_to = current_bet + headroom;
    const int min_raise_to = min_raise_to_total(current_bet, last_full_raise, bb_amt);
    if (max_raise_to >= min_raise_to) {
        if (raise_target_for_action(Action::RAISE_QUARTER, pot, current_bet, min_raise_to) <= max_raise_to)
            mask |= (1 << (int)Action::RAISE_QUARTER);
        if (raise_target_for_action(Action::RAISE_HALF, pot, current_bet, min_raise_to) <= max_raise_to)
            mask |= (1 << (int)Action::RAISE_HALF);
        if (raise_target_for_action(Action::RAISE_THIRD, pot, current_bet, min_raise_to) <= max_raise_to)
            mask |= (1 << (int)Action::RAISE_THIRD);
        if (raise_target_for_action(Action::RAISE_POT, pot, current_bet, min_raise_to) <= max_raise_to)
            mask |= (1 << (int)Action::RAISE_POT);
    }
    mask |= (1 << (int)Action::ALL_IN);
    return mask;
}

int valid_actions_list(int pot, int stack, int to_call,
                       int current_bet, int last_full_raise, int bb_amt,
                       Action* out) {
    uint8_t mask = valid_actions_mask(
        pot, stack, to_call, current_bet, last_full_raise, bb_amt);
    int n = 0;
    for (int a = 0; a < NUM_ACTIONS; a++)
        if (mask & (1 << a)) out[n++] = (Action)a;
    return n;
}

int action_to_chips(Action a, int pot, int stack, int to_call,
                    int current_bet, int last_full_raise, int bb_amt) {
    if (pot <= 0) pot = 1;
    const int min_raise_to = min_raise_to_total(current_bet, last_full_raise, bb_amt);
    const int max_raise_to = current_bet + std::max(0, stack - to_call);
    switch (a) {
        case Action::FOLD:          return 0;
        case Action::CHECK:         return 0;
        case Action::CALL:          return std::min(to_call, stack);
        case Action::RAISE_QUARTER:
        case Action::RAISE_HALF:
        case Action::RAISE_THIRD:
        case Action::RAISE_POT: {
            const int raise_to = std::min(
                raise_target_for_action(a, pot, current_bet, min_raise_to),
                max_raise_to);
            return chips_from_raise_to(raise_to, stack, to_call, current_bet);
        }
        case Action::ALL_IN:        return stack;
        default:                    return 0;
    }
}

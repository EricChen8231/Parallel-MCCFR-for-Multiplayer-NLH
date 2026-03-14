// =============================================================================
// eval.cpp — CPU-side hand evaluator for strategy quality measurement
//
// Plays N hands of 2-player NLH: trained bot vs. scripted opponent.
// Used for Experiments 2, 3, and 4 from the project proposal.
//
// Design notes:
//   - Info-set hash is the same FNV-1a formula as info_set_hash() in cfr_gpu.cu
//     so the lookup key matches what was stored during GPU training.
//   - Positions alternate every hand for an unbiased BB/100 estimate.
//   - "Bot player" index 0 or 1 is passed to the hash; the trained strategy
//     table has entries for both player indices.
//   - All community cards are pre-dealt at game start (matching GPU training).
// =============================================================================
#include "eval.h"
#include "abstraction.h"
#include "hand_eval.h"
#include "card.h"

#include <random>
#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// CPU info-set hash — identical to the GPU kernel's info_set_hash()
// ---------------------------------------------------------------------------
static inline uint32_t info_set_hash_cpu(uint8_t player, uint8_t hole_b,
                                          uint8_t board_b, uint8_t street,
                                          uint32_t action_bits)
{
    uint32_t h = 2166136261u;
    h = (h ^ player)  * 16777619u;
    h = (h ^ hole_b)  * 16777619u;
    h = (h ^ board_b) * 16777619u;
    h = (h ^ street)  * 16777619u;
    h = (h ^ (uint8_t)(action_bits         & 0xFF)) * 16777619u;
    h = (h ^ (uint8_t)((action_bits >>  8) & 0xFF)) * 16777619u;
    h = (h ^ (uint8_t)((action_bits >> 16) & 0xFF)) * 16777619u;
    h = (h ^ (uint8_t)((action_bits >> 24) & 0xFF)) * 16777619u;
    return h & (GPU_TABLE_SIZE - 1);
}

// ---------------------------------------------------------------------------
// Sample one action from a probability array, masked to valid actions.
// ---------------------------------------------------------------------------
static int sample_from_probs(const float* probs, uint8_t valid_mask,
                              std::mt19937& rng)
{
    float total = 0.f;
    for (int a = 0; a < GPU_NUM_ACTIONS; a++)
        if (valid_mask & (1 << a)) total += probs[a];

    if (total < 1e-7f) {
        // Zero-probability table entry: uniform fallback
        std::vector<int> valid;
        for (int a = 0; a < GPU_NUM_ACTIONS; a++)
            if (valid_mask & (1 << a)) valid.push_back(a);
        return valid[rng() % valid.size()];
    }

    float r = std::uniform_real_distribution<float>(0.f, total)(rng);
    float cum = 0.f;
    int last_valid = -1;
    for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
        if (!(valid_mask & (1 << a))) continue;
        cum += probs[a];
        last_valid = a;
        if (r <= cum) return a;
    }
    return last_valid;
}

// ---------------------------------------------------------------------------
// Trained-bot action: look up HostStrategyTable and sample.
// ---------------------------------------------------------------------------
static int trained_action(const HostStrategyTable& strat,
                           uint8_t player, uint8_t hole_b, uint8_t board_b,
                           uint8_t street, uint32_t action_bits,
                           uint8_t valid_mask, std::mt19937& rng)
{
    uint32_t key = info_set_hash_cpu(player, hole_b, board_b, street, action_bits);
    auto it = strat.find(key);
    if (it == strat.end()) {
        // Info set never visited during training: uniform over valid actions
        std::vector<int> valid;
        for (int a = 0; a < GPU_NUM_ACTIONS; a++)
            if (valid_mask & (1 << a)) valid.push_back(a);
        return valid[rng() % valid.size()];
    }
    return sample_from_probs(it->second.probs, valid_mask, rng);
}

// ---------------------------------------------------------------------------
// Scripted-opponent action: fixed fold/call/raise probability distribution.
//
// When to_call == 0 (can CHECK), fold probability is absorbed into check
// probability since folding a free check is dominated.
// Raise probability is split evenly among available raise actions.
// ---------------------------------------------------------------------------
static int scripted_action(OpponentType opp, uint8_t valid_mask, std::mt19937& rng)
{
    float fold_p, call_p, raise_p;
    switch (opp) {
        case OpponentType::CALLING_STATION: fold_p=0.08f; call_p=0.72f; raise_p=0.20f; break;
        case OpponentType::NIT:             fold_p=0.60f; call_p=0.30f; raise_p=0.10f; break;
        case OpponentType::MANIAC:          fold_p=0.10f; call_p=0.20f; raise_p=0.70f; break;
        case OpponentType::BALANCED:        fold_p=0.28f; call_p=0.44f; raise_p=0.28f; break;
        default: /* RANDOM */               fold_p=0.f;   call_p=0.f;   raise_p=0.f;   break;
    }

    // RANDOM: uniform over all valid actions
    if (opp == OpponentType::RANDOM) {
        std::vector<int> valid;
        for (int a = 0; a < GPU_NUM_ACTIONS; a++)
            if (valid_mask & (1 << a)) valid.push_back(a);
        return valid[rng() % valid.size()];
    }

    float probs[GPU_NUM_ACTIONS] = {};
    bool can_check = (valid_mask & (1 << 1)) != 0;
    bool can_fold  = (valid_mask & (1 << 0)) != 0;
    bool can_call  = (valid_mask & (1 << 2)) != 0;

    if (can_check) {
        // Free check available: fold is dominated; absorb fold_p + call_p into check
        probs[1] = fold_p + call_p;
    } else {
        if (can_fold) probs[0] = fold_p;
        if (can_call) probs[2] = call_p;
    }

    int n_raises = 0;
    for (int a = 3; a < GPU_NUM_ACTIONS; a++)
        if (valid_mask & (1 << a)) n_raises++;

    if (n_raises > 0) {
        float per = raise_p / (float)n_raises;
        for (int a = 3; a < GPU_NUM_ACTIONS; a++)
            if (valid_mask & (1 << a)) probs[a] = per;
    } else {
        // No raise possible: add raise_p to the dominant passive action
        if (can_check)     probs[1] += raise_p;
        else if (can_call) probs[2] += raise_p;
        else if (can_fold) probs[0] += raise_p;
    }

    return sample_from_probs(probs, valid_mask, rng);
}

// ---------------------------------------------------------------------------
// Game state for a single 2-player hand
// ---------------------------------------------------------------------------
struct Game2P {
    Card hole[2][2];
    Card community[5];
    int  stacks[2], bets[2], invested[2];
    bool folded[2], all_in_flag[2];
    int  pot, current_bet;
    uint32_t action_bits;
    int  street;
    int  hole_buckets[2];
    int  board_buckets[2][4];  // [player][street]
};

// ---------------------------------------------------------------------------
// Run one betting round.
// first_to_act: which player (0 or 1) acts first this round.
// Returns false if a player folded (hand ends immediately).
// ---------------------------------------------------------------------------
static bool play_betting_round(Game2P& g, int first_to_act, int bot_player,
                                const HostStrategyTable& strat, OpponentType opp,
                                std::mt19937& rng)
{
    bool needs_to_act[2];
    for (int p = 0; p < 2; p++)
        needs_to_act[p] = !g.folded[p] && !g.all_in_flag[p];

    int p = first_to_act;
    for (int iters = 0; iters < 20 && (needs_to_act[0] || needs_to_act[1]); iters++) {
        if (!needs_to_act[p]) { p = 1 - p; continue; }

        int     to_call = g.current_bet - g.bets[p];
        uint8_t vm      = valid_actions_mask(g.pot, g.stacks[p], to_call);

        int chosen;
        if (p == bot_player) {
            chosen = trained_action(strat,
                (uint8_t)p,
                (uint8_t)g.hole_buckets[p],
                (uint8_t)g.board_buckets[p][g.street],
                (uint8_t)g.street,
                g.action_bits, vm, rng);
        } else {
            chosen = scripted_action(opp, vm, rng);
        }

        // Compress 3-bit action into history; mask to 30 bits (matches GPU)
        g.action_bits = ((g.action_bits << 3) | (uint32_t)chosen) & 0x3FFFFFFFu;
        needs_to_act[p] = false;

        if (chosen == 0) { g.folded[p] = true; return false; }  // FOLD

        if (chosen != 1) {  // not CHECK
            int chips = action_to_chips((Action)chosen, g.pot, g.stacks[p], to_call);
            g.stacks[p]   -= chips;
            g.bets[p]     += chips;
            g.invested[p] += chips;
            g.pot         += chips;
            if (g.stacks[p] == 0) { g.all_in_flag[p] = true; needs_to_act[p] = false; }

            if (g.bets[p] > g.current_bet) {   // raise — opponent must respond
                g.current_bet   = g.bets[p];
                needs_to_act[1-p] = !g.folded[1-p] && !g.all_in_flag[1-p];
            }
        }
        p = 1 - p;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Play one complete hand.
//
// bot_player: which player index (0 or 1) the trained bot occupies.
// P0 is always SB (posts SB, acts first preflop), P1 is always BB.
// Postflop: P0 (SB) acts first — matches the GPU training convention where
// the postflop queue starts from sb_pos (see traverse_gpu in cfr_gpu.cu).
//
// Returns net chips won by bot_player (positive = profit).
// ---------------------------------------------------------------------------
static int play_hand(Game2P& g, Deck& deck, std::mt19937& rng,
                     int bot_player, const HostStrategyTable& strat,
                     OpponentType opp, int starting_stack, int sb_amt, int bb_amt)
{
    // Deal
    deck.shuffle(rng);
    for (int p = 0; p < 2; p++) { g.hole[p][0] = deck.deal(); g.hole[p][1] = deck.deal(); }
    for (int i = 0; i < 5; i++) g.community[i] = deck.deal();

    // Init state
    for (int p = 0; p < 2; p++) {
        g.stacks[p] = starting_stack; g.bets[p] = 0; g.invested[p] = 0;
        g.folded[p] = false; g.all_in_flag[p] = false;
    }
    g.pot = 0; g.current_bet = 0; g.action_bits = 0; g.street = 0;

    // Post blinds: P0=SB, P1=BB (matches GPU sb_pos/bb_pos convention)
    int s0 = std::min(sb_amt, g.stacks[0]);
    g.stacks[0] -= s0; g.bets[0] = s0; g.invested[0] = s0;
    int b1 = std::min(bb_amt, g.stacks[1]);
    g.stacks[1] -= b1; g.bets[1] = b1; g.invested[1] = b1;
    g.pot = s0 + b1; g.current_bet = b1;
    if (!g.stacks[0]) g.all_in_flag[0] = true;
    if (!g.stacks[1]) g.all_in_flag[1] = true;

    // Precompute buckets (per-player, matching GPU board_bucket_gpu convention)
    Card hole_flat[4] = { g.hole[0][0], g.hole[0][1], g.hole[1][0], g.hole[1][1] };
    int hb[2], bb_flat[8];
    precompute_buckets(hole_flat, g.community, 2, hb, bb_flat);
    for (int p = 0; p < 2; p++) {
        g.hole_buckets[p] = hb[p];
        for (int s = 0; s < 4; s++) g.board_buckets[p][s] = bb_flat[p * 4 + s];
    }

    // Preflop: P0 (SB) acts first.  Postflop: also P0 acts first (GPU convention).
    static const int first_to_act[4] = { 0, 0, 0, 0 };

    for (int st = 0; st < 4; st++) {
        g.street = st;
        if (st > 0) { g.bets[0] = g.bets[1] = 0; g.current_bet = 0; }

        int active = (!g.folded[0] ? 1 : 0) + (!g.folded[1] ? 1 : 0);
        if (active <= 1) break;

        if (!play_betting_round(g, first_to_act[st], bot_player, strat, opp, rng))
            break;  // someone folded

        active = (!g.folded[0] ? 1 : 0) + (!g.folded[1] ? 1 : 0);
        if (active <= 1 || st == 3) break;
    }

    // Payoff for bot_player
    int opp_player = 1 - bot_player;
    if (g.folded[opp_player]) return g.pot - g.invested[bot_player];
    if (g.folded[bot_player]) return -g.invested[bot_player];

    // Showdown
    uint16_t rb = evaluate_7cards(g.hole[bot_player][0], g.hole[bot_player][1],
        g.community[0], g.community[1], g.community[2], g.community[3], g.community[4]);
    uint16_t ro = evaluate_7cards(g.hole[opp_player][0], g.hole[opp_player][1],
        g.community[0], g.community[1], g.community[2], g.community[3], g.community[4]);

    if (rb > ro) return g.pot - g.invested[bot_player];   // bot wins
    if (ro > rb) return -g.invested[bot_player];           // opponent wins
    return g.pot / 2 - g.invested[bot_player];            // chop (integer rounding OK for eval)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
EvalResult evaluate_strategy(const HostStrategyTable& strat, OpponentType opp,
                              long long n_hands, int stack, int sb, int bb,
                              unsigned seed)
{
    std::mt19937 rng(seed);
    Deck deck;
    Game2P g;

    long long net_chips = 0;
    for (long long h = 0; h < n_hands; h++) {
        // Alternate which player the bot occupies for position fairness.
        // h even → bot is SB (player 0), h odd → bot is BB (player 1).
        int bot_player = (int)(h & 1);
        net_chips += play_hand(g, deck, rng, bot_player, strat, opp, stack, sb, bb);
    }

    EvalResult res;
    res.hands_played = n_hands;
    res.net_bb       = (double)net_chips / (double)bb;
    res.bb_per_100   = res.net_bb / (double)n_hands * 100.0;
    return res;
}

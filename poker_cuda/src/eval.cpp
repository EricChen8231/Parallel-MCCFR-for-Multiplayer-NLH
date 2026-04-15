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

#include <cctype>
#include <cstdlib>
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
static int fallback_action_passive(uint8_t valid_mask);

static int sample_from_probs(const float* probs, uint8_t valid_mask,
                              std::mt19937& rng)
{
    float total = 0.f;
    for (int a = 0; a < GPU_NUM_ACTIONS; a++)
        if (valid_mask & (1 << a)) total += probs[a];

    if (total < 1e-7f) {
        // Degenerate table entry: prefer a passive legal action.
        return fallback_action_passive(valid_mask);
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

static int fallback_action_passive(uint8_t valid_mask)
{
    if (valid_mask & (1 << 1)) return 1;  // CHECK
    if (valid_mask & (1 << 2)) return 2;  // CALL
    if (valid_mask & (1 << 0)) return 0;  // FOLD
    for (int a = 3; a < GPU_NUM_ACTIONS - 1; a++)
        if (valid_mask & (1 << a)) return a;
    if (valid_mask & (1 << 7)) return 7;  // ALL_IN as last resort
    return 1;
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
        // Unseen state: prefer the passive legal action over a random shove.
        return fallback_action_passive(valid_mask);
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
    int  pot, current_bet, last_full_raise, bb_amt;
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
        uint8_t vm      = valid_actions_mask(
            g.pot, g.stacks[p], to_call, g.current_bet, g.last_full_raise, g.bb_amt);

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
            int chips = action_to_chips(
                (Action)chosen, g.pot, g.stacks[p], to_call,
                g.current_bet, g.last_full_raise, g.bb_amt);
            const int prev_bet = g.current_bet;
            g.stacks[p]   -= chips;
            g.bets[p]     += chips;
            g.invested[p] += chips;
            g.pot         += chips;
            if (g.stacks[p] == 0) { g.all_in_flag[p] = true; needs_to_act[p] = false; }

            if (g.bets[p] > g.current_bet) {   // raise — opponent must respond
                const int raise_size = g.bets[p] - prev_bet;
                if (raise_size >= g.last_full_raise || prev_bet == 0)
                    g.last_full_raise = raise_size;
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
// P0 is always SB/button, P1 is always BB.
// Heads-up action order: SB acts first preflop, BB acts first postflop.
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
    g.pot = 0;
    g.current_bet = 0;
    g.last_full_raise = bb_amt;
    g.bb_amt = bb_amt;
    g.action_bits = 0;
    g.street = 0;

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

    // Heads-up NLH: SB/button acts first preflop, BB acts first postflop.
    static const int first_to_act[4] = { 0, 1, 1, 1 };

    for (int st = 0; st < 4; st++) {
        g.street = st;
        if (st > 0) {
            g.bets[0] = g.bets[1] = 0;
            g.current_bet = 0;
            g.last_full_raise = g.bb_amt;
        }

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
// N-player game state
// ---------------------------------------------------------------------------
struct GameNP {
    int  n;
    Card hole[MAX_EVAL_PLAYERS][2];
    Card community[5];
    int  stacks[MAX_EVAL_PLAYERS];
    int  bets[MAX_EVAL_PLAYERS];
    int  invested[MAX_EVAL_PLAYERS];
    bool folded[MAX_EVAL_PLAYERS];
    bool all_in[MAX_EVAL_PLAYERS];
    int  pot, current_bet, last_full_raise, bb_amt;
    uint32_t action_bits;
    int  street;
    int  hole_buckets[MAX_EVAL_PLAYERS];
    int  board_buckets[MAX_EVAL_PLAYERS][4];
};

// ---------------------------------------------------------------------------
// N-player betting round.
// Returns false if only one player remains (everyone else folded).
// ---------------------------------------------------------------------------
static bool play_betting_round_np(GameNP& g, int first_to_act, int bot_player,
                                   const HostStrategyTable& strat, OpponentType opp,
                                   std::mt19937& rng)
{
    int n = g.n;
    bool needs_to_act[MAX_EVAL_PLAYERS] = {};
    for (int p = 0; p < n; p++)
        needs_to_act[p] = !g.folded[p] && !g.all_in[p];

    int p = first_to_act;
    int safety = n * n * 4;
    for (int guard = 0; guard < safety; guard++) {
        bool anyone = false;
        for (int i = 0; i < n; i++) if (needs_to_act[i]) { anyone = true; break; }
        if (!anyone) break;

        if (!needs_to_act[p]) { p = (p + 1) % n; continue; }

        int     to_call = g.current_bet - g.bets[p];
        uint8_t vm      = valid_actions_mask(
            g.pot, g.stacks[p], to_call, g.current_bet, g.last_full_raise, g.bb_amt);

        int chosen;
        if (p == bot_player)
            chosen = trained_action(strat, (uint8_t)p,
                (uint8_t)g.hole_buckets[p],
                (uint8_t)g.board_buckets[p][g.street],
                (uint8_t)g.street, g.action_bits, vm, rng);
        else
            chosen = scripted_action(opp, vm, rng);

        g.action_bits = ((g.action_bits << 3) | (uint32_t)chosen) & 0x3FFFFFFFu;
        needs_to_act[p] = false;

        if (chosen == 0) {  // FOLD
            g.folded[p] = true;
            int active = 0;
            for (int i = 0; i < n; i++) if (!g.folded[i]) active++;
            if (active <= 1) return false;
        } else if (chosen != 1) {  // not CHECK
            int chips = action_to_chips(
                (Action)chosen, g.pot, g.stacks[p], to_call,
                g.current_bet, g.last_full_raise, g.bb_amt);
            const int prev_bet = g.current_bet;
            g.stacks[p]   -= chips;
            g.bets[p]     += chips;
            g.invested[p] += chips;
            g.pot         += chips;
            if (g.stacks[p] == 0) g.all_in[p] = true;

            if (g.bets[p] > g.current_bet) {  // raise: reopen action
                const int raise_size = g.bets[p] - prev_bet;
                if (raise_size >= g.last_full_raise || prev_bet == 0)
                    g.last_full_raise = raise_size;
                g.current_bet = g.bets[p];
                for (int i = 0; i < n; i++)
                    if (i != p && !g.folded[i] && !g.all_in[i])
                        needs_to_act[i] = true;
            }
        }
        p = (p + 1) % n;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Side-pot showdown. Returns net chips won by bot_player.
// ---------------------------------------------------------------------------
static void compute_showdown_winnings_np(const GameNP& g,
                                         uint16_t* ranks,
                                         int* winnings)
{
    int n = g.n;

    // Evaluate hands for all non-folded players
    for (int p = 0; p < n; p++) {
        winnings[p] = 0;
        if (!g.folded[p])
            ranks[p] = evaluate_7cards(
                g.hole[p][0], g.hole[p][1],
                g.community[0], g.community[1], g.community[2],
                g.community[3], g.community[4]);
        else
            ranks[p] = 0;
    }

    // Side-pot algorithm: peel off one level of investment at a time.
    int remaining[MAX_EVAL_PLAYERS];
    for (int p = 0; p < n; p++) remaining[p] = g.invested[p];

    while (true) {
        // Find the smallest non-zero contribution remaining
        int min_inv = INT_MAX;
        for (int p = 0; p < n; p++)
            if (remaining[p] > 0 && remaining[p] < min_inv) min_inv = remaining[p];
        if (min_inv == INT_MAX) break;

        // Build this side pot and mark eligible players
        int  side_pot = 0;
        bool eligible[MAX_EVAL_PLAYERS] = {};
        for (int p = 0; p < n; p++) {
            if (remaining[p] <= 0) continue;
            int contrib = std::min(remaining[p], min_inv);
            side_pot      += contrib;
            remaining[p]  -= contrib;
            if (!g.folded[p]) eligible[p] = true;
        }

        // Find best hand among eligible players
        uint16_t best = 0;
        for (int p = 0; p < n; p++)
            if (eligible[p] && ranks[p] > best) best = ranks[p];

        int n_winners = 0;
        for (int p = 0; p < n; p++)
            if (eligible[p] && ranks[p] == best) n_winners++;

        int share = side_pot / n_winners;
        for (int p = 0; p < n; p++)
            if (eligible[p] && ranks[p] == best) winnings[p] += share;
    }
}

static int resolve_showdown_np(GameNP& g, int bot_player)
{
    uint16_t ranks[MAX_EVAL_PLAYERS] = {};
    int winnings[MAX_EVAL_PLAYERS] = {};
    compute_showdown_winnings_np(g, ranks, winnings);
    return winnings[bot_player] - g.invested[bot_player];
}

// ---------------------------------------------------------------------------
// Play one N-player hand. Returns net chips for bot_player (seat 0).
// dealer: position of the dealer button (0..n-1); rotates each hand.
// ---------------------------------------------------------------------------
static int play_hand_np(GameNP& g, Deck& deck, std::mt19937& rng,
                         int bot_player, const HostStrategyTable& strat,
                         OpponentType opp, int starting_stack,
                         int sb_amt, int bb_amt, int dealer)
{
    int n = g.n;

    deck.shuffle(rng);
    for (int p = 0; p < n; p++) {
        g.hole[p][0] = deck.deal();
        g.hole[p][1] = deck.deal();
    }
    for (int i = 0; i < 5; i++) g.community[i] = deck.deal();

    for (int p = 0; p < n; p++) {
        g.stacks[p]  = starting_stack;
        g.bets[p]    = g.invested[p] = 0;
        g.folded[p]  = g.all_in[p]   = false;
    }
    g.pot = g.current_bet = 0;
    g.last_full_raise = bb_amt;
    g.bb_amt = bb_amt;
    g.action_bits = 0;
    g.street = 0;

    int sb_pos = (n == 2) ? dealer : (dealer + 1) % n;
    int bb_pos = (n == 2) ? ((dealer + 1) % n) : (dealer + 2) % n;

    // Post blinds
    auto post = [&](int p, int amt) {
        int chips = std::min(amt, g.stacks[p]);
        g.stacks[p] -= chips;
        g.bets[p]    = chips;
        g.invested[p]= chips;
        g.pot       += chips;
        if (!g.stacks[p]) g.all_in[p] = true;
    };
    post(sb_pos, sb_amt);
    post(bb_pos, bb_amt);
    g.current_bet = g.bets[bb_pos];

    // Precompute abstraction buckets
    Card hole_flat[MAX_EVAL_PLAYERS * 2];
    for (int p = 0; p < n; p++) {
        hole_flat[p*2]   = g.hole[p][0];
        hole_flat[p*2+1] = g.hole[p][1];
    }
    int hb[MAX_EVAL_PLAYERS], bb_flat[MAX_EVAL_PLAYERS * 4];
    precompute_buckets(hole_flat, g.community, n, hb, bb_flat);
    for (int p = 0; p < n; p++) {
        g.hole_buckets[p] = hb[p];
        for (int s = 0; s < 4; s++)
            g.board_buckets[p][s] = bb_flat[p * 4 + s];
    }

    for (int st = 0; st < 4; st++) {
        g.street = st;
        if (st > 0) {
            for (int p = 0; p < n; p++) g.bets[p] = 0;
            g.current_bet = 0;
            g.last_full_raise = g.bb_amt;
        }

        int active = 0;
        for (int p = 0; p < n; p++) if (!g.folded[p]) active++;
        if (active <= 1) break;

        // Check if all remaining players are all-in (run out the board)
        int can_act = 0;
        for (int p = 0; p < n; p++) if (!g.folded[p] && !g.all_in[p]) can_act++;
        if (can_act <= 1) { if (st == 3) break; continue; }

        // First to act: preflop = UTG (bb+1), postflop = first active left of dealer
        int first = (st == 0) ? (bb_pos + 1) % n : (dealer + 1) % n;
        if (st > 0) {
            for (int i = 0; i < n; i++) {
                int pp = (dealer + 1 + i) % n;
                if (!g.folded[pp] && !g.all_in[pp]) { first = pp; break; }
            }
        }

        if (!play_betting_round_np(g, first, bot_player, strat, opp, rng)) break;

        active = 0;
        for (int p = 0; p < n; p++) if (!g.folded[p]) active++;
        if (active <= 1 || st == 3) break;
    }

    // If only one player remains, they win the pot uncontested
    int last = -1, active = 0;
    for (int p = 0; p < n; p++) if (!g.folded[p]) { last = p; active++; }

    if (active == 1)
        return (last == bot_player ? g.pot : 0) - g.invested[bot_player];

    return resolve_showdown_np(g, bot_player);
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

EvalResult evaluate_strategy_np(const HostStrategyTable& strat, int n_players,
                                  OpponentType opp, long long n_hands,
                                  int stack, int sb, int bb, unsigned seed)
{
    std::mt19937 rng(seed);
    Deck deck;
    GameNP g;
    g.n = n_players;

    long long net_chips = 0;
    for (long long h = 0; h < n_hands; h++) {
        // Rotate dealer button each hand for position fairness.
        // Bot is always seat 0; it cycles through all positions over n_players hands.
        int dealer = (int)(h % n_players);
        net_chips += play_hand_np(g, deck, rng, /*bot_player=*/0,
                                   strat, opp, stack, sb, bb, dealer);
    }

    EvalResult res;
    res.hands_played = n_hands;
    res.net_bb       = (double)net_chips / (double)bb;
    res.bb_per_100   = res.net_bb / (double)n_hands * 100.0;
    return res;
}

// =============================================================================
// Human-play mode
// =============================================================================

// ---------------------------------------------------------------------------
// Hand strength category name (Two Plus Two rank → English)
// ---------------------------------------------------------------------------
static const char* hand_category_name(uint16_t rank)
{
    if (rank <= 1277) return "high card";
    if (rank <= 4137) return "one pair";
    if (rank <= 4995) return "two pair";
    if (rank <= 5853) return "three of a kind";
    if (rank <= 5863) return "straight";
    if (rank <= 7140) return "flush";
    if (rank <= 7296) return "full house";
    if (rank <= 7452) return "four of a kind";
    if (rank <= 7461) return "straight flush";
    return "royal flush";
}

struct HumanDecision {
    int action = 1;
    int chips  = 0;
};

static int raise_to_from_chips(int chips, int to_call, int current_bet)
{
    return current_bet + std::max(0, chips - to_call);
}

static int chips_from_raise_to(int raise_to, int stack, int to_call, int current_bet)
{
    return std::min(to_call + std::max(0, raise_to - current_bet), stack);
}

static int min_raise_to_total(int current_bet, int last_full_raise, int bb_amt)
{
    if (current_bet == 0) return bb_amt;
    return current_bet + std::max(last_full_raise, bb_amt);
}

static int legal_raise_chips_for_action(int action, int pot, int stack, int to_call,
                                        int current_bet, int last_full_raise, int bb_amt)
{
    return action_to_chips(
        (Action)action, pot, stack, to_call,
        current_bet, last_full_raise, bb_amt);
}

static int action_for_custom_raise(int chips, int pot, int stack, int to_call,
                                   int current_bet, int last_full_raise, int bb_amt)
{
    if (chips >= stack) return 7;

    int best_action = 3;
    int best_diff   = std::abs(chips - legal_raise_chips_for_action(
        3, pot, stack, to_call, current_bet, last_full_raise, bb_amt));

    for (int a = 4; a <= 6; a++) {
        const int diff = std::abs(chips - legal_raise_chips_for_action(
            a, pot, stack, to_call, current_bet, last_full_raise, bb_amt));
        if (diff < best_diff) {
            best_diff = diff;
            best_action = a;
        }
    }
    return best_action;
}

// ---------------------------------------------------------------------------
// Print a description of the bot's action.
// ---------------------------------------------------------------------------
// current_bet: highest bet on the table BEFORE this action (used to show
//              "raises to X" total).
static void print_bot_action(int a, int chips, int to_call, int current_bet)
{
    switch (a) {
        case 0: printf("  Bot folds.\n");                         break;
        case 1: printf("  Bot checks.\n");                        break;
        case 2: printf("  Bot calls %d.\n", chips);               break;
        case 7: {
            const int raise_to = raise_to_from_chips(chips, to_call, current_bet);
            printf("  Bot goes all-in: raises to %d (%d chips).\n",
                   raise_to, chips);
            break;
        }
        default: {
            const int raise_to = raise_to_from_chips(chips, to_call, current_bet);
            printf("  Bot raises to %d (puts in %d).\n", raise_to, chips);
            break;
        }
    }
}

static const char* seat_role_tag(int seat, int dealer, int n_players)
{
    if (n_players == 2) {
        if (seat == dealer) return "BTN/SB";
        if (seat == ((dealer + 1) % n_players)) return "BB";
        return "";
    }
    if (seat == dealer) return "BTN";
    if (seat == ((dealer + 1) % n_players)) return "SB";
    if (seat == ((dealer + 2) % n_players)) return "BB";
    return "";
}

static void print_seat_action(int seat, int a, int chips, int to_call, int current_bet)
{
    switch (a) {
        case 0:
            printf("  Seat %d folds.\n", seat);
            break;
        case 1:
            printf("  Seat %d checks.\n", seat);
            break;
        case 2:
            printf("  Seat %d calls %d.\n", seat, chips);
            break;
        case 7: {
            const int raise_to = raise_to_from_chips(chips, to_call, current_bet);
            printf("  Seat %d goes all-in: raises to %d (%d chips).\n",
                   seat, raise_to, chips);
            break;
        }
        default: {
            const int raise_to = raise_to_from_chips(chips, to_call, current_bet);
            printf("  Seat %d raises to %d (puts in %d).\n",
                   seat, raise_to, chips);
            break;
        }
    }
}

static void print_table_state_np(const GameNP& g, int human_player, int dealer,
                                 bool show_all_cards)
{
    printf("  Table:\n");
    for (int p = 0; p < g.n; p++) {
        const char* role = seat_role_tag(p, dealer, g.n);
        printf("    Seat %d", p);
        if (p == human_player) printf(" (You)");
        if (role[0] != '\0')   printf(" [%s]", role);
        printf("  Stack: %d", g.stacks[p]);
        if (g.folded[p]) printf("  [folded]");
        else if (g.all_in[p]) printf("  [all-in]");
        if (show_all_cards || p == human_player) {
            printf("  Cards: %s %s",
                   card_to_str(g.hole[p][0]).c_str(),
                   card_to_str(g.hole[p][1]).c_str());
        }
        printf("\n");
    }
}

// ---------------------------------------------------------------------------
// Interactive action prompt — reads from stdin, returns an abstract action
// for history lookup plus the exact chip amount to put in.
// current_bet: the highest bet on the table this street (used to compute
//              "raise to X" totals for display).
// ---------------------------------------------------------------------------
static HumanDecision human_input_action(uint8_t vm, int pot, int stack,
                                        int to_call, int current_bet,
                                        int last_full_raise, int bb_amt)
{
    const bool can_raise = (vm & ((1 << 3) | (1 << 4) | (1 << 5) | (1 << 6))) != 0;
    const int min_raise_to = min_raise_to_total(current_bet, last_full_raise, bb_amt);
    const int max_raise_to = current_bet + (stack - to_call);
    const int quarter_chips = can_raise
        ? legal_raise_chips_for_action(3, pot, stack, to_call, current_bet, last_full_raise, bb_amt)
        : 0;
    const int half_chips = can_raise
        ? legal_raise_chips_for_action(4, pot, stack, to_call, current_bet, last_full_raise, bb_amt)
        : 0;
    const int pot_chips = can_raise
        ? legal_raise_chips_for_action(6, pot, stack, to_call, current_bet, last_full_raise, bb_amt)
        : 0;

    printf("\n  Pot: %d  |  To call: %d  |  Your stack: %d\n",
           pot, to_call, stack);
    printf("  Your action:\n");
    if (vm & (1 << 0)) printf("    [f] Fold\n");
    if (vm & (1 << 1)) printf("    [k] Check\n");
    if (vm & (1 << 2)) printf("    [c] Call %d\n", std::min(to_call, stack));
    if (can_raise) {
        const int quarter_to = raise_to_from_chips(quarter_chips, to_call, current_bet);
        if (quarter_to == min_raise_to) {
            printf("    [3] Min raise to %d  (put in %d)\n", quarter_to, quarter_chips);
        } else {
            printf("    [3] Quarter-pot raise to %d  (put in %d)\n", quarter_to, quarter_chips);
        }
        if (half_chips != quarter_chips && half_chips < stack) {
            printf("    [4] Half-pot raise to %d  (put in %d)\n",
                   raise_to_from_chips(half_chips, to_call, current_bet), half_chips);
        }
        if (pot_chips != half_chips && pot_chips != quarter_chips && pot_chips < stack) {
            printf("    [6] Pot raise to %d  (put in %d)\n",
                   raise_to_from_chips(pot_chips, to_call, current_bet), pot_chips);
        }
        printf("    [r N] Raise to N  (any legal total from %d to %d)\n",
               min_raise_to, max_raise_to);
    }
    if (vm & (1 << 7)) {
        const int raise_to = raise_to_from_chips(stack, to_call, current_bet);
        printf("    [a] All-in: raise to %d  (put in %d)\n", raise_to, stack);
    }

    char buf[64];
    for (;;) {
        printf("  > ");
        fflush(stdout);
        if (!fgets(buf, sizeof(buf), stdin))
            return { (vm & 1) ? 0 : 1, 0 };

        const char ch = buf[0];
        if ((ch == 'f' || ch == 'F') && (vm & (1 << 0))) return { 0, 0 };
        if ((ch == 'k' || ch == 'K' || ch == 'x') && (vm & (1 << 1))) return { 1, 0 };
        if ((ch == 'c' || ch == 'C') && (vm & (1 << 2)))
            return { 2, std::min(to_call, stack) };
        if ((ch == 'a' || ch == 'A') && (vm & (1 << 7))) return { 7, stack };

        if (can_raise && ch == '3')
            return { 3, quarter_chips };
        if (can_raise && ch == '4' && half_chips != quarter_chips && half_chips < stack)
            return { 4, half_chips };
        if (can_raise && ch == '6' && pot_chips != half_chips &&
            pot_chips != quarter_chips && pot_chips < stack)
            return { 6, pot_chips };

        const char* num_start = buf;
        if (ch == 'r' || ch == 'R') {
            num_start++;
            while (*num_start && std::isspace((unsigned char)*num_start)) num_start++;
        }

        if (can_raise && std::isdigit((unsigned char)*num_start)) {
            char* end = nullptr;
            const long raise_to = std::strtol(num_start, &end, 10);
            if (end != num_start) {
                if (raise_to < min_raise_to || raise_to > max_raise_to) {
                    printf("  Illegal raise total. Enter a total from %d to %d.\n",
                           min_raise_to, max_raise_to);
                    continue;
                }

                const int chips = chips_from_raise_to((int)raise_to, stack, to_call, current_bet);
                return { action_for_custom_raise(chips, pot, stack, to_call,
                                                 current_bet, last_full_raise, bb_amt),
                         chips };
            }
        }

        printf("  Invalid. Options: ");
        if (vm & 1)   printf("f ");
        if (vm & 2)   printf("k ");
        if (vm & 4)   printf("c ");
        if (can_raise) printf("3 ");
        if (can_raise && half_chips != quarter_chips && half_chips < stack) printf("4 ");
        if (can_raise && pot_chips != half_chips &&
            pot_chips != quarter_chips && pot_chips < stack) printf("6 ");
        if (can_raise) printf("r <total> ");
        if (vm & 128) printf("a ");
        printf("\n");
    }
}

// ---------------------------------------------------------------------------
// Play one heads-up hand: human vs. bot.  Returns net chips for human.
// ---------------------------------------------------------------------------
static int play_hand_human(Deck& deck, std::mt19937& rng,
                            int human_player,
                            const HostStrategyTable& strat,
                            int starting_stack, int sb_amt, int bb_amt,
                            bool show_all_cards)
{
    int bot_player = 1 - human_player;

    // Deal
    deck.shuffle(rng);
    Game2P g;
    for (int p = 0; p < 2; p++) {
        g.hole[p][0] = deck.deal();
        g.hole[p][1] = deck.deal();
    }
    for (int i = 0; i < 5; i++) g.community[i] = deck.deal();

    // Init
    for (int p = 0; p < 2; p++) {
        g.stacks[p] = starting_stack;
        g.bets[p]   = g.invested[p] = 0;
        g.folded[p] = g.all_in_flag[p] = false;
    }
    g.pot = 0; g.current_bet = 0; g.action_bits = 0; g.street = 0;

    // Post blinds: P0 = SB, P1 = BB
    int s0 = std::min(sb_amt, g.stacks[0]);
    g.stacks[0] -= s0; g.bets[0] = s0; g.invested[0] = s0;
    int b1 = std::min(bb_amt, g.stacks[1]);
    g.stacks[1] -= b1; g.bets[1] = b1; g.invested[1] = b1;
    g.pot = s0 + b1; g.current_bet = b1;
    if (!g.stacks[0]) g.all_in_flag[0] = true;
    if (!g.stacks[1]) g.all_in_flag[1] = true;
    int last_full_raise = bb_amt;

    // Precompute abstraction buckets
    Card hole_flat[4] = { g.hole[0][0], g.hole[0][1],
                          g.hole[1][0], g.hole[1][1] };
    int hb[2], bb_flat[8];
    precompute_buckets(hole_flat, g.community, 2, hb, bb_flat);
    for (int p = 0; p < 2; p++) {
        g.hole_buckets[p] = hb[p];
        for (int s = 0; s < 4; s++)
            g.board_buckets[p][s] = bb_flat[p * 4 + s];
    }

    static const char* SNAMES[] = { "Preflop", "Flop", "Turn", "River" };
    static const int   N_COMM[] = { 0, 3, 4, 5 };

    for (int st = 0; st < 4; st++) {
        g.street = st;
        if (st > 0) {
            for (int p = 0; p < 2; p++) g.bets[p] = 0;
            g.current_bet = 0;
            last_full_raise = bb_amt;
        }

        int active = (!g.folded[0] ? 1 : 0) + (!g.folded[1] ? 1 : 0);
        if (active <= 1) break;

        // Show street header
        int n_comm = N_COMM[st];
        printf("\n  --- %s ---\n", SNAMES[st]);
        if (n_comm > 0) {
            printf("  Board: ");
            for (int i = 0; i < n_comm; i++) {
                printf("%s", card_to_str(g.community[i]).c_str());
                if (i < n_comm - 1) printf(" ");
            }
            printf("\n");
        }
        printf("  Your hand: %s %s\n",
               card_to_str(g.hole[human_player][0]).c_str(),
               card_to_str(g.hole[human_player][1]).c_str());
        if (show_all_cards) {
            printf("  Bot hand: %s %s\n",
                   card_to_str(g.hole[bot_player][0]).c_str(),
                   card_to_str(g.hole[bot_player][1]).c_str());
        }
        printf("  Stacks  You: %d  |  Bot: %d  |  Pot: %d\n",
               g.stacks[human_player], g.stacks[bot_player], g.pot);

        // If both all-in just run the board silently
        if (g.all_in_flag[0] && g.all_in_flag[1]) {
            if (st < 3) printf("  (all-in — running board)\n");
            if (st == 3) break;
            continue;
        }

        // Betting round
        bool needs_to_act[2] = {
            !g.folded[0] && !g.all_in_flag[0],
            !g.folded[1] && !g.all_in_flag[1]
        };
        bool folded_out = false;

        const int first_to_act = (st == 0) ? 0 : 1;
        for (int guard = 0, p = first_to_act;
             guard < 20 && (needs_to_act[0] || needs_to_act[1]);
             guard++)
        {
            if (!needs_to_act[p]) { p = 1 - p; continue; }

            int     to_call = g.current_bet - g.bets[p];
            uint8_t vm      = valid_actions_mask(
                g.pot, g.stacks[p], to_call, g.current_bet, last_full_raise, bb_amt);

            HumanDecision decision;
            if (p == human_player) {
                decision = human_input_action(vm, g.pot, g.stacks[p],
                                              to_call, g.current_bet,
                                              last_full_raise, bb_amt);
            } else {
                decision.action = trained_action(strat,
                    (uint8_t)p,
                    (uint8_t)g.hole_buckets[p],
                    (uint8_t)g.board_buckets[p][st],
                    (uint8_t)st, g.action_bits, vm, rng);
                decision.chips = legal_raise_chips_for_action(
                    decision.action, g.pot, g.stacks[p], to_call,
                    g.current_bet, last_full_raise, bb_amt);
                print_bot_action(decision.action, decision.chips, to_call, g.current_bet);
            }

            const int chosen = decision.action;
            g.action_bits = ((g.action_bits << 3) | (uint32_t)chosen) & 0x3FFFFFFFu;
            needs_to_act[p] = false;

            if (chosen == 0) {          // FOLD
                g.folded[p] = true;
                folded_out  = true;
                break;
            }
            if (chosen != 1) {          // not CHECK
                const int chips = decision.chips;
                const int prev_bet = g.current_bet;
                g.stacks[p]   -= chips;
                g.bets[p]     += chips;
                g.invested[p] += chips;
                g.pot         += chips;
                if (g.stacks[p] == 0) {
                    g.all_in_flag[p] = true;
                    needs_to_act[p]  = false;
                }
                if (g.bets[p] > g.current_bet) {
                    const int raise_size = g.bets[p] - prev_bet;
                    if (raise_size >= last_full_raise || prev_bet == 0)
                        last_full_raise = raise_size;
                    g.current_bet     = g.bets[p];
                    needs_to_act[1-p] = !g.folded[1-p] && !g.all_in_flag[1-p];
                }
            }
            p = 1 - p;
        }

        if (folded_out) break;

        active = (!g.folded[0] ? 1 : 0) + (!g.folded[1] ? 1 : 0);
        if (active <= 1 || st == 3) break;
    }

    // Resolve
    int net;
    if (g.folded[bot_player]) {
        printf("\n  Bot folds. You win the pot of %d!\n", g.pot);
        net = g.pot - g.invested[human_player];
    } else if (g.folded[human_player]) {
        printf("\n  You fold. Bot wins %d.\n", g.pot);
        net = -g.invested[human_player];
    } else {
        // Showdown
        printf("\n  --- Showdown ---\n");
        printf("  Board: ");
        for (int i = 0; i < 5; i++) {
            printf("%s", card_to_str(g.community[i]).c_str());
            if (i < 4) printf(" ");
        }
        printf("\n");

        uint16_t rh = evaluate_7cards(
            g.hole[human_player][0], g.hole[human_player][1],
            g.community[0], g.community[1], g.community[2],
            g.community[3], g.community[4]);
        uint16_t rb = evaluate_7cards(
            g.hole[bot_player][0], g.hole[bot_player][1],
            g.community[0], g.community[1], g.community[2],
            g.community[3], g.community[4]);

        printf("  You: %s %s  (%s, rank=%u)\n",
               card_to_str(g.hole[human_player][0]).c_str(),
               card_to_str(g.hole[human_player][1]).c_str(),
               hand_category_name(rh), (unsigned)rh);
        printf("  Bot: %s %s  (%s, rank=%u)\n",
               card_to_str(g.hole[bot_player][0]).c_str(),
               card_to_str(g.hole[bot_player][1]).c_str(),
               hand_category_name(rb), (unsigned)rb);

        if (rh > rb) {
            printf("  You win the pot of %d!\n", g.pot);
            net = g.pot - g.invested[human_player];
        } else if (rb > rh) {
            printf("  Bot wins the pot of %d.\n", g.pot);
            net = -g.invested[human_player];
        } else {
            printf("  Chop! You split %d each.\n", g.pot / 2);
            net = g.pot / 2 - g.invested[human_player];
        }
    }
    return net;
}

static int play_hand_human_np(GameNP& g, Deck& deck, std::mt19937& rng,
                              int human_player, int dealer,
                              const HostStrategyTable& strat,
                              int starting_stack, int sb_amt, int bb_amt,
                              bool show_all_cards)
{
    const int n = g.n;

    deck.shuffle(rng);
    for (int p = 0; p < n; p++) {
        g.hole[p][0] = deck.deal();
        g.hole[p][1] = deck.deal();
    }
    for (int i = 0; i < 5; i++) g.community[i] = deck.deal();

    for (int p = 0; p < n; p++) {
        g.stacks[p]  = starting_stack;
        g.bets[p]    = g.invested[p] = 0;
        g.folded[p]  = g.all_in[p]   = false;
    }
    g.pot = g.current_bet = 0;
    g.last_full_raise = bb_amt;
    g.bb_amt = bb_amt;
    g.action_bits = 0;
    g.street = 0;

    const int sb_pos = (n == 2) ? dealer : (dealer + 1) % n;
    const int bb_pos = (n == 2) ? ((dealer + 1) % n) : (dealer + 2) % n;

    auto post = [&](int p, int amt) {
        const int chips = std::min(amt, g.stacks[p]);
        g.stacks[p]  -= chips;
        g.bets[p]     = chips;
        g.invested[p] = chips;
        g.pot        += chips;
        if (!g.stacks[p]) g.all_in[p] = true;
    };
    post(sb_pos, sb_amt);
    post(bb_pos, bb_amt);
    g.current_bet = g.bets[bb_pos];

    Card hole_flat[MAX_EVAL_PLAYERS * 2];
    for (int p = 0; p < n; p++) {
        hole_flat[p*2]   = g.hole[p][0];
        hole_flat[p*2+1] = g.hole[p][1];
    }
    int hb[MAX_EVAL_PLAYERS], bb_flat[MAX_EVAL_PLAYERS * 4];
    precompute_buckets(hole_flat, g.community, n, hb, bb_flat);
    for (int p = 0; p < n; p++) {
        g.hole_buckets[p] = hb[p];
        for (int s = 0; s < 4; s++)
            g.board_buckets[p][s] = bb_flat[p * 4 + s];
    }

    static const char* SNAMES[] = { "Preflop", "Flop", "Turn", "River" };
    static const int   N_COMM[] = { 0, 3, 4, 5 };

    for (int st = 0; st < 4; st++) {
        g.street = st;
        if (st > 0) {
            for (int p = 0; p < n; p++) g.bets[p] = 0;
            g.current_bet = 0;
            g.last_full_raise = g.bb_amt;
        }

        int active = 0;
        for (int p = 0; p < n; p++) if (!g.folded[p]) active++;
        if (active <= 1) break;

        const int n_comm = N_COMM[st];
        printf("\n  --- %s ---\n", SNAMES[st]);
        if (n_comm > 0) {
            printf("  Board: ");
            for (int i = 0; i < n_comm; i++) {
                printf("%s", card_to_str(g.community[i]).c_str());
                if (i < n_comm - 1) printf(" ");
            }
            printf("\n");
        }
        printf("  Your hand: %s %s\n",
               card_to_str(g.hole[human_player][0]).c_str(),
               card_to_str(g.hole[human_player][1]).c_str());
        printf("  Pot: %d\n", g.pot);
        print_table_state_np(g, human_player, dealer, show_all_cards);

        int can_act = 0;
        for (int p = 0; p < n; p++) if (!g.folded[p] && !g.all_in[p]) can_act++;
        if (can_act <= 1) {
            if (st < 3) printf("  (all-in — running board)\n");
            if (st == 3) break;
            continue;
        }

        int first = (st == 0) ? (bb_pos + 1) % n : (dealer + 1) % n;
        if (st > 0) {
            for (int i = 0; i < n; i++) {
                const int pp = (dealer + 1 + i) % n;
                if (!g.folded[pp] && !g.all_in[pp]) { first = pp; break; }
            }
        }

        bool needs_to_act[MAX_EVAL_PLAYERS] = {};
        for (int p = 0; p < n; p++)
            needs_to_act[p] = !g.folded[p] && !g.all_in[p];

        bool folded_out = false;
        int p = first;
        const int safety = n * n * 4;
        for (int guard = 0; guard < safety; guard++) {
            bool anyone = false;
            for (int i = 0; i < n; i++) if (needs_to_act[i]) { anyone = true; break; }
            if (!anyone) break;

            if (!needs_to_act[p]) { p = (p + 1) % n; continue; }

            const int to_call = g.current_bet - g.bets[p];
            const uint8_t vm = valid_actions_mask(
                g.pot, g.stacks[p], to_call, g.current_bet, g.last_full_raise, g.bb_amt);

            int chosen = 1;
            int chips = 0;
            if (p == human_player) {
                HumanDecision decision = human_input_action(
                    vm, g.pot, g.stacks[p], to_call, g.current_bet,
                    g.last_full_raise, g.bb_amt);
                chosen = decision.action;
                chips  = decision.chips;
            } else {
                chosen = trained_action(strat, (uint8_t)p,
                    (uint8_t)g.hole_buckets[p],
                    (uint8_t)g.board_buckets[p][st],
                    (uint8_t)st, g.action_bits, vm, rng);
                chips = legal_raise_chips_for_action(
                    chosen, g.pot, g.stacks[p], to_call,
                    g.current_bet, g.last_full_raise, g.bb_amt);
                print_seat_action(p, chosen, chips, to_call, g.current_bet);
            }

            g.action_bits = ((g.action_bits << 3) | (uint32_t)chosen) & 0x3FFFFFFFu;
            needs_to_act[p] = false;

            if (chosen == 0) {
                g.folded[p] = true;
                int still_active = 0;
                for (int i = 0; i < n; i++) if (!g.folded[i]) still_active++;
                if (still_active <= 1) {
                    folded_out = true;
                    break;
                }
            } else if (chosen != 1) {
                const int prev_bet = g.current_bet;
                g.stacks[p]   -= chips;
                g.bets[p]     += chips;
                g.invested[p] += chips;
                g.pot         += chips;
                if (g.stacks[p] == 0) g.all_in[p] = true;

                if (g.bets[p] > g.current_bet) {
                    const int raise_size = g.bets[p] - prev_bet;
                    if (raise_size >= g.last_full_raise || prev_bet == 0)
                        g.last_full_raise = raise_size;
                    g.current_bet = g.bets[p];
                    for (int i = 0; i < n; i++)
                        if (i != p && !g.folded[i] && !g.all_in[i])
                            needs_to_act[i] = true;
                }
            }
            p = (p + 1) % n;
        }

        if (folded_out) break;

        active = 0;
        for (int pp = 0; pp < n; pp++) if (!g.folded[pp]) active++;
        if (active <= 1 || st == 3) break;
    }

    int last = -1, active = 0;
    for (int p = 0; p < n; p++) if (!g.folded[p]) { last = p; active++; }

    if (active == 1) {
        if (last == human_player) {
            printf("\n  Everyone else folds. You win the pot of %d!\n", g.pot);
            return g.pot - g.invested[human_player];
        }
        printf("\n  Seat %d wins the pot of %d.\n", last, g.pot);
        return -g.invested[human_player];
    }

    uint16_t ranks[MAX_EVAL_PLAYERS] = {};
    int winnings[MAX_EVAL_PLAYERS] = {};
    compute_showdown_winnings_np(g, ranks, winnings);

    printf("\n  --- Showdown ---\n");
    printf("  Board: ");
    for (int i = 0; i < 5; i++) {
        printf("%s", card_to_str(g.community[i]).c_str());
        if (i < 4) printf(" ");
    }
    printf("\n");
    for (int p = 0; p < n; p++) {
        if (g.folded[p]) continue;
        printf("  Seat %d%s: %s %s  (%s, rank=%u)",
               p,
               p == human_player ? " (You)" : "",
               card_to_str(g.hole[p][0]).c_str(),
               card_to_str(g.hole[p][1]).c_str(),
               hand_category_name(ranks[p]),
               (unsigned)ranks[p]);
        if (winnings[p] > 0) printf("  collects %d", winnings[p]);
        printf("\n");
    }

    const int net = winnings[human_player] - g.invested[human_player];
    if (net > 0) printf("  You win chips at showdown.\n");
    else if (net < 0) printf("  You lose chips at showdown.\n");
    else printf("  You break even at showdown.\n");
    return net;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
void play_vs_human(const HostStrategyTable& strat, int n_players, long long n_hands,
                   int starting_stack, int sb_amt, int bb_amt,
                   bool show_all_cards, unsigned seed)
{
    std::mt19937 rng(seed);
    Deck deck;
    long long total_chips  = 0;
    long long hands_played = 0;

    if (n_players <= 2) {
        printf("\n================================================================\n");
        printf("  Human vs Bot  |  Stack: %d  |  Blinds: %d/%d  |  Hands: %lld\n",
               starting_stack, sb_amt, bb_amt, n_hands);
        printf("  Cards:   2-9 T J Q K A   suits: c d h s\n");
        printf("  Actions: [f]old  [k]heck  [c]all  [3]/[4]/[6]=preset raises  [r N]=raise to N  [a]ll-in\n");
        printf("================================================================\n");

        for (long long h = 0; h < n_hands; h++) {
            int human_player = (int)(h & 1);   // alternate SB/BB each hand
            printf("\n[ Hand %lld  |  You: %s  |  Bot: %s ]\n",
                   h + 1,
                   human_player == 0 ? "SB" : "BB",
                   human_player == 0 ? "BB" : "SB");

            int net = play_hand_human(deck, rng, human_player, strat,
                                      starting_stack, sb_amt, bb_amt,
                                      show_all_cards);
            total_chips  += net;
            hands_played++;

            double net_bb = (double)total_chips / (double)bb_amt;
            double bb100  = net_bb / (double)hands_played * 100.0;
            printf("\n  Hand result: %+d chips  |  Total: %+.1f BB  |  BB/100: %+.2f\n",
                   net, net_bb, bb100);

            if (h + 1 < n_hands) {
                printf("  [Enter = next hand, q = quit] ");
                fflush(stdout);
                char buf[8];
                if (!fgets(buf, sizeof(buf), stdin) || buf[0] == 'q' || buf[0] == 'Q')
                    break;
            }
        }

        double net_bb = (double)total_chips / (double)bb_amt;
        printf("\n================================================================\n");
        printf("  Session: %lld hands  |  Net: %+.1f BB  |  BB/100: %+.2f\n",
               hands_played, net_bb,
               hands_played > 0 ? net_bb / (double)hands_played * 100.0 : 0.0);
        printf("================================================================\n");
        return;
    }

    printf("\n================================================================\n");
    printf("  Human vs Bots  |  Players: %d  |  You: Seat 0  |  Stack: %d  |  Blinds: %d/%d  |  Hands: %lld\n",
           n_players, starting_stack, sb_amt, bb_amt, n_hands);
    printf("  Cards:   2-9 T J Q K A   suits: c d h s\n");
    printf("  Actions: [f]old  [k]heck  [c]all  [3]/[4]/[6]=preset raises  [r N]=raise to N  [a]ll-in\n");
    printf("================================================================\n");

    GameNP g;
    g.n = n_players;
    for (long long h = 0; h < n_hands; h++) {
        const int human_player = 0;
        const int dealer = (int)(h % n_players);
        const char* role = seat_role_tag(human_player, dealer, n_players);
        if (role[0] != '\0') {
            printf("\n[ Hand %lld  |  You: Seat %d [%s]  |  Dealer: Seat %d ]\n",
                   h + 1, human_player, role, dealer);
        } else {
            printf("\n[ Hand %lld  |  You: Seat %d  |  Dealer: Seat %d ]\n",
                   h + 1, human_player, dealer);
        }

        int net = play_hand_human_np(g, deck, rng, human_player, dealer, strat,
                                     starting_stack, sb_amt, bb_amt,
                                     show_all_cards);
        total_chips  += net;
        hands_played++;

        double net_bb = (double)total_chips / (double)bb_amt;
        double bb100  = net_bb / (double)hands_played * 100.0;
        printf("\n  Hand result: %+d chips  |  Total: %+.1f BB  |  BB/100: %+.2f\n",
               net, net_bb, bb100);

        if (h + 1 < n_hands) {
            printf("  [Enter = next hand, q = quit] ");
            fflush(stdout);
            char buf[8];
            if (!fgets(buf, sizeof(buf), stdin) || buf[0] == 'q' || buf[0] == 'Q')
                break;
        }
    }

    double net_bb = (double)total_chips / (double)bb_amt;
    printf("\n================================================================\n");
    printf("  Session: %lld hands  |  Net: %+.1f BB  |  BB/100: %+.2f\n",
           hands_played, net_bb,
           hands_played > 0 ? net_bb / (double)hands_played * 100.0 : 0.0);
    printf("================================================================\n");
}

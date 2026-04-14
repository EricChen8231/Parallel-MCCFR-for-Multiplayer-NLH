// =============================================================================
// bot.cpp — Live-play bot: Bayesian opponent modeling + GTO/exploit blending
//
// Algorithm (from project proposal Section 2):
//
//   At each decision point the bot computes:
//     gto_probs    — normalized average strategy from trained table
//     exploit_probs — action probabilities adjusted to exploit observed tendencies
//     w = EXPLOIT_WEIGHT * n / (n + k),  k = 600
//     blended = (1 - w) * gto + w * exploit
//
//   Opponent tendencies are tracked as aggregate fold/call/raise counts.
//   The balanced reference profile is (0.28 / 0.44 / 0.28).
//   Deviations from this reference drive the exploit adjustment:
//     fold_dev  > 0 (folds too much)  → raise more (bluffing is +EV)
//     call_dev  > 0 (calls too much)  → raise more (value betting is +EV)
//     raise_dev > 0 (raises too much) → fold tighter (re-steal is -EV)
//
// The game loop mirrors eval.cpp, with two additions:
//   1. After each opponent decision: observe(action, valid_mask)
//   2. Bot uses blended_probs() instead of pure GTO lookup
//
// Results are accumulated in window_size-hand buckets for Experiment 3's
// rolling BB/100 plot.
// =============================================================================
#include "bot.h"
#include "abstraction.h"
#include "hand_eval.h"
#include "card.h"

#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cassert>

// ---------------------------------------------------------------------------
// OpponentObs rates — uniform prior at balanced baseline when n == 0
// ---------------------------------------------------------------------------
float OpponentObs::fold_rate()  const {
    long long n = total();
    return n ? (float)fold_count  / (float)n : 0.28f;
}
float OpponentObs::call_rate()  const {
    long long n = total();
    return n ? (float)call_count  / (float)n : 0.44f;
}
float OpponentObs::raise_rate() const {
    long long n = total();
    return n ? (float)raise_count / (float)n : 0.28f;
}

// ---------------------------------------------------------------------------
// FNV-1a info-set hash — identical to GPU kernel and eval.cpp
// ---------------------------------------------------------------------------
static inline uint32_t fnv_hash(uint8_t player, uint8_t hole_b, uint8_t board_b,
                                 uint8_t street, uint32_t action_bits)
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
// LiveBot constructor
// ---------------------------------------------------------------------------
LiveBot::LiveBot(const HostStrategyTable& strat, int stack, int sb, int bb)
    : strat_(strat), stack_(stack), sb_(sb), bb_(bb) {}

// ---------------------------------------------------------------------------
// gto_probs — load normalized strategy from table, passive fallback
// ---------------------------------------------------------------------------
static int fallback_action_passive(uint8_t valid_mask)
{
    if (valid_mask & (1 << 1)) return 1;  // CHECK
    if (valid_mask & (1 << 2)) return 2;  // CALL
    if (valid_mask & (1 << 0)) return 0;  // FOLD
    for (int a = 3; a < GPU_NUM_ACTIONS; a++)
        if (valid_mask & (1 << a)) return a;
    return 1;
}

void LiveBot::gto_probs(float* out, uint8_t player, uint8_t hole_b,
                         uint8_t board_b, uint8_t street,
                         uint32_t action_bits, uint8_t valid_mask) const
{
    uint32_t key = fnv_hash(player, hole_b, board_b, street, action_bits);
    auto it = strat_.find(key);
    if (it != strat_.end()) {
        memcpy(out, it->second.probs, GPU_NUM_ACTIONS * sizeof(float));
        return;
    }
    // Info set not seen during training: prefer a passive legal action.
    std::fill(out, out + GPU_NUM_ACTIONS, 0.f);
    out[fallback_action_passive(valid_mask)] = 1.f;
}

// ---------------------------------------------------------------------------
// blended_probs — GTO + confidence-scaled exploit adjustment
//
// Exploit heuristic (applied multiplicatively to GTO probs):
//   raise_signal = fold_dev * 1.5 + call_dev * 0.5   (range: [-0.6, +0.6])
//   fold_signal  = raise_dev * 1.5                    (range: [-0.4, +0.4])
//
//   raise_signal > 0 → raise actions scaled up, passive actions scaled down
//   fold_signal  > 0 → fold action scaled up, call action scaled down
//
// The result is re-normalized and blended with GTO at weight w.
// ---------------------------------------------------------------------------
void LiveBot::blended_probs(float* out, const float* gto, uint8_t vmask) const
{
    long long n = obs_.total();
    float w = EXPLOIT_WEIGHT * (float)n / (float)(n + OPP_MODEL_K);

    if (w < 1e-4f) {
        memcpy(out, gto, GPU_NUM_ACTIONS * sizeof(float));
        return;
    }

    // Deviations from balanced reference (28 / 44 / 28)
    float fold_dev  = obs_.fold_rate()  - 0.28f;
    float call_dev  = obs_.call_rate()  - 0.44f;
    float raise_dev = obs_.raise_rate() - 0.28f;

    float raise_signal = std::clamp(fold_dev * 1.5f + call_dev * 0.5f, -0.6f,  0.6f);
    float fold_signal  = std::clamp(raise_dev * 1.5f,                   -0.4f,  0.4f);

    // Build exploit probs via multiplicative scaling
    float exploit[GPU_NUM_ACTIONS];
    memcpy(exploit, gto, GPU_NUM_ACTIONS * sizeof(float));

    if (std::abs(raise_signal) > 1e-3f) {
        float raise_mult   = 1.f + raise_signal;
        float passive_mult = 1.f - raise_signal * 0.4f;
        for (int a = 3; a < GPU_NUM_ACTIONS; a++)
            if (vmask & (1 << a)) exploit[a] *= raise_mult;
        for (int a = 0; a <= 2; a++)
            if (vmask & (1 << a)) exploit[a] *= passive_mult;
    }

    if (std::abs(fold_signal) > 1e-3f) {
        if (vmask & 1) exploit[0] *= (1.f + fold_signal);
        if (vmask & (1 << 2)) exploit[2] *= (1.f - fold_signal * 0.5f);
    }

    // Clamp, blend, normalize
    float sum = 0.f;
    for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
        out[a] = std::max(0.f, (1.f - w) * gto[a] + w * exploit[a]);
        sum += out[a];
    }
    if (sum > 1e-7f)
        for (int a = 0; a < GPU_NUM_ACTIONS; a++) out[a] /= sum;
    else
        memcpy(out, gto, GPU_NUM_ACTIONS * sizeof(float));
}

// ---------------------------------------------------------------------------
// sample_action
// ---------------------------------------------------------------------------
int LiveBot::sample_action(const float* probs, uint8_t vmask, std::mt19937& rng)
{
    float total = 0.f;
    for (int a = 0; a < GPU_NUM_ACTIONS; a++)
        if (vmask & (1 << a)) total += probs[a];

    if (total < 1e-7f) {
        return fallback_action_passive(vmask);
    }

    float r = std::uniform_real_distribution<float>(0.f, total)(rng);
    float cum = 0.f;
    int last = -1;
    for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
        if (!(vmask & (1 << a))) continue;
        cum += probs[a];
        last = a;
        if (r <= cum) return a;
    }
    return last;
}

// ---------------------------------------------------------------------------
// choose_action — blended strategy lookup
// ---------------------------------------------------------------------------
int LiveBot::choose_action(uint8_t player, uint8_t hole_b, uint8_t board_b,
                            uint8_t street, uint32_t action_bits,
                            uint8_t valid_mask, std::mt19937& rng) const
{
    float gto[GPU_NUM_ACTIONS];
    gto_probs(gto, player, hole_b, board_b, street, action_bits, valid_mask);

    float blended[GPU_NUM_ACTIONS];
    blended_probs(blended, gto, valid_mask);

    return sample_action(blended, valid_mask, rng);
}

// ---------------------------------------------------------------------------
// observe — update opponent model after an opponent decision
// ---------------------------------------------------------------------------
void LiveBot::observe(int action, uint8_t /*valid_mask*/)
{
    if (action == 0)                        obs_.fold_count++;
    else if (action == 1 || action == 2)    obs_.call_count++;
    else                                    obs_.raise_count++;
}

// ---------------------------------------------------------------------------
// Scripted opponent — identical to eval.cpp (reproduced here to avoid coupling)
// ---------------------------------------------------------------------------
static int scripted_action_bot(OpponentType opp, uint8_t valid_mask,
                                std::mt19937& rng)
{
    float fold_p, call_p, raise_p;
    switch (opp) {
        case OpponentType::CALLING_STATION: fold_p=0.08f; call_p=0.72f; raise_p=0.20f; break;
        case OpponentType::NIT:             fold_p=0.60f; call_p=0.30f; raise_p=0.10f; break;
        case OpponentType::MANIAC:          fold_p=0.10f; call_p=0.20f; raise_p=0.70f; break;
        case OpponentType::BALANCED:        fold_p=0.28f; call_p=0.44f; raise_p=0.28f; break;
        default: /* RANDOM */               fold_p=0.f;   call_p=0.f;   raise_p=0.f;   break;
    }

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
        if (can_check)     probs[1] += raise_p;
        else if (can_call) probs[2] += raise_p;
        else if (can_fold) probs[0] += raise_p;
    }

    return LiveBot::sample_action(probs, valid_mask, rng);
}

// ---------------------------------------------------------------------------
// Game state for one hand (2-player)
// ---------------------------------------------------------------------------
struct Hand2P {
    Card hole[2][2];
    Card community[5];
    int  stacks[2], bets[2], invested[2];
    bool folded[2], all_in_flag[2];
    int  pot, current_bet, last_full_raise, bb_amt;
    uint32_t action_bits;
    int  street;
    int  hole_buckets[2];
    int  board_buckets[2][4];   // [player][street 0-3]
};

// ---------------------------------------------------------------------------
// play_betting_round — one street of action.
// Calls observe() when the opponent (non-bot) player acts.
// Returns false if a fold ended the hand.
// ---------------------------------------------------------------------------
static bool play_betting_round_bot(Hand2P& g, int first_to_act, int bot_player,
                                    LiveBot& bot, OpponentType opp,
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
            chosen = bot.choose_action(
                (uint8_t)p,
                (uint8_t)g.hole_buckets[p],
                (uint8_t)g.board_buckets[p][g.street],
                (uint8_t)g.street,
                g.action_bits, vm, rng);
        } else {
            chosen = scripted_action_bot(opp, vm, rng);
            bot.observe(chosen, vm);   // update opponent model
        }

        g.action_bits = ((g.action_bits << 3) | (uint32_t)chosen) & 0x3FFFFFFFu;
        needs_to_act[p] = false;

        if (chosen == 0) { g.folded[p] = true; return false; }

        if (chosen != 1) {
            int chips = action_to_chips(
                (Action)chosen, g.pot, g.stacks[p], to_call,
                g.current_bet, g.last_full_raise, g.bb_amt);
            const int prev_bet = g.current_bet;
            g.stacks[p]   -= chips;
            g.bets[p]     += chips;
            g.invested[p] += chips;
            g.pot         += chips;
            if (g.stacks[p] == 0) { g.all_in_flag[p] = true; needs_to_act[p] = false; }

            if (g.bets[p] > g.current_bet) {
                const int raise_size = g.bets[p] - prev_bet;
                if (raise_size >= g.last_full_raise || prev_bet == 0)
                    g.last_full_raise = raise_size;
                g.current_bet     = g.bets[p];
                needs_to_act[1-p] = !g.folded[1-p] && !g.all_in_flag[1-p];
            }
        }
        p = 1 - p;
    }
    return true;
}

// ---------------------------------------------------------------------------
// play_hand_bot — one complete hand; returns net chips for bot_player.
// ---------------------------------------------------------------------------
static int play_hand_bot(Hand2P& g, Deck& deck, std::mt19937& rng,
                          int bot_player, LiveBot& bot, OpponentType opp,
                          int starting_stack, int sb_amt, int bb_amt)
{
    deck.shuffle(rng);
    for (int p = 0; p < 2; p++) {
        g.hole[p][0] = deck.deal();
        g.hole[p][1] = deck.deal();
    }
    for (int i = 0; i < 5; i++) g.community[i] = deck.deal();

    for (int p = 0; p < 2; p++) {
        g.stacks[p] = starting_stack;
        g.bets[p] = g.invested[p] = 0;
        g.folded[p] = g.all_in_flag[p] = false;
    }
    g.pot = 0;
    g.current_bet = 0;
    g.last_full_raise = bb_amt;
    g.bb_amt = bb_amt;
    g.action_bits = 0;
    g.street = 0;

    // Post blinds: P0 = SB, P1 = BB
    int s0 = std::min(sb_amt, g.stacks[0]);
    g.stacks[0] -= s0; g.bets[0] = s0; g.invested[0] = s0;
    int b1 = std::min(bb_amt, g.stacks[1]);
    g.stacks[1] -= b1; g.bets[1] = b1; g.invested[1] = b1;
    g.pot = s0 + b1; g.current_bet = b1;
    if (!g.stacks[0]) g.all_in_flag[0] = true;
    if (!g.stacks[1]) g.all_in_flag[1] = true;

    // Precompute abstraction buckets
    Card hole_flat[4] = { g.hole[0][0], g.hole[0][1], g.hole[1][0], g.hole[1][1] };
    int hb[2], bb_flat[8];
    precompute_buckets(hole_flat, g.community, 2, hb, bb_flat);
    for (int p = 0; p < 2; p++) {
        g.hole_buckets[p] = hb[p];
        for (int s = 0; s < 4; s++) g.board_buckets[p][s] = bb_flat[p * 4 + s];
    }

    for (int st = 0; st < 4; st++) {
        g.street = st;
        if (st > 0) {
            g.bets[0] = g.bets[1] = 0;
            g.current_bet = 0;
            g.last_full_raise = g.bb_amt;
        }

        int active = (!g.folded[0] ? 1 : 0) + (!g.folded[1] ? 1 : 0);
        if (active <= 1) break;

        const int first_to_act = (st == 0) ? 0 : 1;
        if (!play_betting_round_bot(g, first_to_act, bot_player, bot, opp, rng)) break;

        active = (!g.folded[0] ? 1 : 0) + (!g.folded[1] ? 1 : 0);
        if (active <= 1 || st == 3) break;
    }

    int opp_player = 1 - bot_player;
    if (g.folded[opp_player]) return g.pot - g.invested[bot_player];
    if (g.folded[bot_player]) return -g.invested[bot_player];

    // Showdown
    uint16_t rb = evaluate_7cards(
        g.hole[bot_player][0], g.hole[bot_player][1],
        g.community[0], g.community[1], g.community[2], g.community[3], g.community[4]);
    uint16_t ro = evaluate_7cards(
        g.hole[opp_player][0], g.hole[opp_player][1],
        g.community[0], g.community[1], g.community[2], g.community[3], g.community[4]);

    if (rb > ro) return g.pot - g.invested[bot_player];
    if (ro > rb) return -g.invested[bot_player];
    return g.pot / 2 - g.invested[bot_player];
}

// ---------------------------------------------------------------------------
// play_session — full session with window-by-window reporting
// ---------------------------------------------------------------------------
std::vector<WindowResult> LiveBot::play_session(
    OpponentType opp, long long n_hands, int window_size, unsigned seed)
{
    std::mt19937 rng(seed);
    Deck deck;
    Hand2P g;

    std::vector<WindowResult> windows;
    long long window_chips = 0;
    long long window_start = 1;

    for (long long h = 0; h < n_hands; h++) {
        int bot_player = (int)(h & 1);   // alternate SB/BB each hand
        int chips = play_hand_bot(g, deck, rng, bot_player, *this, opp,
                                  stack_, sb_, bb_);
        window_chips += chips;

        long long hand_num = h + 1;
        if (hand_num % window_size == 0 || hand_num == n_hands) {
            double net_bb     = (double)window_chips / (double)bb_;
            double hands_in_w = (double)(hand_num - window_start + 1);
            double bb100      = net_bb / hands_in_w * 100.0;

            long long n = obs_.total();
            float w = EXPLOIT_WEIGHT * (float)n / (float)(n + OPP_MODEL_K);

            windows.push_back({
                window_start,  hand_num,
                bb100,
                obs_.fold_rate(), obs_.call_rate(), obs_.raise_rate(),
                n, w
            });

            window_chips = 0;
            window_start = hand_num + 1;
        }
    }
    return windows;
}

// ---------------------------------------------------------------------------
// print_summary — Experiment 3 table output
// ---------------------------------------------------------------------------
void LiveBot::print_summary(const std::vector<WindowResult>& windows) const
{
    printf("\n");
    printf("%-14s  %-8s  %-6s  %-6s  %-6s  %-8s  %s\n",
           "Hands", "BB/100", "Fold%", "Call%", "Raise%", "ExplW", "ObsN");
    printf("%s\n", std::string(72, '-').c_str());
    for (const auto& w : windows) {
        printf("[%5lld–%5lld]  %+7.2f  %5.1f%%  %5.1f%%  %5.1f%%   %5.3f  %lld\n",
               w.hand_start, w.hand_end,
               w.bb_per_100,
               w.opp_fold_rate  * 100.f,
               w.opp_call_rate  * 100.f,
               w.opp_raise_rate * 100.f,
               w.exploit_weight,
               w.opp_obs_count);
    }

    if (!windows.empty()) {
        double first_bb100 = windows.front().bb_per_100;
        double last_bb100  = windows.back().bb_per_100;
        printf("\n  Trend (first window → last window): %+.2f → %+.2f BB/100  (Δ = %+.2f)\n",
               first_bb100, last_bb100, last_bb100 - first_bb100);
        printf("  Final exploit weight: %.3f  (n = %lld observations)\n",
               windows.back().exploit_weight, windows.back().opp_obs_count);
    }
}

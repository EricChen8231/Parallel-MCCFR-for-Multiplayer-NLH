// =============================================================================
// test_main.cpp — CPU-only unit tests for poker CFR project
//
// Covers:
//   Module 1: Hand evaluator  (hand_eval.cpp)
//   Module 2: Abstraction     (abstraction.cpp)
//   Module 4: Strategy I/O    (strategy.cpp — save/load only, no GPU)
//
// Run: ./build/test_poker  [path/to/handranks.dat]
//   Default handranks path: poker_cuda/data/handranks.dat
//
// Prints PASS/FAIL per test and a final summary.
// Returns 0 if all tests pass, 1 otherwise.
// =============================================================================

#include "hand_eval.h"
#include "abstraction.h"
#include "strategy.h"
#include "card.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <random>
#include <algorithm>

// ---------------------------------------------------------------------------
// Tiny test framework
// ---------------------------------------------------------------------------
static int  g_pass = 0, g_fail = 0;
static bool g_hand_eval_loaded = false;  // set to true once handranks.dat is loaded

#define EXPECT(cond, name) do { \
    if (cond) { printf("  PASS  %s\n", name); g_pass++; } \
    else       { printf("  FAIL  %s\n", name); g_fail++; } \
} while (0)

#define EXPECT_EQ(a, b, name) EXPECT((a) == (b), name)
#define EXPECT_RANGE(v, lo, hi, name) EXPECT((v) >= (lo) && (v) <= (hi), name)
#define SECTION(s) printf("\n[%s]\n", s)

// ---------------------------------------------------------------------------
// Module 1 — Hand Evaluator
// ---------------------------------------------------------------------------
static void test_hand_evaluator(const char* hr_path)
{
    SECTION("Module 1 — Hand Evaluator");

    // 1.0 Load the table
    bool loaded = hand_eval_init(hr_path);
    EXPECT(loaded, "hand_eval_init returns true");
    g_hand_eval_loaded = loaded;
    if (!loaded) {
        printf("  (skipping remaining hand eval tests — table not loaded)\n");
        return;
    }

    // 1.1 Royal flush: As Ks Qs Js Ts + two irrelevant cards → rank 7462
    // Card encoding: suit*13 + rank,  suit: 0=c,1=d,2=h,3=s,  rank: 0=2,..,12=A
    // As=3*13+12=51  Ks=3*13+11=50  Qs=3*13+10=49  Js=3*13+9=48  Ts=3*13+8=47
    // 2c=0*13+0=0    3d=1*13+1=14
    uint16_t royal = evaluate_7cards(51, 50, 49, 48, 47, 0, 14);
    EXPECT_EQ(royal, 7462, "Royal flush = rank 7462");

    // 1.2 Straight flush (not royal): 9s 8s 7s 6s 5s + 2c 3d
    // 9s=3*13+7=46  8s=3*13+6=45  7s=3*13+5=44  6s=3*13+4=43  5s=3*13+3=42
    uint16_t sf = evaluate_7cards(46, 45, 44, 43, 42, 0, 14);
    EXPECT_RANGE(sf, 7453, 7461, "Straight flush in [7453, 7461]");

    // 1.3 Four of a kind: Ac Ad Ah As + 2c 3d 4h
    // Ac=12 Ad=25 Ah=38 As=51  2c=0  3d=14  4h=2*13+2=28
    uint16_t quads = evaluate_7cards(12, 25, 38, 51, 0, 14, 28);
    EXPECT_RANGE(quads, 7297, 7452, "Four of a kind in [7297, 7452]");

    // 1.4 Full house: Ac Ad Ah Kc Kd + 2h 3s
    // Ac=12 Ad=25 Ah=38 Kc=11 Kd=24  2h=2*13+0=26  3s=3*13+1=40
    uint16_t fh = evaluate_7cards(12, 25, 38, 11, 24, 26, 40);
    EXPECT_RANGE(fh, 7141, 7296, "Full house in [7141, 7296]");

    // 1.5 Flush: Ac 9c 7c 5c 3c + 2d 4h
    // Ac=12 9c=7 7c=5 5c=3 3c=1  2d=13  4h=2*13+2=28
    uint16_t flush = evaluate_7cards(12, 7, 5, 3, 1, 13, 28);
    EXPECT_RANGE(flush, 5864, 7140, "Flush in [5864, 7140]");

    // 1.6 Straight: 5c 4d 3h 2s Ac + 8d 9h
    // 5c=3 4d=1*13+2=15 3h=2*13+1=27 2s=3*13+0=39 Ac=12  8d=1*13+6=19 9h=2*13+7=33
    uint16_t straight = evaluate_7cards(3, 15, 27, 39, 12, 19, 33);
    EXPECT_RANGE(straight, 5854, 5863, "Straight (A-low) in [5854, 5863]");

    // 1.7 Three of a kind: 7c 7d 7h + 2c 3d 5s 9h
    // 7c=5 7d=18 7h=31  2c=0 3d=14 5s=3*13+3=42 9h=2*13+7=33
    uint16_t trips = evaluate_7cards(5, 18, 31, 0, 14, 42, 33);
    EXPECT_RANGE(trips, 4996, 5853, "Three of a kind in [4996, 5853]");

    // 1.8 Two pair: Ac Ad 2c 2d + 3h 5s 7c
    // Ac=12 Ad=25 2c=0 2d=13  3h=2*13+1=27 5s=3*13+3=42 7c=5
    uint16_t twopair = evaluate_7cards(12, 25, 0, 13, 27, 42, 5);
    EXPECT_RANGE(twopair, 4138, 4995, "Two pair in [4138, 4995]");

    // 1.9 One pair: Ac Ad + 2c 3d 5h 7s 9c
    // Ac=12 Ad=25  2c=0 3d=14 5h=2*13+3=29 7s=3*13+5=44 9c=7
    uint16_t onepair = evaluate_7cards(12, 25, 0, 14, 29, 44, 7);
    EXPECT_RANGE(onepair, 1278, 4137, "One pair in [1278, 4137]");

    // 1.10 High card: 2c 4d 6h 8s Tc Qd Ah
    // 2c=0 4d=1*13+2=15 6h=2*13+4=30 8s=3*13+6=45 Tc=8 Qd=1*13+10=23 Ah=2*13+12=38
    uint16_t hc = evaluate_7cards(0, 15, 30, 45, 8, 23, 38);
    EXPECT_RANGE(hc, 1, 1277, "High card in [1, 1277]");

    // 1.11 Determinism: same call twice must return same rank
    uint16_t r1 = evaluate_7cards(51, 50, 49, 48, 47, 0, 14);
    uint16_t r2 = evaluate_7cards(51, 50, 49, 48, 47, 0, 14);
    EXPECT_EQ(r1, r2, "evaluate_7cards is deterministic");

    // 1.12 hand_category boundaries
    const char* cat_1277 = hand_category(1277);
    const char* cat_1278 = hand_category(1278);
    const char* cat_7461 = hand_category(7461);
    const char* cat_7462 = hand_category(7462);
    EXPECT(strcmp(cat_1277, "High Card")     == 0, "hand_category(1277) = High Card");
    EXPECT(strcmp(cat_1278, "One Pair")      == 0, "hand_category(1278) = One Pair");
    EXPECT(strcmp(cat_7461, "Straight Flush")== 0, "hand_category(7461) = Straight Flush");
    EXPECT(strcmp(cat_7462, "Royal Flush")   == 0, "hand_category(7462) = Royal Flush");

    // 1.13 Card encoding: suit-major → rank-major offset
    // to_rm(2c=0) should give offset 1 (standard TwoPlusTwo first card slot)
    // Verify indirectly: evaluating the 5-card royal flush with 2 extra identical
    // cards still gives 7462, confirming the encoding is correct end-to-end.
    uint16_t royal5 = evaluate_7cards(51, 50, 49, 48, 47, 1, 2);  // 2d and 3c extras
    EXPECT_EQ(royal5, 7462, "Royal flush with different irrelevant cards still 7462");

    // 1.14 evaluate_best with n_comm=5 matches evaluate_7cards
    Card comm5[5] = {49, 48, 47, 0, 14};  // Qs Js Ts 2c 3d
    uint16_t eb5 = evaluate_best(51, 50, comm5, 5);
    EXPECT_EQ(eb5, royal, "evaluate_best(n_comm=5) == evaluate_7cards");

    // 1.15 evaluate_best with n_comm=4 (turn): best 5-of-6
    // As Ks + Qs Js Ts 2c (6 cards) — best 5 should be the royal flush
    Card comm4[4] = {49, 48, 47, 0};  // Qs Js Ts 2c
    uint16_t eb4 = evaluate_best(51, 50, comm4, 4);
    EXPECT_EQ(eb4, royal, "evaluate_best(n_comm=4) finds royal among 6 cards");

    // 1.16 evaluate_best with n_comm=3 (flop)
    Card comm3[3] = {49, 48, 47};  // Qs Js Ts
    uint16_t eb3 = evaluate_best(51, 50, comm3, 3);
    EXPECT_RANGE(eb3, 5854, 7462, "evaluate_best(n_comm=3) gives valid rank");

    // 1.17 All ranks must be in [1, 7462]
    {
        // Sample 20 random 7-card deals and verify range
        std::mt19937 rng(12345);
        bool all_valid = true;
        for (int trial = 0; trial < 20; trial++) {
            std::vector<int> deck(52);
            std::iota(deck.begin(), deck.end(), 0);
            std::shuffle(deck.begin(), deck.end(), rng);
            uint16_t r = evaluate_7cards(
                deck[0], deck[1], deck[2], deck[3],
                deck[4], deck[5], deck[6]);
            if (r < 1 || r > 7462) { all_valid = false; break; }
        }
        EXPECT(all_valid, "20 random 7-card evals all in [1, 7462]");
    }
}

// ---------------------------------------------------------------------------
// Module 2 — Abstraction
// ---------------------------------------------------------------------------
static void test_abstraction()
{
    SECTION("Module 2a — Preflop Buckets");

    // 2a.1 Pocket aces: Ac=12, Ad=25 → highest bucket (127)
    int aa_bucket = preflop_bucket(12, 25);
    EXPECT_EQ(aa_bucket, PREFLOP_BUCKETS - 1, "AA bucket == max (127)");

    // 2a.2 Pocket kings: Kc=11, Kd=24 → near top
    int kk_bucket = preflop_bucket(11, 24);
    EXPECT_RANGE(kk_bucket, 116, 127, "KK bucket in [116, 127]");

    // 2a.3 72 offsuit: 2c=0, 7d=18 → near bottom
    int worst = preflop_bucket(0, 18);
    EXPECT_RANGE(worst, 0, 10, "72o bucket in [0, 10]");

    // 2a.4 AKs: Ac=12, Ks=50 → near top
    int aks = preflop_bucket(12, 50);
    EXPECT_RANGE(aks, 110, 127, "AKs bucket in [110, 127]");

    // 2a.5 Symmetry: (h0,h1) and (h1,h0) give same bucket
    int b01 = preflop_bucket(12, 50);
    int b10 = preflop_bucket(50, 12);
    EXPECT_EQ(b01, b10, "preflop_bucket symmetric: (AcKs)==(KsAc)");

    // 2a.6 All 1326 unique combos produce values in [0, 127]
    {
        bool all_valid = true;
        for (int i = 0; i < 52 && all_valid; i++)
            for (int j = i+1; j < 52 && all_valid; j++) {
                int b = preflop_bucket((Card)i, (Card)j);
                if (b < 0 || b >= PREFLOP_BUCKETS) all_valid = false;
            }
        EXPECT(all_valid, "All 1326 preflop combos produce bucket in [0, 127]");
    }

    // 2a.7 AA > KK > AKs > 72o (ordering sanity)
    int pocket_k = preflop_bucket(11, 24);
    EXPECT(aa_bucket >= pocket_k, "AA bucket >= KK bucket");
    EXPECT(pocket_k >= aks,       "KK bucket >= AKs bucket");
    EXPECT(aks > worst,           "AKs bucket > 72o bucket");

    SECTION("Module 2b — Postflop Buckets");

    if (!g_hand_eval_loaded) {
        printf("  (skipping postflop tests — handranks.dat not loaded)\n");
    } else {
        // 2b.1 Royal flush board: Ah=38 Kh=37 Qh=36 Jh=35 Th=34 + AcKs hole
        // Ac=12, Ks=50
        {
            Card royalboard[5] = {38, 37, 36, 35, 34};
            int top_bucket = fast_postflop_bucket(12, 50, royalboard, 5);
            EXPECT_EQ(top_bucket, POSTFLOP_BUCKETS - 1, "Royal flush board → max postflop bucket (255)");
        }

        // 2b.2 Low board: 7c=5 8d=19 9h=33 + hole 2c=0 2d=13
        {
            Card lowboard[3] = {5, 19, 33};
            int low_bucket = fast_postflop_bucket(0, 13, lowboard, 3);
            EXPECT_RANGE(low_bucket, 0, 255, "Low board bucket in [0, 255]");
        }

        // 2b.3 Boundary: max rank maps to bucket 255
        // (7462-1)*256/7462 = 7461*256/7462 = 1910016/7462 = 255 (integer division)
        {
            uint32_t val = (uint32_t)(7462 - 1) * (uint32_t)POSTFLOP_BUCKETS / 7462u;
            EXPECT_EQ((int)val, POSTFLOP_BUCKETS - 1, "rank=7462 maps to bucket 255 via formula");
        }

        // 2b.4 Boundary: rank=1 maps to bucket 0
        {
            uint32_t val = (uint32_t)(1 - 1) * (uint32_t)POSTFLOP_BUCKETS / 7462u;
            EXPECT_EQ((int)val, 0, "rank=1 maps to bucket 0 via formula");
        }

        // 2b.5 All streets produce valid bucket values
        {
            Card comm5[5] = {49, 48, 47, 0, 14};  // Qs Js Ts 2c 3d
            bool ok = true;
            for (int nc : {3, 4, 5}) {
                int b = fast_postflop_bucket(51, 50, comm5, nc);
                if (b < 0 || b >= POSTFLOP_BUCKETS) ok = false;
            }
            EXPECT(ok, "fast_postflop_bucket valid for n_comm=3,4,5");
        }
    }

    SECTION("Module 2c — Valid Action Mask");

    // 2c.1 Free check: to_call=0 → CHECK only (no FOLD, no CALL)
    {
        uint8_t vm = valid_actions_mask(100, 1000, 0, 0, 20, 20);
        bool has_check = (vm >> 1) & 1;
        bool has_fold  = (vm >> 0) & 1;
        bool has_call  = (vm >> 2) & 1;
        EXPECT( has_check, "to_call=0 → CHECK available");
        EXPECT(!has_fold,  "to_call=0 → FOLD NOT available");
        EXPECT(!has_call,  "to_call=0 → CALL NOT available");
    }

    // 2c.2 Forced all-in: to_call >= stack → FOLD | ALL_IN only
    {
        uint8_t vm = valid_actions_mask(100, 30, 50, 50, 20, 20);
        EXPECT_EQ((int)vm, (1 << 0) | (1 << 7), "to_call>=stack → exactly FOLD|ALL_IN");
    }

    // 2c.3 Normal raise available: stack >> to_call
    {
        uint8_t vm = valid_actions_mask(100, 980, 20, 20, 20, 20);
        bool has_fold  = (vm >> 0) & 1;
        bool has_call  = (vm >> 2) & 1;
        bool has_allin = (vm >> 7) & 1;
        bool has_raise = (vm >> 3) & 1;
        EXPECT( has_fold,  "normal situation → FOLD available");
        EXPECT( has_call,  "normal situation → CALL available");
        EXPECT( has_allin, "normal situation → ALL_IN available");
        EXPECT( has_raise, "normal situation → RAISE_MIN available");
    }

    // 2c.4 Stack too small to full-raise: only CALL and ALL_IN above call
    {
        // stack=25, to_call=20 → headroom=5; min raise would be 40 (20+20),
        // max raise = 25+0=25. 25 < 40 → no raise actions except ALL_IN.
        uint8_t vm = valid_actions_mask(100, 25, 20, 20, 20, 20);
        bool has_fold  = (vm >> 0) & 1;
        bool has_call  = (vm >> 2) & 1;
        bool has_allin = (vm >> 7) & 1;
        bool has_rmin  = (vm >> 3) & 1;
        EXPECT( has_fold,  "short stack → FOLD available");
        EXPECT( has_call,  "short stack → CALL available");
        EXPECT( has_allin, "short stack → ALL_IN available");
        EXPECT(!has_rmin,  "short stack (headroom<min_raise) → RAISE_MIN NOT available");
    }

    // 2c.5 Zero headroom: check only
    {
        uint8_t vm = valid_actions_mask(100, 0, 0, 0, 20, 20);
        // stack=0, to_call=0 → can check, headroom=0 → no raises
        bool has_check = (vm >> 1) & 1;
        bool has_raise = (vm >> 3) & 1;
        EXPECT( has_check, "stack=0, to_call=0 → CHECK available");
        EXPECT(!has_raise, "stack=0 → no RAISE available");
    }

    SECTION("Module 2d — Action to Chips");

    // 2d.1 FOLD = 0 chips
    EXPECT_EQ(action_to_chips(Action::FOLD, 100, 1000, 20, 20, 20, 20), 0, "FOLD = 0 chips");

    // 2d.2 CHECK = 0 chips
    EXPECT_EQ(action_to_chips(Action::CHECK, 100, 1000, 0, 0, 20, 20), 0, "CHECK = 0 chips");

    // 2d.3 CALL = min(to_call, stack)
    EXPECT_EQ(action_to_chips(Action::CALL, 100, 1000, 100, 100, 20, 20), 100,
              "CALL full amount");
    EXPECT_EQ(action_to_chips(Action::CALL, 100, 50, 100, 100, 20, 20), 50,
              "CALL capped at stack");

    // 2d.4 ALL_IN = stack
    EXPECT_EQ(action_to_chips(Action::ALL_IN, 100, 500, 20, 20, 20, 20), 500,
              "ALL_IN = stack");

    // 2d.5 RAISE_HALF with pot=100, current_bet=0 → chips = to_call + pot/2 = 0 + 50 = 50
    {
        int chips = action_to_chips(Action::RAISE_HALF, 100, 1000, 0, 0, 20, 20);
        EXPECT_EQ(chips, 50, "RAISE_HALF with pot=100, cur_bet=0 → 50 chips");
    }

    // 2d.6 RAISE_POT with pot=100 → chips = to_call + pot = 100
    {
        int chips = action_to_chips(Action::RAISE_POT, 100, 1000, 0, 0, 20, 20);
        EXPECT_EQ(chips, 100, "RAISE_POT with pot=100, cur_bet=0 → 100 chips");
    }

    // 2d.7 Raise capped at stack
    {
        // stack=30, to_call=20 → headroom=10, max_raise_to=0+10=10 → chips=min(30,30)=30
        int chips = action_to_chips(Action::RAISE_POT, 200, 30, 20, 20, 20, 20);
        EXPECT(chips <= 30, "RAISE_POT capped at stack");
    }

    // 2d.8 RAISE_MIN: result >= to_call (you at least call)
    {
        int chips = action_to_chips(Action::RAISE_MIN, 100, 1000, 20, 20, 20, 20);
        EXPECT(chips >= 20, "RAISE_MIN chips >= to_call");
    }

    SECTION("Module 2e — precompute_buckets consistency");

    // precompute_buckets should fill per-player per-street buckets correctly.
    // Street 0 board bucket must be 0 (no community cards yet).
    // Streets 1-3 must use the same formula as fast_postflop_bucket.
    if (!g_hand_eval_loaded) {
        printf("  (skipping precompute_buckets tests — handranks.dat not loaded)\n");
    } else {
        Card hole[4] = {51, 50, 12, 25};    // P0: As Ks,  P1: Ac Ad
        Card comm[5] = {49, 48, 47, 0, 14}; // Qs Js Ts 2c 3d
        int hb[2], bb[8];
        precompute_buckets(hole, comm, 2, hb, bb);

        EXPECT_RANGE(hb[0], 0, 63, "precompute_buckets: player 0 hole bucket in [0,63]");
        EXPECT_RANGE(hb[1], 0, 63, "precompute_buckets: player 1 hole bucket in [0,63]");
        EXPECT_EQ(bb[0*4+0], 0, "precompute_buckets: player 0 street 0 board bucket = 0");
        EXPECT_EQ(bb[1*4+0], 0, "precompute_buckets: player 1 street 0 board bucket = 0");

        // Street 3 (river): verify matches fast_postflop_bucket directly
        int manual0 = fast_postflop_bucket(51, 50, comm, 5);
        int manual1 = fast_postflop_bucket(12, 25, comm, 5);
        EXPECT_EQ(bb[0*4+3], manual0, "precompute_buckets river matches fast_postflop_bucket P0");
        EXPECT_EQ(bb[1*4+3], manual1, "precompute_buckets river matches fast_postflop_bucket P1");

        // Players have different hole cards → can have different board buckets
        // (they may equal each other coincidentally, but at least verify range)
        EXPECT_RANGE(bb[0*4+3], 0, 127, "P0 river board bucket in [0,127]");
        EXPECT_RANGE(bb[1*4+3], 0, 127, "P1 river board bucket in [0,127]");
    }
}

// ---------------------------------------------------------------------------
// Module 4 — Strategy File I/O (CPU path only, no GPU)
// ---------------------------------------------------------------------------
static void test_strategy_io()
{
    SECTION("Module 4 — Strategy File I/O");

    const std::string tmp_path = "test_strategy_tmp.bin";

    // 4.1 Round-trip: save and reload a synthetic table
    {
        HostStrategyTable orig;
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> dist(0.f, 1.f);

        for (int i = 0; i < 1000; i++) {
            uint32_t key = rng();
            StrategyEntry e;
            float sum = 0.f;
            for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
                e.probs[a] = dist(rng);
                sum += e.probs[a];
            }
            // Normalize so it's a valid probability distribution
            for (int a = 0; a < GPU_NUM_ACTIONS; a++) e.probs[a] /= sum;
            // Avoid inserting uniform placeholder (1/8 each) — it gets skipped on load
            orig[key] = e;
        }

        bool saved = strategy_save(orig, tmp_path);
        EXPECT(saved, "strategy_save returns true for 1000 entries");

        HostStrategyTable loaded;
        bool loaded_ok = strategy_load(loaded, tmp_path);
        EXPECT(loaded_ok, "strategy_load returns true");
        EXPECT_EQ((int)loaded.size(), (int)orig.size(), "round-trip: loaded.size() == orig.size()");

        // Verify all entries match bit-for-bit
        bool match = true;
        for (auto& [k, v] : orig) {
            auto it = loaded.find(k);
            if (it == loaded.end()) { match = false; break; }
            for (int a = 0; a < GPU_NUM_ACTIONS; a++) {
                if (it->second.probs[a] != v.probs[a]) { match = false; break; }
            }
        }
        EXPECT(match, "round-trip: all float values match bit-for-bit");
    }

    // 4.2 Empty table round-trip
    {
        HostStrategyTable empty;
        strategy_save(empty, tmp_path);
        HostStrategyTable loaded;
        bool ok = strategy_load(loaded, tmp_path);
        EXPECT(ok, "strategy_load succeeds on empty table");
        EXPECT_EQ((int)loaded.size(), 0, "empty table loads back as empty");
    }

    // 4.3 Reject V1 strategy magic
    {
        std::ofstream f(tmp_path, std::ios::binary);
        uint64_t bad_magic = 0x53545241540000ULL;  // STRATEGY_MAGIC_V1
        f.write((char*)&bad_magic, sizeof(bad_magic));
        f.close();
        HostStrategyTable t;
        bool ok = strategy_load(t, tmp_path);
        EXPECT(!ok, "strategy_load rejects V1 strategy magic");
    }

    // 4.4 Reject checkpoint V1 magic
    {
        std::ofstream f(tmp_path, std::ios::binary);
        uint64_t ckpt_magic = 0x43465247505500ULL;  // CHECKPOINT_MAGIC_V1
        f.write((char*)&ckpt_magic, sizeof(ckpt_magic));
        f.close();
        HostStrategyTable t;
        bool ok = strategy_load(t, tmp_path);
        EXPECT(!ok, "strategy_load rejects checkpoint V1 magic");
    }

    // 4.5 Reject checkpoint V2 magic
    {
        std::ofstream f(tmp_path, std::ios::binary);
        uint64_t ckpt_magic = 0x43465247505501ULL;  // CHECKPOINT_MAGIC_V2
        f.write((char*)&ckpt_magic, sizeof(ckpt_magic));
        f.close();
        HostStrategyTable t;
        bool ok = strategy_load(t, tmp_path);
        EXPECT(!ok, "strategy_load rejects checkpoint V2 magic");
    }

    // 4.6 Reject garbage file
    {
        std::ofstream f(tmp_path, std::ios::binary);
        uint64_t garbage = 0xDEADBEEFCAFEBABEULL;
        f.write((char*)&garbage, sizeof(garbage));
        f.close();
        HostStrategyTable t;
        bool ok = strategy_load(t, tmp_path);
        EXPECT(!ok, "strategy_load rejects garbage magic");
    }

    // 4.7 Reject wrong table_size
    {
        // Write a valid V2 header but with wrong table_size
        std::ofstream f(tmp_path, std::ios::binary);
        uint64_t magic    = 0x53545241540100ULL;  // STRATEGY_MAGIC_V2
        uint32_t bad_ts   = 12345;
        uint32_t na       = GPU_NUM_ACTIONS;
        uint32_t n        = 0;
        f.write((char*)&magic,  8);
        f.write((char*)&bad_ts, 4);
        f.write((char*)&na,     4);
        f.write((char*)&n,      4);
        f.close();
        HostStrategyTable t;
        bool ok = strategy_load(t, tmp_path);
        EXPECT(!ok, "strategy_load rejects wrong table_size");
    }

    // 4.8 Reject wrong num_actions
    {
        std::ofstream f(tmp_path, std::ios::binary);
        uint64_t magic  = 0x53545241540100ULL;
        uint32_t ts     = (uint32_t)GPU_TABLE_SIZE;
        uint32_t bad_na = 4;
        uint32_t n      = 0;
        f.write((char*)&magic,  8);
        f.write((char*)&ts,     4);
        f.write((char*)&bad_na, 4);
        f.write((char*)&n,      4);
        f.close();
        HostStrategyTable t;
        bool ok = strategy_load(t, tmp_path);
        EXPECT(!ok, "strategy_load rejects wrong num_actions");
    }

    // 4.9 Uniform placeholder rows are skipped on load
    {
        HostStrategyTable orig;
        // Add 5 real entries
        for (int i = 0; i < 5; i++) {
            StrategyEntry e;
            e.probs[0] = 0.5f; e.probs[1] = 0.5f;
            for (int a = 2; a < GPU_NUM_ACTIONS; a++) e.probs[a] = 0.f;
            orig[(uint32_t)(i + 1)] = e;
        }
        // Add 3 placeholder entries (all-uniform = 1/8 each)
        for (int i = 0; i < 3; i++) {
            StrategyEntry e;
            for (int a = 0; a < GPU_NUM_ACTIONS; a++) e.probs[a] = 1.f / GPU_NUM_ACTIONS;
            orig[(uint32_t)(100 + i)] = e;
        }
        strategy_save(orig, tmp_path);

        HostStrategyTable loaded;
        strategy_load(loaded, tmp_path);
        // Exactly the 5 real entries should survive; 3 placeholders skipped
        EXPECT_EQ((int)loaded.size(), 5, "uniform placeholder rows skipped on load");
    }

    // 4.10 strategy_load_auto on a V2 strategy file succeeds
    {
        HostStrategyTable orig;
        StrategyEntry e;
        e.probs[0] = 0.6f; e.probs[1] = 0.4f;
        for (int a = 2; a < GPU_NUM_ACTIONS; a++) e.probs[a] = 0.f;
        orig[42u] = e;
        strategy_save(orig, tmp_path);

        HostStrategyTable loaded;
        // n_players/stack/sb/bb are only used when converting checkpoint; ignored for strategy
        bool ok = strategy_load_auto(loaded, tmp_path, 2, 1000, 10, 20);
        EXPECT(ok, "strategy_load_auto succeeds on V2 strategy file");
        EXPECT_EQ((int)loaded.size(), 1, "strategy_load_auto loads correct entry count");
    }

    // Cleanup temp file
    std::remove(tmp_path.c_str());
    printf("  (temp file removed)\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    const char* hr_path = (argc > 1) ? argv[1]
                                     : "poker_cuda/data/handranks.dat";

    printf("=== Poker CFR CPU Tests ===\n");
    printf("handranks path: %s\n", hr_path);
    printf("PREFLOP_BUCKETS=%d  POSTFLOP_BUCKETS=%d  GPU_NUM_ACTIONS=%d\n",
           PREFLOP_BUCKETS, POSTFLOP_BUCKETS, GPU_NUM_ACTIONS);

    // Must call abstraction_init() before any preflop_bucket usage
    abstraction_init();

    test_hand_evaluator(hr_path);
    test_abstraction();
    test_strategy_io();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}

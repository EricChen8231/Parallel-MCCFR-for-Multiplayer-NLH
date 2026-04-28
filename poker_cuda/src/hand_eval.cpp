#include "hand_eval.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstring>
#include <cassert>
#include <algorithm>

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

// ---------------------------------------------------------------------------
// Convert raw 7-card sequential lookup result to standard rank [1, 7462].
// The handranks.dat table uses encoding: raw = category*4096 + within_rank
//   cat: 1=HC 2=OP 3=TP 4=Trips 5=Str 6=Flush 7=FH 8=Quads 9=SF
//   within_rank is 1-based: 1=weakest in category, N=strongest
// Standard rank ranges:
//   HC 1-1277, OP 1278-4137, TP 4138-4995, Trips 4996-5853,
//   Str 5854-5863, Flush 5864-7140, FH 7141-7296, Quads 7297-7452, SF 7453-7462
// ---------------------------------------------------------------------------
static inline uint16_t convert_raw_rank(int raw) {
    // base[cat] = start-of-category offset so that base[cat]+within = final rank
    static const int base[10] = { 0, 0, 1277, 4137, 4995, 5853, 5863, 7140, 7296, 7452 };
    int cat    = raw >> 12;
    int within = raw & 0xFFF;
    if (cat < 1 || cat > 9 || within < 1) return 0;
    return (uint16_t)(base[cat] + within);
}

// ---------------------------------------------------------------------------
// Standalone 5-card evaluator: does NOT touch the HR table.
// Returns hand rank in [1, 7462] — higher = better.
// s_flush_ranks and s_hc_ranks must be built via build_eval5_tables() first.
// ---------------------------------------------------------------------------
static uint16_t s_flush_ranks[8192];  // 13-bit rank bitmask → flush rank
static uint16_t s_hc_ranks[8192];     // 13-bit rank bitmask → HC rank

static void build_eval5_tables() {
    memset(s_flush_ranks, 0, sizeof(s_flush_ranks));
    memset(s_hc_ranks,    0, sizeof(s_hc_ranks));

    // Enumerate all C(13,5)=1287 combinations of 5 distinct ranks (0=2 .. 12=A).
    // Exclude the 10 straights (9 regular + wheel A-2-3-4-5).
    // The remaining 1277 combinations are used for both flush and HC tables.
    // Sorting by mask descending yields the correct strength ordering because
    // higher-order bits represent higher cards.
    std::vector<uint16_t> masks;
    masks.reserve(1277);

    for (int a = 0; a < 9;  a++)
    for (int b = a+1; b < 10; b++)
    for (int c = b+1; c < 11; c++)
    for (int d = c+1; d < 12; d++)
    for (int e = d+1; e < 13; e++) {
        uint16_t mask = (uint16_t)((1<<a)|(1<<b)|(1<<c)|(1<<d)|(1<<e));
        bool is_str   = (e - a == 4);                        // regular straight
        bool is_wheel = (a==0 && b==1 && c==2 && d==3 && e==12); // A-2-3-4-5
        if (!is_str && !is_wheel)
            masks.push_back(mask);
    }

    // Sort descending: higher mask → stronger hand
    std::sort(masks.begin(), masks.end(), std::greater<uint16_t>());

    // Assign ranks: best mask (index 0) gets the highest rank value
    for (int i = 0; i < (int)masks.size(); i++) {
        s_flush_ranks[masks[i]] = (uint16_t)(7140 - i);  // 7140..5864
        s_hc_ranks[masks[i]]    = (uint16_t)(1277 - i);  // 1277..1
    }
}

static uint16_t eval5_standalone(Card c0, Card c1, Card c2, Card c3, Card c4) {
    int r[5] = { (int)(c0%13), (int)(c1%13), (int)(c2%13), (int)(c3%13), (int)(c4%13) };
    int s[5] = { (int)(c0/13), (int)(c1/13), (int)(c2/13), (int)(c3/13), (int)(c4/13) };

    // Sort ranks descending (insertion sort on 5 elements)
    for (int i = 1; i < 5; i++) {
        int v = r[i], j = i;
        while (j > 0 && r[j-1] < v) { r[j] = r[j-1]; j--; }
        r[j] = v;
    }

    // Count rank frequencies and build 13-bit bitmask
    int cnt[13] = {};
    for (int i = 0; i < 5; i++) cnt[r[i]]++;
    uint16_t mask = 0;
    for (int i = 0; i < 5; i++) mask |= (uint16_t)(1 << r[i]);

    bool is_flush = (s[0]==s[1] && s[1]==s[2] && s[2]==s[3] && s[3]==s[4]);
    bool all_dist = (cnt[r[0]]==1 && cnt[r[1]]==1 && cnt[r[2]]==1
                  && cnt[r[3]]==1 && cnt[r[4]]==1);
    bool is_str   = all_dist && (r[0] - r[4] == 4);      // regular straight
    bool is_wheel = (r[0]==12 && r[1]==3 && r[2]==2      // A-2-3-4-5
                  && r[3]==1  && r[4]==0);
    int str_high  = is_wheel ? 3 : r[0];                 // effective high for straight

    if (is_flush) {
        if (is_str || is_wheel)
            return (uint16_t)(7453 + str_high - 3);  // SF 7453..7462
        return s_flush_ranks[mask];                  // Flush 5864..7140
    }
    if (is_str || is_wheel)
        return (uint16_t)(5854 + str_high - 3);      // Straight 5854..5863

    // Identify hand type
    int quads = -1, trips = -1, pairs[2] = {-1, -1};
    int np = 0;
    for (int rk = 12; rk >= 0; rk--) {
        if      (cnt[rk] == 4) quads = rk;
        else if (cnt[rk] == 3) trips = rk;
        else if (cnt[rk] == 2 && np < 2) pairs[np++] = rk;
    }
    // pairs[0] is the highest pair rank, pairs[1] is the second-highest

    if (quads >= 0) {
        // Four of a kind: 7297..7452 (13 quad ranks × 12 kickers = 156)
        int k = -1;
        for (int i = 0; i < 5; i++) if (r[i] != quads) { k = r[i]; break; }
        int ak = (k < quads) ? k : k - 1;
        return (uint16_t)(7296 + quads * 12 + ak + 1);
    }
    if (trips >= 0 && pairs[0] >= 0) {
        // Full house: 7141..7296 (13 trip ranks × 12 pair ranks = 156)
        int pr = pairs[0];
        int ap = (pr < trips) ? pr : pr - 1;
        return (uint16_t)(7140 + trips * 12 + ap + 1);
    }
    if (trips >= 0) {
        // Three of a kind: 4996..5853 (13 × C(12,2)=66 = 858)
        int k[2], ki = 0;
        for (int i = 0; i < 5; i++) if (r[i] != trips) k[ki++] = r[i];
        if (k[0] < k[1]) { int t = k[0]; k[0] = k[1]; k[1] = t; }
        int a0 = (k[0] < trips) ? k[0] : k[0]-1;
        int a1 = (k[1] < trips) ? k[1] : k[1]-1;
        // Combinatorial index for 2 kickers from 12 slots: C(a0,2)+a1
        return (uint16_t)(4995 + trips * 66 + a0*(a0-1)/2 + a1 + 1);
    }
    if (pairs[0] >= 0 && pairs[1] >= 0) {
        // Two pair: 4138..4995 (C(13,2)=78 pair combos × 11 kickers = 858)
        int p1 = pairs[0], p2 = pairs[1];  // p1 > p2
        int k = -1;
        for (int i = 0; i < 5; i++) if (cnt[r[i]] == 1) { k = r[i]; break; }
        int ak = (k < p2) ? k : (k < p1) ? k-1 : k-2;
        // Pair-pair index: C(p1,2)+p2 = p1*(p1-1)/2+p2  (enumerates {p1,p2} with p1>p2)
        return (uint16_t)(4137 + (p1*(p1-1)/2 + p2)*11 + ak + 1);
    }
    if (pairs[0] >= 0) {
        // One pair: 1278..4137 (13 pair ranks × C(12,3)=220 kicker combos = 2860)
        int p = pairs[0];
        int k[3], ki = 0;
        for (int i = 0; i < 5; i++) if (r[i] != p) k[ki++] = r[i];
        // Sort k descending (3 elements)
        if (k[0]<k[1]) { int t=k[0]; k[0]=k[1]; k[1]=t; }
        if (k[0]<k[2]) { int t=k[0]; k[0]=k[2]; k[2]=t; }
        if (k[1]<k[2]) { int t=k[1]; k[1]=k[2]; k[2]=t; }
        // Adjust kickers to exclude pair slot (0..11 range)
        int a[3];
        for (int i = 0; i < 3; i++) a[i] = (k[i] < p) ? k[i] : k[i]-1;
        // Combinatorial number system for {a[0],a[1],a[2]} with a[0]>a[1]>a[2]
        int c3 = a[0]*(a[0]-1)*(a[0]-2)/6;  // C(a[0],3)
        int c2 = a[1]*(a[1]-1)/2;            // C(a[1],2)
        int c1 = a[2];                        // C(a[2],1)
        return (uint16_t)(1277 + p*220 + c3 + c2 + c1 + 1);
    }
    // High card: 1..1277
    return s_hc_ranks[mask];
}

// ---------------------------------------------------------------------------
// 7-card validation helper: sequential Two Plus Two lookups + conversion.
// Used only during table loading (before the global HR is set).
// ---------------------------------------------------------------------------
static uint16_t eval7_from_table(const std::vector<int32_t>& hr,
                                  Card c0, Card c1, Card c2, Card c3,
                                  Card c4, Card c5, Card c6) {
    int p = hr[53 + to_rm(c0)];
    p = hr[p + to_rm(c1)];
    p = hr[p + to_rm(c2)];
    p = hr[p + to_rm(c3)];
    p = hr[p + to_rm(c4)];
    p = hr[p + to_rm(c5)];
    int raw = hr[p + to_rm(c6)];
    return convert_raw_rank(raw);
}

static bool validate_hand_table(const std::vector<int32_t>& hr, const char* path) {
    if (hr.size() != EXPECTED_HR_ENTRIES) {
        fprintf(stderr,
                "[hand_eval] incompatible handranks table at %s:\n"
                "            expected %zu int32 entries, found %zu.\n",
                path, EXPECTED_HR_ENTRIES, hr.size());
        return false;
    }

    // Royal flush (As Ks Qs Js Ts) + two low cards that don't affect the result.
    // Sequential 7-card lookup + convert_raw_rank must return exactly 7462.
    const uint16_t royal = eval7_from_table(hr,
        /*As*/51, /*Ks*/50, /*Qs*/49, /*Js*/48, /*Ts*/47,
        /*2c*/0,  /*3c*/1);

    if (royal != 7462) {
        fprintf(stderr,
                "[hand_eval] incompatible handranks table at %s:\n"
                "            As Ks Qs Js Ts 2c 3c (7-card eval) returned %u, expected 7462.\n"
                "            Ensure the table was generated by scripts/gen_handranks.py\n"
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
    build_eval5_tables();
    fprintf(stderr, "[hand_eval] loaded %zu entries (%.1f MB)\n", HR.size(), sz / 1e6);
    return true;
}

// ---------------------------------------------------------------------------
// Core 7-card evaluation: sequential Two Plus Two lookups + rank conversion.
// The table's 7-step sequential path always terminates at a valid leaf.
// ---------------------------------------------------------------------------
uint16_t evaluate_7cards(Card c0, Card c1, Card c2,
                          Card c3, Card c4, Card c5, Card c6) {
    int p = HR[53 + to_rm(c0)];
    p = HR[p + to_rm(c1)];
    p = HR[p + to_rm(c2)];
    p = HR[p + to_rm(c3)];
    p = HR[p + to_rm(c4)];
    p = HR[p + to_rm(c5)];
    int raw = HR[p + to_rm(c6)];
    return convert_raw_rank(raw);
}

// ---------------------------------------------------------------------------
// 6-card: best 5-of-6 via brute-force C(6,5)=6 calls to eval5_standalone.
// (Sequential 5/6-step paths in the table don't yield hand ranks — only the
//  full 7-step path does.  Brute-force over the standalone evaluator is safe.)
// ---------------------------------------------------------------------------
static inline uint16_t eval6(Card c0, Card c1, Card c2,
                              Card c3, Card c4, Card c5) {
    Card cs[6] = {c0, c1, c2, c3, c4, c5};
    uint16_t best = 0;
    for (int skip = 0; skip < 6; ++skip) {
        Card h[5]; int k = 0;
        for (int i = 0; i < 6; ++i) if (i != skip) h[k++] = cs[i];
        uint16_t r = eval5_standalone(h[0], h[1], h[2], h[3], h[4]);
        if (r > best) best = r;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Evaluate hole + community (n_comm = 0, 3, 4, 5)
// For n_comm < 3, returns a preflop proxy (higher = better).
// ---------------------------------------------------------------------------
uint16_t evaluate_best(Card h0, Card h1,
                       const Card* community, int n_comm) {
    if (n_comm == 5) {
        return evaluate_7cards(h0, h1,
            community[0], community[1], community[2],
            community[3], community[4]);
    }
    if (n_comm == 4) {
        return eval6(h0, h1,
                     community[0], community[1], community[2], community[3]);
    }
    if (n_comm == 3) {
        return eval5_standalone(h0, h1,
                                community[0], community[1], community[2]);
    }
    // Preflop: no community — return placeholder (higher = better)
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

bool hand_eval_get_eval5_tables(const uint16_t** out_flush_ranks,
                                 const uint16_t** out_hc_ranks) {
    if (HR.empty()) return false;
    *out_flush_ranks = s_flush_ranks;
    *out_hc_ranks    = s_hc_ranks;
    return true;
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

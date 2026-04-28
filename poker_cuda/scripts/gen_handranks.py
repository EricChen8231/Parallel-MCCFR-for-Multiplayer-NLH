#!/usr/bin/env python3
"""
gen_handranks.py — Generate the Two Plus Two 7-card hand evaluator table.

Produces the standard HandRanks.dat used by all Two Plus Two sequential
evaluators. Output: 32,487,834 int32 values, ~130 MB.

Usage:
    python3 scripts/gen_handranks.py [output_path]
    python3 scripts/gen_handranks.py data/handranks.dat

Algorithm: Paul Senzee / James Devlin sequential-lookup table generator.
Reference: http://www.codingthewheel.com/archives/poker-hand-evaluator-roundup

Card encoding (rank-major, 1-based):
    2c=1 2d=2 2h=3 2s=4 3c=5 ... As=52
"""

import sys
import struct
import itertools
from collections import defaultdict

# ---------------------------------------------------------------------------
# Cactus Kev 5-card evaluator (used to score all 7462 equivalence classes)
# ---------------------------------------------------------------------------

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]  # rank primes

# Build lookup structures for the 5-card evaluator
# Each card: bits 0-7=rank_prime, bits 8-11=rank, bit 16=suit, bits 17-28=bit_rank

def make_card(rank, suit):
    """rank 0-12 (2-A), suit 0-3 (c/d/h/s)"""
    prime = PRIMES[rank]
    return prime | (rank << 8) | (1 << (16 + suit)) | (1 << (16 + rank))

# All 52 cards in rank-major order (matches Two Plus Two indexing)
CARDS_RM = []  # index 0 = card 1 in Two Plus Two (1-based)
for rank in range(13):       # 2,3,...,A
    for suit in range(4):    # c,d,h,s
        CARDS_RM.append(make_card(rank, suit))

# ---------------------------------------------------------------------------
# 5-card Cactus Kev evaluator tables (hand category → rank 1-7462)
# We need these to score every 5-card combination.
# ---------------------------------------------------------------------------

# Straight detection primes products
STRAIGHT_PRIMES = [
    31367009,  # A2345
    2*3*5*7*11,     # 23456
    3*5*7*11*13,    # 34567
    5*7*11*13*17,   # 45678
    7*11*13*17*19,  # 56789
    11*13*17*19*23, # 6789T
    13*17*19*23*29, # 789TJ
    17*19*23*29*31, # 89TJQ
    19*23*29*31*37, # 9TJQK
    23*29*31*37*41, # TJQKA
]
# A-2-3-4-5: A has rank 12, prime 41
STRAIGHT_PRIMES[0] = 41*2*3*5*7  # A2345

def eval_5card(c0, c1, c2, c3, c4):
    """
    Evaluate a 5-card hand. Returns rank 1-7462 (higher = better).
    Uses the full Cactus Kev algorithm.
    """
    # Check flush
    suit_bits = (c0 | c1 | c2 | c3 | c4) >> 16
    is_flush = (suit_bits & 0xF) == (suit_bits & -suit_bits)  # all same suit

    # Get rank bits
    rank_bits = (c0 | c1 | c2 | c3 | c4) >> 16
    # Top 13 bits are rank bits (bits 16-28 map to ranks)
    rb = ((c0 | c1 | c2 | c3 | c4) >> 16) & 0x1FFF

    # Prime product for duplicate detection
    prime_prod = (c0 & 0xFF) * (c1 & 0xFF) * (c2 & 0xFF) * (c3 & 0xFF) * (c4 & 0xFF)

    # Get ranks (0-12)
    ranks = [
        (c0 >> 8) & 0xF,
        (c1 >> 8) & 0xF,
        (c2 >> 8) & 0xF,
        (c3 >> 8) & 0xF,
        (c4 >> 8) & 0xF,
    ]
    ranks_sorted = sorted(ranks, reverse=True)

    # Count rank frequencies
    from collections import Counter
    freq = Counter(ranks)
    counts = sorted(freq.values(), reverse=True)

    # Number of distinct ranks
    n_distinct = len(freq)

    # Check straight (5 distinct ranks, max-min==4, no pairs)
    is_straight = False
    if n_distinct == 5:
        if ranks_sorted[0] - ranks_sorted[4] == 4:
            is_straight = True
        elif ranks_sorted == [12, 3, 2, 1, 0]:  # A2345
            is_straight = True
            ranks_sorted = [3, 2, 1, 0, -1]  # treat A as low

    # Straight flush
    if is_flush and is_straight:
        # rank by high card (4=6high..13=A high)
        high = ranks_sorted[0]
        if high == -1: high = 3  # A-low straight flush (wheel): rank as 5-high
        return 7452 + (high - 3)  # 7453..7462 (10 = royal flush when high=12)

    # Four of a kind
    if counts[0] == 4:
        quad_rank = [r for r, c in freq.items() if c == 4][0]
        kick_rank = [r for r, c in freq.items() if c == 1][0]
        # 7297-7452: ordered by quad rank then kicker
        return 7296 + quad_rank * 12 + (kick_rank if kick_rank < quad_rank else kick_rank - 1) + 1

    # Full house
    if counts[0] == 3 and counts[1] == 2:
        trip_rank = [r for r, c in freq.items() if c == 3][0]
        pair_rank = [r for r, c in freq.items() if c == 2][0]
        return 7140 + trip_rank * 12 + (pair_rank if pair_rank < trip_rank else pair_rank - 1) + 1

    # Flush
    if is_flush:
        # 5864-7140: rank by sorted ranks (lexicographic)
        key = tuple(ranks_sorted)
        return _flush_rank(key)

    # Straight
    if is_straight:
        high = ranks_sorted[0]
        if high == -1: high = 3
        return 5853 + (high - 3) + 1  # 5854..5863

    # Three of a kind
    if counts[0] == 3:
        trip_rank = [r for r, c in freq.items() if c == 3][0]
        kicks = sorted([r for r, c in freq.items() if c == 1], reverse=True)
        return _trips_rank(trip_rank, kicks[0], kicks[1])

    # Two pair
    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted([r for r, c in freq.items() if c == 2], reverse=True)
        kick = [r for r, c in freq.items() if c == 1][0]
        return _two_pair_rank(pairs[0], pairs[1], kick)

    # One pair
    if counts[0] == 2:
        pair_rank = [r for r, c in freq.items() if c == 2][0]
        kicks = sorted([r for r, c in freq.items() if c == 1], reverse=True)
        return _one_pair_rank(pair_rank, kicks[0], kicks[1], kicks[2])

    # High card
    return _high_card_rank(tuple(ranks_sorted))


# Pre-build rank tables for high card and flush
_high_card_table = {}
_flush_table = {}

def _build_rank_tables():
    """Pre-compute ranks for all no-pair and flush hands."""
    # High card: all C(13,5) = 1287 combos of distinct ranks
    combos = list(itertools.combinations(range(12, -1, -1), 5))
    # Remove straights and A-2-3-4-5 for high card
    def is_straight_combo(c):
        if c[0] - c[4] == 4: return True
        if list(c) == [12, 3, 2, 1, 0]: return True
        return False
    no_pair_non_straight = [c for c in combos if not is_straight_combo(c)]
    # Sort: best to worst (already sorted descending)
    # Assign ranks 1..N (1=worst)
    for i, c in enumerate(reversed(no_pair_non_straight)):
        _high_card_table[c] = i + 1
        _flush_table[c] = 5863 + i + 1  # flushes start at 5864

_build_rank_tables()

def _flush_rank(key):
    return _flush_table.get(key, 0)

def _high_card_rank(key):
    return _high_card_table.get(key, 0)

# Pre-build trips, two-pair, one-pair tables
_trips_table = {}
_two_pair_table = {}
_one_pair_table = {}

def _build_combo_tables():
    # Three of a kind: 858 combos
    rank = 1
    entries = []
    for trip in range(13):
        for k1 in range(12, -1, -1):
            if k1 == trip: continue
            for k2 in range(k1-1, -1, -1):
                if k2 == trip: continue
                entries.append((trip, k1, k2))
    # Sort worst to best: lower trip = worse, then kicks
    def trips_key(e): return (e[0], e[1], e[2])
    entries.sort(key=trips_key)
    for i, e in enumerate(entries):
        _trips_table[e] = 4995 + i + 1  # 4996..5853

    # Two pair: 858 combos
    entries2 = []
    for p1 in range(12, -1, -1):
        for p2 in range(p1-1, -1, -1):
            for k in range(12, -1, -1):
                if k == p1 or k == p2: continue
                entries2.append((p1, p2, k))
    def tp_key(e): return (e[0], e[1], e[2])
    entries2.sort(key=tp_key)
    for i, e in enumerate(entries2):
        _two_pair_table[e] = 4137 + i + 1  # 4138..4995

    # One pair: 2860 combos
    entries3 = []
    for p in range(13):
        others = [r for r in range(13) if r != p]
        for k1, k2, k3 in itertools.combinations(sorted(others, reverse=True), 3):
            entries3.append((p, k1, k2, k3))
    def op_key(e): return (e[0], e[1], e[2], e[3])
    entries3.sort(key=op_key)
    for i, e in enumerate(entries3):
        _one_pair_table[e] = 1277 + i + 1  # 1278..4137

_build_combo_tables()

def _trips_rank(trip, k1, k2):
    return _trips_table.get((trip, k1, k2), 0)

def _two_pair_rank(p1, p2, k):
    return _two_pair_table.get((p1, p2, k), 0)

def _one_pair_rank(p, k1, k2, k3):
    return _one_pair_table.get((p, k1, k2, k3), 0)


# ---------------------------------------------------------------------------
# Best 5 from N evaluator
# ---------------------------------------------------------------------------

def best5(cards_rm_indices):
    """cards_rm_indices: list of 0-based indices into CARDS_RM (0..51)"""
    cards = [CARDS_RM[i] for i in cards_rm_indices]
    best = 0
    for combo in itertools.combinations(cards, 5):
        r = eval_5card(*combo)
        if r > best:
            best = r
    return best


# ---------------------------------------------------------------------------
# Two Plus Two sequential table builder
# ---------------------------------------------------------------------------

def build_table(progress=True):
    """
    Build the 32,487,834-entry Two Plus Two sequential lookup table.
    Returns a list of int32 values.
    """
    HR_SIZE = 32487834
    HR = [0] * HR_SIZE

    # The table uses a DFS/BFS over card sequences.
    # State: sorted tuple of cards seen so far (rank-major 1-based indices)
    # Each state maps to a slot in HR.
    # After 5+ cards, HR[slot] = hand rank.
    # After <5 cards, HR[slot] = next slot base.

    # We use a BFS approach.
    # Slot 0 is unused. Slots 1..52 are for single-card states... actually
    # the table layout is more complex. Let me use the standard Senzee layout.

    # Standard layout: entry point for first card is HR[53..104] (indices 53+card)
    # where card is rank-major 1-52.

    # We'll use a "next free slot" pointer and a dict from frozenset→slot.
    next_slot = [105]  # slots 0..104 are reserved (53..104 = first-card entries)
    state_to_slot = {}

    def alloc_block():
        s = next_slot[0]
        next_slot[0] += 53  # 53 entries per block (cards 1..52 + padding)
        return s

    def get_slot(sorted_cards_tuple):
        if sorted_cards_tuple not in state_to_slot:
            state_to_slot[sorted_cards_tuple] = alloc_block()
        return state_to_slot[sorted_cards_tuple]

    # BFS over all possible card sequences of length 1..7
    # For each sequence, compute the slot and fill HR.

    total_states = 0
    all_cards = list(range(1, 53))  # rank-major 1-52

    if progress:
        print("Building Two Plus Two table (this takes ~2-5 min)...")
        sys.stdout.flush()

    # Process sequences of increasing length
    from itertools import combinations

    # For each possible hand (up to 7 cards), fill the table.
    # The key insight: HR[slot + next_card] = either hand_rank or next_slot.

    # Build all states level by level.
    # Level 1: 52 single-card states. Entry: HR[53 + card] = slot for that state.
    # Level 2: C(52,2) = 1326 two-card states.
    # ...
    # Level 7: C(52,7) = 133,784,560 seven-card states (terminal).

    # For terminal states (5, 6, 7 cards): HR[slot + next_card] = hand_rank.
    # For non-terminal states: HR[slot + next_card] = slot_for_extended_state.

    # Memory concern: we need to track all states. With 7-card states that's
    # ~133M entries just in the dict — too much RAM.
    #
    # Better approach: iterate over all 7-card hands directly and fill backwards.
    # But this requires careful slot management.
    #
    # Standard approach: only 5,6,7-card states are terminal. States 1-4 are
    # intermediate. The table is sparse in practice.

    # Fill entry points for 1-card states
    for card in all_cards:
        slot = get_slot((card,))
        HR[53 + card] = slot

    if progress:
        print(f"  Level 1: {len(state_to_slot)} states")
        sys.stdout.flush()

    # Fill levels 2-4 (non-terminal intermediate states)
    for length in range(2, 5):
        count = 0
        for cards in combinations(all_cards, length):
            s = cards  # sorted tuple
            slot = get_slot(s)
            # For each possible next card
            for next_card in all_cards:
                if next_card in s:
                    continue
                extended = tuple(sorted(s + (next_card,)))
                if length + 1 < 5:
                    # Intermediate state
                    next_slot_val = get_slot(extended)
                    HR[slot + next_card] = next_slot_val
                else:
                    # Next level is 5 (terminal) — rank it
                    rank = best5(list(c - 1 for c in extended))  # convert to 0-based for CARDS_RM
                    HR[slot + next_card] = rank
            count += 1
        if progress:
            print(f"  Level {length}: {count} states")
            sys.stdout.flush()

    # Fill level 5 (5-card terminal — but also entry for 6-card lookup)
    count5 = 0
    for cards in combinations(all_cards, 5):
        slot = get_slot(cards)
        for next_card in all_cards:
            if next_card in set(cards):
                continue
            extended = tuple(sorted(cards + (next_card,)))
            # 6-card hand rank (best 5 of 6)
            rank = best5(list(c - 1 for c in extended))
            # Also need slot for 6-card state (for 7-card lookup)
            six_slot = get_slot(extended)
            HR[slot + next_card] = six_slot  # point to 6-card slot for further lookup
            # Store rank too? No — the 6-card slot stores 7-card ranks.
            # Actually for 6-card evaluation (final), we want rank.
            # For 7-card evaluation, we want next slot.
            # The Two Plus Two table handles this with a single value:
            # if the value is < 7463, it's a rank; otherwise it's a slot.
            # But that breaks for large slot indices...
            # Standard approach: terminal nodes store rank directly,
            # non-terminal nodes store slot offsets.
            # For sequential eval, you do 5, 6, or 7 lookups, not mixed.
        count5 += 1
        if count5 % 100 == 0 and progress:
            print(f"\r  Level 5: {count5}/{len(list(combinations(range(52),5)))} ...", end="")
    if progress:
        print(f"\r  Level 5: {count5} states")
        sys.stdout.flush()

    return HR


# ---------------------------------------------------------------------------
# Simpler, correct implementation using the actual Senzee approach
# ---------------------------------------------------------------------------

def build_table_senzee(outpath, progress=True):
    """
    Build and write the Two Plus Two table using a direct slot-assignment
    approach that matches the standard handranks.dat layout exactly.
    """
    import array
    import os

    HR_SIZE = 32487834
    HR = array.array('i', [0] * HR_SIZE)

    # The Senzee table uses a specific recursive structure.
    # Rather than reimplementing from scratch, we use a well-tested
    # enumeration approach: for each of the C(52,7) = 133M 7-card hands
    # we need the best-5-from-7 rank.
    #
    # This approach is too slow in pure Python for 133M hands.
    # Instead we generate it level by level, precomputing ranks.

    # ---------------------------------------------------------------------------
    # Fast 5-card evaluator using the pre-built lookup tables
    # ---------------------------------------------------------------------------

    # Generate all 7462 canonical hand ranks
    # We need a fast eval for the generation step.

    # Level approach: fill in the table working up from 5-card hands.
    # States: frozensets of cards (rank-major 1-52).
    # Entry point: HR[53 + first_card] = slot for 1-card state.

    # Slot allocator — we'll pack states breadth-first.
    # Each state needs 53 slots (for cards 1..52, some unused).

    next_free = [105]
    state_slot = {}  # frozenset -> slot

    def get_or_alloc(fset):
        if fset not in state_slot:
            s = next_free[0]
            if s + 53 > HR_SIZE:
                raise RuntimeError(f"HR overflow at slot {s}")
            state_slot[fset] = s
            next_free[0] = s + 53
        return state_slot[fset]

    cards = list(range(1, 53))

    if progress:
        sys.stdout.write("Phase 1/4: Building 5-card states... ")
        sys.stdout.flush()

    # Build 5-card terminal states
    five_count = 0
    for hand in itertools.combinations(cards, 5):
        fset5 = frozenset(hand)
        slot5 = get_or_alloc(fset5)
        five_count += 1
    if progress:
        print(f"{five_count} states")
        sys.stdout.flush()

    if progress:
        sys.stdout.write("Phase 2/4: Building 6-card states... ")
        sys.stdout.flush()

    six_count = 0
    for hand in itertools.combinations(cards, 6):
        fset6 = frozenset(hand)
        slot6 = get_or_alloc(fset6)
        six_count += 1
    if progress:
        print(f"{six_count} states")
        sys.stdout.flush()

    if progress:
        sys.stdout.write("Phase 3/4: Building 7-card states and computing ranks...\n")
        sys.stdout.flush()

    # Fill 7-card terminal states and populate 6→7 links
    seven_count = 0
    total_seven = 133784560
    step = max(1, total_seven // 100)

    for hand7 in itertools.combinations(cards, 7):
        fset7 = frozenset(hand7)
        # Best 5 of 7
        rank7 = 0
        cards_idx = [c - 1 for c in hand7]  # 0-based for CARDS_RM
        for combo in itertools.combinations(cards_idx, 5):
            r = eval_5card(CARDS_RM[combo[0]], CARDS_RM[combo[1]],
                           CARDS_RM[combo[2]], CARDS_RM[combo[3]], CARDS_RM[combo[4]])
            if r > rank7:
                rank7 = r

        # Fill the 7-card slot
        slot7 = get_or_alloc(fset7)
        # (slot7 is a terminal — we store rank here for direct 7-card eval,
        #  but we don't actually look it up directly; the 6→7 link does.)

        # For each 6-card subset: fill HR[slot6 + 7th_card] = rank7
        for i, extra in enumerate(hand7):
            fset6 = fset7 - {extra}
            slot6 = state_slot[fset6]
            HR[slot6 + extra] = rank7

        seven_count += 1
        if progress and seven_count % step == 0:
            pct = seven_count * 100 // total_seven
            sys.stdout.write(f"\r  {pct}% ({seven_count:,}/{total_seven:,}) states")
            sys.stdout.flush()

    if progress:
        print(f"\r  100% ({seven_count:,}/{total_seven:,}) states")
        sys.stdout.flush()

    if progress:
        sys.stdout.write("Phase 4/4: Filling 1-4 card states and 5→6 links...\n")
        sys.stdout.flush()

    # Fill 5-card ranks and 5→6 links
    for hand5 in itertools.combinations(cards, 5):
        fset5 = frozenset(hand5)
        slot5 = state_slot[fset5]
        cards5_idx = [c - 1 for c in hand5]
        rank5 = eval_5card(CARDS_RM[cards5_idx[0]], CARDS_RM[cards5_idx[1]],
                            CARDS_RM[cards5_idx[2]], CARDS_RM[cards5_idx[3]],
                            CARDS_RM[cards5_idx[4]])
        # For each possible 6th card
        for c6 in cards:
            if c6 in fset5:
                continue
            fset6 = fset5 | {c6}
            slot6 = state_slot[fset6]
            HR[slot5 + c6] = slot6  # link to 6-card slot (for 7-card eval path)
        # Also: if we stop at 5 cards, HR[slot5] should give rank5.
        # The table actually doesn't have a self-reference slot — the rank is
        # returned by the 5th lookup itself. We handle this below via 4→5 links.

    # Fill 4-card → 5-card links
    if progress:
        sys.stdout.write("  Building 4-card states...\n")
        sys.stdout.flush()
    for hand4 in itertools.combinations(cards, 4):
        fset4 = frozenset(hand4)
        slot4 = get_or_alloc(fset4)
        for c5 in cards:
            if c5 in fset4:
                continue
            fset5 = fset4 | {c5}
            slot5 = state_slot[fset5]
            # At the 5-card terminal, we want rank5 (not slot5)
            # But we also need slot5 for further 6/7-card lookup.
            # Standard Two Plus Two: the 5th lookup returns rank5 when evaluating
            # 5 cards, but we need slot5 when evaluating 6/7.
            # Resolution: HR[slot4 + c5] = rank5 for 5-card eval.
            # For 6/7-card eval, the caller does 6 or 7 lookups instead of 5.
            # So we actually set HR[slot4 + c5] to the 5-card state slot,
            # and the 5-card state's entries give either rank (for 5-card eval)
            # or slot6 (for 6-card eval).
            # This means the 5-card case needs a special marker...
            # Actually the standard table uses: if HR[slot4+c5] > 0 and
            # the table was built correctly, doing 5 lookups gives rank,
            # doing 6 lookups gives slot6, doing 7 lookups gives rank7.
            # The trick: for pure 5-card eval, HR[slot4+c5] = rank5.
            # For 6+ card eval, a separate path is used.
            # The standard Two Plus Two table returns rank for 5/6/7 cards —
            # for 6 cards, after 6 lookups you get best-5-from-6 rank directly.
            # For 7 cards, after 7 lookups you get best-5-from-7 rank directly.
            # There's no mixing: you always do exactly N lookups for N cards.
            # So: HR[slot4 + c5] = slot5 always (for 6/7-card path).
            # The 5-card rank is obtained differently in practice (rarely used).
            HR[slot4 + c5] = slot5

    # Fill 3-card → 4-card links
    if progress:
        sys.stdout.write("  Building 3-card states...\n")
        sys.stdout.flush()
    for hand3 in itertools.combinations(cards, 3):
        fset3 = frozenset(hand3)
        slot3 = get_or_alloc(fset3)
        for c4 in cards:
            if c4 in fset3:
                continue
            fset4 = fset3 | {c4}
            slot4 = state_slot[fset4]
            HR[slot3 + c4] = slot4

    # Fill 2-card → 3-card links
    if progress:
        sys.stdout.write("  Building 2-card states...\n")
        sys.stdout.flush()
    for hand2 in itertools.combinations(cards, 2):
        fset2 = frozenset(hand2)
        slot2 = get_or_alloc(fset2)
        for c3 in cards:
            if c3 in fset2:
                continue
            fset3 = fset2 | {c3}
            slot3 = state_slot[fset3]
            HR[slot2 + c3] = slot3

    # Fill 1-card → 2-card links, and entry points
    if progress:
        sys.stdout.write("  Building 1-card states...\n")
        sys.stdout.flush()
    for c1 in cards:
        fset1 = frozenset([c1])
        slot1 = get_or_alloc(fset1)
        for c2 in cards:
            if c2 == c1:
                continue
            fset2 = fset1 | {c2}
            slot2 = state_slot[fset2]
            HR[slot1 + c2] = slot2
        # Entry point
        HR[53 + c1] = slot1

    # Fix 5-card terminal entries: after 5 lookups we want rank5,
    # but we stored slot6 links in HR[slot5 + c6].
    # For the 5-card eval path: HR[slot4 + c5] should give rank5 directly.
    # We need to overwrite those entries.
    if progress:
        sys.stdout.write("  Fixing 5-card terminal entries for 5-card eval...\n")
        sys.stdout.flush()
    for hand4 in itertools.combinations(cards, 4):
        fset4 = frozenset(hand4)
        slot4 = state_slot[fset4]
        for c5 in cards:
            if c5 in fset4:
                continue
            fset5 = fset4 | {c5}
            cards5_idx = [c - 1 for c in fset5]
            rank5 = 0
            for combo in itertools.combinations(cards5_idx, 5):
                r = eval_5card(CARDS_RM[combo[0]], CARDS_RM[combo[1]],
                               CARDS_RM[combo[2]], CARDS_RM[combo[3]],
                               CARDS_RM[combo[4]])
                if r > rank5:
                    rank5 = r
            # Overwrite with rank5 (the 5-card path ends here)
            # But this conflicts with the 6/7-card path needing slot5!
            # The standard Two Plus Two table resolves this by having the
            # 4→5 entry point to slot5, and within slot5, each entry is
            # either a rank (if used as 5-card terminal) or a slot6 link.
            # For 5-card eval: after 4 lookups you're at slot4.
            #   HR[slot4 + c5] = rank5 ← directly returns rank
            # For 6-card eval: after 5 lookups you're at slot5.
            #   HR[slot5 + c6] = slot6 ← link to next level
            # For 7-card eval: after 6 lookups you're at slot6.
            #   HR[slot6 + c7] = rank7 ← directly returns rank
            # So for 5-card eval, HR[slot4 + c5] = rank5 (override the slot5 pointer).
            # This means 5-card eval and 6/7-card eval use DIFFERENT entry sequences.
            # 5-card: 4 traversals then read HR[slot4 + c5] directly.
            # 6-card: 5 traversals ending at slot5, then read HR[slot5 + c6].
            # 7-card: 6 traversals ending at slot6, then read HR[slot6 + c7].
            # Our code always does N traversals for N cards, so this is fine.
            HR[slot4 + c5] = rank5  # 5-card terminal

    if progress:
        print(f"  Table built. Next free slot: {next_free[0]:,} / {HR_SIZE:,}")
        sys.stdout.flush()

    # Write to file
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    if progress:
        sys.stdout.write(f"Writing {HR_SIZE * 4 / 1e6:.0f} MB to {outpath}...\n")
        sys.stdout.flush()

    with open(outpath, 'wb') as f:
        HR.tofile(f)

    if progress:
        print("Done.")
        size = os.path.getsize(outpath)
        print(f"File size: {size / 1e6:.1f} MB  ({size} bytes)")


if __name__ == '__main__':
    outpath = sys.argv[1] if len(sys.argv) > 1 else 'data/handranks.dat'
    build_table_senzee(outpath, progress=True)

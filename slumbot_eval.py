#!/usr/bin/env python3
"""
slumbot_eval.py — Play the trained CFR strategy vs Slumbot's public API.

Usage:
    pip install requests
    python slumbot_eval.py --strategy strategy_5h.bin --hands 200
    python slumbot_eval.py --strategy test_strategy.bin --hands 200 --verbose

Slumbot parameters: SB=50, BB=100, stack=20,000 (200 BB).
Our training: SB=10, BB=20, stack=1000 (50 BB).
The abstraction (Chen buckets + equity buckets + action-type history) is
stack-size-agnostic, so the strategy transfers without modification.

Results are reported as mBB/hand (millibig-blinds per hand) and BB/100,
matching the standard poker research metric.
"""

import argparse
import bisect
import itertools
import random
import struct
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

# =============================================================================
# Constants (must match cfr_gpu.cuh)
# =============================================================================
GPU_NUM_ACTIONS = 8
GPU_TABLE_SIZE  = 1 << 26          # 67,108,864 slots
STRATEGY_MAGIC_V2 = 0x53545241540100

PREFLOP_BUCKETS  = 64    # matches abstraction.h
POSTFLOP_BUCKETS = 128   # matches abstraction.h

# Action indices (must match abstraction.h)
FOLD, CHECK, CALL, RAISE_MIN, RAISE_HALF, RAISE_TWO_THIRDS, RAISE_POT, ALL_IN = range(8)

# Slumbot API
SLUMBOT_URL = "https://slumbot.com"   # note: no www — redirects cause 400

# =============================================================================
# Strategy file loading
# =============================================================================
class StrategyTable:
    """
    Memory-efficient strategy lookup table.
    Uses a direct-index numpy array (GPU_TABLE_SIZE × 8 float32) so that
    67M-entry strategy files don't need 67M Python list objects (~34 GB).
    Instead: ~2.1 GB float32 array + 67 MB bool mask = ~2.2 GB total.
    Supports the same .get(key) / len() interface as a regular dict.
    """
    def __init__(self, np, valid_keys, valid_probs, n_total):
        self._n = n_total
        # Allocate direct-access arrays
        print(f"  Allocating {GPU_TABLE_SIZE * GPU_NUM_ACTIONS * 4 / 1e9:.1f} GB table ...", flush=True)
        self._table = np.zeros((GPU_TABLE_SIZE, GPU_NUM_ACTIONS), dtype=np.float32)
        self._valid = np.zeros(GPU_TABLE_SIZE, dtype=bool)
        self._table[valid_keys] = valid_probs
        self._valid[valid_keys] = True

    def get(self, key, default=None):
        if self._valid[key]:
            return self._table[key].tolist()   # 8-float list, one entry only
        return default

    def __len__(self):
        return self._n


def load_strategy(path):
    """Load strategy.bin → StrategyTable (numpy direct-access) or plain dict for small files."""
    print(f"Loading strategy from {path} ...", flush=True)
    t0 = time.time()

    with open(path, 'rb') as f:
        magic, = struct.unpack('<Q', f.read(8))
        if magic != STRATEGY_MAGIC_V2:
            raise ValueError(f"Unexpected magic: {hex(magic)} in {path}")
        table_size, num_actions, n = struct.unpack('<III', f.read(12))
        assert table_size == GPU_TABLE_SIZE,  f"Table size mismatch: {table_size}"
        assert num_actions == GPU_NUM_ACTIONS, f"Num actions mismatch: {num_actions}"
        raw = f.read()  # read rest of file

    entry_bytes = 4 + GPU_NUM_ACTIONS * 4  # uint32 key + 8 floats
    expected = n * entry_bytes
    if len(raw) < expected:
        raise ValueError(f"Truncated file: expected {expected} bytes, got {len(raw)}")

    try:
        import numpy as np
        print(f"  Parsing {n:,} entries via numpy ...", flush=True)
        dtype = np.dtype([('key', '<u4'), ('probs', '<f4', GPU_NUM_ACTIONS)])
        arr = np.frombuffer(raw[:n * entry_bytes], dtype=dtype)
        # Filter out uniform-placeholder entries (all probs = 0.125)
        uniform = np.all(np.abs(arr['probs'] - 0.125) < 1e-6, axis=1)
        valid = arr[~uniform]
        n_valid = int(valid.shape[0])

        if n_valid > 10_000_000:
            # Large strategy (>10M entries): plain dict would need ~5 GB/10M entries
            # and would OOM for 67M entries (~34 GB). Use direct numpy table instead.
            strat = StrategyTable(np, valid['key'], valid['probs'], n_valid)
        else:
            # Small strategy: plain dict is fine (faster construction, less total RAM)
            keys  = valid['key'].tolist()
            probs = valid['probs'].tolist()
            strat = dict(zip(keys, probs))
    except ImportError:
        # Slow path: pure-Python struct loop (no numpy)
        print("  numpy not found — using slow loader ...", flush=True)
        fmt = '<I8f'
        chunk_size = struct.calcsize(fmt)
        strat = {}
        for i in range(n):
            off = i * chunk_size
            vals = struct.unpack_from(fmt, raw, off)
            key, probs = vals[0], list(vals[1:])
            if all(abs(p - 0.125) < 1e-6 for p in probs):
                continue
            strat[key] = probs

    print(f"  Loaded {len(strat):,} info sets in {time.time()-t0:.1f}s")
    return strat

# =============================================================================
# Card utilities
# =============================================================================
# Internal encoding: suit-major → card = suit*13 + rank  (0-based, 0-51)
#   suit: 0=clubs 1=diamonds 2=hearts 3=spades
#   rank: 0=2 … 12=A

RANK_CHAR = '23456789TJQKA'
SUIT_CHAR = 'cdhs'

def parse_card(s):
    """Slumbot 'Ac' → internal index."""
    r = RANK_CHAR.index(s[0])
    u = SUIT_CHAR.index(s[1])
    return u * 13 + r

def card_rank(c): return c % 13
def card_suit(c): return c // 13

# =============================================================================
# Hand evaluator (pure Python — replicates GPU eval5_gpu logic)
# =============================================================================
# Category offset table (base[cat] from cfr_gpu.cu)
_CATEGORY_BASE = [0, 0, 1277, 4137, 4995, 5853, 5863, 7140, 7296, 7452]

def eval5(cards):
    """Evaluate 5 cards → rank in [1, 7462]. Higher = stronger."""
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]
    cnt = {}
    for r in ranks:
        cnt[r] = cnt.get(r, 0) + 1
    counts = sorted(cnt.values(), reverse=True)

    is_flush  = len(set(suits)) == 1
    all_dist  = (len(cnt) == 5)
    is_str    = all_dist and (ranks[0] - ranks[4] == 4)
    is_wheel  = (set(ranks) == {12, 0, 1, 2, 3})
    str_high  = 3 if is_wheel else ranks[0]

    if is_flush and (is_str or is_wheel):    # Straight flush / Royal
        return _CATEGORY_BASE[9] + str_high - 3 + 1
    if is_flush:
        # Encode flush as descending rank quintuple packed into int
        v = ranks[0]*13**4 + ranks[1]*13**3 + ranks[2]*13**2 + ranks[3]*13 + ranks[4]
        return _CATEGORY_BASE[6] + (v % (_CATEGORY_BASE[7] - _CATEGORY_BASE[6])) + 1
    if is_str or is_wheel:
        return _CATEGORY_BASE[5] + str_high - 3 + 1

    quads = trips = pair1 = pair2 = None
    for r in range(12, -1, -1):
        if cnt.get(r) == 4:
            quads = r
        elif cnt.get(r) == 3:
            trips = r
        elif cnt.get(r) == 2:
            if pair1 is None: pair1 = r
            else:              pair2 = r

    if quads is not None:
        k = next(r for r in ranks if r != quads)
        ak = k if k < quads else k - 1
        return _CATEGORY_BASE[8] + quads * 12 + ak + 1

    if trips is not None and pair1 is not None:
        ap = pair1 if pair1 < trips else pair1 - 1
        return _CATEGORY_BASE[7] + trips * 12 + ap + 1

    if trips is not None:
        ks = [r for r in ranks if r != trips]
        ks.sort(reverse=True)
        a0 = ks[0] if ks[0] < trips else ks[0] - 1
        a1 = ks[1] if ks[1] < trips else ks[1] - 1
        return _CATEGORY_BASE[4] + trips * 66 + a0*(a0-1)//2 + a1 + 1

    if pair1 is not None and pair2 is not None:
        k = next(r for r in ranks if cnt[r] == 1)
        ak = k if k < pair2 else (k-1 if k < pair1 else k-2)
        return _CATEGORY_BASE[3] + (pair1*(pair1-1)//2 + pair2)*11 + ak + 1

    if pair1 is not None:
        ks = [r for r in ranks if r != pair1]
        ks.sort(reverse=True)
        a = [ki if ki < pair1 else ki-1 for ki in ks]
        return _CATEGORY_BASE[2] + pair1*220 + a[0]*(a[0]-1)*(a[0]-2)//6 + a[1]*(a[1]-1)//2 + a[2] + 1

    # High card: encode as big number preserving rank order
    v = ranks[0]*13**4 + ranks[1]*13**3 + ranks[2]*13**2 + ranks[3]*13 + ranks[4]
    return 1 + (v % 1277)

def eval_best(hole0, hole1, community):
    """Best 5-card rank from 2 hole + community."""
    all_cards = [hole0, hole1] + list(community)
    if len(all_cards) < 5:
        return 0
    return max(eval5(list(combo)) for combo in itertools.combinations(all_cards, 5))

# =============================================================================
# Abstraction buckets (mirrors abstraction.cpp)
# =============================================================================
def chen_score(c0, c1):
    r0, r1 = card_rank(c0), card_rank(c1)
    suited = card_suit(c0) == card_suit(c1)
    if r0 < r1:
        r0, r1 = r1, r0
    # Must mirror abstraction.cpp chen_score exactly (same special cases)
    if   r0 == 12: base = 10.0  # A
    elif r0 == 11: base = 8.0   # K
    elif r0 == 10: base = 7.0   # Q
    elif r0 ==  9: base = 6.0   # J
    else:          base = max(0.5, r0 / 2.0)  # 2-T
    if r0 == r1:
        return max(5.0, base * 2.0)
    score = base + (2.0 if suited else 0.0)
    gap = r0 - r1 - 1
    score += [0, -1, -2, -4, -5][min(gap, 4)]
    if gap <= 1 and r0 < 12:
        score += 1.0
    return score

_ALL_CHEN = sorted(chen_score(i, j) for i in range(52) for j in range(i+1, 52))

def preflop_bucket(c0, c1):
    pct = bisect.bisect_left(_ALL_CHEN, chen_score(c0, c1)) / len(_ALL_CHEN)
    return min(int(pct * PREFLOP_BUCKETS), PREFLOP_BUCKETS - 1)

def postflop_bucket(hole0, hole1, community):
    if not community:
        return 0
    rank = eval_best(hole0, hole1, community)
    return min(int((rank - 1) * POSTFLOP_BUCKETS // 7462), POSTFLOP_BUCKETS - 1)

# =============================================================================
# Info-set hash (FNV-1a — must match info_set_hash in cfr_gpu.cu exactly)
# =============================================================================
def info_set_hash(player, hole_b, board_b, street, action_bits):
    h = 2166136261
    for byte in [player, hole_b, board_b, street,
                 action_bits & 0xFF,
                 (action_bits >>  8) & 0xFF,
                 (action_bits >> 16) & 0xFF,
                 (action_bits >> 24) & 0xFF]:
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h & (GPU_TABLE_SIZE - 1)

# =============================================================================
# Valid actions mask (mirrors valid_mask_gpu in cfr_gpu.cu)
# =============================================================================
def valid_actions_mask(pot, stack, to_call, current_bet, last_full_raise, bb):
    if pot <= 0: pot = 1
    mask = 0
    if to_call == 0:
        mask |= (1 << CHECK)
    elif to_call >= stack:
        return (1 << FOLD) | (1 << ALL_IN)
    else:
        mask |= (1 << FOLD) | (1 << CALL)

    head = stack - to_call
    if head <= 0:
        return mask | (1 << ALL_IN)

    min_rt = (current_bet + max(last_full_raise, bb)) if current_bet > 0 else max(1, bb)
    max_rt = current_bet + head

    def _valid(rt): return min_rt <= rt <= max_rt

    if max_rt >= min_rt:
        if _valid(min_rt):                                  mask |= (1 << RAISE_MIN)
        if _valid(max(min_rt, current_bet + max(1, pot//2))): mask |= (1 << RAISE_HALF)
        if _valid(max(min_rt, current_bet + max(1, 2*pot//3))): mask |= (1 << RAISE_TWO_THIRDS)
        if _valid(max(min_rt, current_bet + max(1, pot))):   mask |= (1 << RAISE_POT)
    mask |= (1 << ALL_IN)
    return mask

# =============================================================================
# Action → chip computation (mirrors action_to_chips in abstraction.cpp)
# =============================================================================
def action_to_chips(action, pot, stack, to_call, current_bet, last_full_raise, bb):
    if pot <= 0: pot = 1
    min_rt = (current_bet + max(last_full_raise, bb)) if current_bet > 0 else max(1, bb)
    max_rt = current_bet + max(0, stack - to_call)

    def clamp(rt): return min(rt, max_rt)

    if action == FOLD:   return 0
    if action == CHECK:  return 0
    if action == CALL:   return min(to_call, stack)
    if action == RAISE_MIN:        rt = clamp(min_rt)
    elif action == RAISE_HALF:     rt = clamp(max(min_rt, current_bet + max(1, pot//2)))
    elif action == RAISE_TWO_THIRDS: rt = clamp(max(min_rt, current_bet + max(1, 2*pot//3)))
    elif action == RAISE_POT:      rt = clamp(max(min_rt, current_bet + max(1, pot)))
    elif action == ALL_IN:         return stack
    else: return 0
    return min(to_call + max(0, rt - current_bet), stack)

# =============================================================================
# Slumbot action string → abstract action index
# =============================================================================
def slumbot_bet_to_abstract(bet_to, current_bet, bets_actor, pot):
    """Map Slumbot's bet_to (total commitment this street) to our action index."""
    # Net raise above current_bet
    raise_above = bet_to - current_bet
    if raise_above <= 0:
        return CALL

    # All-in if player is committing everything
    frac = raise_above / max(1, pot)
    if frac >= 1.5 or raise_above >= 15000:
        return ALL_IN
    if frac >= 0.85:
        return RAISE_POT
    if frac >= 0.55:
        return RAISE_TWO_THIRDS
    if frac >= 0.35:
        return RAISE_HALF
    return RAISE_MIN

# =============================================================================
# Game state tracker
# =============================================================================
class GameTracker:
    """
    Tracks NLH game state from Slumbot API responses.
    Computes: pot, stacks, current_bet, to_call, last_full_raise, action_bits.

    Convention: player 0 = SB (acts first preflop), player 1 = BB.
    Slumbot client_pos=0 is SB.
    """

    def __init__(self, sb=50, bb=100, stack_size=20000):
        self.sb = sb
        self.bb = bb
        self.stack_size = stack_size

    def parse(self, action_str, client_pos, hole_cards_int, board_int):
        """
        Parse the full Slumbot action string and return info needed for strategy lookup.

        Returns dict with:
          player, hole_b, board_b, street,
          action_bits, valid_mask, pot, to_call, current_bet, our_stack, last_full_raise
        """
        streets = action_str.split('/')
        street = len(streets) - 1  # 0=pre 1=flop 2=turn 3=river

        # Rebuild state from scratch
        stacks = [self.stack_size - self.sb, self.stack_size - self.bb]
        pot    = self.sb + self.bb
        bets   = [0, 0]  # current-street bets (above blind level doesn't matter for pot)
        # Actually: bets[0] = amount put in this street (for SB = sb initial, BB = bb initial)
        bets   = [self.sb, self.bb]   # preflop starting bets (blinds)
        current_bet   = self.bb
        last_full_raise = self.bb
        accumulated_pot = 0           # pot from completed streets

        action_bits = 0

        for st_idx, st_acts in enumerate(streets):
            if st_idx > 0:
                # Close previous street
                accumulated_pot += sum(bets)
                bets = [0, 0]
                current_bet = 0
                last_full_raise = self.bb

            # Preflop: SB (pos 0) acts first
            # Postflop: BB (pos 1) acts first
            actor = 0 if st_idx == 0 else 1

            i = 0
            while i < len(st_acts):
                c = st_acts[i]

                if c == 'f':
                    action_bits = ((action_bits << 3) | FOLD) & 0x3FFFFFFF
                    i += 1
                elif c == 'k':
                    action_bits = ((action_bits << 3) | CHECK) & 0x3FFFFFFF
                    i += 1
                elif c == 'c':
                    action_bits = ((action_bits << 3) | CALL) & 0x3FFFFFFF
                    to_c = current_bet - bets[actor]
                    chips = min(to_c, stacks[actor])
                    stacks[actor] -= chips
                    bets[actor]   += chips
                    i += 1
                elif c == 'b':
                    j = i + 1
                    while j < len(st_acts) and st_acts[j].isdigit():
                        j += 1
                    bet_to = int(st_acts[i+1:j])
                    eff_pot = accumulated_pot + sum(bets)
                    abstract = slumbot_bet_to_abstract(bet_to, current_bet, bets[actor], eff_pot)
                    action_bits = ((action_bits << 3) | abstract) & 0x3FFFFFFF

                    chips = bet_to - bets[actor]
                    chips = min(chips, stacks[actor])
                    stacks[actor] -= chips
                    bets[actor]   += chips

                    if bets[actor] > current_bet:
                        raise_inc = bets[actor] - current_bet
                        if raise_inc >= last_full_raise or current_bet == 0:
                            last_full_raise = raise_inc
                        current_bet = bets[actor]
                    i = j
                else:
                    i += 1  # skip unexpected chars
                    continue

                actor = 1 - actor  # alternate actors

        # Slumbot client_pos convention: client_pos=1 = SB (acts first preflop),
        # client_pos=0 = BB.  Our internal model: player 0 = SB, player 1 = BB.
        # So: internal_player = 1 - client_pos.
        internal_player = 1 - client_pos  # 0=SB, 1=BB in our training convention

        # Current state: it's now our turn (or end of hand if winnings present)
        our_stack = stacks[internal_player]
        to_call   = current_bet - bets[internal_player]

        pot = accumulated_pot + sum(bets)

        # Community cards by street
        comm_counts = [0, 3, 4, 5]
        n_comm = comm_counts[min(street, 3)]
        community = board_int[:n_comm]

        # Compute abstraction buckets
        h0, h1  = hole_cards_int[0], hole_cards_int[1]
        hole_b  = preflop_bucket(h0, h1)
        board_b = postflop_bucket(h0, h1, community)

        vm = valid_actions_mask(pot, our_stack, to_call, current_bet, last_full_raise, self.bb)

        return dict(
            player=internal_player,   # 0=SB, 1=BB — matches training convention
            hole_b=hole_b, board_b=board_b,
            street=min(street, 3),
            action_bits=action_bits,
            valid_mask=vm,
            pot=pot,
            to_call=to_call,
            current_bet=current_bet,
            our_stack=our_stack,
            last_full_raise=last_full_raise,
        )

# =============================================================================
# Strategy lookup + action selection
# =============================================================================
def sample_action(strat, state, rng):
    """Look up info set, sample valid action. Falls back to passive if unseen."""
    key = info_set_hash(state['player'], state['hole_b'], state['board_b'],
                        state['street'], state['action_bits'])
    entry = strat.get(key)
    vm = state['valid_mask']

    if entry is None:
        # Unseen state: prefer passive (check > call > fold, no raise)
        for a in [CHECK, CALL, FOLD]:
            if vm & (1 << a):
                return a
        return CALL

    # Mask to valid actions and renormalize
    masked = [entry[a] if (vm & (1 << a)) else 0.0 for a in range(GPU_NUM_ACTIONS)]
    total = sum(masked)
    if total < 1e-9:
        for a in [CHECK, CALL, FOLD]:
            if vm & (1 << a):
                return a
        return CALL

    r = rng.random() * total
    cum = 0.0
    last_valid = -1
    for a in range(GPU_NUM_ACTIONS):
        if not (vm & (1 << a)): continue
        cum += masked[a]
        last_valid = a
        if r <= cum:
            return a
    return last_valid

def action_to_incr(action, state, bb):
    """Convert abstract action to Slumbot 'incr' string."""
    pot = state['pot']
    stack = state['our_stack']
    to_call = state['to_call']
    current_bet = state['current_bet']
    last_full_raise = state['last_full_raise']

    if action == FOLD:  return 'f'
    if action == CHECK: return 'k'
    if action == CALL:  return 'c'
    if action == ALL_IN:
        # All-in: total commitment = chips already in this street + remaining stack
        already_in = current_bet - to_call   # what we've put in so far this street
        return f'b{already_in + stack}'

    # Raise actions
    min_rt = (current_bet + max(last_full_raise, bb)) if current_bet > 0 else max(1, bb)
    max_rt = current_bet + (stack - to_call)

    if action == RAISE_MIN:
        rt = min_rt
    elif action == RAISE_HALF:
        rt = max(min_rt, current_bet + max(1, pot // 2))
    elif action == RAISE_TWO_THIRDS:
        rt = max(min_rt, current_bet + max(1, 2 * pot // 3))
    elif action == RAISE_POT:
        rt = max(min_rt, current_bet + max(1, pot))
    else:
        rt = min_rt

    rt = min(rt, max_rt)
    if rt <= 0 or rt < min_rt:
        return 'c'  # fallback
    return f'b{rt}'

# =============================================================================
# Slumbot API helpers
# =============================================================================
ACTION_NAMES = ['fold','check','call','raise_min','raise_half','raise_2/3','raise_pot','all_in']

def new_hand(token=None, timeout=15):
    # Slumbot API accepts JSON body; do NOT use www.slumbot.com (returns 400)
    payload = {'token': token} if token else {}
    r = requests.post(f"{SLUMBOT_URL}/api/new_hand", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def act(token, incr, timeout=15):
    r = requests.post(f"{SLUMBOT_URL}/api/act",
                      json={'token': token, 'incr': incr}, timeout=timeout)
    r.raise_for_status()
    return r.json()

# =============================================================================
# Main evaluation loop
# =============================================================================
def play_hand(strat, tracker, rng, token, verbose=False):
    """
    Play one hand against Slumbot.
    Returns (net_chips_won, new_token).
    net_chips_won: positive = we won, negative = we lost.
    """
    state_data = new_hand(token)
    token = state_data.get('token', token)

    client_pos = state_data.get('client_pos', 0)
    bb = tracker.bb

    while True:
        # Check if hand is over
        if 'winnings' in state_data and state_data['winnings'] is not None:
            net = int(state_data['winnings'])
            if verbose:
                hole = state_data.get('hole_cards', [])
                bot_hole = state_data.get('bot_hole_cards', [])
                board = state_data.get('board', [])
                print(f"  Hand over: net={net:+d} chips ({net/bb:+.1f} BB)")
                print(f"  Our cards: {' '.join(hole)}  Bot cards: {' '.join(bot_hole)}  Board: {' '.join(board)}")
                print(f"  Action: {state_data.get('action','')}")
            return net, token

        action_str  = state_data.get('action', '')
        hole_strs   = state_data.get('hole_cards', [])
        board_strs  = state_data.get('board', [])

        hole_int  = [parse_card(c) for c in hole_strs] if len(hole_strs) >= 2 else [0, 1]
        board_int = [parse_card(c) for c in board_strs]

        info = tracker.parse(action_str, client_pos, hole_int, board_int)

        chosen = sample_action(strat, info, rng)
        incr   = action_to_incr(chosen, info, bb)

        if verbose:
            board_disp = ' '.join(board_strs) if board_strs else '(no board)'
            streets = ['Pre','Flop','Turn','River']
            print(f"  {streets[info['street']]}  cards={' '.join(hole_strs)}  board={board_disp}")
            print(f"  pot={info['pot']} to_call={info['to_call']} stack={info['our_stack']}")
            print(f"  action -> {ACTION_NAMES[chosen]} (incr='{incr}')")

        state_data = act(token, incr)
        token = state_data.get('token', token)

def main():
    ap = argparse.ArgumentParser(description="Play trained CFR strategy vs Slumbot")
    ap.add_argument('--strategy', required=True, help='Path to strategy .bin file')
    ap.add_argument('--hands', type=int, default=200, help='Number of hands to play')
    ap.add_argument('--seed', type=int, default=42, help='RNG seed')
    ap.add_argument('--verbose', action='store_true', help='Print each hand decision')
    ap.add_argument('--sb', type=int, default=50, help='Small blind (Slumbot default: 50)')
    ap.add_argument('--bb', type=int, default=100, help='Big blind (Slumbot default: 100)')
    ap.add_argument('--stack', type=int, default=20000, help='Starting stack (Slumbot default: 20000)')
    args = ap.parse_args()

    strat   = load_strategy(args.strategy)
    tracker = GameTracker(sb=args.sb, bb=args.bb, stack_size=args.stack)
    rng     = random.Random(args.seed)

    print(f"\nPlaying {args.hands} hands vs Slumbot  (BB={args.bb}, stack={args.stack})")
    print(f"Strategy info sets: {len(strat):,}\n")

    total_chips = 0
    token = None
    errors = 0
    t0 = time.time()

    for hand_num in range(1, args.hands + 1):
        try:
            net, token = play_hand(strat, tracker, rng, token, verbose=args.verbose)
            total_chips += net

            if hand_num % 25 == 0 or hand_num == args.hands:
                bb_won   = total_chips / args.bb
                bb_per_100 = bb_won / hand_num * 100
                elapsed  = time.time() - t0
                print(f"  [{hand_num:4d}/{args.hands}]  "
                      f"net={bb_won:+.1f} BB  "
                      f"BB/100={bb_per_100:+.2f}  "
                      f"({elapsed:.0f}s)", flush=True)

        except requests.exceptions.RequestException as e:
            errors += 1
            print(f"  [hand {hand_num}] API error: {e} (skipping)", file=sys.stderr)
            token = None  # reset token on error
            if errors > 10:
                print("Too many API errors, stopping.", file=sys.stderr)
                break
            continue

    elapsed = time.time() - t0
    bb_won = total_chips / args.bb
    hands_played = hand_num - errors
    bb_per_100 = bb_won / max(1, hands_played) * 100

    # 95% CI (approximate — poker has ~30-50 BB/hand std dev)
    import math
    std_per_hand_bb = 30.0  # conservative estimate for HUNL
    ci95 = 1.96 * std_per_hand_bb / math.sqrt(max(1, hands_played)) * 100

    print(f"\n{'='*55}")
    print(f"  Hands played : {hands_played}")
    print(f"  Net BB       : {bb_won:+.2f}")
    print(f"  BB/100       : {bb_per_100:+.2f}  (95%% CI: ±{ci95:.1f})")
    print(f"  Time         : {elapsed:.0f}s  ({elapsed/max(1,hands_played):.1f}s/hand)")
    print(f"{'='*55}")
    print()
    print("  Interpretation:")
    print("  * +0 BB/100 is expected for a GTO strategy (Nash equilibrium)")
    print("    vs any opponent, since GTO can't lose in expectation.")
    print("  * Negative = Slumbot is exploiting our abstraction errors")
    print("  * Positive = we're exploiting Slumbot's weaknesses")
    print(f"  * Need ~{int(9*(std_per_hand_bb)**2 / (bb_per_100**2+0.01)):,} hands for 95% significance")
    print()
    print("  Note: our bot was trained at 50BB depth; Slumbot plays 200BB.")
    print("  Deep-stack spots have distribution shift — expect moderate noise.")

if __name__ == '__main__':
    main()

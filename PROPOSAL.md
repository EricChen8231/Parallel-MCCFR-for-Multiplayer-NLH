# EE/CSCI 451 — Project Proposal
## Parallel Monte Carlo Counterfactual Regret Minimization for No-Limit Texas Hold'em Poker

**Team:** [Your Name(s)]
**Category:** Research, Implement, and Evaluate
**Platform:** CUDA C++ · cuRAND · OpenMP · MPI · USC CARC A100 GPUs
**Proposal Due:** Wednesday, March 11, 2026 at 11:59 PM

---

## Introduction

### Background

No-Limit Texas Hold'em (NLHE) poker is a canonical benchmark for computing Nash equilibria in large imperfect-information games. The game has an estimated 10^160 reachable game states, making exact computation infeasible. The state-of-the-art approach is Counterfactual Regret Minimization (CFR) (Zinkevich et al., NeurIPS 2007), and specifically its Monte Carlo variant External-Sampling MCCFR (ES-MCCFR) (Lanctot et al., NeurIPS 2009), which approximates Nash equilibrium through millions of self-play iterations.

Each ES-MCCFR iteration is an independent random sample of the game tree — structurally embarrassingly parallel. However, practical parallelization is non-trivial. Our baseline Python implementation reveals the dominant cost is not compute but worker serialization overhead: pickling Python strategy dictionaries to disk consumes approximately 57 seconds per worker, capping speedup at roughly 7.6× on a 20-core workstation despite near-perfect compute parallelism. This is a direct instance of Amdahl's Law — the serial fraction driven by serialization dominates, not the parallel work of game tree traversal.

This project addresses these bottlenecks by porting the entire training system from Python to CUDA C++, running thousands of simultaneous poker games on a single A100 GPU, and augmenting the base algorithm with Linear CFR+ — the algorithm used by Pluribus (Brown & Sandholm, *Science* 2019) — to reduce the number of iterations required for convergence. The full training system targets USC CARC's A100 GPUs.

### Contributions and Hypotheses

The project advances four concrete hypotheses. First, GPU batched simulation is expected to achieve 100–1000× throughput over single-threaded CPU by running 65,536 simultaneous games per GPU kernel launch on an A100. The baseline Python implementation achieves approximately 230 iterations per second; a C++ CPU implementation achieves roughly 50,000; and an A100 GPU with a batch size of 65,536 is projected to reach approximately 13 million hands per second, representing a roughly 56,000× speedup over the Python baseline.

Second, a Structure-of-Arrays (SoA) memory layout combined with warp shuffle intrinsics is expected to eliminate the memory bottleneck. Storing regret tables in SoA format enables fully-coalesced warp reads yielding 128-byte memory transactions, compared to 8-byte strided accesses in Array-of-Structures (AoS) layout. The warp shuffle primitive `__shfl_down_sync` further reduces regret-sum reductions from a for-loop with shared memory to purely register-based operations, freeing L1 bandwidth for other computation.

Third, Linear CFR+ convergence is expected to be 2–10× faster than vanilla CFR by weighting iteration t's contribution by t rather than uniformly, which accelerates convergence from empirical O(1/ε²) toward near O(1/ε). At equal wall-clock time, Linear CFR+ is expected to reach the same exploitability level 2–10× faster. Fourth, multi-GPU MPI scaling across four A100 nodes on the CARC Discovery GPU partition should achieve approximately 3.8× speedup, bounded by the Amdahl limit set by AllReduce synchronization consuming roughly 2% of total time.

---

## Description of Implementation

### Core Algorithm: External-Sampling MCCFR with CFR+

ES-MCCFR traverses the game tree for N-player NLHE. At each information set, the update player's nodes explore all available actions and accumulate counterfactual regret, while other players' nodes sample a single action from the current regret-matched strategy, and terminal nodes compute chip payoffs via hand evaluation. The average strategy accumulated across all iterations converges to Nash equilibrium.

CFR+ modifies the base algorithm in three ways relative to vanilla CFR. Rather than updating all N players per iteration, CFR+ updates one player per iteration in alternating fashion. Rather than allowing regret to accumulate without bound, CFR+ applies the ReLU clamp `r = max(0, r + delta)`, preventing large negative regrets from early random-play from slowing convergence — the so-called "digging out" problem. Finally, rather than weighting all iterations equally in the strategy average, CFR+ weights iteration t's contribution by t, further accelerating convergence. These three changes together are what enabled Pluribus to achieve superhuman performance in six-player NLHE with a tractable compute budget.

### Transforming the Python Implementation to CUDA C++

The Python baseline, while functionally correct, carries several structural inefficiencies that motivate a ground-up rewrite in CUDA C++. In the Python implementation, cards are stored as integer indices 0–51 (where rank = idx % 13 and suit = idx // 13), hand evaluation calls a Python-level `best_hand_int()` function operating on tuples, game state is managed by a mutable `_State` class using a snapshot-and-restore pattern, and information set keys are Python strings assembled from concatenated action histories and card bucket labels. Strategy storage uses a Python dictionary mapping string keys to lists of floats, and multi-process parallelism is achieved by pickling these dictionaries to disk for inter-process merging.

Each of these design choices is appropriate for Python but becomes a performance liability at scale. The C++ CUDA rewrite addresses each layer systematically. Cards remain `uint8_t` values in the range 0–51, but 7-card hand evaluation is replaced by a precomputed Two Plus Two lookup table of approximately 130 MB, reducing per-hand evaluation cost from roughly 2 microseconds in Python to roughly 100 nanoseconds in C++. A `precompute_ranks()` function evaluates all players' best hands across all streets exactly once per iteration before any tree traversal begins, amortizing this cost across the entire game tree.

Game state is restructured as a fixed 256-byte `GameState` struct that is entirely stack-allocated. Snapshot and restore operations use `memcpy` rather than Python object copying, eliminating all heap allocation from the recursion hot path. Action history, which in Python is a dynamic list of strings, is encoded as a compact `uint8_t[4][16]` array per street, keeping the entire state representation in CPU registers or L1 cache.

Information set keys, which in Python are variable-length strings assembled with string concatenation, are replaced by 64-bit FNV-1a hashes of the tuple (player index, hole bucket, board bucket, action sequence). This eliminates both string allocation overhead and dictionary key hashing overhead, since integer key lookup in `std::unordered_map<uint64_t, std::array<float,8>>` is substantially cheaper than Python string dictionary access. The regret and strategy sum arrays, previously Python lists of floats, become fixed-size `float[8]` arrays directly embedded in the hash map value, enabling cache-line-aligned access.

The most significant transformation addresses the parallelization bottleneck directly. The Python implementation achieves parallelism by spawning worker processes, each of which trains independently and serializes its strategy dictionary to a shared file using Python's `pickle` module; the main process then reads and merges all files. This file-based inter-process communication consumes approximately 57 seconds per worker for a 50,000-iteration run, which Amdahl's Law translates into a hard speedup ceiling of roughly 7.6× regardless of core count. The C++ implementation eliminates this bottleneck entirely: worker threads sharing a process address space merge their strategy tables in O(total_info_sets) time with no locking and no I/O, and multi-node scaling uses MPI `Allreduce` over a flat float array, which requires no disk access and synchronizes in roughly 2% of total training time. Concretely, each MPI rank trains its share of iterations independently, then exports its strategy sums to a flat float array and participates in a single collective reduction via `MPI_Allreduce` with `MPI_SUM`, reconstructing a globally merged strategy without any per-node file I/O.

Card abstraction is also refined in the C++ version. The Python baseline uses 8 preflop buckets derived from Chen formula percentiles and 16 postflop buckets derived from hand rank. The C++ CARC version expands these to 50 preflop buckets and 50 postflop equity-percentile buckets, and adds a fifth abstract bet size (quarter-pot) alongside the existing half-pot, third-pot, pot, and all-in sizes. This finer abstraction reduces the information loss from bucketing, producing a closer approximation to the full-game Nash equilibrium at the cost of a larger strategy table, which the A100's 80 GB HBM2e memory can accommodate.

The live-play bot mirrors this transformation: the Python `CFRBot` class, which performs dictionary lookup against a pickled strategy file, is replaced by a C++ bot that loads the binary strategy table into memory at startup and performs hash-based lookup with sub-microsecond latency. Opponent modeling retains the same Bayesian confidence-scaling formula — `effective_weight(n) = EXPLOIT_WEIGHT × n / (n + k)` where k = 600 — but the observation recording and exploitation delta computation are implemented in C++ for compatibility with the real-time decision loop.

Parallelization in the C++ system operates at three levels simultaneously. Within a single node, OpenMP spawns independent trainer threads that each run their own copy of the game tree traversal loop; because threads never write to shared memory during traversal, no synchronization is needed until the final merge. Across CARC nodes, MPI provides the AllReduce collective. The two mechanisms compose cleanly: each MPI rank runs OpenMP-parallel trainers internally, so a four-node deployment with 48 cores per node runs 192 concurrent trainers, all contributing to a single globally merged strategy through one MPI collective. The corresponding CARC Slurm job requests four nodes with one MPI task per node, 48 CPUs per task, and eight hours of wall time, running the combined OpenMP-MPI binary with ten million total iterations and CFR+ enabled.

---

## Evaluation / Criteria of Success

### Experiment 1: Speedup Curve

The first experiment measures wall-clock speedup over a fixed budget of 50,000 iterations with two players. Speedup is defined as S(N) = T_serial / T_parallel(N). Four conditions are compared: the Python file-based reduce baseline at worker counts of 1, 2, 4, 8, 12, and 20; a Python numpy-based reduce at the same worker counts; C++ with OpenMP at 1, 4, 8, 16, and 48 threads; and C++ with MPI on CARC at 48, 96, 144, and 192 total workers. Results are plotted as speedup curves with Amdahl's theoretical prediction overlaid, using serial fractions of s = 0.40 for the pickle baseline, s = 0.013 for numpy reduce, and s ≈ 0.001 for C++ MPI. The experiment succeeds if the Python numpy reduce achieves S(12) ≥ 15× (compared to 7.6× for the pickle baseline), the C++ OpenMP implementation achieves S(48) ≥ 40×, and the C++ MPI deployment achieves S(192) ≥ 150×.

### Experiment 2: CFR+ Convergence

The second experiment compares vanilla CFR and CFR+ given equal wall-clock training time of 30 minutes with two players. Success is evaluated on three metrics: win rate against a uniform random opponent over 10,000 hands, number of distinct information sets discovered as a function of iterations, and exploitability proxy measured in BB/100 against a fixed calling-station archetype. The experiment succeeds if CFR+ achieves at least 2× better win rate in BB/100 at equal wall-clock time.

### Experiment 3: Real-Time Adaptation

The third experiment measures how effectively the opponent model transitions from pure GTO play toward exploitative adjustments over 500 hands of live play against four scripted archetypes. The archetypes are a calling station (8% fold, 72% call, 20% raise), a nit (60% fold, 30% call, 10% raise), a maniac (10% fold, 20% call, 70% raise), and a balanced player (28% fold, 44% call, 28% raise). BB/100 is measured in 50-hand rolling windows to observe the transition. Success requires a positive BB/100 trend from hands 1–50 to hands 250–300 against the calling station and nit archetypes, and no significant trend against the balanced archetype.

### Experiment 4: Abstraction Quality

The fourth experiment directly evaluates the impact of the Python-to-C++ rewrite's abstraction improvements. Strategies trained with the coarse abstraction (8 preflop and 16 postflop buckets, trained on a laptop) are compared against strategies trained with the fine abstraction (50 preflop and 50 postflop buckets, trained on CARC) at equal iteration count. The fine abstraction should achieve at least 10 BB/100 improvement against a calling-station opponent, demonstrating that the larger information set space made feasible by GPU training translates to meaningfully better strategy quality.

---

## Performance Projections

The performance gains from the Python-to-C++ CUDA transformation are substantial across the entire throughput hierarchy. The Python serial baseline achieves approximately 230 iterations per second, yielding 6.6 million iterations in an 8-hour overnight run. A Python numpy reduce with 12 workers scales this to roughly 4,000 iterations per second and 115 million overnight iterations, already a meaningful improvement but still constrained by Python's per-object overhead. The C++ single-core implementation, freed from interpreter overhead, Python object allocation, and string-based dictionary access, reaches approximately 50,000 iterations per second and 1.44 billion overnight iterations — a roughly 220× improvement over the Python serial baseline on the same hardware. C++ with OpenMP across 48 CARC cores reaches approximately 2.4 million iterations per second and 69 billion overnight, and the full four-node MPI deployment reaches approximately 9.6 million iterations per second and 276 billion iterations overnight. At 276 billion iterations, every information set in the abstracted game is visited thousands of times, exceeding the total training compute of Pluribus on a comparable abstraction.

---

## References

Zinkevich, M. et al. (2007). Regret Minimization in Games with Incomplete Information. *NeurIPS*. Lanctot, M. et al. (2009). Monte Carlo Sampling for Regret Minimization in Extensive Games. *NeurIPS*. Tammelin, O. (2014). Solving Large Imperfect Information Games Using CFR+. *arXiv:1407.5042*. Bowling, M. et al. (2015). Heads-Up Limit Hold'em Poker Is Solved. *Science*, 347(6218). Brown, N., Sandholm, T. (2017). Superhuman AI for Heads-Up No-Limit Poker: Libratus. *Science*. Brown, N., Sandholm, T. (2019). Superhuman AI for Multiplayer Poker (Pluribus). *Science*. Brown, N., Sandholm, T. (2019). Solving Imperfect-Information Games via Discounted Regret Minimization. *AAAI*. Kim, J. (2024). GPU-Accelerated Counterfactual Regret Minimization. *arXiv:2408.14778*. Schmid, M. et al. (2019). Variance Reduction in Monte Carlo CFR (VR-MCCFR). *AAAI*. Jackson, E. (2013). Slumbot NL. *AAAI Workshop on Computer Poker*. Southey, F. et al. (2005). Bayes' Bluff: Opponent Modelling in Poker. *UAI*. Ganzfried, S., Sandholm, T. (2012). Safe Opponent Exploitation. *ACM EC*. Li, X., Miikkulainen, R. (2018). Opponent Modeling and Exploitation via Evolved RNNs. *GECCO*.

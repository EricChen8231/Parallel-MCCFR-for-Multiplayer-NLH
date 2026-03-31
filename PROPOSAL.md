# EE/CSCI 451 Project Proposal
## Parallel Monte Carlo Counterfactual Regret Minimization for Abstracted No-Limit Texas Hold'em

**Team:** Yun Han, Eric Chen, Dave Rodriguez  
**Category:** Research, Implement, and Evaluate  
**Platform:** CUDA C++, cuRAND, optional MPI/NCCL, USC CARC GPUs

---

## Introduction

Counterfactual Regret Minimization (CFR) and Monte Carlo CFR (MCCFR) are standard methods for computing approximate equilibrium strategies in large imperfect-information games such as poker. No-Limit Texas Hold'em is a particularly challenging target because players have hidden cards, actions are sequential, the betting tree is large, and exact solving is infeasible even for heavily abstracted versions of the game. We therefore train on an abstracted poker game that compresses private hands, public boards, and bet sizes into a tractable strategy space while still preserving the key imperfect-information structure.

Accelerating MCCFR is non-trivial. Although each sampled traversal is largely independent, the workload is irregular: different trajectories reach different streets, touch different information sets, and terminate at different depths. This creates control-flow divergence on GPUs and makes memory access patterns difficult to optimize. In addition, training repeatedly updates large regret and average-strategy tables, so parallel speedup depends not only on raw compute throughput but also on how efficiently these shared statistics are stored, accumulated, and merged.

Our project is to design, implement, and evaluate a parallel MCCFR training system for abstracted no-limit poker, with a particular focus on a CUDA implementation that can train many sampled traversals concurrently on CARC GPUs. The main deliverables are:

- a correct MCCFR trainer for an abstracted poker environment in CUDA C++,
- optional multi-node scaling through coarse-grained MPI table merging,
- evaluation code for runtime, throughput, and poker-playing quality,
- CARC build and training scripts for reproducible experiments, and
- plots and analysis for speedup, efficiency, and learned-strategy quality.

Our hypotheses are:

- Batched GPU MCCFR will achieve much higher training throughput than a single-process CPU-style traversal loop by running many sampled traversals concurrently.
- Compact array-based strategy storage, structure-of-arrays layouts, and block-level accumulation will improve scalability by reducing memory overhead and synchronization pressure.
- CFR+ style regret clamping and linear averaging will reach stronger strategies than vanilla MCCFR under the same wall-clock budget.
- Optional MPI-based table merging will provide useful multi-node scaling because ranks can train independently for long intervals and communicate only through bulk reductions.

---

## Description of the Implementation

### Core System

The project uses external-sampling MCCFR on an abstracted no-limit Texas Hold'em game. Hole cards are mapped to preflop buckets, public boards are mapped to postflop strength buckets, and the betting abstraction uses a small set of legal no-limit actions such as fold, call/check, minimum raise, half-pot raise, pot raise, and all-in. This keeps the information-set space manageable while still allowing strategically meaningful betting.

The implementation platform is CUDA C++ with cuRAND for random sampling. A precomputed hand-evaluation table is loaded once and used for fast seven-card hand ranking during training and evaluation. Regret and average-strategy statistics are stored in compact flat arrays rather than pointer-heavy object graphs, making them suitable for GPU kernels and bulk synchronization.

### GPU Thread Management Techniques

The CUDA implementation applies several techniques from EE 451:

- **Thread-per-traversal assignment:** each CUDA thread simulates one sampled poker traversal within a batch.
- **Block-level batching:** threads are grouped into blocks so that many independent traversals run concurrently.
- **Structure-of-arrays storage:** regret tables and strategy sums are laid out contiguously by action to improve coalesced access.
- **Shared-memory accumulation:** per-block temporary updates are accumulated locally before being flushed to global memory.
- **Parallel reduction / warp-level primitives:** reductions over action values use warp-synchronous operations where possible to reduce overhead.
- **Occupancy-aware launch configuration:** batch size and block size are tuned to keep the GPU busy without excessive register pressure.
- **Warp-divergence mitigation:** state encoding and traversal structure are kept compact so that divergence costs are reduced as much as possible, even though MCCFR remains inherently irregular.

These techniques are important because MCCFR is not a regular dense linear-algebra workload. Performance depends on reducing the cost of irregular branching, scattered table access, and frequent updates to regret statistics.

### Optional MPI Usage

MPI is used only as an optional coarse-grained scaling mechanism. Each rank runs local GPU training independently on its own device and maintains its own regret and average-strategy tables. After a training segment, ranks export these tables and participate in `MPI_Allreduce` to sum the accumulators across ranks. This avoids fine-grained communication inside the traversal loop and matches the embarrassingly parallel structure of batched MCCFR. If NCCL is available, multi-GPU synchronization can also be tested on supported systems.

### Benchmark Selection

Our benchmark selection is driven by abstraction size and training budget rather than by many separate games. We will primarily evaluate:

- heads-up training, which is smaller and converges faster,
- six-player training, which is more realistic and stresses scalability,
- multiple batch sizes to study GPU throughput, and
- multiple wall-clock budgets to study strategy quality over time.

This selection is appropriate because the main research question is how well MCCFR scales on GPU hardware for an abstracted poker environment, not whether one poker variant is easier than another.

### Scope

The core deliverable is the CUDA/MPI training system and its evaluation. A pure CPU OpenMP baseline is not the primary focus of the current implementation and will be treated as a stretch goal rather than a required deliverable. This keeps the proposal aligned with the project as implemented while preserving room for additional comparison if time permits.

---

## Evaluation / Criteria of Success

We will evaluate both **system performance** and **strategy quality**.

### Performance Metrics

The performance evaluation will use:

- total wall-clock runtime,
- hands or traversals processed per second,
- speedup from increasing GPU batch size or number of MPI ranks, and
- parallel efficiency for multi-rank runs.

We will use fixed configurations on CARC so that comparisons are reproducible. Our code deliverables for this portion are the training binary, CARC submission scripts, and benchmark scripts for repeated runs.

### Strategy-Quality Metrics

The instructor feedback asked what metric we will use to evaluate poker quality relative to optimal. Our answer is:

- **Primary metric for the full abstracted game:** approximate exploitability proxies based on performance against fixed opponents and stability of the learned average strategy over training time.
- **Secondary metric:** BB/100 against several scripted opponent archetypes such as calling-station, nit, maniac, balanced, and random baselines.
- **If time permits:** exact or smaller-scale best-response evaluation on a reduced benchmark setting, or a Monte Carlo best-response style approximation, to connect the learned policy more directly to optimal play.

For the large abstracted no-limit game, exact exploitability is too expensive to compute directly. Therefore, the practical evaluation will focus on head-to-head results against controlled baselines, comparison between earlier and later checkpoints/strategy exports, and CFR vs CFR+ behavior under the same time budget. This still gives a meaningful measure of whether the parallel implementation preserves or improves strategic quality while accelerating training.

### Planned Experiments

We plan to report:

1. **Throughput vs batch size:** measure how GPU throughput changes with different batch sizes.
2. **Single-node vs multi-node scaling:** compare one-GPU and MPI-merged multi-GPU training runs.
3. **CFR vs CFR+:** compare strategy quality under the same wall-clock budget.
4. **Heads-up vs six-player training:** compare convergence behavior and practical play quality for different game sizes.

Success means:

- demonstrating clear acceleration from GPU batching and useful scaling from optional MPI merging,
- identifying the main bottlenecks in memory access, divergence, and synchronization,
- showing that longer or better-configured training produces stronger strategies in evaluation, and
- delivering reproducible code, scripts, and plots consistent with the implemented system.

---

## Deliverables

The final project deliverables will include:

- the CUDA C++ MCCFR trainer for abstracted poker,
- optional MPI/NCCL support for coarse-grained distributed merging,
- evaluation modes for scripted-opponent testing and human play,
- CARC build and training scripts,
- benchmark results and plots, and
- a final report discussing performance bottlenecks, scaling behavior, and strategy quality.

---

## References

Zinkevich, M. et al. (2007). *Regret Minimization in Games with Incomplete Information*. NeurIPS.  
Lanctot, M. et al. (2009). *Monte Carlo Sampling for Regret Minimization in Extensive Games*. NeurIPS.  
Tammelin, O. (2014). *Solving Large Imperfect Information Games Using CFR+*. arXiv.  
Brown, N., Sandholm, T. (2019). *Superhuman AI for Multiplayer Poker*. Science.  
Schmid, M. et al. (2019). *Variance Reduction in Monte Carlo Counterfactual Regret Minimization*. AAAI.

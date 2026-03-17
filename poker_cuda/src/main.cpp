// =============================================================================
// main.cpp — CLI entry point for CUDA-accelerated poker CFR trainer
//
// Modes:
//   train       — GPU MCCFR training, MPI AllReduce, save strategy
//   benchmark   — compare throughput at different batch sizes
//   eval        — play trained strategy vs scripted opponent, report BB/100
//   info        — print GPU device info (useful on CARC)
// =============================================================================
#include "cfr_gpu.cuh"
#include "hand_eval.h"
#include "abstraction.h"
#include "strategy.h"
#include "eval.h"
#include "bot.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

static void print_usage(const char* prog)
{
    printf(
        "Usage: %s [options]\n"
        "\n"
        "Training / benchmark flags:\n"
        "  --mode <train|benchmark|eval|info>  (default: train)\n"
        "  --players  <N>                      (default: 2)\n"
        "  --iters    <N>                      (default: 1000)\n"
        "  --batch    <N>                      (default: 65536)\n"
        "  --stack    <N>                      (default: 1000)\n"
        "  --sb       <N>                      (default: 10)\n"
        "  --bb       <N>                      (default: 20)\n"
        "  --save     <file>                   (default: strategy.bin)\n"
        "  --load     <file>                   load checkpoint before training\n"
        "  --handranks <file>                  (default: data/handranks.dat)\n"
        "  --no-cfrplus                        disable CFR+ clamping\n"
        "  --no-lcfr                           disable Linear CFR weighting\n"
        "\n"
        "Eval flags (--mode eval):\n"
        "  --strategy  <file>                  normalized strategy.bin to evaluate\n"
        "  --opponent  <type>                  calling_station|nit|maniac|balanced|random\n"
        "                                      (default: calling_station)\n"
        "  --hands     <N>                     number of hands to play (default: 10000)\n"
        "\n"
        "Play flags (--mode play):\n"
        "  --strategy  <file>                  strategy to use for the bot\n"
        "  --opponent  <type>                  scripted archetype (default: calling_station)\n"
        "  --hands     <N>                     total hands in session (default: 500)\n"
        "  --window    <N>                     rolling BB/100 window size (default: 50)\n",
        prog);
}

static void print_gpu_info()
{
    int n;
    cudaGetDeviceCount(&n);
    printf("GPUs found: %d\n", n);
    for (int i = 0; i < n; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("  [%d] %s  SM=%d  VRAM=%.0fGB  BW=%.0fGB/s  L2=%.0fMB\n",
               i, p.name, p.multiProcessorCount,
               p.totalGlobalMem / 1e9,
               2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1e6,
               p.l2CacheSize / 1e6);
    }
}

int main(int argc, char* argv[])
{
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#else
    int mpi_rank = 0, mpi_size = 1;
#endif

    // --- Parse arguments ---
    std::string mode           = "train";
    int         n_players      = 2;
    long long   iters          = 1000;
    int         batch_size     = 65536;
    int         stack_size     = 1000;
    int         sb             = 10;
    int         bb             = 20;
    bool        use_cfr_plus   = true;
    bool        use_linear_cfr = true;
    std::string save_path      = "strategy.bin";
    std::string load_path;
    std::string hr_path        = "data/handranks.dat";
    // eval / play mode flags
    std::string strategy_path;
    std::string opponent_str   = "calling_station";
    long long   eval_hands     = 10000;
    int         window_size    = 50;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--mode")       && i+1<argc) mode          = argv[++i];
        else if (!strcmp(argv[i], "--players")    && i+1<argc) n_players     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")      && i+1<argc) iters         = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--batch")      && i+1<argc) batch_size    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stack")      && i+1<argc) stack_size    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sb")         && i+1<argc) sb            = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bb")         && i+1<argc) bb            = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save")       && i+1<argc) save_path     = argv[++i];
        else if (!strcmp(argv[i], "--load")       && i+1<argc) load_path     = argv[++i];
        else if (!strcmp(argv[i], "--handranks")  && i+1<argc) hr_path       = argv[++i];
        else if (!strcmp(argv[i], "--strategy")   && i+1<argc) strategy_path = argv[++i];
        else if (!strcmp(argv[i], "--opponent")   && i+1<argc) opponent_str  = argv[++i];
        else if (!strcmp(argv[i], "--hands")      && i+1<argc) eval_hands    = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--window")     && i+1<argc) window_size   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-cfrplus"))  use_cfr_plus   = false;
        else if (!strcmp(argv[i], "--no-lcfr"))     use_linear_cfr = false;
        else if (!strcmp(argv[i], "--help"))      { print_usage(argv[0]); return 0; }
    }

    // --- GPU info mode (no GPU required for display) ---
    if (mode == "info") {
        print_gpu_info();
        return 0;
    }

    // =========================================================================
    // EVAL mode: CPU-only, no GPU needed
    // =========================================================================
    if (mode == "eval") {
        // Load hand evaluator (needed for showdown evaluation)
        abstraction_init();
        if (!hand_eval_init(hr_path.c_str())) {
            fprintf(stderr, "ERROR: Could not load handranks.dat from %s\n"
                    "  Download: https://github.com/tangentforks/TwoPlusTwoHandEvaluator\n",
                    hr_path.c_str());
            return 1;
        }

        std::string strat_file = strategy_path.empty() ? save_path : strategy_path;
        printf("Eval: loading strategy from %s ...\n", strat_file.c_str());
        HostStrategyTable strat;
        if (!strategy_load(strat, strat_file)) {
            fprintf(stderr, "ERROR: Could not load strategy from %s\n", strat_file.c_str());
            return 1;
        }
        printf("  Loaded %zu info sets.\n", strat.size());

        // Map opponent string → enum
        OpponentType opp = OpponentType::CALLING_STATION;
        if      (opponent_str == "nit")             opp = OpponentType::NIT;
        else if (opponent_str == "maniac")          opp = OpponentType::MANIAC;
        else if (opponent_str == "balanced")        opp = OpponentType::BALANCED;
        else if (opponent_str == "random")          opp = OpponentType::RANDOM;

        printf("Opponent: %s  |  Hands: %lld\n\n", opponent_str.c_str(), eval_hands);
        EvalResult res = evaluate_strategy(strat, opp, eval_hands,
                                           stack_size, sb, bb, /*seed=*/42);

        printf("=== Eval Results ===\n");
        printf("  Hands played : %lld\n", res.hands_played);
        printf("  Net BB       : %.1f\n", res.net_bb);
        printf("  BB/100       : %+.2f\n", res.bb_per_100);

#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 0;
    }

    // =========================================================================
    // PLAY mode: CPU-only adaptive bot session (Experiment 3)
    //
    // Plays n_hands against a scripted archetype, adapting through Bayesian
    // opponent modeling.  Reports rolling BB/100 in window_size-hand windows
    // to show the GTO→exploit transition over the session.
    // =========================================================================
    if (mode == "play") {
        abstraction_init();
        if (!hand_eval_init(hr_path.c_str())) {
            fprintf(stderr, "ERROR: Could not load handranks.dat from %s\n", hr_path.c_str());
            return 1;
        }

        std::string strat_file = strategy_path.empty() ? save_path : strategy_path;
        printf("Play: loading strategy from %s ...\n", strat_file.c_str());
        HostStrategyTable strat;
        if (!strategy_load(strat, strat_file)) {
            fprintf(stderr, "ERROR: Could not load strategy from %s\n", strat_file.c_str());
            return 1;
        }
        printf("  Loaded %zu info sets.\n", strat.size());

        OpponentType opp = OpponentType::CALLING_STATION;
        if      (opponent_str == "nit")      opp = OpponentType::NIT;
        else if (opponent_str == "maniac")   opp = OpponentType::MANIAC;
        else if (opponent_str == "balanced") opp = OpponentType::BALANCED;
        else if (opponent_str == "random")   opp = OpponentType::RANDOM;

        printf("Opponent: %s  |  Hands: %lld  |  Window: %d\n\n",
               opponent_str.c_str(), eval_hands, window_size);
        printf("Bayesian model: EXPLOIT_WEIGHT=%.2f  k=%d\n",
               EXPLOIT_WEIGHT, OPP_MODEL_K);
        printf("  (exploit weight = %.2f * n / (n + %d))\n\n",
               EXPLOIT_WEIGHT, OPP_MODEL_K);

        LiveBot bot(strat, stack_size, sb, bb);
        auto windows = bot.play_session(opp, eval_hands, window_size, /*seed=*/42);
        bot.print_summary(windows);

        const auto& obs = bot.observations();
        printf("\n=== Final Opponent Model ===\n");
        printf("  Observations  : %lld decisions\n",   obs.total());
        printf("  Fold rate     : %.1f%%  (reference: 28%%)\n", obs.fold_rate()  * 100.f);
        printf("  Call rate     : %.1f%%  (reference: 44%%)\n", obs.call_rate()  * 100.f);
        printf("  Raise rate    : %.1f%%  (reference: 28%%)\n", obs.raise_rate() * 100.f);

#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 0;
    }

    // =========================================================================
    // TRAIN / BENCHMARK modes: GPU required
    // =========================================================================
    if (mpi_rank == 0) {
        printf("=== Poker CUDA CFR Trainer ===\n");
        print_gpu_info();
    }

    // Each MPI rank trains on its own GPU (rank i → device i % num_devices)
#ifdef USE_MPI
    {
        int n_dev = 0;
        cudaGetDeviceCount(&n_dev);
        if (n_dev > 0) cudaSetDevice(mpi_rank % n_dev);
    }
#endif

    GPUCFRTrainer trainer(n_players, stack_size, sb, bb,
                          use_cfr_plus, use_linear_cfr);

    if (!trainer.load_hand_table(hr_path.c_str())) {
        if (mpi_rank == 0)
            fprintf(stderr,
                "WARNING: Could not load %s\n"
                "  Download: https://github.com/tangentforks/TwoPlusTwoHandEvaluator\n"
                "  Continuing without exact hand evaluation (postflop approximate)\n",
                hr_path.c_str());
    }

    if (!load_path.empty()) {
        if (mpi_rank == 0) printf("Loading checkpoint from %s ...\n", load_path.c_str());
        if (!trainer.load_checkpoint(load_path))
            fprintf(stderr, "[rank %d] WARNING: Could not load checkpoint %s\n",
                    mpi_rank, load_path.c_str());
    }

    // -------------------------------------------------------------------------
    if (mode == "train") {
        if (mpi_rank == 0)
            printf("\nTraining: players=%d  iters=%lld  batch=%d  ranks=%d\n",
                   n_players, iters, batch_size, mpi_size);

        std::string ckpt_path = (mpi_rank == 0 && !save_path.empty()) ? save_path + ".ckpt" : "";
        trainer.train(iters, batch_size, mpi_rank == 0, ckpt_path, 10000);

        // -----------------------------------------------------------------------
        // MPI AllReduce: merge regret tables from all ranks into a single global
        // strategy.  Each rank trained independently on its GPU; their regret /
        // strategy_sum tables are summed element-wise across all ranks so the
        // final strategy reflects the combined training budget.
        // -----------------------------------------------------------------------
#ifdef USE_MPI
        if (mpi_size > 1) {
            if (mpi_rank == 0) printf("\nMPI AllReduce: merging %d ranks...\n", mpi_size);

            std::vector<float> reg, ssum;
            trainer.export_tables(reg, ssum);   // device → host

            MPI_Allreduce(MPI_IN_PLACE, reg.data(),  (int)reg.size(),
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, ssum.data(), (int)ssum.size(),
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            trainer.import_tables(reg, ssum);   // host → device

            if (mpi_rank == 0) printf("AllReduce complete.\n");
        }
#endif

        if (mpi_rank == 0) {
            printf("\nInfo sets active: %d\n", trainer.num_info_sets_active());

            // Save checkpoint (raw regrets + strategy_sum for resuming)
            std::string ckpt = save_path + ".ckpt";
            if (trainer.save_checkpoint(ckpt))
                printf("Checkpoint saved to %s\n", ckpt.c_str());

            // Save normalized average strategy (Nash approximation)
            auto strat = trainer.get_strategy();
            if (strategy_save(strat, save_path))
                printf("Strategy saved to %s  (%zu info sets)\n",
                       save_path.c_str(), strat.size());
            else
                fprintf(stderr, "ERROR: could not save strategy to %s\n", save_path.c_str());
        }
    }
    // -------------------------------------------------------------------------
    else if (mode == "benchmark") {
        if (mpi_rank == 0) printf("\nBenchmark: batch sizes vs throughput\n");
        for (int bs : {4096, 16384, 65536, 262144}) {
            GPUCFRTrainer t2(n_players, stack_size, sb, bb, use_cfr_plus, use_linear_cfr);
            t2.load_hand_table(hr_path.c_str());
            auto t0 = std::chrono::high_resolution_clock::now();
            t2.train(10, bs, false);
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            long long hands = 10LL * bs * n_players;
            if (mpi_rank == 0)
                printf("  batch=%6d  hands=%.0fM  time=%.2fs  speed=%.1fM/s\n",
                       bs, hands / 1e6, sec, hands / sec / 1e6);
        }
    }
    else {
        if (mpi_rank == 0) {
            fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
            print_usage(argv[0]);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}

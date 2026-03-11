// =============================================================================
// main.cpp — CLI entry point for CUDA-accelerated poker CFR trainer
//
// Modes:
//   train       — GPU MCCFR training, save strategy
//   benchmark   — compare throughput at different batch sizes
//   info        — print GPU device info (useful on CARC)
// =============================================================================
#include "cfr_gpu.cuh"
#include "hand_eval.h"
#include "abstraction.h"
#include "strategy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cuda_runtime.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

static void print_usage(const char* prog)
{
    printf(
        "Usage: %s [options]\n"
        "  --mode <train|benchmark|info>  (default: train)\n"
        "  --players  <N>                 (default: 2)\n"
        "  --iters    <N>                 (default: 1000)\n"
        "  --batch    <N>                 (default: 65536)\n"
        "  --stack    <N>                 (default: 1000)\n"
        "  --sb       <N>                 (default: 10)\n"
        "  --bb       <N>                 (default: 20)\n"
        "  --save     <file>              (default: strategy.bin)\n"
        "  --load     <file>\n"
        "  --handranks <file>             (default: data/handranks.dat)\n"
        "  --no-cfrplus                   disable CFR+ clamping\n"
        "  --no-lcfr                      disable Linear CFR weighting\n",
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
#endif

    // --- Parse arguments ---
    std::string mode        = "train";
    int   n_players         = 2;
    long long iters         = 1000;
    int   batch_size        = 65536;
    int   stack_size        = 1000;
    int   sb                = 10;
    int   bb                = 20;
    bool  use_cfr_plus      = true;
    bool  use_linear_cfr    = true;
    std::string save_path   = "strategy.bin";
    std::string load_path;
    std::string hr_path     = "data/handranks.dat";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--mode")       && i+1<argc) mode       = argv[++i];
        else if (!strcmp(argv[i], "--players")    && i+1<argc) n_players  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")      && i+1<argc) iters      = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--batch")      && i+1<argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stack")      && i+1<argc) stack_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sb")         && i+1<argc) sb         = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bb")         && i+1<argc) bb         = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save")       && i+1<argc) save_path  = argv[++i];
        else if (!strcmp(argv[i], "--load")       && i+1<argc) load_path  = argv[++i];
        else if (!strcmp(argv[i], "--handranks")  && i+1<argc) hr_path    = argv[++i];
        else if (!strcmp(argv[i], "--no-cfrplus"))  use_cfr_plus   = false;
        else if (!strcmp(argv[i], "--no-lcfr"))     use_linear_cfr = false;
        else if (!strcmp(argv[i], "--help"))      { print_usage(argv[0]); return 0; }
    }

    // --- GPU info mode ---
    if (mode == "info") {
        print_gpu_info();
        return 0;
    }

    printf("=== Poker CUDA CFR Trainer ===\n");
    print_gpu_info();

    // --- Initialize trainer ---
    GPUCFRTrainer trainer(n_players, stack_size, sb, bb,
                          use_cfr_plus, use_linear_cfr);

    if (!trainer.load_hand_table(hr_path.c_str())) {
        fprintf(stderr,
            "WARNING: Could not load %s\n"
            "  Run: ./gen_table data/handranks.dat\n"
            "  Continuing without exact hand evaluation (postflop will be approximate)\n",
            hr_path.c_str());
    }

    // --- Load checkpoint if specified ---
    if (!load_path.empty()) {
        printf("Loading checkpoint from %s ...\n", load_path.c_str());
        if (!trainer.load_checkpoint(load_path))
            fprintf(stderr, "WARNING: Could not load checkpoint %s\n", load_path.c_str());
    }

    // -------------------------------------------------------------------------
    if (mode == "train") {
        printf("\nTraining: players=%d  iters=%lld  batch=%d\n",
               n_players, iters, batch_size);

        trainer.train(iters, batch_size, true);

        printf("\nInfo sets active: %d\n", trainer.num_info_sets_active());

        // Save checkpoint (raw regrets + strategy_sum for resuming)
        std::string ckpt = save_path + ".ckpt";
        if (trainer.save_checkpoint(ckpt))
            printf("Checkpoint saved to %s\n", ckpt.c_str());

        // Save normalized average strategy
        auto strat = trainer.get_strategy();
        if (strategy_save(strat, save_path))
            printf("Strategy saved to %s  (%zu info sets)\n",
                   save_path.c_str(), strat.size());
        else
            fprintf(stderr, "ERROR: could not save strategy to %s\n", save_path.c_str());
    }
    // -------------------------------------------------------------------------
    else if (mode == "benchmark") {
        printf("\nBenchmark: batch sizes vs throughput\n");
        for (int bs : {4096, 16384, 65536, 262144}) {
            GPUCFRTrainer t2(n_players, stack_size, sb, bb, use_cfr_plus, use_linear_cfr);
            t2.load_hand_table(hr_path.c_str());
            auto t0 = std::chrono::high_resolution_clock::now();
            t2.train(10, bs, false);
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            long long hands = 10LL * bs * n_players;
            printf("  batch=%6d  hands=%.0fM  time=%.2fs  speed=%.1fM/s\n",
                   bs, hands / 1e6, sec, hands / sec / 1e6);
        }
    }
    else {
        fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
        print_usage(argv[0]);
        return 1;
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}

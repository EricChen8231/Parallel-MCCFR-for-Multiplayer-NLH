#include "strategy.h"
#include <fstream>
#include <cstdio>

// Binary format:
//   [uint64_t magic] [uint32_t num_entries]
//   For each: [uint32_t key] [float probs[8]]

static constexpr uint64_t STRATEGY_MAGIC   = 0x53545241540000ULL;
static constexpr uint64_t CHECKPOINT_MAGIC = 0x43465247505500ULL;

static bool peek_magic(const std::string& path, uint64_t& magic)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read((char*)&magic, sizeof(magic));
    return (bool)f;
}

bool strategy_save(const HostStrategyTable& strat, const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t magic = STRATEGY_MAGIC;
    uint32_t n = (uint32_t)strat.size();
    f.write((char*)&magic, sizeof(magic));
    f.write((char*)&n,     sizeof(n));
    for (auto& [k, v] : strat) {
        f.write((char*)&k,        sizeof(k));
        f.write((char*)v.probs,   GPU_NUM_ACTIONS * sizeof(float));
    }
    return (bool)f;
}

bool strategy_load(HostStrategyTable& strat, const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t magic; uint32_t n;
    f.read((char*)&magic, sizeof(magic));
    f.read((char*)&n,     sizeof(n));
    if (!f) return false;
    if (magic == CHECKPOINT_MAGIC) {
        fprintf(stderr,
                "ERROR: %s is a training checkpoint, not a playable strategy file.\n"
                "  Use the exported .bin strategy, not the .ckpt resume file.\n",
                path.c_str());
        return false;
    }
    if (magic != STRATEGY_MAGIC) {
        fprintf(stderr,
                "ERROR: %s does not have a valid strategy file header.\n",
                path.c_str());
        return false;
    }
    strat.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t k;
        StrategyEntry e;
        f.read((char*)&k,      sizeof(k));
        f.read((char*)e.probs, GPU_NUM_ACTIONS * sizeof(float));
        strat.emplace(k, e);
    }
    return (bool)f;
}

bool strategy_load_auto(HostStrategyTable& strat, const std::string& path,
                        int n_players, int stack, int sb, int bb)
{
    uint64_t magic = 0;
    if (!peek_magic(path, magic)) return false;

    if (magic == STRATEGY_MAGIC)
        return strategy_load(strat, path);

    if (magic != CHECKPOINT_MAGIC) {
        fprintf(stderr,
                "ERROR: %s is neither a strategy .bin nor a checkpoint .ckpt file.\n",
                path.c_str());
        return false;
    }

    fprintf(stderr,
            "NOTE: %s is a checkpoint; converting it to a playable strategy in memory...\n",
            path.c_str());

    GPUCFRTrainer trainer(n_players, stack, sb, bb);
    if (!trainer.load_checkpoint(path)) {
        fprintf(stderr, "ERROR: Could not load checkpoint from %s\n", path.c_str());
        return false;
    }

    strat = trainer.get_strategy();

    std::string converted_path = path;
    if (converted_path.size() >= 5 &&
        converted_path.compare(converted_path.size() - 5, 5, ".ckpt") == 0) {
        converted_path.erase(converted_path.size() - 5);
    } else {
        converted_path += ".bin";
    }

    if (strategy_save(strat, converted_path)) {
        fprintf(stderr, "NOTE: Exported playable strategy to %s\n", converted_path.c_str());
    } else {
        fprintf(stderr,
                "WARNING: Converted checkpoint in memory, but could not save %s\n",
                converted_path.c_str());
    }
    return true;
}

void strategy_print_top(const HostStrategyTable& strat, uint32_t key, int n)
{
    static const char* ACTION_NAMES[] = {
        "fold", "check", "call",
        "raise_quarter", "raise_half", "raise_third", "raise_pot", "all_in"
    };
    auto it = strat.find(key);
    if (it == strat.end()) { printf("(info set not found)\n"); return; }
    const auto& e = it->second;

    // Sort by probability
    int order[GPU_NUM_ACTIONS];
    for (int i = 0; i < GPU_NUM_ACTIONS; i++) order[i] = i;
    for (int i = 0; i < GPU_NUM_ACTIONS - 1; i++)
        for (int j = i+1; j < GPU_NUM_ACTIONS; j++)
            if (e.probs[order[j]] > e.probs[order[i]]) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
    for (int i = 0; i < n && i < GPU_NUM_ACTIONS; i++) {
        int a = order[i];
        if (e.probs[a] > 0.001f)
            printf("  %-16s %.1f%%\n", ACTION_NAMES[a], e.probs[a] * 100.f);
    }
}

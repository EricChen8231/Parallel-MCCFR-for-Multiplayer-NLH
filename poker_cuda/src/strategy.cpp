#include "strategy.h"
#include <fstream>
#include <cstdio>

// Binary format:
//   [uint64_t magic] [uint32_t num_entries]
//   For each: [uint32_t key] [float probs[8]]

bool strategy_save(const HostStrategyTable& strat, const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t magic = 0x53545241540000ULL;
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

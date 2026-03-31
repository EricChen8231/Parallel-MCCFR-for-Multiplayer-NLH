#pragma once
#include "cfr_gpu.cuh"
#include <string>

bool strategy_save(const HostStrategyTable& strat, const std::string& path);
bool strategy_load(HostStrategyTable& strat, const std::string& path);
bool strategy_load_auto(HostStrategyTable& strat, const std::string& path,
                        int n_players, int stack, int sb, int bb);
void strategy_print_top(const HostStrategyTable& strat, uint32_t key, int n = 3);

#pragma once
#include "cfr_gpu.cuh"
#include <string>

bool strategy_save(const HostStrategyTable& strat, const std::string& path);
bool strategy_load(HostStrategyTable& strat, const std::string& path);
void strategy_print_top(const HostStrategyTable& strat, uint32_t key, int n = 3);

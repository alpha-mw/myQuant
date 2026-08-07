# Silent Exception Inventory

Generated for Phase 8 on 2026-07-06 after routing
`quant_investor/market/download_cn.py` silent exception swallows into
data-quality diagnostics.

Command:

```bash
rg -n -U "except Exception[^\n]*:\n\s+(pass|continue)" quant_investor web scripts daily_runner.py tests -g '*.py'
```

Out-of-scope silent catch sites left unchanged in Phase 8:

- `quant_investor/enhanced_data_layer.py`: 148, 212, 221, 236, 254
- `quant_investor/sec_fundamental.py`: 85, 102, 153, 169
- `quant_investor/fundamental_branch.py`: 515
- `quant_investor/agents/orchestrator.py`: 355
- `quant_investor/agents/master_agent.py`: 135, 146
- `quant_investor/agents/fundamental_agent.py`: 30
- `quant_investor/global_context/builder.py`: 61
- `quant_investor/bayesian/calibration.py`: 106
- `quant_investor/monitoring/cn_aggressive_market_metrics.py`: 87, 94
- `quant_investor/monitoring/cn_aggressive_portfolio_tracker.py`: 216, 223
- `quant_investor/market/branch_readiness.py`: 135, 320
- `quant_investor/market/fundamental_mart.py`: 198
- `quant_investor/market/market_data_reader.py`: 256
- `quant_investor/market/download.py`: 482, 484
- `quant_investor/market/name_map.py`: 53
- `scripts/factor_health_automation.py`: 467, 739

Notes:

- `quant_investor/market/download_cn.py` had no remaining
  `except Exception: pass/continue` matches after the Phase 8 change.
- These entries are inventory only. No behavior outside `download_cn.py` was
  changed in Phase 8.

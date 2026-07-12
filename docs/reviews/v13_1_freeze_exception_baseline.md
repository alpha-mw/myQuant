# v13.1 Freeze-Exception Baseline

Captured before implementation on 2026-07-12 (Asia/Shanghai).

## Isolated implementation baseline

- Source commit: `463140137ee1ddb2cecacc6aeb6edb226a6d67ef`
- Implementation branch: `codex/myquant-governance-dashboard-v2`
- Original worktree branch: `codex/weekly-factor-auto-promotion`

## Original dirty worktree fingerprints

- Tracked diff SHA-256: `867401872e4c815c792d256a3e47374076890cd39a8031a3bff4363b647b3813`
- Current registry SHA-256: `4c8680a9cccf08c188c072564312b17eabd61cb82487d89b2338fe488ad05806`

The original worktree remains review input only. None of its uncommitted changes
are transplanted wholesale into this branch.

Dirty paths at capture time:

- `.agent/CONTINUITY.md`
- `docs/factor_governance.md`
- `quant_investor/factor_registry/mined_factors.json`
- `quant_investor/factors/pit_fundamentals.py`
- `quant_investor/market/branch_readiness.py`
- `quant_investor/market/fundamental_mart.py`
- `scripts/daily_factor_mining_automation.py`
- `scripts/mine_quant_branch_factors.py`
- `scripts/retest_aquant_alpha_mix_8gate.py`
- `tests/unit/test_branch_data_readiness.py`
- `tests/unit/test_factor_governance.py`
- `tests/unit/test_fundamental_mart.py`
- `tests/unit/test_quant_branch_factor_mining.py`
- `quant_investor/market/fundamental_generation.py` (untracked)

## Freeze-exception boundary

This branch may change Theme, factor-governance, and local dashboard contracts
only. It must not add broker execution, network-dependent verification, Web
workspace maintenance, or silent legacy fallbacks. Merge to the frozen v13
mainline remains subject to Maxwell's explicit approval after the required
offline acceptance gates pass.

# v15 No-Theme Architecture Review

Status: approved for isolated implementation with the constraints below.

- Keep the canonical branch set `quant/fundamental/macro` and the likelihood
  set `quant/fundamental`.
- Keep `RiskGuard -> ICCoordinator -> PortfolioConstructor` authoritative.
  FactorGovernanceProtocol v3 canonical replay binds the complete six-stage
  runtime context rather than a Quant-only approximation.
- Produce one hash-bound `v15_run_readiness` object.  Formal reports and the
  Dashboard consume it and must not independently recompute authorization.
- Preserve the implemented Fundamental v3 two-stage promotion contract.
- Preserve `market macro-refresh` as the sole live-explicit authoritative Macro
  publisher and keep its lock, dual CAS, journal, readback and recovery rules.
- Switch package/report/Factor/Dashboard contracts together.  A v14 or v2
  artifact may remain historical evidence but cannot be relabelled as v15.
- Delete active Theme producers and consumers only after their dependencies
  have been replaced.  Historical prose and negative tests may mention Theme;
  current imports, configuration, schemas, schedules and artifacts may not.
- Do not mutate the Factor registry, data pointers, private strategy records or
  contained ledger hashes during implementation or provider-hard-off checks.

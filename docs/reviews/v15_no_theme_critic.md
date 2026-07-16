# v15 No-Theme Critic Review

Status: approved after the following corrections were incorporated.

- Fundamental v3 is already on clean `main`; it needs v15 adaptation and
  regression coverage, not migration of a dirty patch.
- Keep `market macro-refresh`; `macro-maintain` remains observer/staging only
  and no `macro-promote` command is introduced.
- PortfolioConstructor positive weights are a valid subset of the eligible
  decision domain.  Risk and IC need explicit outcomes for every eligible
  symbol; the Portfolio may omit symbols as zero-weight outcomes.
- Theme zero-surface validation uses an explicit active denylist and historical
  allowlist rather than an impossible whole-repository word ban.
- Dashboard v3 switches schema, exporter, checker, UI, samples and tests in one
  change and rejects v2 current inputs.
- The first provider-hard-off v15 run is successful when the contract is v15,
  Theme is absent and new risk remains blocked.  It need not activate Factor or
  Macro production controls.
- Theme private evidence purge is intentionally outside this implementation
  run.  It is irreversible and requires a separate authorization after the
  zero-consumer gate and protected-tree hash comparison pass.

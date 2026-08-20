# CN Same-Day Post-Close Maintenance

`quant-investor market daily-maintain` is the registered coordinator for
same-day CN data maintenance after the Shanghai close. It orchestrates
component-owned capture, validation, publication, and recovery; it does not
replace the canonical Market, PIT, Fundamental, Macro, or release-calendar
stores.

## Schedule and slots

The Codex automation runs in `Asia/Shanghai` on open-session weekdays at
16:20, 17:20, 18:20, and 20:20. `--attempt-slot auto` maps delayed starts to:

```text
[16:20, 17:20) -> 1620
[17:20, 18:20) -> 1720
[18:20, 20:20) -> 1820
[20:20, 24:00) -> 2020
otherwise       -> OUTSIDE_SLOT
```

Early provider-not-ready results are `RETRY_PENDING` and exit successfully so
only final failures notify. An unresolved final attempt becomes
`SAME_DAY_SLA_MISSED` and exits nonzero. There is no 01:00 fallback.

## Authority and data flow

The coordinator obtains a fresh exact `trade_cal(exchange="SSE")` response
through `OfficialTushareHttpsClient`. It seals the raw response, request
projection, ordered fields/items/counts, request-ID hash, local observation
time, selected closed session, and attempt slot. Cache and local-date fallback
never authorize publication.

The component DAG is:

```text
PIT -> Market -> 100-session History
                         |-> Fundamental integrity readback
                         `-> Macro/release transaction
```

- PIT acquires the three `stock_basic` partitions once, then validates and
  publishes from the same sealed capture. The immutable generation and
  discovery pointer are canonical; the compatibility JSON is derived after
  CAS and has no strict-reader authority.
- Market seals `daily`, `daily_basic`, and `adj_factor` before publication.
  Exact keyset equality, duplicate rejection, target-date equality, disjoint
  classification, zero true missing, scope/PIT binding, and expected Market
  pointer SHA are pre-CAS gates. A missed-day catch-up is the exact ordered
  `(parent,target]` open-session window and is capped at five sessions. If the
  Market date already equals the target but its PIT binding is older, the
  coordinator performs an explicit full same-target recapture and publishes a
  new immutable snapshot without advancing the date.
- History is a fresh 100-open-session recomputation. Shadow mode binds it to a
  complete private Market/PIT candidate; it never falls back to canonical.
- Fundamental is health/readback-only in v1. Its age follows
  `ADVISORY_NO_FIXED_MAXIMUM`; the separately governed weekly safe-successor
  workflow is not inferred or invoked by this coordinator.
- Macro/release first prepares two private immutable candidates. Commit uses a
  durable journal and the fixed Market-to-PIT-to-release-to-observations lock
  order, revalidating the sealed Market and PIT pointer bytes throughout the
  transition. Automatic recovery may only complete a deterministic forward
  transition. Rollback is operator-only and requires all four old/new pointer
  SHAs.

## Commands

Shadow or execute one scheduled attempt:

```bash
./.venv/bin/quant-investor market daily-maintain \
  --market CN \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --run-root /Users/maxwell/mySpace/myQuant/data/private/cn_daily_maintenance \
  --mode shadow \
  --attempt-slot auto
```

An execute-mode deterministic safety blocker creates an owner-only write veto.
The automation cannot clear it. An operator may archive and clear one exact
veto with:

```bash
./.venv/bin/quant-investor market clear-write-veto \
  --market CN \
  --run-root /Users/maxwell/mySpace/myQuant/data/private/cn_daily_maintenance \
  --expected-veto-sha256 <exact-sha256> \
  --reason <bounded-operator-reason>
```

Use `--lane macro` for `MACRO_WRITE_VETO.json`; the default `--lane global`
retains legacy/global `WRITE_VETO.json` behavior. A Macro veto is never removed
manually and does not authorize bypassing an unresolved Macro journal.

Macro transaction modes are mutually exclusive:

```text
--prepare-transaction
--commit-prepared
--recover
--recover --execute-forward
--recover --execute-rollback <four exact old/new pointer SHAs>
```

Read-only recovery is the default. Daily automation can use forward completion
only; it never calls rollback or the legacy sequential `--commit` route.

## Machine states

The attempt receipt reports `same_day_status`,
`fundamental_integrity_status`, `fundamental_refresh_status=HEALTH_ONLY`,
`maintenance_status`, and the existing registered research-readiness result
or `UNCONFIRMED`. The coordinator never derives a new meaning for
`usable_for_investment_research`.

The overall same-day SLA still includes the target-aligned Macro/release pair.
Factor inputs are a separate dependency lane: exact PIT, Market and 100-session
History may be ready while the overall run remains `PARTIAL` because Macro is
blocked. Receipts expose `factor_input_readiness`,
`factor_input_shadow_readiness`, `core_blockers`, `macro_status`, and
`macro_blockers`. LOW/W80 never consume Macro or Fundamental merely because the
coordinator reports those lanes.

## Shadow and cutover

Stage A changes the existing automation to the four post-close slots in
`shadow` mode. A complete open-session day must produce all four attempt
receipts, leave every protected canonical byte and inventory unchanged, and
produce at least one private candidate `SHADOW_COMPLETE`.

Stage B changes only the automation mode to `execute`. It is prohibited until
Stage A evidence has been independently reviewed. Core PIT/Market/History
failures retain the global/core write veto. A disjoint Macro-only failure keeps
the overall run partial and uses its own Macro veto/journal without
retroactively invalidating an exact Factor-input closure. Existing legacy
global vetoes keep their global meaning until explicitly cleared by exact SHA.

No attempt or journal evidence is deleted automatically in v1. Every capture
and promotion performs resource preflight and fails closed unless required
space plus a 25% margin is available.

This workflow never authorizes Factor/Mainline activation, Dashboard, Paper,
holdings, broker, order, trade, or funds-transfer changes.

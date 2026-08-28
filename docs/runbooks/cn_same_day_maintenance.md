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

All four live slots use the same release-owned launcher. The launcher verifies
the installed import origin, then reads only `TUSHARE_TOKEN` from the
workspace-root `.env` without sourcing or executing that file. The `.env` must
be an owner-owned regular file with one link, mode `0600`, valid UTF-8, and
exactly one valid `TUSHARE_TOKEN` key. The launcher records a non-secret access
receipt, injects the token only into the maintenance child, and unsets it on
every exit path. It never calls macOS Keychain:

```bash
scripts/operations/run_cn_daily_slot.sh \
  --python <exact-installed-python> \
  --expected-import-root <exact-install-root> \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --run-root /Users/maxwell/mySpace/myQuant/data/private/cn_daily_maintenance \
  --attempt-slot <1620|1720|1820|2020>
```

The credential receipt contains only source/env-file/env-key, slot, time,
`READY|BLOCKED`, and explicit `token_material_recorded=false` /
`token_hash_recorded=false`. A missing, unsafe, ambiguous, or invalid `.env`
token stops in the launcher before `daily-maintain`, so it cannot create a new
coordinator veto.

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

One global veto class has an automatic governed recovery path: exact
`TUSHARE_TOKEN_MISSING` from a zero-write attempt. Recovery additionally
requires a fresh READY credential receipt, unchanged Market/PIT/Factor/Store
pointer preimages, the original attempt's `canonical_unchanged=true` and empty
stage set, the exact veto SHA, and a nonempty token only in process memory:

```bash
quant-investor market recover-transient-write-veto \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --run-root /Users/maxwell/mySpace/myQuant/data/private/cn_daily_maintenance \
  --expected-veto-sha256 <exact-veto-sha> \
  --credential-preflight-receipt <exact-absolute-path> \
  --expected-credential-preflight-sha256 <exact-sha>
```

The veto is archived and an immutable recovery receipt is written. Replay is
`NO_ACTION`. Pointer drift, partial writes, storage/security/schema/lineage
failures, non-credential blockers, unsafe evidence, or receipt mismatch remain
blocked. Macro vetoes are never eligible for this automatic path.

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

After the 20:20 Factor/Store branches finish, the automation may call the exact
`research morning-cutover` evaluator. Macro, Fundamental, Theme exposure,
Paper, I6 and benchmark-relative gaps are auxiliary; they do not reverse a
closed Market/PIT/History/Calendar/Factor/Store core. The evaluator writes one
date-bound decision receipt and returns only a scheduler action. The Codex
automation performs and reads back any scheduler change; repository code does
not write `~/.codex`.

Once a real 09:45 run exists for the same session, the 20:20 workflow may also
call `research morning-evaluate` after the strict Market close is complete. The
exact request binds the immutable `0945-run.v1.json`, the associated Sina quote
capture, and the expected Market pointer SHA. `PREFLIGHT` computes deterministic
09:45-to-close outcomes without writing. After Codex writes the bounded
`eod-evaluation.md` with the fixed research-only authority declarations, `SEAL`
publishes the immutable `eod-evaluation.v1.json`. The receipt reports
`operational_success` independently from `decision_quality`; missing benchmark
or individual close observations are auxiliary and cannot rewrite a successful
09:45 operational result. The evaluator never creates Paper fills, orders,
portfolio state, or holdings changes.

This workflow never authorizes Mainline activation, portfolio mutation,
holdings mutation, broker, order, trade, or funds-transfer changes.

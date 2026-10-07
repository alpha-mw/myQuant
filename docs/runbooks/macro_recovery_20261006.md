# Macro catch-up recovery (packet 2026-10-06, executed 2026-10-07)

The macro lane has been blocked since 2026-09-22: `MACRO_WRITE_VETO.json` is on
disk, and the observations catch-up cannot satisfy
`production_observation_readiness_not_pass` because the canonical release plan
only carried notices through June (fixed by the published release extension
`release-macro-recovery-20261006-v1`, pointer `5fdf4256…`).

## Why the original plan needed correcting

Every later step below was proven in code and in sandbox before any canonical
write; the original "refresh at 20260905, then 4 chunks at the live clock"
plan cannot pass:

- The observation roll judges readiness (`build_macro_snapshot`: freshness
  `FRESHNESS_MAX_AGE_DAYS`, period lag `PERIOD_MAX_LAG_DAYS`) at the **decision
  cutoff**, and `compile_local_market_breadth_observation` additionally
  requires every input artifact's **stamp and file mtime** to be ≤ that
  cutoff. A replay executed days later therefore cannot run at "now": each
  chunk needs its own declared historical decision clock.
- A chunk's earliest target must sit within 7 calendar days of its clock
  (daily breadth period lag), and the clock can never precede the sealed
  capture evidence of any session in the window — the two bounds fix each
  chunk's cutoff to a single day.
- The mid-September official scope must be refreshed before rolling past
  2026-09-19 (PMI 202607 availability ages out at 50 days). That refresh
  (`bundle#1`) swaps the PMI third vintage to 202606/07/08 — newest
  availability 2026-09-15 — so it can be published at a September cutoff.
  The June PMI page is staged in the packet (`sources/nbs-pmi-202606.html`).
- An official-scope refresh generation carries `latest_local_trade_date`
  (not `local_target_trade_date`); `_parent_local_target` now anchors the
  next window on it so rolls can follow a refresh.

## Code (commit 69d1aee, tests in `tests/unit/test_macro_maintenance.py`, `test_macro_retrospective_recovery.py`)

- `run_cn_macro_maintenance` accepts `decision_cutoff_at`, bounded
  fail-closed (`_resolve_roll_decision_cutoff`): never after the live capture
  clock, never before the target's own close. The retrospective v2 contract
  may carry the key; the prepare threads it into the rolls.
- `build_retrospective_market_projections` gains `identity_stamp_at`: the
  projections carry the declared historical identity (snapshot id **and**
  mtime), rejected when it would predate any sealed input's real production
  time (source snapshot stamp, capture `captured_at`, attempt `started_at`)
  or lie in the future. Legacy live mode (`reconstructed_at`, within 1h of
  now) is unchanged and mutually exclusive with it.

## Exact execution (private packet `data/private/macro_recovery_20261006/`)

Each chunk: `run_macro_chunk.py --target T --cutoff C --label L --execute`
(rebuilds projections at the declared clock, contracts, prepares, commits
through the journaled API; requires `SUCCESS`/`terminal: true` and pointer
equality). Each refresh: `refresh_official.py --bundle B --target T --cutoff C
--run-id R`.

All ten steps executed 2026-10-07, each commit `SUCCESS`/`terminal: true`
with pointer equality verified; receipts in the packet
(`chunk-r*-committed.json`, `readiness-closure.json`).

| # | step | window | target | cutoff (+08:00) | state |
| --- | --- | --- | --- | --- | --- |
| 1 | chunk r1 | 09-04..09-10 | 20260910 | 2026-09-11T23:00:00 | COMMITTED → observations `d33cd2d0…`, release `a7be2cfb…` |
| 2 | chunk r2 | 09-11..09-16 | 20260916 | 2026-09-16T23:00:00 | COMMITTED → `f128c534…`, `a812caff…` |
| 3 | refresh#1 | — | 20260916 | 2026-09-16T23:30:00 | PROMOTED (`bundles1/CN/mreb120261007T021842`) → `96b55944…` |
| 4 | chunk r3 | 09-17..09-22 | 20260922 | 2026-09-23T23:00:00 | COMMITTED → `51707c6b…`, `52d3e3ce…` |
| 5 | chunk r4 | 09-23..09-24 | 20260924 | 2026-09-24T23:00:00 | COMMITTED → `2a513b31…`, `4111de94…` |
| 6 | chunk r5 | 09-28..09-30 | 20260930 | 2026-10-02T22:00:00 | COMMITTED → `0c8d9cff…`, `0913e3f0…`; terminal journal sha `33cfbb2e…` |
| 7 | refresh#2 | — | 20260930 | 2026-10-03T22:00:00 | PROMOTED (`bundles/CN/mrext20261006T160041`) → `e438f6b2…` |
| 8 | closure pre-check | — | 20260930 | — | refused with exactly `MACRO_READINESS_VETO_ARCHIVE_INVALID`, archive genuinely absent |
| 9 | veto clear | — | — | — | CLEARED at 2026-10-07T02:48:13Z, archive `veto_archive/7f920f08….json`, reason `phase12-macro-recovery:20260930:terminal-sha256=33cfbb2e…` |
| 10 | closure | — | 20260930 | — | `READY` + `veto_lifecycle CLEARED`; `validate_macro_readiness_closure` replays; saved as `readiness-closure.json` (sha `2d608732…`) |

Cutoff derivations (evidence bound → clock):
r1 ≥ 2026-09-11T13:51:56Z (9/10+9/11 capture), first target 09-04 → 09-11.
r2 ≥ 2026-09-16T08:22:33Z, first target 09-11 → 09-16.
r3 ≥ 2026-09-22T08:24:01Z (9/17+9/18 evidence is 09-19T03:45Z), first 09-17 → 09-23.
r4 ≥ 2026-09-24T12:58:27Z, first 09-23 → 09-24.
r5 ≥ 2026-10-02T00:40:24Z (9/30 capture ran on 10/2), first 09-28 → 10-02.
refresh#1 cutoff ≥ 2026-09-15T02:00Z (bundle#1 newest) and > r2's; breadth 09-16 → 09-16T23:30.
refresh#2 cutoff ≥ max(2026-09-30T01:30Z, r5's) and breadth 09-30 ≤ 7 days → 10-03.

The closure requires the terminal transaction's target to equal the frozen
market frontier (20260930) and the PIT generation `pit-20260930-…`, so the
terminal journal must be r5's canonical-layout journal
(`data/private/macro_recovery_transactions/<txn>/journals/<txn>/0007-terminal.json`).

## Sandbox rehearsal (2026-10-07, before any canonical write)

The whole sequence ran on private copies of both macro stores with real
captures and live coverage fetches: `sb1 → sb2 → refresh#1 → sb3 → sb4 → sb5
→ refresh#2`, each commit `SUCCESS`/terminal; the final state validated at 39
rows, chain validated, `readiness: pass`, local breadth 09-28/29/30. Artifacts
under `data/private/macro_recovery_20261006/` (`chunk-sb*-committed.json`,
`sandbox-obs/`, `sandbox-release/`).

## Completed state (2026-10-07)

Final store: generation `macro-recovery-refresh2`, target 20260930, 39 rows,
chain validated, `readiness: pass` at its recorded clock (2026-10-03T14:00Z);
newest official rows PMI 202609 (available 2026-09-30) and the 202608
economy/money vintages. Market and PIT pointers unchanged (`9a07ed56…`,
`2f3c758c…`); release `0913e3f0…`, observations `e438f6b2…`; veto archived
with its exact bytes and a unique clear receipt.

The next daily window (20261008) resolves to `['20261008']`, so the 10-08
daily maintenance can roll the macro chain forward normally; October releases
will still need a further release-plan extension packet when they come due
(the roll copies the parent plan verbatim by design).

Interim note: canonical execution was briefly paused after step 1 by the
session's permission classifier (canonical writes beyond the first required
an explicit in-session owner authorization, which was then given). The
partial state was consistent: store at target 20260910, chain valid,
readiness `pass` at its own clock, veto still present, nothing regressed.

## Findings recorded while preparing

- The roll copies the parent's `plan.json` verbatim and only fetches the two
  coverage index pages as evidence, so a stale plan can never pick up later
  releases — the plan has to be extended by this kind of packet.
- The `official_web` compiler rejects a plan whose compiled scope is ambiguous
  (the H1 economy page already emits `cn.gdp_yoy`), and a refresh bundle must
  contain exactly 36 official rows with three vintages per indicator available
  by its cutoff.
- The daily factor loop's `ValueError` on same-target re-runs is
  `DAILY_FACTOR_STATE_CHECKPOINT_MISMATCH` (the release's error mapping at
  `daily_maintenance.py` discards the message and records only the type name).
  It is fail-closed and does not affect a new session; it only makes holiday
  and second-slot re-runs report BLOCKED.

## Bridge until the active release carries the fix (needed for 2026-10-08)

The active release (`660f066`, and the pending cutover commit `2259530`)
computes the catch-up window from `local_target_trade_date` alone:

```python
parent_target = str(existing_metadata.get("local_target_trade_date") or "")
```

The store's newest generation is now `macro-recovery-refresh2` — an
official-scope refresh generation, which records `latest_local_trade_date`
instead — so the 2026-10-08 daily macro stage reads an empty anchor and the
182 pinned open dates fail `macro_observation_catch_up_window_invalid`. The
fix is `_parent_local_target` (commit `69d1aee`, `69d1aee` not in `2259530`).
Until a release carries it:

The failing daily macro stage also **writes a new `MACRO_WRITE_VETO.json`**
(the live `daily_maintenance.py` does so on `macro_status == "BLOCKED"`), and
the daily stage short-circuits on that file before it ever calls the macro
component, so a bridge roll alone does not restore the lane: the full 10-08
evening is four steps, each separately authorized.

1. **Landing check** — no network, no prepare, only the private receipt:

   ```bash
   PYTHONPATH=~/mySpace/myQuant-worktrees/macro-bridge \
     /Users/maxwell/mySpace/myQuant/.venv/bin/python \
     ~/mySpace/myQuant-worktrees/macro-bridge/scripts/operations/cn_macro_forward_roll.py \
     --workspace /Users/maxwell/mySpace/myQuant --target 20261008 --check
   ```

   `CHECK_OK` means the lock is free, the newest execute attempt for the
   target has `ended.json` with empty `core_blockers`, and the market
   frontier is on the target; the receipt also reports the fresh
   `MACRO_WRITE_VETO.json`'s sha256 for step 3. (Verified 2026-10-07 against
   the live workspace: today the same check passes for `20260930` and
   refuses `20261008` with "no execute attempt yet".)
2. **Bridge roll** — the daily component's own call (`daily_components.macro`)
   on the live clock, no reconstruction. Run it from the pinned clean worktree
   (the script refuses to run from uncommitted code, refuses while the daily
   lock is held, and requires step 1's state:

   ```bash
   PYTHONPATH=~/mySpace/myQuant-worktrees/macro-bridge \
     /Users/maxwell/mySpace/myQuant/.venv/bin/python \
     ~/mySpace/myQuant-worktrees/macro-bridge/scripts/operations/cn_macro_forward_roll.py \
     --workspace /Users/maxwell/mySpace/myQuant --target 20261008 --execute
   ```

   (`PYTHONPATH` makes the process import the worktree's committed code —
   verified to win over the main tree's venv — while `--workspace` keeps every
   data read/write on the live workspace.) It acquires
   `data/private/cn_daily_maintenance/.daily-maintenance.lock`, fetches the two
   official coverage index pages (live, like the daily stage), and must return
   `SUCCESS` with `terminal: true`; the receipt records the code commit, the
   run-landing evidence, the veto sha for step 3, and the terminal journal sha.
   A repeat reports `NO_ACTION`.)
3. **Clear the new veto** with the exact sha from step 1/2 and a reason bound
   to the bridge roll's terminal journal sha, then verify the archive bytes
   and the clear receipt (same clean-worktree interpreter):

   ```bash
   PYTHONPATH=~/mySpace/myQuant-worktrees/macro-bridge \
     /Users/maxwell/mySpace/myQuant/.venv/bin/python - <<'PY'
   import json
   from pathlib import Path
   from quant_investor.market.daily_maintenance import clear_cn_daily_write_veto
   WS = Path("/Users/maxwell/mySpace/myQuant")
   receipt = json.loads((WS / "data/private/macro_recovery_transactions/macro-forward-20261008/receipt.json").read_text())
   print(clear_cn_daily_write_veto(
       run_root=WS / "data/private/cn_daily_maintenance",
       expected_veto_sha256=receipt["macro_write_veto"]["sha256"],
       reason=f"phase12-macro-recovery:20261008:terminal-sha256={receipt['terminal_journal_sha256']}",
       lane="macro",
   ))
   PY
   ```
4. **Rebuild the readiness closure** on the bridge roll's terminal journal —
   expect `READY` with `veto_lifecycle` `NOT_PRESENT` (no veto was bound into
   this plain transaction) and a replaying `validate_macro_readiness_closure`:

   ```bash
   PYTHONPATH=~/mySpace/myQuant-worktrees/macro-bridge \
     /Users/maxwell/mySpace/myQuant/.venv/bin/python - <<'PY'
   import hashlib, json
   from pathlib import Path
   from quant_investor.macro.readiness_closure import (
       build_macro_readiness_closure, validate_macro_readiness_closure,
   )
   WS = Path("/Users/maxwell/mySpace/myQuant")
   rel = ("data/private/macro_recovery_transactions/macro-forward-20261008/"
          "journals/macro-forward-20261008/0007-terminal.json")
   tsha = hashlib.sha256((WS / rel).read_bytes()).hexdigest()
   closure = build_macro_readiness_closure(workspace_root=WS, terminal_path=rel, terminal_sha256=tsha)
   print("status:", closure["status"], "| veto_lifecycle:", closure["veto_lifecycle"]["state"])
   print("replays:", validate_macro_readiness_closure(workspace_root=WS, closure=closure) == closure)
   PY
   ```

   (Both the step-3 and step-4 command shapes were exercised on the live
   workspace 2026-10-07: the clear call returned `NO_ACTION` with no veto
   present and zero side effects, and the closure rebuilt `READY`/replaying on
   the recovery's real terminal.)

After step 3 the later same-day attempts self-heal (their macro fast path
needs `local_target_trade_date == target` plus the exact market binding, which
step 2 restores), and 10-09 onward runs normally under the old release code.

Durable follow-up — the fix must ride the next release switch, whose shape is
now fixed by `docs/runbooks/release_repoint_20261008.md`: that switch is
deferred until a session publishes a capture under the current boot (the
2026-10-04 device-number drift makes `market native-seal` refuse the retained
2026-09-30 capture), and it currently targets commit `2259530`, which does
**not** contain `69d1aee`/`53822bc`. So option (a) means: rebuild the release
from a new commit that carries the anchor fix (and, as reviewed, the
`daily_maintenance.py` `str(exc)` retention and the same-target checkpoint
idempotence fixes), regenerate the factor-loop context, and then follow the
same `release_repoint_20261008.md` seal/continue flow. `69d1aee` cherry-picks
cleanly onto `2259530` (verified 2026-10-07).

Cutover-commit checklist (verified 2026-10-07): that new clean commit needs,
in addition to the release-side work of record —

- `69d1aee` (anchor fix) and `53822bc`/`1cf8acd`/`55de41c` (replay provenance
  and the bridge) from this workstream;
- the two reviewed `daily_maintenance.py` fixes, which are currently
  **uncommitted** in the working tree (`daily_maintenance.py` carries
  `"blocker": str(exc) or type(exc).__name__`; `daily_factor_loop.py` carries
  the checkpoint-state changes) — their author must commit them first, since
  `release-prepare` builds from a clean checkout;

Deadline: before the next official refresh, hard stop when PMI 202609 ages
out of the 50-day window (~2026-11-19). Until then the bridge stays in
service.

## Provenance of replayed projections (`built_at_wall_clock`)

Replay mode stamps each projection's identity (snapshot id and file mtime —
the clocks readers treat as availability) with the declared historical clock,
bounded below by the sealed capture evidence (source snapshot stamp, capture
`captured_at`, attempt `started_at`). To keep the real build time discoverable
inside the artifacts, each projection's `metadata.built_at_wall_clock` and the
candidate manifest's `built_at_wall_clock` now record the actual wall clock at
which the bytes were authored. Reader compatibility was verified: the breadth
compile reads named manifest fields only (never `metadata` key sets), the roll
hash-checks the manifest bytes, and chain validation binds the projections by
path+sha — so unknown metadata keys are invisible to current and older
readers.

Downstream status: the reconstruction label lives in the projected manifests
(retained in the packet) and the store's local-breadth evidence binds them by
path+sha, so an auditor can trace a store row to its reconstruction and its
true build time. No consumer currently excludes or downgrades rows
mechanically on this label (the only reader is the breadth compile itself);
a forward-out-of-sample consumer must resolve the evidence refs before
treating replayed rows as pre-available — recorded as a follow-up.

# Macro catch-up recovery (prepared 2026-10-06)

The macro lane has been blocked since 2026-09-22: `MACRO_WRITE_VETO.json` is on
disk, and the observations catch-up cannot satisfy
`production_observation_readiness_not_pass` because the canonical release plan
(`release-macro-catchup-20260903-v1`) only carries notices through June.

This packet contains the governed preparation for the recovery described in
`docs/plans/phase12_macro_two_chunk_recovery.md`. Everything below was
validated read-only; every step is a canonical write and needs owner approval.

## Staged artifacts (all under `data/private/macro_recovery_20261006/`)

| artifact | sha256 |
| --- | --- |
| `release-extension/plan.json` | `4857240a5ae85fcfcaf0b0348be51ab7889e82e1efc50addc3058293abe0081a` |
| `release-extension/capture_manifest.json` | `f0b288c498e59db6fbd59f1f48946cc431543ef8127724b541f94a69a2d8867e` |

Validation (`release_calendar._compile_inputs`, no write): **11 events,
35 resolutions, 142 sources**; the child is the parent's exact prefix plus:

- `nbs-pmi-202607`, `nbs-economy-202607`, `pbc-money-stock-202607`,
  `nbs-pmi-202608`, `pbc-money-stock-202608`, `nbs-economy-202608`,
  `nbs-pmi-202609`

built from one official-web compilation of the 12-page window
(3 economy months, 3 PMI months, 4 money months, 2 consecutive GDP quarters —
the third GDP quarter is contributed by the H1 economy page, which emits
`cn.gdp_yoy` itself; adding a separate 2026Q2 GDP page makes the compiled
scope ambiguous).

## Execution order (owner-approved, one step at a time)

1. `publish_release_extension.py` — CAS-publishes the release generation
   `release-macro-recovery-20261006-v1`; verify the new pointer sha.
2. Four catch-up chunks, each ≤5 open sessions:
   `run_macro_chunk.py --target 20260910 --execute`, then `20260916`,
   `20260922`, `20260930`. Each rebuilds fresh retrospective projections
   (the builder requires `reconstructed_at` within one hour), re-derives the
   contract + lineage inventory, prepares, then commits through the native
   journaled API (`commit_prepared_macro_transaction`), which requires
   status `SUCCESS` and `terminal: true` and CASes both macro pointers.
   Stop on any non-terminal result; do not retry under a new run id.
3. Only after the final chunk: build the macro readiness closure and
   `clear_cn_daily_write_veto(lane="macro", expected_veto_sha256=<original>,
   reason="phase12-macro-recovery:20260930:terminal-sha256=<actual>")` —
   the reason digest is read from the sealed terminal journal, never predicted.

The veto stays in place through steps 1-2 by design: it blocks the daily
maintenance's macro stage, not the governed recovery transaction, and the
recovery's own postcheck must pass while it is still present.

## Findings recorded while preparing

- The roll copies the parent's `plan.json` verbatim and only fetches the two
  coverage index pages as evidence, so a stale plan can never pick up later
  releases — the plan has to be extended by this kind of packet.
- The `official_web` compiler rejects a plan whose compiled scope is ambiguous:
  the H1 economy release already emits `cn.gdp_yoy`, so a separate 2026Q2 GDP
  page must not be planned alongside it.
- The daily factor loop's `ValueError` on same-target re-runs is
  `DAILY_FACTOR_STATE_CHECKPOINT_MISMATCH` (the release's error mapping at
  `daily_maintenance.py` discards the message and records only the type name).
  It is fail-closed and does not affect a new session; it only makes holiday
  and second-slot re-runs report BLOCKED.

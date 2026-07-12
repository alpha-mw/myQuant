# myQuant v13.1 Freeze-Exception Handoff

Date: 2026-07-12 (Asia/Shanghai)

## Outcome

The requested safety mechanisms were implemented and verified on an isolated,
offline review branch before integration. Production activation remains
intentionally incomplete. The local Dashboard is usable, Theme v2 is available
in observer mode, and Factor Governance v2 is report-only. No Theme formal
switch, forward Factor registry transition, joint production path, broker, LLM,
or Web API was enabled.

Worktree label and branch:

- `myQuant-governance-dashboard-v2` (isolated sibling worktree)
- `codex/myquant-governance-dashboard-v2`
- source commit `463140137ee1ddb2cecacc6aeb6edb226a6d67ef`

## Dashboard acceptance

The final real local export and semantic checker passed with no checker errors
and no sample fallback. All private JSON, JavaScript, CSV, and export-summary
artifacts are ignored by Git and mode `0600`; JSON/JavaScript readback is exact.

The latest real-account export remains private. Its exact dates, balances,
exposures, counts, holdings, trades, and reconciliation statistics are not
stored in this tracked handoff. The contract correctly reports `partial` and
the semantic checker surfaces its warnings. Explicit blockers include partial
attribution, excluded non-formal dates, position rows without an effective
contribution date, a legacy manifest
that does not declare its ledger SHA, missing Theme/Factor production artifacts,
unknown trade fees/NAV fee provenance, missing Theme as-of, and quote/benchmark
dates after the analysis date. The exporter still binds that legacy manifest to
a contained ledger path, manifest SHA, computed ledger SHA, and stable double
readback; the missing declaration remains visible rather than silently trusted.
No annualized or attribution claim may hide insufficient strict-calendar
coverage or unreconciled residuals.

## Theme v2 state

Implemented and verified:

- technology/advanced-manufacturing taxonomy and supply-chain roles;
- independent canonical multi-membership v2 loading, PIT revision selection,
  half-open validity, tombstones, and immediate broken state;
- 5/20/60/120-day axes, long-horizon coverage, industrial freshness, lifecycle,
  no forced admission, and bounded PE/VC rank prior;
- separate observation/watch/formal lanes and Markov non-technology caps;
- PortfolioConstructor cap proof, post-control reconciliation, durable 0600
  artifact, idempotence, kill switch, and target-weight clearing;
- complete runtime joint-manifest verification.

Formal Theme remains code-level blocked by
`canonical_joint_replay_producer_not_implemented`. It also lacks the required
20 distinct live-shadow days and trusted private production evidence for
membership coverage, PIT valuation/crowding, and industrial confirmation.

PE/VC knowledge uses an explicitly initialized 0600 HMAC key, chained approval
ledger, archived reviewed drafts/migration evidence, immutable natural
revisions, Asia/Shanghai availability floor, and recoverable transaction WAL.
Import remains draft-only until the local approve command succeeds.

## Factor Governance v2 state

The baseline registry remains one legacy-selectable record; historical records
were not bulk-restored. Metadata count, names, and set hash match that selectable
set, but selectable is not equivalent to v2 production-ready. The strict v2
runtime reports `governance_blocked`, confidence `0`, and no legacy fallback.
Current blockers include missing v2 protocol/evidence metadata, a single
factor's 100% normalized allocation violating the 20% factor and 35% family
budgets, and the unavailable canonical producer. The release gate verifies the
registry SHA and selectable-set metadata rather than documenting a potentially
stale hash here.

The private `old14_manifest.json` was built and read back mode `0600`, with a
self-hash and a source-record hash for each explicit factor. It reconstructs
`1 current + 13 historical` only as a Quant report-only shadow, carries zero
production confidence, leaves the formal registry SHA unchanged, and is not a
canonical full-DAG replay producer.

Weekly mining/health are report-only. Maturity, family FDR, purged walk-forward,
30-day embargo, independent failure windows, A/B/C/D recomputation, slot risk
budgets, monthly budget, targeted CAS/WAL/readback, and inverse rollback are
implemented. The current JSON evidence builder is explicitly a report-only
normalizer. New forward mutation is code-level blocked by
`canonical_full_chain_replay_producer_unavailable`; a hand-written evidence
artifact cannot create a WAL, consume the monthly budget, or change registry
bytes. Historical inverse-WAL rollback remains available.

## Joint replay and production stop

Threshold sealing, hash-chained seal ledger, 60/20/20 chronological split,
five-scenario artifact verification, shadow evidence, acceptance verification,
and atomic 0600 manifest persistence are implemented. The fixed cycle ID is
`myquant-v13.1-freeze-exception`; a permanent mode-`0600` lock and exactly one
ledger entry prevent a second dataset or threshold seal from defining a new
cycle. The actual canonical five-path historical DAG producer is not
implemented. This is an explicit stop condition, not a successful production
replay.

Consequently, Theme formal, forward Factor transition, and the joint path all
remain off. Dashboard read-only analysis is independently available.

## Verification

Final unified quality gate:

- focused Dashboard/Theme/Factor/joint: `362 passed`
- Dashboard Node contract: `1 passed`; all relevant JavaScript syntax checks
  passed
- full Theme sweep: `312 passed`
- full Factor/Quant-governance sweep: `455 passed`
- public CLI/package/integration smoke: `38 passed`
- staged-upgrade compatibility: `105 passed`
- staged-upgrade mypy: 9 source files, no issues
- `git diff --check`: passed

Read-only storage validation, executed with isolated code against the original
canonical data root:

- strict CN Parquet: passed for the locally selected canonical snapshot; exact
  snapshot identifiers and dates remain in private validation output
- clean/factor-readiness/cleaning-report validation: passed

The installed in-app browser runtime and local Playwright CLI were unavailable,
so exact viewport regression used local headless Chrome against the private
snapshot. Fresh 1280px, 390px and 844x390 landscape renders passed visual
review; measured document widths exactly matched their viewports with no page
horizontal overflow. The compact alert summary keeps all detail in an
expandable local panel, places desktop KPIs and the primary chart in the first
screen, and prioritizes NAV/sleeve exposure plus the discipline table on mobile.

## Isolation and residual assumptions

The original worktree changes were reviewed, committed separately, and then
integrated without discarding their strict full-A, restricted exposure, or PIT
fundamental evidence repairs. Release identifiers and current registry hashes
are recorded by Git and generated validation manifests.

Low-risk residual assumptions:

- the PE/VC HMAC key is a local same-user trust root; an attacker who can replace
  the key and every private artifact can replace the trust history unless its
  fingerprint is pinned outside that directory;
- Theme membership approval is hash-pinned for formal use but does not yet have
  a signed approval ledger equivalent to PE/VC;
- production activation remains blocked until the missing canonical producers,
  private PIT evidence, live-shadow period, and rollback drills are complete.

Production activation still requires every stop condition above to pass.

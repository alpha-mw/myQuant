# myQuant v13.1 Freeze-Exception Handoff

Date: 2026-07-12 (Asia/Shanghai)

## Outcome

The requested safety mechanisms are implemented on an isolated, offline review
branch; production activation is intentionally incomplete. The local Dashboard
is usable, Theme v2 is available in observer mode, and Factor Governance v2 is
report-only. No Theme formal switch, forward Factor registry transition, joint
production path, broker, LLM, Web API, commit, push, or merge was enabled.

Worktree label and branch:

- `myQuant-governance-dashboard-v2` (isolated sibling worktree)
- `codex/myquant-governance-dashboard-v2`
- source commit `463140137ee1ddb2cecacc6aeb6edb226a6d67ef`

## Dashboard acceptance

The final real local export and semantic checker passed with no checker errors
and no sample fallback. All private JSON, JavaScript, CSV, and export-summary
artifacts are ignored by Git and mode `0600`; JSON/JavaScript readback is exact.

Safe aggregates from the latest private snapshot:

- strategy record date: `2026-07-12`
- analysis trading date: `2026-07-08`
- quote and benchmark value date: `2026-07-10`
- gross/net NAV exposure: `0.04257703`
- cash weight: `0.95742297`
- latest NAV-weight sum: `0.04257703`
- latest equity-sleeve-weight sum: `1.0`
- latest position count: `4`
- NAV rows / position rows / executed trade rows: `74 / 260 / 25`

The contract correctly remains `partial` with 12 blockers; the independent
semantic checker reports 24 warnings. Explicit blockers include partial
attribution, six excluded non-formal dates,
ten position rows without an effective contribution date, a legacy manifest
that does not declare its ledger SHA, missing Theme/Factor production artifacts,
unknown trade fees/NAV fee provenance, missing Theme as-of, and quote/benchmark
dates after the analysis date. The exporter still binds that legacy manifest to
a contained ledger path, manifest SHA, computed ledger SHA, and stable double
readback; the missing declaration remains visible rather than silently trusted.
Strict-calendar reconciliation currently has 67 valid return days, 7 fully
covered days, and 0 days reconciled within one basis point; no annualized or
attribution claim should hide those limitations.

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
budgets, and the unavailable canonical producer. Registry file SHA-256 is
`104382f4c24b926fb44cd832845e4279295da945f7aa70fc463ff0f546804760`.

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

- focused Dashboard/Theme/Factor/joint: `356 passed`
- Dashboard Node contract: `1 passed`; all relevant JavaScript syntax checks
  passed
- full Theme sweep: `312 passed`
- full Factor/Quant-governance sweep: `449 passed`
- public CLI/package/integration smoke: `38 passed`
- staged-upgrade compatibility: `105 passed`
- staged-upgrade mypy: 9 source files, no issues
- `git diff --check`: passed

Read-only storage validation, executed with isolated code against the original
canonical data root:

- strict CN Parquet: passed, snapshot `20260710T034319Z`, latest complete trade
  date `20260708`, coverage `1.0`
- clean/factor-readiness/cleaning-report validation: passed

The installed in-app browser runtime and local Playwright CLI were unavailable,
so exact viewport regression used local headless Chrome against the private
snapshot. Fresh 1280px, 390px and 844x390 landscape renders passed visual
review; measured document widths exactly matched their viewports with no page
horizontal overflow. The compact alert summary keeps all detail in an
expandable local panel, places desktop KPIs and the primary chart in the first
screen, and prioritizes NAV/sleeve exposure plus the discipline table on mobile.

## Isolation and residual assumptions

The original dirty worktree was not edited by this implementation. Its registry
SHA remains `4c8680a9cccf08c188c072564312b17eabd61cb82487d89b2338fe488ad05806`.
Its tracked diff changed independently after baseline capture from
`867401872e4c815c792d256a3e47374076890cd39a8031a3bff4363b647b3813` to
`baae7b202677681d8b5999aa4c00a076d48b85b5f407df1bd31ff96fa31ad140`;
those external changes were not transplanted or reverted.

Low-risk residual assumptions:

- the PE/VC HMAC key is a local same-user trust root; an attacker who can replace
  the key and every private artifact can replace the trust history unless its
  fingerprint is pinned outside that directory;
- Theme membership approval is hash-pinned for formal use but does not yet have
  a signed approval ledger equivalent to PE/VC;
- production activation remains blocked until the missing canonical producers,
  private PIT evidence, live-shadow period, and rollback drills are complete.

Merge still requires Maxwell's explicit confirmation.

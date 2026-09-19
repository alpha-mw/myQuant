# CN weekly review improvements

User request: execute the six recommendations in order (2026-09-07).

## Scope and ownership

This task owns the existing weekly exporter/checker/runbook, a shared read-only
research-risk calculator and its daily consumer script, report rendering, focused
tests and a read-only daily acceptance/attribution consumer. It does not modify
Fundamental source or the daily Factor producer. The task titled
`修复myQuant因子效果闭环` owns producer/outcome/deployment work and is active.
Reuse its exact delivery/registered outcomes rather than build another producer.
No Git mutations, credentials, provider calls, actual holdings, event declarations,
Store/Factor/System pointers or trade/Paper writers are part of local verification.
Missing historical events require factual evidence, not inferred CLOSED_EMPTY.

## Ordered implementation

1. Exercise existing close-through-latest preparation against exact current inputs.
   Report all missing required session/event/benchmark dates independently of
   successful Store integrity. Repair reporting so FRESH does not hide incomplete
   week coverage. Reuse existing event and benchmark loaders and official-close
   writer; do not create a substitute accounting or publication path.
2. Implement one deterministic read-only risk calculation, shared by a daily
   report script and weekly exporter. Pin and validate existing owner policy
   `owner-trailing-anchor-policy-20260901-v1` and owner-stop policy exact SHA.
   Resolve current/baseline ledger and lineage only from Store-v3; verify costs,
   shares, subsequent position lifecycle/add/corporate-action invalidation and
   exact entry refs. Respect explicit owner resets for three legacy names.
   Require all expected strict-close sessions from tracking start, reject duplicate,
   nonfinite, nonpositive and corporate-action-incompatible prices. Use Decimal
   retention 0.80/0.65 and policy CNY0.01 HALF_UP. Missing evidence yields typed
   non-executable rows. Stale holdings never become current trading authority.
3. Read exact daily producer receipts/Factor lineage and same-date artifacts to
   distinguish functional implementation, installed/deployed state, on-time daily
   completion, late recovery and missing dates. Add a deterministic acceptance
   projection; five real days is an interim checkpoint, existing ten-day operational
   observation remains distinct. No simulated days or extra scheduler.
4. Consume explicitly SHA-bound existing outcome delivery manifest via prescribed
   candidate installed validator when necessary. Keep raw-close IC/RankIC diagnostic,
   cost-adjusted performance unavailable without cost evidence, no Factor-to-actual-
   portfolio attribution without consumption binding. Preserve correction lineage.
5. Introduce explicit v2 daily narrative input: automation_id, thread_id, run_id,
   started_at, completed_at, trade_date, run_status and research_status. Last-run is
   audit-only. Validate chronological/local-trade-date consistency, dedupe immutable
   run IDs and reject conflicting identity rows. Separate task/research/formal
   closure coverage; no narrative-provided formal authority. Maintain explicit v1
   historical readability, but current production runbook and consumers use v2.
6. Export a compact investment summary and separate evidence appendix from the same
   checked bundle. Label full/partial week and baseline/end dates explicitly;
   invalid/retired values only in audit columns. Checker must recompute added
   coverage/period/risk projections and fail on tampering. Update daily/weekly
   consumer contract documentation and prepare exact existing automation updates;
   stop after preparing exact consumer update payloads and readback procedures;
   no Git, installed-release, automation, provider or Store publication until a
   concrete publication decision binds payload, destination and release.

## Verification and stop conditions

Architect amendments (accepted): new weekly bundle schema is
`cn_weekly_portfolio_evidence.v2`; historical v1 remains explicit readback only.
Daily narrative schema `cn_weekly_daily_review_input.v2` uses immutable
`(automation_id, run_id)` and canonical UTC started/completed timestamps; run_status
is COMPLETED/FAILED/IN_PROGRESS, research_status is COMPLETE/PARTIAL/BLOCKED/NOT_RUN.
Daily timestamps describe factual SAME_SESSION or LATE_RECOVERY completion using
Asia/Shanghai dates. Exact scheduler punctuality is NOT_CONFIGURED without a
registered dispatch/SLA receipt; no additional clock cutoff is invented. FAILED
and IN_PROGRESS prove execution only; completed research requires COMPLETED run.
task execution, research completion, receipt continuity and formal closure have
independent sets. FULL_WEEK refers to complete valuation/benchmark performance only;
overall research completeness additionally requires all expected completed daily
reviews. Formal authority stays an independent blocking domain and never follows
from narrative flags. Conflicting duplicate run identities fail.

Pure risk module: `quant_investor/strategy_records/research_risk.py`; resolved-source
adapter/daily consumer: `scripts/export_cn_research_risk.py`; weekly uses the same
adapter. Policy baseline record must be ancestor of current head, with identical
fee-inclusive cost and shares throughout validated forward lifecycle. Pin trailing
policy SHA b313aa91e1f7ca1e8922b2d22f7735ceee3190675c2e8dab3955c69f0f1d342a and stop
policy SHA 11eb7018ff6abde2d276c7b997e0e61e2cd6b170407872734bb8d407b334c178.
Unproven lifecycle/current holdings yields non-executable diagnostics only. No
reanchoring to later valuation prices. Missing dates/nonfinite/adjustment changes
block calculations; retired fields appear only in audit evidence.

Official-close coverage analyzer will be one pure function shared by dry-run and
execute validation, enumerating event, held-close and three-index benchmark gaps
for every required date. Execute stays atomic/fail-closed. Current real gaps are
both9/4event closure and9/4benchmark; no replacement close or empty declaration.

Factor handoff is owned by active task01a07192-032d-7410-966f-3f1ea5d8e944. Existing
delivery `reports/operations/daily_factor_loop/20260906-220e6d1/evidence-manifest.json`
contains no versioned handoff schema; therefore do not treat it as a new standalone
trusted contract. Read-only audit may verify its exact refs through its pinned
220e6d1 candidate interpreter/registered artifact validator and label the output
DESCRIPTIVE_DELIVERY_AUDIT. Production weekly integration remains DEPENDENCY_BLOCKED
until that owner supplies a versioned validated read-only consumer handoff. No copy,
directory scan, new evaluator, or change to producer files/schedules. Continuous
real-day proof cannot be manufactured by this local implementation.

- Architect then Critic review before schema/policy interpretation implementation.
- Focused meaningful tests: previous-run timestamp versus current run, duplicate
  runs, malformed chronology, receipt versus narrative authority, missing Friday
  close, correct period baseline, stale holdings, owner resets, add invalidation,
  missing session and adjustment changes, positive/no-positive peak, rounding,
  shared daily/weekly parity and checker rejection of resealed tampering.
- Real offline Store verify, prescribed installed reads, close preparation,
  new daily risk export and new weekly export/check/render, exact before/after
  protected-pointer SHAs. No repository-wide cleanup of unrelated failures.
- Future real sessions cannot be completed now. Report the exact number proven and
  remaining; use existing producer automation for future evidence. If deployment
  belongs to the active producer task, deliver explicit consumer requirements and
  current acceptance status rather than overwrite its release binding.
- Missing event evidence or benchmark capture blocks official 9/4 publication only;
  finish independent code/consumer/report work. Prepare a concrete final action
  before any unapproved production or external write permission question.

Implementation review amendments: risk lifecycle also consumes the registered Event
Store and requires every expected post-baseline session. Common lifecycle, trailing
and owner-stop blockers are separate; both lanes affect readiness. Risk, Factor
daily coverage and effectiveness have explicit completeness domains. Checker
recomputes aggregate warnings/blockers, and Factor daily component acceptance reads
exact maintenance refs plus policy-replayed Top100; it never infers unattended
dispatch from artifact timestamps.

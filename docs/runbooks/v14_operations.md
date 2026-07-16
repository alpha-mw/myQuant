# myQuant v14 Operations (Historical)

This is the historical operating contract for the retired v14 DAG. Current
operations use `docs/runbooks/v15_operations.md`; this file is retained only
for immutable artifact interpretation and must not be used by current runs.
The repository runs offline by default and keeps deterministic data, risk, and
production-authorization gates authoritative.

## Active architecture

- Formal branches: `quant`, `fundamental`, and `macro`.
- Likelihood inputs: `quant` and `fundamental`.
- Macro v2 remains an observer and control-context input; it is not a Bayesian
  stock-selection likelihood.
- Intelligence is retired from the active DAG and current catalog. Historical
  Intelligence marts, tags, reports, and replay evidence remain immutable.
- The v13 incubation/freeze and freeze-exception merge protocol are retired.
  They do not constrain current `main` or schedules.

## Independent safety boundaries

Retiring the v13 freeze does not relax these controls:

- Canonical CN data must pass `quant-investor market storage-validate --market CN`.
- Branch readiness and data governance fail closed when v14 provenance or
  generation evidence is missing.
- Current full-A Quant readiness is bound to the complete canonical coverage
  scope and its component hash. Evidence-backed non-trading absences are
  excluded; historical serving files are never added to the current scope.
  Coverage, classification, evidence, or PIT hash drift returns a structured
  blocked report.
- RiskGuard and deterministic control-chain vetoes remain authoritative.
- Quant FactorGovernanceProtocol v2 stays report-only until its independent
  production-apply authorization is implemented; PR4 stops at
  `forward_factor_apply_not_authorized_pr4`.
- No schedule may auto-checkout, cherry-pick, merge, call a broker, or create
  orders/trades.
- Legacy Macro catalog entries are diagnostic only and cannot claim v14
  production readiness.
- A missing, stale, or unverifiable canonical Macro generation blocks branch
  fusion before MacroAgent, Markov, RiskGuard, and the investable funnel. The
  blocked path emits a neutral VETO diagnostic, keeps the 0.55/0.50 baseline,
  preserves existing holdings, and authorizes no new decision or order.

## Offline preflight

Run from the repository root with the repository virtual environment:

```bash
./.venv/bin/quant-investor market storage-validate --market CN
./.venv/bin/quant-investor market data-governance --market CN
PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
```

For operator-facing commands, verify the exact CLI help before changing a
schedule. Missing source data, provenance, receipts, or approvals is a blocker;
do not substitute CSV, stale snapshots, inferred values, or hand-written pass
flags.

## Fundamental authoritative rebuild

An authoritative full-A Fundamental refresh is an explicit two-stage operation:

1. `market fundamental-maintain --allow-live --authoritative-full-rebuild`
   fetches into an isolated data root and a separate v3 checkpoint root. It
   binds the exact full-A scope, market pointer, PIT membership, request
   outcomes, financial-period coverage, canonical market-bar file set and
   per-symbol first/last trade bounds, and Parquet readback fingerprints.
2. `market fundamental-promote --expected-pointer-sha256 <sha>` independently
   revalidates the staged generation and advances the canonical Fundamental
   pointer with compare-and-swap semantics.

Financial completeness is measured only against matured post-listing quarters;
an observed pre-listing period in one endpoint does not expand another
endpoint's denominator, and a quarter whose reporting lag ends after the bound
eligibility end is not mature. Daily history is intersected with the exact
canonical bar bounds; the authoritative boundary tolerance is 62 days while
monthly coverage still requires at least 90% and no run of more than two
missing months. The bar file-set SHA, bounds SHA, and policy are replayed during
promotion.

The live rebuild must use a new run/checkpoint root after any v2 checkpoint,
scope drift, audit-policy or evidence-binding change, malformed response, or
failed request. Do not edit or weaken an older checkpoint binding to make it
resume. Promotion rejects legacy primary provenance, checkpoint or canonical
bar drift, incomplete endpoint coverage, or a derived mart that cannot be
reproduced from the accepted raw checkpoint and the exact bound PIT membership.
Weekly readiness schedules remain local and read-only; they must never add
`--allow-live` or perform promotion.

## Macro authoritative refresh

Macro primary refresh is a separate, explicit live operation. First record the
exact SHA-256 of `data/parquet/cn/_catalog.json` and `_latest.json`, verify that
the requested `as_of` is the latest complete session, then run:

```bash
./.venv/bin/quant-investor market macro-refresh \
  --market CN \
  --as-of <latest_complete_trade_date> \
  --data-root data/parquet/cn/macro_daily \
  --run-id <new_generation_id> \
  --expected-catalog-sha256 <catalog_sha256> \
  --expected-market-pointer-sha256 <market_pointer_sha256> \
  --nbs-cn-pmi-url <nbs_official_release_https_url> \
  --allow-live
```

`--nbs-cn-pmi-url` and `--allow-live` are both mandatory. The URL has no
repository default: the operator must supply the issuer-bound HTTPS URL of the
National Bureau of Statistics formal release page for the intended month.
`cn-macro-provider-bundle.v2` applies the
`official-first-per-endpoint.v1` policy: `cn_pmi` uses that NBS release as its
production primary source, while `cn_cpi`, `cn_ppi`, `sf_month`, and `cn_m`
remain Tushare primary endpoints for now. This changes neither the Quant nor
the Fundamental source policy.

`--allow-tushare-fallback` is optional and off by default. It authorizes a
`cn_pmi` Tushare fallback only after a classified transient NBS transport
failure. A redirect-policy violation, malformed or ambiguous page, unexpected
title/record/month/value, parser mismatch, or any other content/semantic error
still fails closed. A fallback generation records the failed official attempt
and the selected fallback role explicitly; it never labels Tushare data as
official or combines the two values.

A same-`run_id` crash retry is policy-bound as well as byte-bound. Repeat the
exact initial NBS URL and the exact `--allow-tushare-fallback` choice used to
create the landed generation. A changed URL or changed fallback authorization
is rejected before provider I/O or catalog promotion; use a new run id for a
new request policy.

On an official success, the generation stores the NBS HTML under
`provider_captures/`. Its byte SHA-256 and size, parser version and parser
contract hash, issuer URL and record ID, `source_release_at`, fetch start and
completion times, and redirect chain are bound by `provider_bundle.json`. The
`nbs-cn-pmi-html.v2` parser additionally rejects a stated PMI month later than
the formal page's `PubDate` month. The
manifest binds the exact capture-file set and aggregate hash; primary
provenance and the strict catalog carry the same aggregate capture binding.
Strict readback and transaction recovery re-read the sidecar, verify its
hash/size/path, reparse it, and compare the parsed release identity, month, and
value with the provider bundle before accepting the generation.

Keep the source clock, observation clock, and commit clock separate:
`source_release_at` is the timestamp published by NBS,
`fetch_completed_at` is when this run actually obtained an endpoint response,
and the transaction journal's `committed_at` is when the catalog transaction
finished. The decision cutoff is the latest endpoint I/O completion, never the
page publication time. Every endpoint completion must be timezone-aware and
exactly equal its selected observation timestamp; the maximum of those times
must equal both `decision_cutoff_at` and bundle `fetched_at`. When fallback is
authorized, the recorded official failure must complete no later than the
fallback response. Endpoint I/O completion must remain in the 72-hour capture
window, and every selected month must meet its declared
`month_end + max_release_lag_days` lower bound.

The publisher also binds the exact canonical bar pointer and input partitions,
the deterministic transform, and the generated Parquet. It publishes through
the shared catalog-writer lock, two-object CAS, and a recoverable transaction
journal. The formula cross-section uses only symbols whose terminal bar is the
requested session. Before switch and during recovery, the publisher verifies
the complete strict-catalog required table closure, the exact market-pointer
SHA, every generation/input hash, and the v2 capture closure; orphaned partial
transactions may be retried only after deterministic cleanup.
`cn-macro-provider-bundle.v1` remains readable only for historical
compatibility and can never satisfy the v2 current-equivalent check. Never
hand-edit a generation or replay it historically;
`historical_replay_eligible=false` is permanent. A schedule may report this
command as a manual remediation step but must never run it.

The one-time post-cutover event-score schema migration is
`scripts/retire_event_score_catalog_residual.py`. It requires the exact current
catalog SHA, market-pointer SHA, legacy event-score Parquet SHA, a new run id,
`--apply`, and `RETIRE_EVENT_SCORE_SCHEMA_V14`. It preserves the legacy source
bytes, publishes an immutable filtered generation, and commits only after a
fresh strict reader proves that `intelligence_score` is no longer visible.
Do not remove only the catalog column declaration or rewrite the legacy source
file in place.

## Schedule routing

Current schedules must name the three active branches, omit Intelligence from
active DAG work, and point to this runbook. Weekly Quant governance may report
proposals but must not mutate the production registry or weights. Fundamental
and Macro jobs must preserve their own provenance and activation gates.

Schedule edits are complete only after TOML parse, invariant readback, and
entrypoint/help validation. Unrelated schedule metadata must remain unchanged.

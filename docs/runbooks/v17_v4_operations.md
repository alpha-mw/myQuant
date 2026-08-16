# V17 v4 Mainline Operations

> **Current availability:** this repository implements exact, fail-closed
> mainline readers but no public production publisher or activation command.
> Steps 2 and 3 specify the required contract for a future or external governed
> writer; they are not commands that can currently be run from this repository.
> Do not use low-level storage/test helpers as an operational substitute.

## 1. Preconditions

- Work from a clean, reviewed build of the V17 v4 contract and runtime.
- Use strict CN Parquet canonical data; validate the active market pointer,
  manifest, PIT membership, Fundamental and Macro generations, and cutoff.
- Require the complete Factor, risk, holdings, and portfolio closure expected
  by the mainline run contract.
- Keep provider acquisition or data maintenance separate from a decision run.

Do not proceed on missing data, missing receipts, stale pointers, schema/hash
drift, or unresolved readiness blockers.

## 2. Build an immutable run

The governed mainline writer produces one closed
`myquant.v17.v4.mainline-run.v1` at:

```text
results/v17_mainline/strategies/<strategy-id>/runs/<run-id>/run.json
```

Validate the exact run bytes and every transitive reference after persistence.
Creating this run does not make it public.

## 3. Activate operationally

Operational activation is a separate act from installing or deploying code.
Advance only:

```text
results/v17_mainline/strategies/<strategy-id>/_active.json
```

The activation writer must receive the expected pointer prevalue, revalidate
the immutable run, perform an atomic CAS replacement, and read back the exact
proposed bytes. Any mismatch stops with no pointer change. Never activate by
renaming a run, editing the pointer manually, or scanning for the latest run.

## 4. Read the public run

All commands below read the same pointer and return the same
`myquant.v17.v4.mainline-public-run.v1` authority chain:

```bash
quant-investor research run \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>

quant-investor market analyze \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>

quant-investor market run \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>
```

The current repository has no Web reader. Python and CLI readers resolve the
same exact pointer. When a caller supplies an expected pointer SHA, a stale
value is a blocker, not permission to read another generation.

## 5. Expected fail-closed states

| Condition | Result | Writes |
|---|---|---:|
| Active pointer absent | `V17_MAINLINE_UNINITIALIZED` | 0 |
| Pointer/run/closure invalid | `V17_MAINLINE_BLOCKED:<blocker>` | 0 |
| Market is not CN | `V17_MARKET_UNSUPPORTED` | 0 |
| `market backtest` | `V17_BACKTEST_UNAVAILABLE` | 0 |

After any failure, inspect the exact pointer and run reference. Do not create a
placeholder pointer, substitute a Shadow session, scan an older result, or
infer success from a run directory.

## 6. Forward Shadow research

Forward evidence is deliberately separate:

```bash
quant-investor-v17-v4 run-forward \
  --workspace-root /absolute/path/to/myQuant \
  --request-path data/private/v17_v4_runs/forward_requests/<request_id>.json \
  --request-sha256 <sha256>
```

Its final session reference proves only a completed Shadow observation. It
does not activate the mainline, publish a public run, or authorize new risk,
broker, order, execution, or trade activity.

## 7. Evaluate matured research evidence

Run R2.2 only with the exact content-bound request path and byte SHA:

```bash
quant-investor-v17-v4 research-evaluate \
  --workspace-root /absolute/path/to/myQuant \
  --request-path data/private/research_intelligence/evaluation_requests/forward-evaluation-request-<sha256>.json \
  --request-sha256 <sha256>
```

The command is offline and stdout-only. It does not write a result, append
memory, call a provider or LLM, change a Factor tier or weight, choose a
portfolio, or read or update the active pointer. Its authority fields must
remain `decision_protocol=myquant.v17.v4`, `mainline_authority=false`, and
`operational_activation_unchanged=true`.

## 8. Verification

Run the narrow contract and CLI tests for the changed surface, then run the
full unit suite plus the flake8 and mypy steps from
`.github/workflows/ci-cd.yml` for a broad release. Verification
must include missing-pointer and invalid-pointer no-write assertions, CN-only
routing, public DTO equality across surfaces, unsupported-backtest no-write,
and Shadow/mainline isolation.

## 8. CN Macro and release-calendar maintenance

`quant-investor market macro-maintain` is the only registered writer for the
daily CN Macro observation roll and official release-calendar coverage
extension. It is dry-run by default. A committing run requires both
`--allow-live` and `--commit`, exact expected-before SHA-256 values for the
release-calendar and Macro-observation pointers, and exact path+SHA bindings
for the immutable market snapshot manifest, its target-date coverage manifest,
and the full-A scope artifact.

The command calls only the NBS and PBC official coverage URLs. It stores each
complete HTTP response entity as hash-bound `coverage_response` evidence and a
v2 coverage receipt that binds that response. The common cutoff is frozen after
both response entities have completed and before either pointer write. The
release calendar is published first by CAS; the local breadth observation then
rolls against the same cutoff, target, market-open-days evidence and canonical
market inputs. If the second stage fails, the release child may remain as an
independently healthy promotion and the receipt reports `PARTIAL`. A failed
stage never substitutes stale, inferred, public-fallback or manual values.

The command must not import or restore the retired
`quant_investor.market.macro_mart` producer. It does not run analysis, create
candidates or portfolios, activate Factor or Mainline state, render Dashboard
output, connect a broker, create an order, or trade.

## 9. CN Fundamental safe-successor maintenance

Production cutoff advancement must use the registered safe-successor mode.
Do not use the ordinary live merge as a fallback: that compatibility path can
represent a partial provider result and does not prove an operator-frozen
predecessor or an append-only historical prefix.

The registered publication workflow remains two phase, but a deferred source
classification is not eligible to enter it. For an unpaired opaque
balancesheet observation, first use the exclusive live-source diagnostic:

```text
market fundamental-maintain \
  --taint-analysis-dry-run \
  --allow-live \
  --universes full_a \
  --audit-run-root /absolute/private/new-run-root \
  ...the frozen predecessor/market/PIT/scope/history arguments...
```

Despite the name `dry-run`, this command calls the live registered provider.
Its only write-set is the private audit root (`frozen_state.json`, `capture/`,
`analysis/`, and `execution_receipt.json`). It never creates a staging
generation, installs a generation, advances a pointer, authorizes promotion,
or returns a staging-compatible object. `PASS` means only that every deferred
observation was proven target-bounded non-reachable through the requested
cutoff. It does not make the source authoritative or the canonical Fundamental
current. `BLOCKED` returns a sealed terminal receipt and the same run id/root
cannot later be resumed as `PASS`.

The publication workflow is deliberately two phase and remains unavailable to
any current-window capture marked `DEFERRED_UNSUPPORTED_OBSERVATIONS`:

1. `market fundamental-maintain --safe-incremental-successor` captures the
   exact Fundamental predecessor, market pointer and PIT pointer bytes before
   acquisition. It fetches the registered target support window into an
   isolated fileset, requires zero failed, malformed or paginated responses,
   freezes a canonical subject set equal to the union of predecessor subjects,
   every delta session's PIT-expected and observed-bar subjects, and the target
   full-A scope, proves the daily-basic keyset against canonical bars and
   reason-coded non-bar classifications, freezes the predecessor prefix,
   derives only the open successor window, and writes a promotion-ready
   staging generation only when the source is independently authoritative.
   A zero-deferred capture is sealed as `AUTHORITATIVE_DELTA_COMPLETE` and may
   cross the source boundary; a nonzero deferred inventory remains
   capture-only, and authoritative open/load, staging and promotion consumers
   reject it even when a diagnostic reports `PASS`.
2. `market fundamental-promote --safe-incremental-successor` performs a
   read-only preflight by default. Canonical mutation additionally requires
   `--execute`, an explicit journal root, the expected predecessor pointer SHA,
   and unchanged captured market and PIT pointer bytes. The promoter acquires
   the market, PIT and Fundamental locks in that order, installs an immutable
   generation, advances the pointer by CAS, and performs exact post-write
   readback. A failed post-check rolls back with the sealed predecessor pointer
   bytes when the candidate pointer is still current. Ordinary JSON artifacts
   retain the 64 MiB safety ceiling; only the exact predecessor Fundamental
   manifest may use the dedicated 128 MiB ceiling, and it is still read
   stably, schema-checked, rehashed, and matched to every predecessor table.

An owner-approved append-first successor is an explicit modifier of phase 1,
not a fallback and not an instruction to choose a historical winner:

```text
market fundamental-maintain \
  --safe-incremental-successor \
  --append-first-successor \
  --historical-taint-failure-evidence \
    /absolute/private/legacy-capture-failures#3564 \
  --successor-income-support 689009.SH@20250630 \
  --successor-financial-support cashflow:920198.BJ@20260630 \
  --allow-live \
  --universes full_a \
  ...the frozen predecessor/market/PIT/scope/history/staging arguments...
```

This mode makes the exact predecessor cutoff the publication boundary and
requests every natural announcement date in `(parent_cutoff,target_cutoff]`,
plus every target-window open-session `daily_basic` partition. If that clean
delta proves a production fallback needs an earlier comparison row, repeat
`--successor-income-support TS_CODE@YYYYMMDD` for a prior-year income key, or
`--successor-financial-support TABLE:TS_CODE@YYYYMMDD` for an exact
income/balancesheet/cashflow key reported by the registered derivation blocker.
Each such request
is provider-bounded by both subject and report period. The response may seed
only the hidden calculation state: it cannot produce a predecessor winner,
period suffix, daily suffix, or canonical row. The derivation must prove the
declared keyset equals both the captured pre-cutoff keyset and the fallback
read-set; an absent, extra, cross-symbol, cross-period, post-cutoff, or unused
support row blocks. Fina-indicator, forecast, daily-basic, or any other table
cannot enter this support capability. An exact empty provider response is sealed as an absence
proof and must likewise equal an actually consumed fallback read-set; it is
not synthetic data or an allowlist. The mode never reinterprets the historical conflict. The
supplied failure artifact must independently replay to the same material
source error and is copied, with its exact source binding and raw response,
into permanent generation evidence.

Before staging, a sealed historical-taint registry must prove all of the
following: each ambiguity has availability on or before the predecessor
cutoff; the exact business key is already present in the immutable predecessor
period table; it is not the predecessor's current daily winner; no financial
row for the same `(ts_code,end_date)` appears anywhere in the complete delta;
the current-window source has zero failed, malformed, paginated, deferred or
material-conflict observations; and the only pre-cutoff source reads are the
sealed, exact, consumed financial fallback dependencies described above. A new
same-period event, an incomplete same-period/fallback dependency,
an unsealed failure artifact, registry drift or predecessor drift blocks the
run. This is whole-key poisoning across the delta, not a symbol/date allowlist.

The registry, old failure JSON, raw response and old capture binding are copied
under `provider_evidence/historical_taint/`. Their file hashes are in the
provider manifest; the registry's semantic seal is in the support plan,
derivation evidence and metadata. Both staging and promotion provenance
validation rebuild the registry against the installed delta source fileset and
the predecessor's immutable table refs. A staged result is still not canonical:
run the normal read-only `fundamental-promote` preflight first, and use
`--execute` only after every readback and live market/PIT binding remains exact.

The successor is a versioned mixed generation. The original seam remains the
first trusted predecessor cutoff, while later daily successors append after
the immediate predecessor cutoff without rewriting an earlier successor.
Pointer, manifest, readiness and binding-aware loader output must agree on:

```text
mixed=true
legacy_direct_reader_provenance=limited
binding_aware_research_ready=true
homogeneous_history_ready=false
```

These fields are a limitation, not a warning that may be ignored. Research
that crosses the seam must preserve the verified derivation binding and treat
the seam as a methodology boundary. A direct-path consumer that drops pointer
and manifest provenance cannot claim seam-aware or homogeneous-history
readiness.

Exact-date provider endpoints may return a global symbol partition that is a
strict superset of the frozen canonical subject set. The source fileset must
retain the exact raw response bytes and every normalized row, bind separate
request-envelope and canonical-subject scope references, and reconcile each
response as disjoint in-scope plus out-of-scope observations. Out-of-scope
observations remain permanent audit evidence but cannot enter canonical winner
selection. This authority-bound projection is not an allowlist: any malformed
row, ambiguous code identity, scope collision, count/hash mismatch, or changed
in-scope candidate fingerprint blocks staging and leaves canonical unchanged.

Provider-external identities are accepted only as out-of-scope evidence when
the exact, unmodified value matches the registered external-identity grammar:
one uppercase letter plus five digits and a `.SH`, `.SZ` or `.BJ` suffix, or a
six-digit code plus `!` and a positive one-to-three-digit version before the
same suffix. Do not trim, case-fold, alias or map these values to canonical
symbols. They remain non-authoritative, staging-ineligible and
promotion-ineligible. An in-scope business key with materially different
supported `comp_type` candidates is not an external-identity case and must
terminate capture as `BLOCKING_UNKNOWN`; update flags do not establish
cross-physical-lane authority.

Statement physical classification remains endpoint-specific. Income and
cashflow accept only `comp_type` 1-4. For balancesheet, a `comp_type=7` row
with a projection-identical supported 1-4 peer remains
`OPAQUE_EQUIVALENCE_ONLY`: the raw `7` stays in evidence and only the supported
peer enters the accepted table. A changed value/date/availability,
supported-peer conflict or conflicting update winner still blocks at source.

After physical update dominance, an in-scope balancesheet `7` with no supported
peer is instead preserved as `TAINTED_PENDING_ANALYSIS`. The exact raw response
remains immutable, the opaque row is excluded from the accepted support table,
and the capture is mechanically non-authoritative and promotion-ineligible.
The analyzer shares the versioned production financial event kernel: state is
keyed by `(table,ts_code,end_date)`; a complete
`(ts_code,availability_date)` batch becomes visible atomically; any four-table
event rereads same-period fina/income/balancesheet/cashflow state plus the
previous-year income fallback; period/forecast winners retain production
ordering. Strict prelisting (`end_date` and availability both before exact PIT
`list_date`) is only an eligibility gate. PASS also requires seam-anchor
equality, no post-seam same-period event, no taint binding in lineage, and no
target winner selection. Unknown dependency, missing anchor, tie, authority
drift or resource drift is `BLOCKING_UNKNOWN`. This does not assign meaning to
`7`, map it to another company type, extend provider taxonomy generally, or
create a symbol/date allowlist.

Before the first provider call, seal a resource receipt that covers current
available RAM, process RSS and `RLIMIT_AS`/`RLIMIT_DATA` headroom as well as
physical RAM, source capture, staging temp, canonical
temp/final/orphan/rollback, fsync reserve, rolling free-disk protection and a
25% margin. A taint diagnostic additionally budgets the deferred inventory,
per-observation proofs, event trace and audit receipts; it does not gain
permission to write the staging/canonical reserves. Repeat the receipt with exact source/table sizes after capture and
before opening any support table. The production handoff is path-backed:
request evidence is streamed, records are independently replayed from exact
raw bytes, and aggregate support tables are deterministic sorted Parquet with
at most 2,048 rows and 16 MiB per stream batch. The production store rejects
full-table access; financial replay keeps at most one symbol's four-endpoint
hidden state resident under one aggregate per-symbol byte cap (not four
independent caps), and forecast/daily-basic replay is batched. Staging and
promotion validators must both replay the same versioned source fileset; file
SHA closure alone is insufficient.

Any source, schema, canonicalization, hidden PIT dependency, keyset, SHA, CAS,
readback, journal-recovery or pointer-drift blocker leaves the current healthy
Fundamental pointer unchanged. The automation must report `blocked` or
`partial`; it must not retry through ordinary `--allow-live`, generate
synthetic rows, apply an allowlist, activate Factor or Mainline state, write a
Dashboard or Paper ledger, connect a broker, create an order, or trade.

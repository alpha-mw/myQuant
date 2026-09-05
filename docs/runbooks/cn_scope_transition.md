# Explicit CN full-A scope transition

Ordinary `market daily-maintain` does not refresh an existing `full_a` scope.
The ordinary PIT capture path continues to reject scope changes and implicit
pending-stock admission. A scope transition requires a separately authorized,
owner-only `cn-scope-transition-request.v1` input with its exact byte SHA.

The existing `run_cn_daily_slot.sh` supports paired
`--scope-transition-request` and `--expected-scope-transition-sha256` arguments
only for execute slot 2020. They route before token-only veto recovery to the
registered transition branch of `market daily-maintain`. No schedule change or
automatic periodic scope expansion is implied.

The request binds immutable old/new scope bytes, exact added/removed sets,
effective date, captured stock_basic L/D/P evidence, close-session receipt,
predecessor Market/PIT pointers, protected Fundamental/checkpoint pointers, and
the exact zero-write failed attempt and non-transient veto. The candidate must
equal the official listed set. Additions require effective listing evidence;
removals require effective delisting evidence. Historical PIT records remain
nonshrinking. Explicit capture validation may read the immutable candidate
without writing; publication requires the locked operation context and marker.

The operation uses the existing PIT validate/publish and Market component
capture/replay/publish functions. Its journal is keyed by request SHA beneath
`data/private/cn_daily_maintenance/scope_transitions/`. While its active marker
exists, canonical scope/Market/PIT/Fundamental readers return
`SCOPE_TRANSITION_IN_PROGRESS`. Before PIT CAS, a failed run may restore exact
old scope only with unchanged PIT and Market pointers. After PIT CAS the marker
and veto remain, and the same exact request recovers forward. Predecessor files
are never rewritten. A conflicting request or drift fails closed.

History repair is restricted to missing symbol-date keys for the request's
additions, from each listing date through the target open sessions. It obtains
daily OHLCV, daily_basic and adj_factor through the existing provider/maintainer,
accepts only registered suspension evidence for absent bars, and uses
`MarketDataStore.upsert_bars` with the expected Market pointer. Existing keys are
not replaced. The current-date coverage is preserved by the normal historical
upsert contract. Full history audit and final scope/Market/PIT byte bindings must
pass before one readiness receipt can report Claude input ready.

Only verified terminal closure permits the existing exact-SHA veto archive/clear
operation. A token-only recovery, manual veto deletion, count/SHA edit, lower-level
legacy PIT refresh, or old generation overwrite is not a scope-transition path.
Terminal replay validates the same refs and returns NO_ACTION. Fundamental
promotion, Factor/Store/portfolio activation, holdings and trading are outside
this operation.

Use a clean detached checkout and the existing `system release-prepare` process
to obtain the actual new installed interpreter and release-input SHA. Never
edit the previous installed package or select dirty workspace code for a live
transition. Existing checkpoint bytes remain intact; a changed canonical scope
or Market/PIT binding requires a new rebuild binding rather than relabeling the
old checkpoint as current.

An obsolete per-run `daily_basic_coverage_boundaries.json` may independently
block the Fundamental consumer after Market/PIT closure. The explicit paired
`--retire-coverage-declaration-sha256 <exact-sha>` option archives only the fixed
canonical file, after verified transition readiness, if its sealed v2 content
has exclusively older cutoffs. It preserves exact bytes and a retirement receipt,
creates no replacement exemptions, and does not rebind old interval identities.
The new rebuild starts with the normal strict no-exemption default and may create
new evidence-backed declarations from its own raw capture. Receipt replay checks
the exact schema, invariant values, archived bytes and readiness ref. The option
is unavailable without the paired scope-transition request and its SHA.

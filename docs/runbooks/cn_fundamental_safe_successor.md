# CN Fundamental Safe-Successor Operations

Production Fundamental cutoff advancement uses the stable, unversioned
safe-successor commands. The ordinary live merge remains a compatibility path;
it is not a fallback because it can represent a partial provider result and
does not prove an operator-frozen predecessor or an append-only prefix.

## Acquisition and staging

`market fundamental-maintain --safe-incremental-successor` freezes the exact
Fundamental predecessor, market pointer, PIT pointer, membership, scope, and
history-audit bytes before acquisition. It writes only to an isolated staging
root and requires zero failed, malformed, conflicting, or paginated responses.

The source fileset retains exact raw response bytes and every normalized row.
It binds request-envelope and canonical-subject scopes separately, preserves
out-of-scope observations as audit evidence, and permits only the frozen
in-scope subject set to participate in canonical winner selection. Malformed
rows, ambiguous identities, scope collisions, count/hash drift, resource-gate
failure, or a changed in-scope fingerprint block staging.

The canonical subject set is the union of predecessor subjects, every delta
session's PIT-expected and observed-bar subjects, and the target full-A scope.
The operator proves the daily-basic keyset against canonical bars and
reason-coded non-bar classifications, freezes the predecessor prefix, and
derives only the open successor window.

An unpaired opaque balancesheet observation is never eligible for publication.
Use the exclusive live-source diagnostic first:

```text
market fundamental-maintain \
  --taint-analysis-dry-run \
  --allow-live \
  --universes full_a \
  --audit-run-root /absolute/private/new-run-root \
  ...the frozen predecessor/market/PIT/scope/history arguments...
```

Despite the name, this command calls the registered live provider. Its only
write-set is the private audit root; it cannot create a staging generation,
install a generation, advance a pointer, or authorize promotion. `PASS` proves
only that the deferred observation is target-bounded and non-reachable through
the requested cutoff. A terminal `BLOCKED` run id/root cannot later resume as
`PASS`.

An owner-approved append-first successor is an explicit modifier of this
phase, never a historical-winner fallback:

```text
market fundamental-maintain \
  --safe-incremental-successor \
  --append-first-successor \
  --historical-taint-failure-evidence /absolute/private/failure#ordinal \
  --successor-income-support TS_CODE@YYYYMMDD \
  --successor-financial-support TABLE:TS_CODE@YYYYMMDD \
  --allow-live \
  --universes full_a \
  ...the frozen predecessor/market/PIT/scope/history/staging arguments...
```

Support requests are bounded by exact subject and period and may seed only
hidden calculation state. They cannot create a predecessor winner, period or
daily suffix, or canonical row. The captured pre-cutoff keyset must equal the
actually consumed fallback read-set; absent, extra, cross-symbol, cross-period,
post-cutoff, or unused support blocks. Empty provider responses are sealed as
absence proofs, not synthesized data or allowlists. Registry, raw response,
failure evidence, capture binding, and immutable predecessor table refs are
replayed again at staging and promotion.

## Promotion

`market fundamental-promote --safe-incremental-successor` is read-only unless
`--execute` is explicit. Execution additionally requires an owner-only durable
journal root, an exact expected predecessor pointer SHA, and unchanged captured
market and PIT pointer bytes.

The promoter acquires market, PIT, and Fundamental locks in that order,
installs an immutable generation, advances the Fundamental pointer by CAS, and
performs exact post-write readback. Recovery uses only the journal and sealed
predecessor bytes. Any source, schema, canonicalization, hidden-PIT,
resource-budget, keyset, SHA, CAS, readback, or pointer-drift blocker leaves the
current healthy Fundamental pointer unchanged.

Path-backed staging and promotion use the same exact-byte custody boundary as
sealed source filesets. Each owner-only, regular, single-link Parquet file is
hashed from an `O_NOFOLLOW` descriptor before decoding, rewound and decoded
through that same descriptor, then rehashed and inode-checked after decoding.
Logical table equivalence never replaces the manifest's exact Parquet byte SHA.

Before the first provider call, the operator seals a resource receipt covering
available RAM, process RSS, `RLIMIT_AS`/`RLIMIT_DATA` headroom, physical RAM,
source capture, staging and canonical temp/final/orphan/rollback space, fsync
reserve, rolling free-disk protection, and a 25% margin. It repeats the receipt
after capture with exact source/table sizes. Request evidence is streamed;
records replay from exact raw bytes; aggregate support Parquet is deterministic
and sorted with at most 2,048 rows and 16 MiB per stream batch. Financial replay
holds at most one symbol's four-endpoint hidden state under one aggregate
per-symbol byte cap, while forecast and daily-basic replay remain batched.

## Derivation boundary

A safe successor is a mixed generation. The original seam remains the first
trusted predecessor cutoff; later successors append after the immediate parent
without rewriting an already published suffix. Pointer, manifest, readiness,
and the binding-aware loader must agree on:

```text
mixed=true
legacy_direct_reader_provenance=limited
binding_aware_research_ready=true
homogeneous_history_ready=false
```

These are hard methodology boundaries. A consumer that resolves a Parquet path
but discards the verified derivation binding cannot claim seam-aware or
homogeneous-history readiness.

Balancesheet `comp_type=7` may be retained only as opaque evidence when a
supported `comp_type` 1-4 peer has the same business key and identical complete
business projection after physical update dominance. It is never interpreted,
mapped, generalized, or admitted by symbol allowlist. Unpaired or conflicting
opaque observations remain a blocker for authoritative staging.

## Authority boundary

Fundamental maintenance does not activate System, Factor, Mainline, Dashboard,
Paper, portfolio, broker, order, execution, trade, or funds-transfer authority.
It must never recover through a retired numeric runtime, inferred source,
synthetic row, stale pointer, or ordinary live-merge fallback.

## Unified production age policy

Fundamental remains a mandatory, integrity-valid System source. Its pointer,
manifest, three canonical Parquet tables, provider evidence, historical
market/PIT/scope bindings, PIT availability dates, schemas, and exact hashes
must all replay. Missing, corrupt, future-dated, or self-binding-incomplete
evidence blocks production.

Daily forecast evidence is PIT-relative to each row: a non-null
`forecast_ann_date` must be a canonical date no later than that row's
`trade_date`, the Fundamental cutoff, and the System cutoff. Global cutoff
compliance alone does not admit a forecast published after its row date.

Snapshot age is different. The unified runtime uses the last known good
Fundamental generation under `ADVISORY_NO_FIXED_MAXIMUM`. It seals the snapshot
cutoff, current System cutoff, calendar-day age, exact open-session age, and
latest admitted `availability_date`. Age has no `fresh`/`stale` bucket, hidden
maximum, or automatic blocking threshold. LOW, W80, and the W75 control depend
only on exact current market, PIT membership, and exchange calendar inputs, so
Fundamental age cannot change their weights or readiness. Investment remains
`BLOCKED`, and mixed-history limitations remain visible.

An operator may attach an immutable, subject-bound
`system.fundamental_operator_veto` before generation assembly. The artifact is
VETO-only: there is no ALLOW or waiver artifact. A valid veto stops before an
operational generation, final authorization, prepared transaction, or CAS. A
veto created after activation does not rewrite active history; use a new
generation or the governed emergency suspension path.
The current VETO builder derives `actor_uid` and `os_actor` from the process
effective UID. Admission independently requires the sealed actor and the
securely opened VETO source owner to equal that effective UID. Historical
readback preserves the originally sealed actor without reinterpreting it as a
descendant process identity.

For routine maintenance, use a weekly incremental append as the default and a
monthly reconciliation/compaction review. A manual refresh remains appropriate
before a Fundamental-dependent decision or during concentrated disclosure
periods. This cadence is operational guidance only: it does not enable an
automation, call a provider, or publish a generation by itself.

The resource policy remains role-specific:

```text
ordinary JSON/evidence                     64 MiB
exact predecessor manifest                128 MiB
canonical Fundamental table Parquet       512 MiB compressed
canonical fundamental_daily work cells    256,000,000
other Fundamental table work cells        100,000,000
decoded Parquet batch                     2,048 rows / 16 MiB
```

The larger Parquet ceiling applies only to the exact daily, period, and
quarantine table roles reached through the request, manifest, receipt, and
System source-object closure. It does not widen JSON, provider evidence,
market/PIT/calendar, or generic Factor sources.
The daily cell limit is a deterministic full-table cardinality/work ceiling,
not a peak-memory allowance or an estimate of total scalar operations across
the two-pass semantic fingerprint.
Direct JSON replay uses one no-follow descriptor, a current-user-owned regular
single-link file with no executable or non-owner writable bits, stable
device/inode/mode/size timestamps, exact byte count, and exact SHA. No pathname
reopen is authoritative in this validation path.

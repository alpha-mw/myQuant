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

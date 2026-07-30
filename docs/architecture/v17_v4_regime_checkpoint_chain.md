# V17 v4 bounded causal regime checkpoint chain

## Status and boundary

`myquant.v17.v4.regime-evidence.v3` is an additive, research-only contract.
It does not replace or reinterpret Regime Evidence v1 or v2. The v1 schema,
v2 schema, v1 inference policy, and v2 producer remain byte-for-byte frozen.

V3 exists to remove two structural defects in the v2 predecessor chain:

- replay of the third contiguous v2 session exceeds the sealed 16-node closure
  budget, and then exceeds the depth budget if the node check is bypassed;
- a missed v2 publication permanently prevents later contiguous publication.

V3 has no Factor, portfolio, selector, governance, provider, broker, order,
execution, trade, promotion, formal activation, or production-default
authority. V5 consumption is a separate work package.

## Publication model

The publication mode remains
`PRIOR_SESSION_EFFECTIVE_NEXT_SESSION`. Inference remains
`FILTERED_CAUSAL`; smoothing and retrospective causal backfill remain
forbidden.

Only the fixed Regime Evidence v3 strategy/session slot finalizes a session.
Chain and segment anchors and state checkpoints are immutable prerequisites,
not completion markers. An unreferenced checkpoint is an orphan and cannot be
used to continue the chain or appear as an audit head.

The V3 chain starts from the policy-v2 bootstrap prior. It does not import a
v1 or v2 posterior. Migration of an old posterior is outside this contract.
The chain anchor is only the bootstrap seed: it binds `chain_id`,
`first_eligible_session`, the policy-v2 bootstrap prior, `global_seed`, and
the compact calendar-prefix commitment. It is not a finalized evidence record
and therefore has no `record_commitment`, `segment_accumulator`, or
`global_accumulator`.

## Deterministic paths and locking

One exact packaged inference policy v2 is admitted. A per-strategy lock covers
the fixed chain anchor, all content-addressed segment/checkpoint prerequisites,
and every fixed evidence slot for that strategy. There is no mutable head,
`latest` pointer, directory scan, glob, or fallback discovery.

The path grammar is:

```text
data/private/v17_v4_sources/regime_v3/<strategy_id>/chain_anchor.v1.json
data/private/v17_v4_sources/regime_v3/<strategy_id>/segments/<segment_id>.v1.json
data/private/v17_v4_sources/regime_v3/<strategy_id>/checkpoints/<effective_session>/<checkpoint_id>.v1.json
data/private/v17_v4_sources/regime_evidence/<strategy_id>/<effective_session>/regime_evidence.v3.json
data/private/v17_v4_sources/regime_v3/<strategy_id>/.producer.lock
```

All current policy, feature, transition, model, calendar, PIT, market, and
locator inputs are validated and re-read by exact SHA under the lock before
the first irreversible write.

## Session and segment state machine

Let `C` be the ordered sealed Shanghai open-session calendar, `F` the chain
anchor's first eligible session, `D` the current effective session, and `P`
the prior finalized evidence's effective session when one exists. `F`, `P`,
and `D` must be members of `C`.

For a finalized predecessor:

```text
M = [s in C where P < s < D]
```

For an anchor-only lineage:

```text
M = [s in C where F <= s < D]
```

`g = len(M)` is the missed-evidence count and is bounded by policy at 260.
The current observed-through session must be the open session immediately
before `D`.

The evidence ordinal counts finalized v3 evidence, not calendar sessions.
It starts at zero. Segment positions count finalized records since the most
recent segment reset and range from 0 through 63.

Phase precedence is normative:

| Condition | Phase | Segment action |
|---|---|---|
| no predecessor, `D == F`, `g == 0` | `GENESIS` | create segment 0, position 0 |
| `g > 0` | `RECOVERY` | create next segment, position 0 |
| `g == 0` and prior position is 63 | `ROLLOVER` | create next segment, position 0 |
| otherwise | `CONTIGUOUS` | retain segment, increment position |

`RECOVERY` takes precedence over natural rollover. Checkpoint-only artifacts
never change the ordinal or position.

## Recovery inference

All arithmetic uses the existing Decimal inference implementation, native
state order, 12-place quantization, round-half-even normalization, and
largest-remainder state-order tie break.

For each missing session:

```text
p = Q(normalize(p T))
```

Quantization `Q` is applied after every transition-only step. The current
feature is then incorporated exactly once:

```text
p = Q(normalize((p T) elementwise L_current))
```

For contiguous publication, `g == 0`, so only the current step is performed.
No missing-session likelihood is invented. Recovery records the exact ordered
missing-session list and resets the segment accumulator while continuing the
global accumulator.

## Calendar prefix commitment

Future calendar entries are never committed. The historical prefix is the
exact ordered list of sealed open sessions at or before `D`, and its final
item must equal `D`.

```text
H(canonical({
  "domain": "myquant.v17.v4.regime-calendar-prefix.v1",
  "open_sessions": <C entries <= D>,
  "policy_byte_sha256": <sha256>,
  "prefix_end_session": D,
  "prefix_length": <integer>,
  "strategy_id": <strategy_id>
}))
```

Every later checkpoint must reproduce the predecessor prefix exactly and then
extend it. Truncation, reorder, historical mutation, or a same-length
alternative fails closed. A future-only extension is permitted.

## Commitment algebra

`H` means SHA-256 of UTF-8 canonical JSON. Every payload below has exactly the
listed keys. SHA values are lowercase 64-hex strings.

```text
chain_id = H({
  "domain": "myquant.v17.v4.regime-chain-id.v1",
  "policy_byte_sha256": <sha256>,
  "policy_semantic_sha256": <sha256>,
  "strategy_id": <strategy_id>
})
```

```text
segment_id = H({
  "chain_id": <chain_id>,
  "domain": "myquant.v17.v4.regime-segment-id.v1",
  "segment_index": <integer>,
  "segment_start_session": <date>,
  "start_phase": "GENESIS" | "RECOVERY" | "ROLLOVER"
})
```

`record_core` contains exactly:

- version `myquant.v17.v4.regime-checkpoint-record.v1`;
- chain, strategy, and exact policy reference;
- evidence ordinal and phase;
- segment id, index, and position;
- observed-through and effective sessions;
- ordered missing sessions and their domain-separated digest;
- calendar prefix end, count, and SHA;
- null predecessor, or the prior finalized evidence/checkpoint identity,
  byte SHA, semantic SHA, global accumulator, segment id/index/position, and
  effective session;
- complete seven-field current feature, model, and transition artifact refs;
- sealed posterior and native hard state.

Current input byte and semantic hashes are included. Checkpoint/evidence
envelope IDs, their self-hashes, and volatile publication timestamps are the
only excluded envelope fields.

Serialized `calendar_prefix` is the compact object
`{prefix_end_session, prefix_length, prefix_sha256}`. The full open-session
prefix is held by the sealed calendar input and is not embedded in each
record. Serialized `prior_finality` is scalar-only and contains the prior
evidence/checkpoint IDs, byte SHAs, semantic SHAs, prior global accumulator,
prior segment id/index/position, prior finalized evidence ordinal, and prior
effective session. It never contains a prior segment artifact ref.

```text
record_commitment = H({
  "domain": "myquant.v17.v4.regime-record-commitment.v1",
  "record_core": <record_core>
})
```

```text
global_seed = H({
  "chain_id": <chain_id>,
  "domain": "myquant.v17.v4.regime-global-seed.v1",
  "policy_byte_sha256": <sha256>,
  "strategy_id": <strategy_id>
})
```

```text
segment_seed = H({
  "chain_id": <chain_id>,
  "domain": "myquant.v17.v4.regime-segment-seed.v1",
  "policy_byte_sha256": <sha256>,
  "previous_global_accumulator": <sha256 or null>,
  "segment_id": <segment_id>,
  "segment_index": <integer>,
  "segment_start_session": <date>,
  "start_phase": "GENESIS" | "RECOVERY" | "ROLLOVER",
  "strategy_id": <strategy_id>
})
```

The null previous global accumulator is allowed only for `GENESIS` at
ordinal zero.

```text
segment_accumulator = H({
  "domain": "myquant.v17.v4.regime-segment-accumulator.v1",
  "previous_accumulator": <segment_seed at position 0, otherwise prior segment accumulator>,
  "record_commitment": <record_commitment>,
  "segment_id": <segment_id>,
  "segment_position": <integer>
})
```

```text
global_accumulator = H({
  "chain_id": <chain_id>,
  "domain": "myquant.v17.v4.regime-global-accumulator.v1",
  "evidence_ordinal": <integer>,
  "previous_accumulator": <global_seed at ordinal 0, otherwise prior global accumulator>,
  "record_commitment": <record_commitment>
})
```

## Finality, forks, and crashes

Every non-genesis build requires explicit path/SHA pairs for the prior
finalized v3 evidence and its exact checkpoint. The producer fully replays
that evidence's bounded direct closure. Older lineage is represented only by
scalar commitments.

While holding the same strategy lock, the producer enumerates every sealed
open session between predecessor and current and checks each deterministic v3
evidence slot for absence. It does not scan a directory. If any later
finalized slot exists, a stale predecessor cannot create a fork.

Crash behavior:

- anchor-only: a future `RECOVERY` can continue from the bootstrap prior after
  proving all intervening evidence slots absent;
- segment/checkpoint-only: these are content-addressed orphans and are not
  final;
- evidence written before client readback: an exact retry replays the existing
  evidence and may complete after the wall-clock window;
- no evidence was written before cutoff: that old session cannot be finalized
  later.

New absent-slot publication uses whole-second UTC timestamps and inclusive
boundaries:

```text
abs(now_utc - created_at) <= 300 seconds
created_at >= observed-session Shanghai 15:00
created_at <= cutoff
Shanghai-date(cutoff) == D
```

Exact existing-slot retry skips only the wall-clock freshness check. It still
requires identical build arguments, lineage, source closure, and bytes.

## Replay and full audit

Daily replay receives explicit path/SHA inputs for the chain anchor, active
segment anchor, current checkpoint, current snapshots, and immediate prior
evidence/checkpoint when present. It verifies the prior evidence's complete
direct closure but never follows scalar commitments into older records.
Daily replay is therefore constant in lineage length, although source bytes
and market symbol count remain data-dependent.

Full audit receives:

- explicit chain-anchor and sealed-calendar path/SHA pairs;
- an ordered stream of explicit evidence, checkpoint, and segment-anchor
  path/SHA entries;
- an expected head evidence SHA/session;
- an audit-as-of session and cutoff.

There is no head discovery. The expected head must be the final stream item.
Audit recomputes calendar prefixes, phases, ordinals, both accumulators, and
deterministic evidence slots. An internal absent slot is legal only when a
later `RECOVERY` records it. A due missing tail after the trusted head produces
a fail-closed head/as-of mismatch, never a complete audit.

A slot is due after the policy-v2 local deadline
`23:59:59+08:00`, inclusive. Audit-as-of and every stream session must be a
member of the sealed calendar.

Audit budgets cover the calendar, anchors, segment anchors, evidence,
checkpoints, and recursively read current direct source closure:

```text
records                 <= 1,000
aggregate evidence bytes <= 128 MiB
per-record depth/nodes  <= 5 / 16
elapsed monotonic time  <= 120 seconds
```

Each record separately inherits the sealed policy's JSON, raw-byte, reference,
Parquet-row, and no-follow closure limits. Per-record caches are released, so
memory is bounded by one record's closure plus the at-most-1,000-row compact
audit summary and accumulator state. Full audit
proves deterministic
continuity relative to the supplied trusted head and calendar; it does not
establish external authenticity.

## Deployment and rollback quarantine

Sprint 1E-0A does not invoke the producer in any non-temporary strategy
workspace, even if a plausible input is discovered. Real inspection is
read-only. There is no migration, schedule, live artifact, pointer, cleanup,
or deletion.

Rollback means disabling the v3 commands or future scheduling while retaining
all immutable artifacts. An incorrect committed calendar prefix stops that
lineage; replacement or migration is a separate governed work package.

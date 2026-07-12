# Theme Protocol v2

Theme Protocol v2 is the deterministic, point-in-time Theme observer and formal
eligibility contract for the frozen v13.1 exception. It remains local and
offline. It does not call an LLM, a network provider, a broker, or a Web API.

## Runtime modes

- `market_observation` contains every scored theme.
- `tech_thesis_watch` contains technology / advanced-manufacturing themes, or a
  theme with an approved PE/VC thesis, that has not passed every formal gate.
- `formal_investable` is populated only when every formal gate passes and the
  explicit formal switch is enabled. The pre-ranking evaluator never produces
  this lane; only the post-control reconciliation artifact may do so.

The repository defaults are observer-only:

```text
THEME_PROTOCOL_V2_ENABLED=1
THEME_MEMBERSHIP_V2_ENABLED=1
THEME_MEMBERSHIP_V2_PATH=private/theme_knowledge/theme_membership.v2.jsonl
THEME_MEMBERSHIP_V2_REQUIRED=0
THEME_MEMBERSHIP_V2_EXPECTED_SHA256=
THEME_V2_FORMAL_ENABLED=0
THEME_V2_FORMAL_KILL_SWITCH=1
```

Setting a legacy `THEME_POOL_MIN_ADMITTED_THEMES` value above zero cannot force
admission. The formal pool is empty when no theme naturally passes. Rollback is
the explicit formal kill switch, which returns Theme v2 to observer-only and
emits `rollback_status`, `rollback_reason`, and an empty `formal_pool`.

Every protocol result includes a deterministic `protocol_hash`. The hash covers
the protocol version, taxonomy, thresholds, ranking weights, lifecycle, and
Markov tactical caps. Dashboard and joint replay gates should read that hash,
`status`, `formal_enabled`, `formal_kill_switch`, and `rollback_status`.

## Versioned contracts

Schemas are tracked in `quant_investor/themes/schemas/`:

- `theme_taxonomy.v2` defines parent / child nodes, aliases, mandate, tradable
  nodes, and supply-chain roles.
- `theme_membership.v2` supports multiple memberships per security, supply-chain
  role, revenue exposure or `null`, effective dates, and `available_at`.
- `theme_evidence_event.v1` records orders, capacity, certification, products,
  customer validation, policy, attention, and hard-kill evidence.
- `theme_state.v2` carries the six independent score / risk axes, lifecycle,
  lane, ranks, blockers, and persistence state.
- `pevc_thesis.v1` is the private PE/VC knowledge contract.

The packaged taxonomy starts with AI, semiconductors, optical interconnect,
commercial space, and humanoid robots, including their main supply-chain roles.
Industry mappings remain one membership source; they are no longer the only
source. Any valid secondary membership can qualify a security, independent of
the primary display label.

Formal prequalification requires at least one active `theme_membership.v2`
detail with `available_at <= as_of`. An `industry_map.v1` label is observation
metadata only and produces `pit_membership_missing`; it cannot prequalify a
theme. Legacy membership ID lists are retained only as diagnostics and are not
accepted as PIT evidence.

The canonical membership v2 loader is independent of the legacy concept path.
Observer mode may read an unpinned local mode-0600 store and labels it
`hash_unpinned_observer_only`. Formal mode additionally requires
`THEME_MEMBERSHIP_V2_REQUIRED=1`, an exact expected byte SHA-256, successful PIT
coverage, and a current active winning revision. Revision selection happens
before active/inactive filtering, so a latest tombstone cannot expose an older
active membership.

All membership, evidence, and thesis reads enforce `available_at <= as_of`.
Membership validity is half-open: `effective_from <= as_of < effective_to`;
the `effective_to` hand-over date belongs only to the successor revision.
Duplicate `(symbol, theme_id)` revisions are resolved deterministically from
PIT-known, already-effective records by `updated_at`; compatibility rows without
it fall back conservatively to `available_at/effective_from`, then
`membership_id` and canonical content hash. Canonical approval requires a valid
`updated_at`, and formal mode blocks if any loaded v2 revision lacks it.
Future-available or future-effective revisions cannot mask the revision that
was valid at the historical `as_of`.
Changing `THEME_METADATA_SYMBOL_LIMIT` changes only the `display` payload; full
machine maps remain untruncated and therefore produce the same candidate set.

## Ranking, gates, and lifecycle

The base rank is:

```text
45% attention + 35% industrial validation + 20% market confirmation
```

An approved, in-date PE/VC thesis can add at most `0.10` percentile rank inside
the PIT prequalified set. CandidatePool consumes this adjusted protocol rank,
not the legacy scanner rank. The prior cannot change eligibility or remove/offset
attention, industrial, market, crowding, valuation, stale-evidence, positive
edge / BUY, RiskGuard, or PortfolioConstructor blockers.

Crowding and valuation are independent availability-gated axes. A missing,
stale, or unverified axis is `null` and adds `crowding_unavailable` or
`valuation_risk_unavailable`; unknown risk is never coerced to zero.

The scanner/DAG output contains a `prequalified_pool` for research candidates.
Every such symbol remains a `theme_v2_prequalified_research_candidate`. After
Bayesian edge/BUY, RiskGuard, and PortfolioConstructor, call
`reconcile_theme_protocol_v2()` with explicit per-symbol boolean outcomes. The
resulting immutable `theme_formal_reconciliation.v1` artifact is the only source
of `formal_pool`; missing gates fail closed. It is hash-bound to protocol,
`as_of`, and the active PIT membership snapshot. A production-valid result must
be persisted under the configured ignored private directory with mode `0600`
and pass readback verification; disabled or failed persistence clears the
formal pool. Existing content is accepted only as an identical idempotent
readback, never overwritten.

Runtime formal reconciliation also consumes a fully verified canonical joint
manifest and explicit PortfolioConstructor Theme-cap proof (enabled, matching
protocol hash, valid Markov lane status, and no malformed diagnostic). The
current branch's canonical five-path producer is intentionally unavailable, so
`canonical_joint_replay_producer_not_implemented` blocks before any supplied
manifest is read. If the producer is implemented later, runtime reuses the full
dataset/split/seal/scenario/shadow/acceptance verifier; a minimal self-hashed
manifest is insufficient.

Lifecycle is:

```text
discovery -> warming -> validated_trend -> crowded -> cooling -> broken
```

Upward transitions require three distinct trade dates. Cooling requires two.
Dates count only when backed by the local trading-calendar artifact, are not in
the future, and are not weekends. Repeated evaluation on one date is
idempotent. Expiry/revocation of the last PIT membership or a PIT hard-kill
event immediately sets `broken`.

Formal prequalification also requires both 60-day and 120-day attention axes
and at least 95% long-horizon history coverage. Short windows remain visible in
observer mode but cannot be re-normalized into formal eligibility. Industrial
freshness is based only on positive, unexpired order/capacity/certification/
product/customer-validation evidence; a recent attention event cannot refresh
stale industrial evidence.

Non-technology tactical caps are returned with every evaluation:

| Markov state | NAV cap | Position cap |
| --- | ---: | ---: |
| 趋势上涨 | 15% | 2 |
| 震荡低波 | 10% | 1 |
| 震荡高波 | 5% | 1 |
| 趋势下跌 | 0% | 0 |

Before PortfolioConstructor, tactical limits are applied prospectively only to
active PIT memberships in the protocol `prequalified_pool`. Unknown membership,
mixed technology/non-technology mandates, and an unadmitted technology
secondary membership all fail closed into the non-technology lane. The
post-control reconciliation then rechecks both position count and aggregate NAV
weight. RiskGuard and PortfolioConstructor remain authoritative downstream.

## Private PE/VC knowledge workflow

`private/` is ignored by Git. The canonical default is
`private/theme_knowledge/pevc_theses.jsonl`; drafts are stored under
`private/theme_knowledge/drafts/`. Word, Markdown, JSON, and local Notion exports
can create drafts. Import never writes the canonical store.

```bash
python scripts/manage_pevc_knowledge.py init-key
python scripts/manage_pevc_knowledge.py import-draft thesis.docx --source-type word
python scripts/manage_pevc_knowledge.py import-draft notion-export.md --source-type notion_export
python scripts/manage_pevc_knowledge.py approve private/theme_knowledge/drafts/<id>.json \
  --expected-draft-hash <reviewed-sha256>
python scripts/manage_pevc_knowledge.py validate
```

Approval validates every required thesis field, verifies the reviewed draft
hash, sets canonical `available_at` to
`max(source_available_at, approval_recorded_date)`, and uses an explicitly
initialized mode-0600 HMAC key, chained ledger, archived reviewed draft, and
recoverable transaction WAL. Approved revisions are immutable; changed content
requires a new natural version. Reads revalidate signatures, chain, archives,
permissions, and `approved_at <= as_of`. Migration evidence is copied into a
content-addressed mode-0600 archive and revalidated on every read; approval and
availability are floored to the recorded Asia/Shanghai business date, so a
backdated draft cannot leak knowledge before approval. There is no
Notion API path: `notion_export` means a local export or structured local input.

## Membership v2 migration

No symbol/theme mapping is inferred. A trusted local symbol master is mandatory
before approval; without one the tool emits `coverage_blocked` and Theme formal
eligibility remains closed.

```bash
python scripts/manage_theme_membership_v2.py build-draft local-memberships.jsonl \
  --symbol-master local-symbol-master.json
python scripts/manage_theme_membership_v2.py approve <draft.json> \
  --expected-draft-hash <reviewed-sha256>
python scripts/manage_theme_membership_v2.py validate \
  --symbol-master local-symbol-master.json --as-of 2026-07-10
```

Draft and canonical outputs default to ignored `private/theme_knowledge/` paths,
are atomic and mode `0600`, and record source/master hashes plus
`mapping_inferred=false`.

## Offline protocol evaluation

```bash
python scripts/run_theme_protocol_v2.py \
  --theme-snapshot results/theme_snapshots/<snapshot>.json \
  --as-of 2026-07-10 \
  --markov-regime 趋势上涨
```

`--formal-enabled` alone is insufficient if the kill switch remains active or
downstream gate evidence is missing. This is intentionally fail closed.

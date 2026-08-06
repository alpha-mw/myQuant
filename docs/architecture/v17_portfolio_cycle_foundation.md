# ADR: V17 Portfolio-Cycle Read-Only Foundation

- Status: implemented foundation; operational portfolio cycle remains blocked
- Protocol: `myquant.v17.v4`
- Implementation basis: `11d485b322fd12eb95d465b8387030cf5dd33598`
- Prior documentation-review basis: `3aa82552f9577fd8fc27df03be3be1669ccd4b51`
- Parent source basis: `c8795fcf3be19476ba1dc2720e1665879950ccab`

## Decision

Introduce a bounded, read-only foundation for diagnosing whether the inputs to a
V17 portfolio decision are closed. Phase 1 provides exact identity and holdings
validation plus deterministic readiness documents. It does **not** implement a
full-A producer, publish an immutable mainline run, activate a pointer, mutate a
portfolio or paper ledger, evaluate learning metrics, or call a provider,
broker, order, execution, or trade surface.

The public diagnostic is:

```text
quant-investor portfolio cycle-status
```

It is a no-write command. It accepts only explicit canonical paths and exact
SHA-256 bindings; it never discovers a strategy, selects a newest directory,
converts a historical label into an ID, or falls back to CSV, Shadow, Forward,
cached, inferred, or synthetic operational data.

## Source-basis note

The architecture review first froze `3aa82552…`. Before Phase 1 implementation,
the same branch advanced to `11d485b…` through a focused test-maintenance commit;
the worktree was clean and the new commit did not add the missing portfolio-cycle
runtime. Phase 1 therefore uses `11d485b…` as its implementation basis and keeps
`3aa82552…` as review provenance. `c8795fcf…` is the parent commit, not an
additional capability source. These refs are not unioned with another branch,
installed package, historical worktree, or runtime artifact.

## Current source truth

At the Phase 1 basis:

- `QuantInvestor.run()`, `research run`, `market analyze`, and `market run` are
  read-only adapters over one explicit canonical strategy active pointer.
- No registered full-A decision producer exists.
- No registered deterministic `RiskGuard` or `PortfolioConstructor` exists.
- No governed high-level mainline publisher or owner-operated activation CLI
  exists.
- `MainlineStore` exposes low-level immutable-write/CAS infrastructure, not
  production publication authority.
- The historical label `aggressive_tech_manufacturing` is not a canonical V17
  strategy ID.
- No current-holdings resolver, paper simulator, manual-fill writer, or learning
  orchestrator existed before this foundation.

Accordingly, a green Phase 1 result means only that the read-only foundation has
validated the evidence explicitly supplied to it. It never means that the
business loop is operationally closed.

## Authority domains

Authority is deliberately non-transitive. Each domain must be reported and
authorized separately:

| Domain | Phase 1 behavior | What it does not grant |
|---|---|---|
| Git/release | freezes one source basis | runtime activation or artifact validity |
| strategy identity | validates one exact declaration | a general runtime alias or pointer selection |
| holdings baseline | resolves one exact pointer → manifest → Parquet closure | holdings mutation or trade authority |
| canonical data | remains unverified until exact-ref validators exist | maintenance, provider access, or data promotion |
| Factor active set | remains unverified until an exact active-set validator exists | registry/active-set mutation or mainline authority |
| risk policy | remains unverified until an exact policy validator exists | an implemented `RiskGuard` |
| portfolio policy | remains unverified until an exact policy validator exists | an implemented portfolio producer |
| pure producer | unavailable in Phase 1 | publication or activation |
| immutable publication | unavailable in Phase 1 | active visibility |
| mainline activation | unavailable in Phase 1 | holdings/paper/trade mutation |
| paper ledger | unavailable in Phase 1 | broker/order/execution/trade |
| learning | unavailable in Phase 1 | production rule or Factor mutation |
| broker/order/execution/trade | always false | any real-world action |

Phase 1 intentionally exposes no `--*-verified` boolean flags. A boolean cannot
prove artifact bytes or lineage. All non-foundation gates therefore stay false
with explicit blockers until later exact path+SHA+schema validators exist.

## Identity declaration

Schema ID:

```text
myquant.v17.v4.strategy-identity-declaration.v1
```

The declaration is canonical JSON, exact-byte SHA bound, and requires exactly:

```text
schema_id
protocol
historical_label
canonical_strategy_id
declared_by
declared_at
authority_kind
provenance
semantic_sha256
```

`canonical_strategy_id` is lowercase, path-safe, and hyphen-delimited. The
declaration preserves the historical label as migration evidence only. Runtime
commands continue to accept the canonical ID; the historical label never
selects a strategy, holdings path, mainline run, or fallback.

`authority_kind=owner_declaration` means an explicit declaration whose bytes and
provenance validate but whose cryptographic signing authority is not proven.
Phase 1 rejects `signed_attestation` because it has no signature verifier. A
future independently verifiable signing contract would require a new versioned
schema. File ownership alone is not proof of signing authority.

If both `--strategy-id` and a declaration are supplied, they are an equality
constraint. A mismatch blocks the diagnostic; neither value silently replaces
the other.

## Holdings closure

The resolver accepts only an explicit workspace-relative pointer path and its
exact byte SHA. It follows references already contained in validated documents:

```text
holdings pointer
  -> holdings manifest
    -> accounting policy JSON
    -> price-source JSON
    -> current-holdings Parquet ledger
```

It does not enumerate timestamped strategy records or choose by mtime. The only
directory enumeration permitted is bounded casefold-collision checking for each
explicit path component; it is a security check, not discovery or selection.

Schema IDs:

```text
myquant.v17.v4.holdings-pointer.v1
myquant.v17.v4.holdings-manifest.v1
myquant.v17.v4.current-holdings-ledger.v1
myquant.v17.v4.holdings-accounting-policy.v1
myquant.v17.v4.holdings-price-source.v1
```

The pointer binds the canonical strategy, update timestamp, and exact manifest
ref. The manifest binds:

- canonical strategy ID and account ID;
- `currency=CNY`;
- trade date, UTC `as_of`, valuation time, decision cutoff, and pointer time;
- exact accounting-policy, price-source, and ledger refs;
- contributed capital, cash, cost basis, market value, unrealized PNL, realized
  PNL, and NAV as scale-4 decimal strings.

Chronology is fail-closed:

```text
trade_date <= as_of.date
trade_date <= valuation_at.date
trade_date <= decision_cutoff.date
as_of <= valuation_at <= decision_cutoff <= pointer.updated_at
```

The CLI's required `--decision-cutoff` is a canonical UTC expected equality
constraint, not a second authoritative timestamp. A match sets
`decision_cutoff_verified=true`; a mismatch makes the holdings input invalid and
never overwrites the manifest.

### Parquet ledger

The exact stable-read byte buffer is parsed directly; the path is not reopened
after hashing. The ledger is bounded to 64 MiB by the exact reader, 10,000 rows,
and 32 columns. Its schema metadata must carry the ledger schema ID. Required,
non-null columns are:

| Column | Arrow type |
|---|---|
| `symbol` | `string` |
| `name` | `string` |
| `shares` | `int64` |
| `avg_cost` | `decimal128(20,4)` |
| `market_price` | `decimal128(20,4)` |
| `cost_basis` | `decimal128(20,4)` |
| `market_value` | `decimal128(20,4)` |
| `unrealized_pnl` | `decimal128(20,4)` |
| `realized_pnl` | `decimal128(20,4)` |

Symbols use `000000.SH|SZ|BJ` form and are strictly ascending and unique.
Shares are positive integers. Binary floating-point equality is never used.
Calculations use `Decimal`, scale 4, and `ROUND_HALF_EVEN`:

```text
row.cost_basis = shares * avg_cost
row.market_value = shares * market_price
row.unrealized_pnl = market_value - cost_basis
manifest totals = exact sums of corresponding rows
NAV = cash + total_market_value
NAV = contributed_capital + total_realized_pnl + total_unrealized_pnl
```

The exact accounting-policy artifact must freeze `currency=CNY`, `money_scale=4`,
`rounding_mode=ROUND_HALF_EVEN`, and
`capital_identity=NAV_EQUALS_CONTRIBUTED_CAPITAL_PLUS_REALIZED_PLUS_UNREALIZED`.
The exact price-source artifact must bind the same CNY currency, canonical source
ID, `as_of`, and `valuation_at` as the manifest. These policies are deliberately
narrow; changing them requires a versioned contract rather than a silent
default.

### Exact-read security

Every artifact path must be a canonical ASCII POSIX path relative to the
workspace. Exact reads reject traversal, absolute paths, backslashes, symlinks,
non-regular files, owner mismatch, modes other than `0600`, hard links, casefold
collisions, oversized input, byte-SHA mismatch, semantic-SHA mismatch, and
concurrent byte or metadata mutation. When an artifact is supplied, its
workspace must already exist and be owned by the current user. Reads create no
directory, lock, cache, receipt, or pointer.

## Readiness documents

Phase 1 defines two different documents to prevent the first-run dependency
cycle between “inputs ready” and “mainline active.”

### `decision-input-readiness.v1`

Schema ID:

```text
myquant.v17.v4.decision-input-readiness.v1
```

This is a pre-run diagnostic. It binds the verified identity and holdings
projections plus these exact gate keys in this fixed order:

```text
strict_cn_data
pit
fundamental
macro
release_calendar
factor_active_set
risk_policy
portfolio_policy
mainline_publisher
paper_simulation
learning_runtime
```

Each gate is `{verified: boolean, ref: exact-ref-or-null}`. A verified gate must
carry a valid exact ref; an unverified gate must carry `ref=null`. The Phase 1
CLI supplies no gate evidence, so all 11 gates remain false.

Status values:

```text
BLOCKED
FOUNDATION_VALIDATED
```

Truth table:

| Identity | Holdings | Gate evidence | State |
|---|---|---|---|
| missing/invalid | any | any | `BLOCKED` |
| verified | missing/invalid | any | `BLOCKED` |
| verified | verified and cutoff-matched | any | `FOUNDATION_VALIDATED` |

Gate failures remain in `blockers` even when foundation state is validated.
`FOUNDATION_VALIDATED` means only that identity and holdings closed; it is not
`business_ready` or `FULL_CYCLE_CLOSED`. It always carries
`business_ready=false`, `operational_authority=false`, and
`phase_capability=FOUNDATION_ONLY`.

### `public-cycle-status.v1`

Schema ID:

```text
myquant.v17.v4.public-cycle-status.v1
```

This is a post-run read-only diagnostic. It binds one accepted
`decision-input-readiness` document and one exact mainline resolution. Status
values are:

```text
BLOCKED
PUBLIC_CLOSURE_ACTIVE_FOUNDATION_ONLY
```

Truth table:

| Input foundation state | Exact mainline public closure | State |
|---|---|---|
| blocked/invalid | any | `BLOCKED` |
| foundation-validated | missing/blocked/lineage mismatch | `BLOCKED` |
| foundation-validated | active exact public closure with matching canonical strategy | `PUBLIC_CLOSURE_ACTIVE_FOUNDATION_ONLY` |

Even the second status remains foundation-only. It does not prove that a
producer exists, that a publication or activation action was authorized, or
that paper and learning are available.

### Blocker rules

Blockers are unique and sorted by UTF-8/ASCII byte order. The same input always
produces the same order. Missing is distinct from invalid. The readiness layer
uses stable codes for, at minimum:

```text
V17_STRATEGY_ID_UNCONFIRMED
IDENTITY_DECLARATION_INVALID
HOLDINGS_BASELINE_UNAVAILABLE
HOLDINGS_BASELINE_INVALID
STRICT_CN_DATA_UNVERIFIED
PIT_UNVERIFIED
FUNDAMENTAL_UNVERIFIED
MACRO_UNVERIFIED
RELEASE_CALENDAR_UNVERIFIED
FACTOR_ACTIVE_SET_UNAVAILABLE
RISK_POLICY_UNAVAILABLE
PORTFOLIO_POLICY_UNAVAILABLE
CAPABILITY_BLOCKED_V17_MAINLINE_PUBLISHER
PAPER_SIMULATION_UNAVAILABLE
LEARNING_RUNTIME_UNAVAILABLE
V17_MAINLINE_UNINITIALIZED
V17_MAINLINE_BLOCKED:<exact blocker>
```

Low-level `PORTFOLIO_CYCLE_*` validation exceptions are converted at the CLI
boundary into either a stable business blocker or a redacted JSON error. They
never authorize fallback behavior.

All readiness documents expose the fixed authority map below; every value is
false in Phase 1:

```text
broker
order
execution
trade
provider
mainline_write
factor_write
holdings_write
paper_ledger_write_authorized
```

Synthetic fixture results additionally require:

```text
synthetic_only=true
operational_authority=false
phase_capability=FOUNDATION_ONLY
```

Synthetic success proves validator behavior only and cannot upgrade any real
authority or current-state claim.

## CLI contract

```text
quant-investor portfolio cycle-status \
  --workspace-root <existing-root> \
  --historical-label <exact-historical-label> \
  --decision-cutoff <YYYY-MM-DDTHH:MM:SSZ> \
  [--strategy-id <canonical-id>] \
  [--identity-path <relative-path> --identity-sha256 <sha256>] \
  [--holdings-pointer-path <relative-path> \
   --holdings-pointer-sha256 <sha256>]
```

Rules:

- workspace root, exact historical label, and decision cutoff are required;
- historical label is used only to check identity-declaration equality and
  never selects a strategy or path;
- strategy ID is optional; omission produces a blocker unless an exact identity
  declaration establishes it;
- each path/SHA pair must be supplied together or omitted together;
- path arguments are canonical workspace-relative POSIX paths;
- SHA arguments are lowercase 64-character hex;
- decision cutoff is canonical UTC with seconds and `Z`, and must equal the
  holdings manifest cutoff exactly;
- all success or blocked stdout is one JSON document and nothing else;
- `BLOCKED` exits `2` as an expected business state;
- malformed arguments exit `2` through `argparse`, with stderr and no JSON
  stdout;
- security or internal failures exit `1` with a redacted JSON error and no
  traceback;
- the command never creates a missing workspace or writes within an existing
  one.

Existing `research run`, `market analyze`, `market run`, `market backtest`, Web,
and Python public behavior remains unchanged.

## Future separation

Phase 1 deliberately reserves three different future operations:

```text
pure deterministic producer
  -> immutable exact-once publication
  -> explicit owner-operated active-pointer CAS
```

The pure producer must have no filesystem or pointer side effects. Publication
may write an immutable closed run but must not activate it. Activation requires
a separate exact authorization, intended strategy/run, pointer path, expected
prevalue SHA, CAS, and readback. Failure preserves the prior pointer bytes.

Paper is a separate authority domain. A future successful paper append must
explicitly carry `paper_ledger_write_authorized=true` for that invocation while
broker/order/execution/trade, mainline, Factor, and holdings authorities remain
false. A schedule or feature flag is never sufficient authorization.

Learning is metric-specific. Each metric becomes available only when its exact
transitive sources close:

- 1/3/5/10-session return: authoritative open-session calendar and total-return
  labels;
- realized PNL: exact fills, fees, cash flows, and corporate actions;
- avoided drawdown: sealed rejected-action counterfactual;
- opportunity cost and replacement alpha: sealed benchmark/replacement policy;
- cash drag: exact cash-return convention;
- turnover and slippage: exact proposed/filled/rejected events and price refs.

Unavailable metrics remain `UNAVAILABLE` with blockers. Learning may emit an
immutable research proposal; it may not modify live rules, Factor state,
holdings, mainline pointers, automations, or trading permissions.

## Non-goals and stop conditions

Phase 1 stops before any operational mutation. It does not:

- restore retired `run_pipeline`, V2/V3 agents, funnel, optimizer, or learning
  modules;
- invent the canonical ID for `aggressive_tech_manufacturing`;
- scan historical holdings or use CSV fallback;
- activate or mutate a Factor set;
- implement placeholder scoring, risk, or portfolio policies;
- publish or activate mainline output;
- create paper/manual fills or learning records;
- call live Tushare, yfinance, an LLM/provider, broker, order, execution, or
  trade surface.

Operational work must stop until the owner supplies exact identity and holdings
artifacts, an accounting policy, an activated Factor set, deterministic scoring,
risk and portfolio policy artifacts, and separate publication/activation
authority.

## Verification

Focused Phase 1 acceptance includes:

```text
pytest -q tests/unit/test_portfolio_cycle_identity_holdings.py
pytest -q tests/unit/test_portfolio_cycle_readiness.py
pytest -q tests/unit/test_portfolio_cycle_cli.py
pytest -q tests/unit/test_v17_mainline_runtime.py \
  tests/unit/test_v17_public_python.py \
  tests/unit/test_v17_public_cli.py \
  tests/unit/test_v17_public_web.py
git diff --check
```

Coverage must include no-write snapshots, path/SHA pair validation, both status
truth tables, synthetic authority non-upgrade, stable-read mutation, permissions,
symlink/hardlink/traversal/casefold/size failures, Parquet schema and Decimal
identities, chronology, deterministic blocker order, and existing public CLI
regression. PyArrow absence or incompatible runtime behavior is a blocker; Phase
1 must not upgrade dependencies ad hoc.

# Daily Investment Intelligence Compilation

`quant-investor research compile-daily` is the stable, offline, research-only
composition path from one exact Factor production head into source-bound
Industry/Theme gates and five-state Investment Decisions. It does not restore
the retired version-named Intelligence package or any versioned CLI.

## Authority boundary

The command returns one canonical machine-JSON line and writes nothing. It
cannot call a provider or model, publish a System or Mainline generation,
construct a portfolio, write a Paper ledger, mutate Strategy Record Store or
holdings, connect a broker, create an order, execute, or trade.

Every emitted artifact remains:

```text
research_only = true
production = false
run_state = INACTIVE
authority.* = false
```

## Required sequence

```text
verified Factor production head + exact LOW/W80 OPEN observations
  -> effective-dated daily_research_policy
  -> deterministic factor_research_rank
  -> exact SW2021 Industry projection
  -> exact DC/registered-TDX-fallback Theme membership projection
  -> decision_context
  -> five-state investment_decision
  -> research_evaluation
  -> evidence_bundle
```

The command consumes a canonical exact request below `--workspace-root` and an
expected request SHA. LOW/W80 observations are separate workspace-relative
path/SHA inputs. The Factor head is copied under the Factor lock from the exact
expected pointer; the pointer is rechecked under the lock after compilation.
Any pointer, generation, signal, symbol-set, observation, PIT, Market, date, or
company-keyset drift blocks the whole command.

## Policy is mandatory

There are no implicit policy defaults. `daily_research_policy` must seal:

- explicit research strategy identity and effective interval;
- exact LOW/W80 IDs, `HIGHER_IS_BETTER` directions and weights;
- average-tie zero-to-one cross-sectional percentile algorithm;
- missing/nonfinite blocker, minimum cohort, exact pool size, pool-boundary
  tie-break and final ordering; the fixed safety ceiling is 200 companies;
- provider-qualified technology Theme IDs and provider precedence;
- `RESEARCH_APPROVED` and `PAPER_CANDIDATE` thresholds;
- Fundamental freshness policy `ADVISORY_NO_FIXED_MAXIMUM`.

The owner-approved prospective Phase A policy is code-owned and published only
through `research policy-publish`. It uses:

```text
strategy_id = aggressive_tech_manufacturing
created_at = 2026-08-21T16:00:00Z
effective_signal_date = 20260822
pool_size = 100
minimum_cohort = 3000
LOW/W80 = 0.5 / 0.5
RESEARCH_APPROVED = 0.80
PAPER_CANDIDATE = 0.90
technology_policy_state = UNCONFIGURED
technology_theme_ids = []
```

`UNCONFIGURED` is not equivalent to an allowlist that rejects every company.
It may be used only to publish the Factor Top100 research pool. Theme projection
and `compile-daily` reject it. A later ACTIVE allowlist must be a distinct,
later-effective immutable policy revision; v1 is never rewritten.

The compiler does not infer the System strategy identity from a historical
Strategy Record directory. A test or implementation-smoke policy is not an
owner policy and cannot be used for an investment conclusion.

## Source projections

`project_tushare_industry_source` replays the exact stable SW2021 taxonomy and
membership plans, captures, and all partition documents. Exactly one effective
L3 membership is `AVAILABLE`; none is `UNMAPPED`; conflicting memberships are
`AMBIGUOUS`. `stock_basic.industry`, company names, and inferred mappings are
forbidden.

`project_tushare_theme_source` replays the exact DC plan/capture/partitions and
uses TDX only for the exact fallback keyset returned by the registered DC
fallback rule. Provider membership is `MEMBERSHIP_ONLY`: it can pass or reject
the technology hard gate but cannot fabricate revenue or economic exposure.
Complete empty membership is `NO_MEMBERSHIP`. A company needs a separate exact
`theme_assessment` before Theme becomes `AVAILABLE` in Decision.

## Fundamental, hypothesis, and risk

This first milestone deliberately does not accept caller-supplied Theme
exposure, Fundamental scores, hypothesis status, risk status, or generic
supporting artifacts. Provider Theme membership can pass or reject the
technology hard gate, but it is not economic exposure. Decision therefore
remains `INSUFFICIENT_EVIDENCE` and the evidence bundle remains `BLOCKED` until
separately reviewed deterministic producers with deep source replay exist.

Fundamental freshness remains disclosed under
`ADVISORY_NO_FIXED_MAXIMUM`; the compiler does not invent a same-day rule or a
Fundamental score.

## CLI

```bash
quant-investor research compile-daily \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --request <workspace-relative-canonical-request.json> \
  --expected-request-sha256 <exact-sha256>
```

Publish the code-owned immutable Phase A policy:

```bash
quant-investor research policy-publish \
  --workspace-root /Users/maxwell/mySpace/myQuant
```

Publish one eligible immutable Factor Top100 pool:

```bash
quant-investor research pool-publish \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --request <workspace-relative-canonical-request.json> \
  --expected-request-sha256 <exact-sha256>
```

`pool-publish` derives the rank itself from the expected Factor pointer, exact
LOW/W80 observation path/SHAs and the exact v1 policy path/SHA. It does not
accept a rank, selected symbols, output path, manifest or receipt. A signal
before `20260822` fails before any pool-store directory is created.

The immutable pool root is strategy-scoped:

```text
results/intelligence/research_pool/aggressive_tech_manufacturing/YYYY-MM-DD/
  factor_research_rank.json
  manifest.json
  publish_receipt.json
  selected_symbols.json
```

Publication uses an owner-only same-filesystem sibling staging directory and a
native atomic no-replace directory rename. There is no active/current/latest
pointer or mtime resolver. Exact replay returns `NO_ACTION`; any different or
unsafe existing closure blocks without replacement.

Provider capture, scheduled routing, System assembly/activation, Mainline
candidate publication, I6 portfolio construction, and Paper execution remain
separate later phases. The policy and pool writers own only their exact
research-only roots.

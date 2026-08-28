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

Maxwell approved the ACTIVE Theme seed after the 2026-08-26 close. The immutable
v2 policy is therefore prospective from Factor signal date `20260827`; it never
relabels the 20260824 pool:

```text
primary provider = TUSHARE_DC
fallback provider = TUSHARE_TDX
fallback rule = ONLY_REGISTERED_DC_FALLBACK_COMPANY_KEYSET
technology policy state = ACTIVE
primary DC IDs = 11 owner-approved seed themes
TDX aliases = six exact-name mappings plus two explicit owner-approved aliases
```

The corresponding `theme_governance_policy` also fixes seven top-level domains
and the economic-exposure states `HIGH / MEDIUM / LOW / UNVERIFIED`.
Membership is explicitly not economic exposure. Until annual/interim reports,
company announcements, IR records, revenue/product/customer, order/capacity or
capex evidence is replayed, a Theme-pass company remains `UNVERIFIED` with
`ECONOMIC_EXPOSURE_SOURCE_REQUIRED` and cannot become a formal Decision.

The compiler does not infer the System strategy identity from a historical
Strategy Record directory. A test or implementation-smoke policy is not an
owner policy and cannot be used for an investment conclusion.

### Source-bound exposure and Fundamental MVP

Maxwell approved the first research-only producer policy on 2026-08-28. Theme
membership never supplies an exposure level. A company can receive one only
from an exact annual report, interim report, company announcement, IR record or
revenue-structure file that quantifies Theme-related revenue share:

```text
HIGH       revenue share >= 30%
MEDIUM     10% <= revenue share < 30%
LOW        0% < revenue share < 10%
UNVERIFIED no positive quantitative revenue share
```

Product, customer, order, capacity and capex evidence may support the research
narrative but cannot alone upgrade the level. `LOW` is a hard research veto and
cannot become `PAPER_CANDIDATE`. Every evidence artifact binds a workspace-
relative source path, source SHA, availability time and exact page when
applicable.

The Fundamental MVP reads one exact registered, binding-aware and Gate-2-passed
`fundamental_daily.parquet`. It computes full-cohort average-tie zero-to-one
percentiles for ROE, ROA, inverse debt/assets, OCF/profit, FCF/profit, net-profit
growth, forecast revision and FCF/price. The five components are equal-weighted
at 20%; minimum coverage is 60%; `industry_cycle` remains explicitly missing.
The registered snapshot cutoff is disclosed but is not mechanically required
to equal the Factor signal date. Missing metrics are never imputed.

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

Publish the later-effective owner Theme v2 bundle:

```bash
quant-investor research theme-policy-publish \
  --workspace-root /Users/maxwell/mySpace/myQuant
```

This publishes `theme-governance.v1.json` first and `v2.json` second. An
interruption after the governance leaf is inert; exact replay completes or
returns `NO_ACTION`. The v1 bytes are never modified.

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

For `signal_date >= 20260827`, the request must instead bind the exact v2
policy path/SHA. The writer accepts only the code-owned v1 or v2 bytes. The v2
pool remains Factor-only until same-date DC/registered-TDX-fallback source
replay completes; its manifest says `PENDING_SOURCE_REPLAY`, never a fabricated
technology shortlist.

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

### PROJECT_ENV Theme replay

After one v2 Top100 is immutably published, the release-owned Theme replay
entrypoint builds the exact ASCII-sorted DC plan from that pool, captures DC as
primary, derives the registered fallback company keyset, and captures TDX only
for that keyset:

```bash
<installed-python> -I <clean-release-repository>/scripts/operations/run_cn_theme_replay.py \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --expected-import-root <installed-root> \
  --selected-symbols results/intelligence/research_pool/aggressive_tech_manufacturing/<YYYY-MM-DD>/selected_symbols.json \
  --expected-selected-symbols-sha256 <exact-sha256> \
  --policy results/policies/research/aggressive_tech_manufacturing/v2.json \
  --expected-policy-sha256 <exact-sha256> \
  --allow-live
```

The entrypoint reuses installed `read_project_env_token` and reads only the
owner-controlled workspace `.env` key `TUSHARE_TOKEN`. It never sources the
file, reads Keychain, logs/hashes/persists the token, retries a provider call,
or creates a second membership path. Existing exact capture roots replay with
zero network calls. A DC registry failure blocks before TDX; incomplete DC
company partitions alone form the registered TDX fallback keyset. Capture
completion remains membership-only and cannot upgrade economic exposure. The
TDX member endpoint may return industry or broad-index identities in addition
to concepts; they remain sealed in raw partitions, while projection admits
only IDs present in the exact same-date captured TDX concept registry.

## 09:45 morning strategy

`research morning-strategy` extends the same stable research lane; it does not
create a second data or investment system. The automation first captures one
credential-free Sina response with `scripts/capture_sina_cn_quotes.py`, then
runs `PREFLIGHT`, produces the Codex research narrative, and runs `SEAL`.

Required core closure:

```text
previous-trade-date Factor VERIFIED / ACTIVE / READY
exact LOW/W80 OPEN observations
Store-v3 registered pointer and active_closure
09:30-09:46 Asia/Shanghai Sina capture with exact raw SHA
installed/scheduler origin verified by the surrounding automation
```

An absent same-date Top100, Macro blocker, Fundamental partial state, Theme
economic-exposure gap, or benchmark-relative tail is auxiliary and may produce
`PARTIAL`; it cannot be silently filled. Stale Factor, invalid Store,
unavailable quote, wrong time/date, unsafe output, SHA drift or authority drift
is a core blocker.

The deterministic output and machine receipt are:

```text
results/operations/morning_strategy/CN/<YYYYMMDD>/0945-strategy.md
results/operations/morning_strategy/CN/<YYYYMMDD>/0945-run.v1.json
```

The receipt binds previous trade date, Factor pointer, observations, Store
pointer, quote request/response/raw SHA, optional Top100 manifest, output SHA,
core/auxiliary blockers and false broker/order/execution/holdings-mutation
flags. Exact replay is `NO_ACTION`; different bytes conflict.

The 20:20 `research morning-cutover` receipt separates
`core_production_status`, `holdings_status`, and `auxiliary_status`. Its state
machine is:

```text
EVENING_PRIMARY
  -> eligible core -> DUAL_RUN
     (primary automation 09:45 + temporary 21:00 fallback automation;
      Dashboard 21:30 retained)

DUAL_RUN
  -> two successful real 09:45 receipts -> MORNING_PRIMARY

MORNING_PRIMARY
  -> missing/invalid current 09:45 receipt -> DUAL_RUN fallback resumed
```

The retired one-time 20:40 task is not part of this path. Scheduler updates
must preserve the pre-update automation configs, use Codex `automation_update`,
read back Shanghai slots, and roll back on any partial update. The source code
only seals the deterministic eligibility/action receipt.

A single RRULE cannot safely encode both 09:45 and 21:00 without creating a
cross-product of hours and minutes. During `DUAL_RUN`, the existing `automation`
ID is the 09:45 primary and a distinct temporary `cn-evening-review-fallback`
retains the former 21:00 prompt. Promotion pauses that fallback; rollback
resumes it. No duplicate data DAG is created.

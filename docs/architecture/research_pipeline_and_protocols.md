# Research Pipeline And Protocols

This page distinguishes the current callable V17 surface from the upstream
producer topology that remains a governance target. The distinction is
load-bearing: public command names such as `market run` are compatibility names
and do not prove that a decision DAG was executed.

## Implemented public surface

`quant-investor research run`, `quant-investor market analyze`, and
`quant-investor market run` all resolve the same exact strategy active pointer.
The `QuantInvestor` Python facade uses the same reader. The repository has no
public Web reader.

```text
exact strategy active pointer
  -> exact immutable mainline run
  -> exact formal / portfolio / source closure
  -> myquant.v17.v4.mainline-public-run.v1
```

The reader never builds a run, scans for a recent artifact, selects another
protocol, falls back to Shadow, or writes a bootstrap. Mainline backtesting is
unsupported. The current repository has no public production publisher or
activation CLI.

## Implemented research surfaces

### V4 Forward / Shadow

`quant-investor-v17-v4 run-forward` processes an exact, content-bound request
and produces research-only Forward evidence. Shadow artifacts cannot advance a
mainline pointer or be returned as the public run.

### I0 Investment Intelligence

`quant_investor.intelligence` is a deterministic, research-only library. It
accepts explicit V4 observation and source closures and provides:

- source-bound Evidence records;
- Bayesian likelihood diagnostics;
- a three-layer causal Regime filter;
- availability-aware branch Fusion;
- falsifiable Hypotheses;
- immutable append-only Memory values and receipts.

It has no provider, model, selector, portfolio, broker, order, execution,
trade, persistence-writer, or active-pointer authority.

### R2.2 Forward Research Evaluator

`quant-investor-v17-v4 research-evaluate` accepts one exact canonical request,
replays its V4 and I0 inputs, and emits one canonical JSON envelope to stdout.
It may return a Memory Append Proposal, but it does not persist that proposal.
There is no V17 I0/R2.2 daily scheduler, request generator, daemon or automatic
paper-portfolio adapter in the current tree.

### I1 Investment Decision Intelligence

`quant_investor.intelligence.decision` is a deterministic, content-addressed,
library-only decision layer. It accepts the complete I0 replay closure and may
also accept one exact R2.2 request path and byte SHA-256. It rebuilds I0 and,
when supplied, R2.2 before it admits the research context; it never trusts a
summary-shaped or merely resealed receipt.

```text
exact V4 / I0 replay + optional exact R2.2 replay
  -> Investment Decision Context
  -> four-dimensional Risk Assessment
  -> THESIS_INVALIDATED | INSUFFICIENT_EVIDENCE | WATCHLIST
     | RESEARCH_APPROVED | PAPER_CANDIDATE
  -> deterministic Investment Memo
  -> append-only Decision Discipline Chain
  -> optional external-review Paper Intake Proposal
```

The five outcomes are research states, not actions. `PAPER_CANDIDATE` means
only that a replayed decision passed the stricter eligibility gates for an
external paper-review workflow. It does not select a stock, admit a holding,
construct a portfolio, set a weight or authorize an order or trade.

I1 has no public CLI, Web route, scheduler, daemon, persistence or Memory
writer, Paper adapter implementation, selector, portfolio, provider, model,
broker, order, execution or trade authority. It does not modify the existing
Memory contract or R2.2 Memory Proposal. See
[V17 I1 Investment Decision Intelligence](v17_i1_investment_decision_intelligence.md)
for the closed replay, state precedence, memo and discipline contracts.

### I2-I6 Investment Intelligence v2

`quant_investor.intelligence_v2` implements the next research-only layers while
keeping I0/R2.2/I1 byte-frozen:

```text
B0 readiness + Quant v5
  -> I2 Industry
  -> I3 Theme
  -> I4 Fundamental Profile
  -> I4.5 same-closure Decision v2
  -> I5 skeptical open-Web advisory review
  -> I6 research portfolio / paper / graduation
  -> signed research-mainline publication capability
```

Industry/Theme identity, score and risk policies, admission and portfolio
constraints remain deterministic and owner-sealed. I5 may search the public Web
without a domain whitelist, but citations are only leads until a local
DNS-pinned collector validates the exact HTML. The private committee has no
tools and cannot change Decision state, risk, veto or admission. AI rank and
capital influence are capped at 10% with deterministic fallback.

I6 artifacts are research/paper evidence, never broker holdings or executable
orders. The marked-run reader fail-closes on an invalid v2 closure, sidecar,
legacy projection, Ed25519 permit or quarantine marker. The current publication
boundary does not yet persist every closure-dependent builder's full validation
capsule and does not provide a formal publisher or an owner-confirmed CAS
coordinator, so it remains `CAPABILITY_PARTIAL`. It does not sign with, load or
store an owner private key. No pointer is written without a later exact-target
display and fresh user confirmation.

### Tushare 10,000 governed source path

The v2 source package reuses the official V4 HTTPS transport in strict-decimal
mode. It separates a dry-run capability probe, same-as-of Fundamental VIP
shadow acquisition, exact v3/VIP reconciliation, isolated staging and the
journaled Fundamental-only CAS. Physical network attempts, including retries,
must satisfy `vip_attempts * 10 <= baseline_provider_calls_attempted`.

I2 consumes sealed SW2021 taxonomy and membership partitions. I3 uses complete
DC snapshots first and TDX only for a sealed DC-incomplete company keyset.
Neither compiler permits display labels or diagnostics to become identity or
Decision authority. `MarketRiskProjection.v1` accepts only same-session daily,
daily-basic, suspension and price-limit refs and can only tighten I6 risk.

The Fundamental promotion authorization does not authorize a V17 active
pointer write. See
[Tushare 10,000 Investment Intelligence Flow](tushare_10000_investment_intelligence.md).

## Causal Regime contract

The current Markov implementation is under
`quant_investor/intelligence/regime/`, not a production overlay configured by
environment variables.

The fixed layers are:

- Market: `BULL`, `RANGE`, `HIGH_VOL`, `BEAR`;
- Industry: `EARLY_EXPANSION`, `EXPANSION`, `PEAK`, `DECLINE`, `RECOVERY`;
- Theme: `EMERGING`, `ACCELERATING`, `MAINSTREAM`, `CROWDED`, `DECLINING`.

Each layer performs exactly one forward filter step:

```text
predicted[j] = sum(previous[i] * transition[i,j])
posterior[j] = predicted[j] * emission[j] / normalization
selected     = maximum posterior with fixed-domain tie break
```

Every posterior-driving source must belong to the exact observation closure.
There is no backward smoothing, parameter fitting, hidden history discovery,
persistence writer, position cap, risk overlay or portfolio mutation.

R2.2 accepts an optional Regime binding per origin. If supplied, the Input,
Evidence and Receipt must be content-bound, replay exactly, use only authorized
sources, and satisfy `input.available_at <= receipt.timestamp <= origin cutoff`.
The aggregation consumes selected states only; missing bindings stay explicit
and invalid bindings block the evaluation.

## Implemented producer contracts and operationally blocked inputs

The v2 library implements governed producer contracts for this topology:

```text
strict CN Parquet + PIT membership
  -> deterministic candidate set
  -> Quant / Fundamental evidence + Macro context
  -> deterministic risk and portfolio controls
  -> immutable mainline run
  -> expected-prevalue CAS activation + exact readback
```

This remains distinct from the current public read commands and is not evidence
that a real run can be published. Real Factor v5 prospective evidence,
Industry/Theme catalogs, owner policies, ZDR capability and matured paper
outcomes may be absent; missing authority keeps the candidate blocked. The
following invariants remain mandatory:

- deterministic controls remain authoritative;
- advisory model output cannot alter admission or limits and can affect only
  admitted-set ordering within the explicit 10% rank/capital bounds;
- missing or invalid evidence is explicit, never silently replaced;
- only an exact active pointer grants public visibility;
- code deployment and operational activation remain separate actions.

The intended evidence roles remain asymmetric: Quant and Fundamental may enter
symbol likelihood; Macro is context/prior and cannot vote as a third symbol
likelihood. Fundamental evidence must prove exact generation provenance or
remain neutral/unavailable. These are producer requirements, not claims that a
callable public producer exists in this repository.

## Reporting and bucket rules

Any future structured producer should keep coverage, diagnostics and investment
risk separate:

- `investment_risks` may affect investability or governed risk limits;
- `coverage_notes` describe availability and provider gaps;
- `diagnostic_notes` describe engineering failures and degradation.

Coverage and diagnostic notes cannot directly become portfolio weights. A
read-only narrator or advisory review model may explain a structured result but
cannot mutate it.

## Standalone legacy automation

`quant_investor/automation/` is a restored, incomplete legacy lane. It has no
public V17 entrypoint and retains lazy references to modules not present in the
current tree. It is neither the public V17 reader nor the I0/R2.2 research loop,
and it grants no mainline authority. Its repair, publication or removal requires
a separate compatibility audit.

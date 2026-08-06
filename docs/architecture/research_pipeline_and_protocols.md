# Research Pipeline And Protocols

This page distinguishes the current callable V17 surface from the upstream
producer topology that remains a governance target. The distinction is
load-bearing: public command names such as `market run` are compatibility names
and do not prove that a decision DAG was executed.

## Implemented public surface

`quant-investor research run`, `quant-investor market analyze`, and
`quant-investor market run` all resolve the same exact strategy active pointer.
The `QuantInvestor` Python facade and `GET /api/research/{strategy_id}` use the
same reader.

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

## Governance target for an upstream producer

An eventual governed producer is expected to close this topology:

```text
strict CN Parquet + PIT membership
  -> deterministic candidate set
  -> Quant / Fundamental evidence + Macro context
  -> deterministic risk and portfolio controls
  -> immutable mainline run
  -> expected-prevalue CAS activation + exact readback
```

This is a normative topology, not the implementation behind the current public
read commands. A future producer must keep the following invariants:

- deterministic controls remain authoritative;
- advisory model output cannot alter candidates, limits or weights;
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

# V17 Sprint I0 Investment Intelligence Architecture

## Purpose and scope

Sprint I0 restores an additive, research-only intelligence layer between V4
Forward Observation artifacts and future evaluation work. It does not replace
or modify the V4 observation runtime, the V5 baseline, the current research CLI,
public Python/Web surfaces, factor governance, portfolio construction or any
execution surface.

The package root is `quant_investor/intelligence`. Its cross-layer dependencies
are limited to read-only V4 canonical/schema and typed artifact validators.
There is no provider, model, broker, order, trade, selector, portfolio or
filesystem writer in the package.

## Runtime flow

```text
exact V4 Session + Run + explicit typed Observation refs
            + exact recursive source closure refs
                       |
                       v
             Observation Evidence Bundle
                       |
             source-bound Evidence records
                       |
       +---------------+----------------+
       |               |                |
       v               v                v
   Bayesian       3-layer Regime   Quant/Fundamental
       |               |              branches
       +---------------+----------------+
                       |
          availability-aware Fusion
                       |
          falsifiable Hypothesis set
                       |
          append-only Memory hash chain
                       |
          Research Intelligence Receipt
```

Every I0 artifact is deterministic and content addressed. Canonical JSON uses
the V4 canonicalizer, Decimal arithmetic uses precision 50, output decimals use
12 places and round-half-even, and set-like arrays have an explicit ASCII order.

## Bayesian contract

For prior probability `p`, the engine computes prior odds `p / (1 - p)`. Each
evidence record has a direction, likelihood ratio `LR` and strength `s`. Its
effective likelihood is:

```text
effective_LR = 1 + s * (LR - 1)
combined_LR  = product(effective_LR in evidence_id order)
posterior    = prior_odds * combined_LR / (1 + prior_odds * combined_LR)
confidence   = 1 - product(1 - s)
uncertainty  = 1 - confidence
```

Positive evidence requires `LR > 1`; negative and contrary evidence require
`0 < LR < 1`; neutral evidence requires `LR = 1`. Evidence is source-bound,
timestamped, independently content addressed and rejected when its availability
time is after the receipt cutoff.

## Regime contract

The three required state domains are fixed:

- Market: `BULL`, `RANGE`, `HIGH_VOL`, `BEAR`.
- Industry: `EARLY_EXPANSION`, `EXPANSION`, `PEAK`, `DECLINE`, `RECOVERY`.
- Theme: `EMERGING`, `ACCELERATING`, `MAINSTREAM`, `CROWDED`, `DECLINING`.

Each layer performs exactly one forward Markov filter step. The posterior-driving
previous distributions, transition matrices and emission likelihoods first enter
a content-addressed `RegimeInput` carrying observation time, availability time
and exact source refs. Runtime construction admits those source refs only when
they are part of the Observation bundle's recursively verified closure.

```text
predicted[j]   = sum(previous[i] * transition[i,j])
posterior[j]   = predicted[j] * emission[j] / normalization
selected_state = maximum posterior, fixed-domain order as tie break
```

There is no backward pass, future observation input, smoothing, persistence
writer, position cap or risk overlay. The receipt contains the selected state
for every layer, full posterior maps, selected-state transition probabilities
and exact evidence refs.

## Branches and fusion

The required Quant branch records factor score, RankIC, ICIR, exposure and
coverage. The Fundamental branch records quality, earnings, valuation and
industry position. Industry and Theme are reserved optional branch types.

Quant is mandatory. For every supplied branch, evidence mass is:

```text
mass = availability * confidence * reliability
normalized_weight = mass / sum(mass)
```

The output `research_confidence_score` is the weighted research score multiplied
by weighted branch confidence and availability coverage. It is not a trade
score, portfolio weight, selection score or governance decision. No fixed 50/50
fallback exists.

## Hypotheses and memory

A hypothesis must state the thesis, why it may be true, what would make it fail,
at least one positive evidence ref, at least one negative/contrary evidence ref,
an expected forward window, related companies and industries, and one or more
machine-checkable falsification conditions. A condition has a metric id,
operator, Decimal threshold and a 1-252 session window.

Investment memory is a pure immutable tuple of content-addressed entries. Each
entry commits to the previous entry semantic SHA. Appending requires an expected
tip, and validation with the expected tip detects mutation, reordering, middle
deletion and tail deletion. `FAILED_CASE` entries require status `FAILED`; no API
can delete or overwrite them. Persistence remains the caller's responsibility.

## V4 Observation adapter

The adapter accepts:

- an absolute workspace root;
- one exact Session relative path and byte SHA;
- explicit Factor/Strategy Observation refs;
- every exact recursively referenced request, stage receipt, factor definition,
  source and inventory ref required by the supplied roots;
- optional explicit Label refs;
- optional explicit Evaluation Receipt refs;
- an intelligence `as_of` cutoff.

The Session resolves exactly one Run. Typed observations are not discovered from
the Run stage payload. Every typed artifact must be explicitly supplied and must
match its version-specific path prefix, byte SHA, semantic SHA, identity,
strategy, decision session, request/run backlink and cutoff rules. The adapter
then walks exact refs recursively from Session, Run, Observation, Label and
Evaluation roots. Undeclared refs, conflicting refs, unused refs, cycles and
depth/cardinality overflow all fail closed. Labels and evaluations may mature
after the observation cutoff but never after `as_of`. An absent optional list is
accepted; an invalid supplied optional ref blocks the entire bundle.

The descriptor-relative reader rejects absolute/traversal paths, symlinks,
hardlinks, unsafe ownership/modes, case-fold ambiguity, version/path drift,
noncanonical JSON, files above 8 MiB and closures above 64 MiB.

Recursive closure admission is version sealed. Registered V4 request, stage and
inventory artifacts are passed through the V4 typed validator. The small set of
external terminal source versions has an explicit identity field, closed shape
and version-specific path prefix. Unknown versions fail closed. Every edge must
use an allowlisted target version, retain the parent strategy and satisfy
`child cutoff <= parent cutoff <= as_of`.

`forward-stage-output.v1` is included in that recursive closure, including its
request and lineage-receipt refs. Its canonical `payload_json` remains an opaque,
non-authorizing runtime payload: exact refs encoded inside that JSON string are
not admitted as evidence. I0 accepts typed Observation, Label and Evaluation
artifacts only through their separately supplied and version-validated refs.

## AI boundary and R2.2 evaluator

I0 provides only a pure wrapper for externally supplied AI drafts. Allowed kinds
are extraction, summary, hypothesis draft and contrary-evidence draft. Every
draft requires exact source refs, a timestamp and confidence. Recursive payload
validation rejects posterior, likelihood, weight, governance, portfolio,
selector, provider, broker, execution, order and trade control fields. I0 does
not import or invoke a model/provider.

Sprint R2.2 implements the Protocol seam as an additive, offline evaluator in
`quant_investor/intelligence/evaluator`. It accepts one canonical, content-bound
request by exact relative path and byte SHA. That request binds every V4
Session, Run, typed Factor Observation, matured Label, Evaluation Receipt,
recursive closure ref, hypothesis, evidence item, evaluation policy and memory
tip. The evaluator reruns the I0 exact adapter; it never discovers a `latest`
artifact and never reads a provider.

The result is a self-contained
`myquant.v17.research-intelligence.forward-evaluation-envelope.v1`. Its primary
artifact is the immutable
`myquant.v17.research-intelligence.forward-evaluation-receipt.v1`; factor,
variant, hypothesis, calibration, regime and append-only memory proposal
receipts are embedded and content addressed independently. The CLI is:

```bash
quant-investor-v17-v4 research-evaluate \
  --workspace-root /absolute/path/to/myQuant \
  --request-path data/private/research_intelligence/evaluation_requests/forward-evaluation-request-<sha256>.json \
  --request-sha256 <exact-byte-sha256>
```

All non-`research-evaluate` arguments are delegated unchanged to the existing
V4 dispatcher. The new command prints exactly one canonical JSON envelope and
writes no result file. Evaluation may propose memory suffix entries, but the
caller owns persistence and compare-and-swap of the expected tip.

R2.2 computes research diagnostics only. It cannot change factor weights,
Bayesian posteriors, factor governance, selectors, portfolio state or any
production/runtime route. Post-hoc policy conclusions are downgraded, failed
hypotheses remain in memory as `FAILED_CASE`, and regime evaluation consumes
only the state selected by the causal I0 one-step filter; it performs no
backward smoothing.

## Authority and verification

Every I0 runtime artifact carries `research_only=true`, `production=false` and
the same closed authority object. Broker, execution, factor-governance write,
formal activation, LLM, order, portfolio, provider, research-runtime default,
selector and trade flags are all false.

The package manifest provides manifest-relative drift detection for the exact
Python source set and byte hashes. Its trust anchor is the reviewed Git
checkpoint; the manifest is not represented as an external signature or an
independent immutable root. `verify_package` checks the working source tree
against that checkpointed manifest. Likewise, `verify_runtime_receipt` verifies
the closed content-addressed summary shape. A bundle summary is not an authority
token: `build_intelligence_runtime_receipt` requires the original workspace,
Session path/SHA and explicit typed/closure refs, reruns the exact Observation
adapter, and requires byte-for-byte equality with the supplied bundle before it
performs complete component replay and source-authorization checks.

Required verification is: I0 tests, Forward tests, V4 regression, V5 baseline,
public boundaries, staged quality gate, mypy, Black, Flake8, package verification
and a full runtime-receipt replay. Source Truth, Authority, Security Master and
Census artifacts are protected and must have an empty diff.

# V17 I1 Investment Decision Intelligence

Sprint I1 adds a deterministic, content-addressed investment-decision library
above the existing I0 Investment Intelligence runtime and the optional R2.2
Forward Research Evaluator. Its purpose is to turn a fully replayed research
closure into a reviewable investment memo and a disciplined research state.

I1 is **library-only and research-only**. It does not expose a public CLI, Web
route, scheduler, daemon, persistence writer or paper-portfolio implementation.
It does not select securities, construct or mutate a portfolio, call a provider
or model, connect to a broker, create an order, execute or trade.

## Decision flow

```text
exact V4 / I0 replay + optional exact R2.2 replay
  -> Investment Decision Context
  -> Risk Assessment
  -> five-state Investment Decision
  -> deterministic Investment Memo
  -> append-only Decision Discipline Chain
  -> external-review-only Paper Intake Proposal
```

Every step returns a new canonical sealed value. Builders do not mutate their
inputs or persist outputs. Validators do not accept a receipt merely because
its hash is internally consistent: they rerun the corresponding builder from
the complete supplied replay closure and require byte-for-byte equality.

## Fixed authority boundary

Every I1 artifact carries the same closed authority posture:

```text
research_only=true
production=false
decision_protocol=myquant.v17.v4
mainline_authority=false
operational_activation_unchanged=true
broker=false
execution=false
order=false
trade=false
```

The wider I1 boundary also keeps selector, portfolio, provider and LLM
authority closed. Context notes and allowlisted AI drafts may contribute
source-bound research text; they cannot set a score, posterior, threshold,
risk, decision state, conclusion, candidate, weight or action.

I1 does not register a second V17 decision protocol. `myquant.v17.v4` remains
the sole public decision protocol, and an I1 artifact can never substitute for
an exact active mainline pointer.

## Exact replay closure

The Decision Context collector accepts an explicit I0 replay closure. It must
contain exactly the V4 observation bundle, exact Session path and byte hash,
observation and source-closure references, Evidence, Bayesian receipts, Regime
input and receipt, Quant and Fundamental branches, Fusion receipt, Hypotheses,
Memory chain state, and explicit label/evaluation references. Missing, extra or
future-dated inputs fail closed.

The collector calls the existing I0 runtime builder and therefore replays the
complete V4/I0 closure. It does not scan a `latest` location and does not accept
a summary-shaped runtime receipt. The selected company must be bound to exactly
one selected Hypothesis and its matching Bayesian receipt. Quant,
Fundamental, Fusion, Regime, Evidence and authorized source references must all
agree with the replayed runtime.

The two replay layers retain separate canonical times. The frozen I0 closure is
replayed at its own Fusion/runtime timestamp, which must be no later than the
Decision Context `as_of`; I1 does not rewrite that historical closure to the
later review time. Optional matured R2.2 is replayed independently at the
Decision Context `as_of`, so later outcomes can be evaluated without making the
original I0 decision evidence future-dated.

R2.2 is optional, but it is never partially trusted. When supplied, the exact
request path and request byte SHA-256 are both required. I1 reruns the existing
Forward Research Evaluator and validates the complete successful envelope,
main receipt and embedded factor, variant, hypothesis, calibration, regime and
memory topology. A blocked or internal-error envelope is a contract failure,
not an unavailable research result.

Only a preregistered R2.2 hypothesis evaluation whose status is `FAILED` may
invalidate a thesis. The current compact R2.2 hypothesis receipt omits the
`preregistered` flag, so I1 independently replays the exact request and origin
closure to derive the positive preregistration fact. The selected receipt must
also be `FAILED`, and the replayed main receipt must not contain
`POSTHOC_POLICY_CONCLUSIONS_DOWNGRADED`. The existing evaluator contract emits
`FAILED` only for a preregistered hypothesis; a triggered post-hoc hypothesis is
`UNCERTAIN`. Therefore `UNCERTAIN`, post-hoc failure-like evidence or any
non-preregistered conclusion cannot produce thesis invalidation.

## Content-addressed artifacts

I1 defines eight immutable-value artifact families:

| Artifact | Purpose |
|---|---|
| Decision Policy | Research and paper-review evidence, confidence, posterior and risk gates. |
| Context Note | Source-bound company display name, why-now, industry, theme or valuation context. |
| Decision Context | Exact I0/R2.2 replay bindings and explicit evidence availability. |
| Risk Assessment Receipt | Four-dimensional risk summary with explicit unavailable dimensions and hard vetoes. |
| Investment Decision Receipt | One of five research states plus stable reason and blocker codes. |
| Investment Memo | Deterministic projection of validated Hypothesis, Evidence, notes, risks and allowlisted drafts. |
| Decision Discipline Entry | Append-only review lifecycle entry bound to a replayed decision. |
| Paper Intake Proposal | Minimal request for external paper review of an eligible decision. |

Artifacts use the I0 canonical JSON and SHA-256 conventions. Decimal values are
compared as fixed-precision decimal values, timestamps are UTC seconds, set-like
arrays use defined ASCII ordering, duplicate identifiers and references are
rejected, and future evidence is not admitted.

## Evidence availability and risk

The I0 research closure is always mandatory. A Decision Policy may additionally
require any of these availability classes at the research or paper-review tier:

- `INDUSTRY_CONTEXT`
- `THEME_CONTEXT`
- `VALUATION_CONTEXT`
- `WHY_NOW`
- `AI_DRAFT`
- `R22_EVALUATION`

The paper-review requirements must be at least as strict as the research
requirements. Industry, Theme, Valuation and Why Now are proven by validated,
same-company Context Notes. An internal Fundamental valuation metric is not a
substitute for source-bound `VALUATION_CONTEXT`.

Risk is assessed independently across `BUSINESS`, `FINANCIAL`, `MARKET` and
`THESIS`. Every available assessment is bound to admitted Evidence or an exact
authorized source reference. Unavailable dimensions stay explicit. Overall
severity is the maximum available dimension severity, and a hard veto fires
only when a named veto reaches the policy threshold.

## Five research states

I1 returns one of five states in a fixed precedence order:

| Priority | State | Meaning |
|---:|---|---|
| 1 | `THESIS_INVALIDATED` | A preregistered R2.2 hypothesis evaluation is `FAILED`. |
| 2 | `INSUFFICIENT_EVIDENCE` | A research-required availability class or risk dimension is unavailable. |
| 3 | `WATCHLIST` | Research inputs exist, but a veto or research confidence, posterior, risk or supported-R2.2 gate fails. |
| 4 | `RESEARCH_APPROVED` | Research gates pass, but a stricter paper-review input or gate does not. |
| 5 | `PAPER_CANDIDATE` | Research and paper-review gates both pass. |

These are research workflow states, not market actions. In particular,
`PAPER_CANDIDATE` means only that the decision is eligible to be submitted for
**external paper-review**. It is not a stock selector result, portfolio
admission, position, target price, `BUY`, `SELL`, `HOLD`, order or trade signal.

Higher-priority state selection does not erase other diagnosed reasons or
blockers. For example, a preregistered failed thesis remains
`THESIS_INVALIDATED` while any missing evidence remains visible in the receipt.

## Deterministic memo and AI boundary

The Investment Memo is a projection, not a narrator. It may copy only:

- the validated Hypothesis thesis, supporting rationale and falsification
  conditions;
- admitted Evidence direction, strength, reason, source type and exact ref;
- validated risk-assessment reasons;
- validated Context Notes;
- allowlisted fields from validated `SUMMARY`, `EXTRACTION` and
  `CONTRARY_EVIDENCE_DRAFT` values.

The memo generator cannot invent an investment narrative. Missing Why Now
context remains unavailable. A `HYPOTHESIS_DRAFT` is not admitted to the
decision context, and no AI draft can alter the deterministic decision.

## Discipline chain

I1 keeps decision learning separate from the existing Investment Memory
contract. It introduces a dedicated append-only hash chain rooted at the zero
SHA-256 value:

```text
DECISION_CREATED / ACTIVE
  -> DECISION_REVIEWED / OUTCOME_AVAILABLE
  -> THESIS_CONFIRMED / CONFIRMED
     or THESIS_FAILED / FAILED
  -> LESSON_LEARNED / LEARNED
```

Every entry is bound to a fully replayed Decision Receipt. When a later decision
uses a new Context, the exact added and removed Evidence/R2.2 evaluation refs
must explain the change. Price observations may be recorded but do not count as
Evidence changes and cannot by themselves justify a different risk or decision
state. I1 does not write formal Investment Memory or alter R2.2 Memory Proposal
semantics.

## Paper-review seam

The paper seam is deliberately minimal. The library defines a
`PaperPortfolioAdapter` protocol but supplies no implementation, creates no
instance and makes no call. A Paper Intake Proposal can be built only after the
Decision Receipt is fully replayed and proven to be a genuine
`PAPER_CANDIDATE`.

The proposal contains only the exact Decision reference and
`PENDING_EXTERNAL_REVIEW`. It contains no action, side, quantity, weight,
holding, cash or order field. Submission, acceptance and any paper-ledger
effects belong to a separately governed external workflow.

## Library surface

The public Python library is exported from
`quant_investor.intelligence.decision`:

```python
from quant_investor.intelligence.decision import (
    DecisionContractError,
    PaperPortfolioAdapter,
    append_decision_discipline,
    assess_investment_risk,
    build_context_note,
    build_decision_policy,
    build_investment_memo,
    build_paper_intake_proposal,
    collect_investment_decision_context,
    make_investment_decision,
    validate_context_note,
    validate_decision_discipline_chain,
    validate_decision_policy,
    validate_investment_decision_context,
    validate_investment_decision_receipt,
    validate_investment_memo,
    validate_paper_intake_proposal,
    validate_risk_assessment_receipt,
)
```

There is intentionally no corresponding public CLI, Web route, scheduler,
automatic Memory writer, Paper adapter implementation or portfolio workflow.

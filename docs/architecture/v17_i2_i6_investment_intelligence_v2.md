# V17 I2-I6 Investment Intelligence v2

This document describes the implemented, research-only v2 library. It does not
describe a live strategy, an initialized mainline pointer, or an automated
trading system.

## Purpose and authority boundary

The v2 package closes the research path that follows frozen I0/R2.2/I1:

```text
exact data-readiness receipt + governed Factor v5 set
  -> deterministic full-A initial pool and Quant branch
  -> source-bound Industry identity and component
  -> source-bound Theme identity, component and risk
  -> Fundamental Intelligence Profile
  -> same-run Evidence Graph / Fusion / Decision v2
  -> skeptical open-Web advisory review
  -> deterministic research portfolio and paper evaluation
  -> signed research-mainline publication capability
```

All builders are library functions. They do not register a public CLI, Web
endpoint, scheduler or daemon. Ordinary verification performs no live Provider,
model, broker, order, execution or trade call and writes no active pointer.

The original `quant_investor.intelligence` package is byte-frozen. New work is
isolated under `quant_investor.intelligence_v2`, whose package manifest binds
the frozen-v1 manifest identity instead of changing I0/R2.2 implementation
identity on every later sprint.

## B0: readiness and deterministic Quant production

The readiness receipt keeps `AVAILABLE`, `STALE` and `BLOCKED` distinct.
Market/PIT must match the target session. Fundamental may be one canonical open
session behind only when an owner policy explicitly allows it; it is never
relabeled `AVAILABLE`. Macro freshness is source- and release-calendar-bound,
but Macro does not enter company alpha.

The Quant producer consumes an exact full-A PIT universe, an explicitly chosen
Factor Governance v5 admitted set and an owner-sealed pool policy. It ranks by
score and breaks ties by company code. It does not scan `latest`, mutate a
factor registry, write a production factor set, or create a portfolio.

## I2 and I3: Industry and Theme identity

Industry classification is selected only from effective, available,
source-bound membership under owner-sealed provider and taxonomy precedence.
Conflicting equal-precedence classifications are `AMBIGUOUS`; absent mappings
are `UNMAPPED`. Either state blocks Decision v2 admission. The LLM cannot infer
an industry.

Theme identity distinguishes:

- `AVAILABLE`: validated effective membership exists;
- `NO_MEMBERSHIP`: a complete catalog explicitly covers the company but has no
  effective membership;
- `UNMAPPED`, `AMBIGUOUS`, and `RETIRED`: non-admissible identity states.

`NO_MEMBERSHIP` is a cap bucket, not a neutral score. Theme components remain
missing unless source-bound exposure and an owner-sealed component policy exist.
The LLM cannot create a theme, membership, lifecycle or score.

## I4 and I4.5: Fundamental profile and Decision v2

The Fundamental Intelligence Profile wraps the frozen v3 scorer once per
build/replay. Its authoritative values are 12-place Decimal projections; the
raw binary-float representation is audit-only. Missing components are not
filled with zero, a mean, or model prose.

Decision v2 binds one B0/I2/I3/I4 run to fresh frozen-I0-compatible Evidence,
branches, Fusion, Bayesian and Hypothesis artifacts, then replays the exact
R2.2 request for that Hypothesis. The frozen v3 fusion function is invoked once
and retains 50% Quant, 50% Fundamental and Fundamental-coverage attenuation.
Industry and Theme are not counted again as extra alpha branches; Macro and AI
do not enter Fusion.

The five states retain the I1 meanings and priority:

1. `THESIS_INVALIDATED` for a same-closure preregistered R2.2 failure;
2. `INSUFFICIENT_EVIDENCE` for missing required identity/evidence/risk;
3. `WATCHLIST` for deterministic gate failure;
4. `RESEARCH_APPROVED` when research gates pass but paper gates do not;
5. `PAPER_CANDIDATE` when all deterministic paper gates pass.

These are research states, not BUY/SELL/HOLD or portfolio instructions.

## I5: open-Web skeptical committee

I5 uses the fixed role `怀疑型 AI 投委会`. A new live run has exactly three
public Responses Web Search rounds: discovery, contrary gaps and verification
closure. Domain filtering is deliberately not configured. This freedom applies
to public-source discovery, not to authority: model citations are URL leads
until the local collector safely captures and validates the cited HTML.

The local collector permits only HTTP/HTTPS on ports 80/443, rejects private or
reserved DNS results, pins the connection to validated IPs, verifies the peer,
rechecks every redirect, bypasses proxy/cookie/auth state and enforces fixed
time, header, body, decompression, MIME, charset and HTML limits. PDF is
discovery-only. Historical replay consumes frozen receipts and performs zero
DNS, socket, HTTP, credential or model calls.

The private committee uses a distinct project, credential identity, client,
history, cache and receipt-store namespace. It receives only the replayed
Decision v2 projection and locally validated facts, performs exactly one model
call, and has `tools=[]`, `store=false`, no Web/file/MCP/code tool, no background
mode and no response continuation. A live call additionally requires an
unexpired owner-sealed capability receipt with exact organization, project,
model, endpoint, prompt/schema hashes and ZDR control evidence.

AI influence is advisory-only. Rank weight and absolute rank movement are each
capped at 10%. A non-zero change requires a validated first-party fact or two
independent original sources. Unresolved conflicts force zero change. I5 cannot
change admission, Decision state, posterior, risk or veto.

## I6: research portfolio, paper evidence and publication capability

The portfolio constructor starts only from replayed `PAPER_CANDIDATE` Decision
v2 receipts. Owner policies have no implementation defaults. Decimal weight
quanta are allocated in a fixed constraint order: security, liquidity,
industry, theme, gross, cash and turnover. Macro can only tighten gross, cash,
risk or vetoes. The advisory ordering is separately constrained and falls back
byte-for-byte to the deterministic portfolio when capital total variation
exceeds 10%.

Paper artifacts are explicitly research-only. Effective-dated A-share policy
binds the exchange calendar, T+1 availability, lots, suspension, price limits,
IPO windows, price source, partial fills, fees, corporate-action handling,
listing/delisting and expiry/cancellation. Missing rules block simulation;
legacy backtest defaults are not used. Graduation is a research receipt and
requires owner-selected horizons from 1D/5D/20D/60D, an exact benchmark,
matured coverage, cost-adjusted excess return, drawdown, regime stability and
no hard-risk breach.

The publication profile is `INTELLIGENCE_V2_PUBLICATION_V1`. Legacy runs with
no marker continue through the unchanged v4 reader. A marked formal output must
carry exactly one ref to an upstream v2 publication closure. That closure binds
only research artifacts; it contains no run, formal, pointer, activation
sidecar or permit ref. The reader derives the downstream activation-sidecar
path from the closure path, and derives the external Ed25519 `ACTIVATE` permit
path from the actual target pointer bytes. The sidecar and permit therefore sit
outside the pointer's transitive content closure:

```text
research artifacts -> publication closure -> formal -> run -> pointer
                                  ^             ^       ^        ^
                                  +------ external activation sidecar ------+
                                                  ^
                                      external ACTIVATE permit
```

This ordering avoids both `run -> formal -> sidecar -> run` and
`pointer -> ... -> sidecar -> pointer` content-addressing cycles. Quarantine is
checked at the fixed per-run path.

The marked-run reader now validates the fixed publication-closure, sidecar,
legacy projection, Ed25519 `ACTIVATE` permit and quarantine path before it will
return the public DTO. The low-level CAS remains the only commit point, but this
repository does not implicitly authorize its use. Implementation and ordinary
tests stop before an active-pointer write. Activation requires a later display
of the exact strategy, run, target pointer SHA, expected predecessor and
rollback target, followed by fresh explicit user confirmation and a short-lived
owner signature.

The publication boundary is intentionally still fail-closed. Closure-dependent
Decision v2, Evidence Graph, advisory, portfolio, paper and graduation builders
do not yet have persisted validation-capsule artifacts that let the public
reader rerun every builder from raw inputs. The repository also does not yet
provide the formal immutable-store publisher or the owner-confirmed
expected-current CAS coordinator. A synthetic signed marked-run test proves the
reader boundary and legacy compatibility, not production readiness.

## Current readiness

The deterministic B0-I6 research libraries and signed marked-reader boundary
are implemented, but the end-to-end publication capability is not yet complete.
Real readiness also requires prospective Factor v5 evidence, complete
Industry/Theme catalogs, owner policies, same-run data closures, verified ZDR
capability, owner/recovery public-key policy, matured paper outcomes and the
owner evidence-archive gate. Fixtures cannot substitute for any of them.

Until those inputs exist and a separate activation confirmation is granted,
the correct state is:

```text
CAPABILITY_PARTIAL
RESEARCH_MAINLINE_CANDIDATE_BLOCKED
active pointer unchanged
```

# v16 Evidence-v2 Contract

Status: **disconnected and permanently nonauthorizing**.

This contract defines the prospective evidence needed before v16 can be
considered for production. The initial implementation under
`quant_investor/v16/evidence_v2/` is a pure validation foundation. Existing
v15/v16 commands, readiness writers, Dashboard surfaces, Factor registries,
production/default pointers, portfolios, and execution paths do not import it.
No artifact described here is an activation receipt or human authorization.

## Authority boundary

- v15 remains the production/default authority.
- Every evidence-v2 builder writes `activation_candidate=false`,
  `new_risk_authorized=false`, and `production_apply_enabled=false`.
- Evidence-v2 validation success is necessary evidence only. It cannot mutate
  canonical data, Factor state, Dashboard state, capital, holdings, orders, or
  the production/default pointer.
- A failed mandatory gate leaves `readiness_status=no_new_risk` with exact
  blockers. A caller-supplied readiness boolean or metric is not evidence.
- Integration into an authorizing consumer is forbidden until all migration
  and prospective epochs below are complete and separately reviewed.

## One attempt and three epochs

There is exactly one v16 evidence attempt. Its immutable genesis binds the
proposed Factor transition graph, hermetic runtime capsule, and open-session
calendar. Any terminal failure exhausts v16. A retry requires a new major
protocol; relabeling, resetting, deleting, or reusing an attempt ID is
forbidden.

The attempt has three disjoint prospective epochs:

1. **Epoch A, training capture.** Capture predictions/features/outcomes under
   the frozen proposed Factor graph, then train and seal model bundles. No A row
   may enter B or C qualification metrics.
2. **Epoch B, Factor qualification.** Recompute the exact A/B/C/D Factor replay
   using only frozen A model bundles. B is research replay only, with zero
   capital and no production apply. No B outcome may enter C metrics.
3. **Epoch C, post-activation validation.** Validate each scheduled `s0`
   continuation under a separately authorized Factor activation. C consumes no
   A/B outcome. Factor activation is not v16 activation.

Schedules must be declared before their first `s0` opens. Schedule slots and
target windows are immutable, ordered, non-overlapping, and bound to the same
attempt, runtime capsule, and open-session calendar. Epoch B/C schedules bind
exactly one frozen model bundle for each of `quant`, `fundamental`, `macro`, and
`llm`; Epoch A binds none. Each B/C schedule also binds one calibration-universe
artifact that already fixes every branch/slot/symbol, cohort ID, future artifact
path, and lambda-fold reference. The schedule's independent RFC3161 receipt
must anchor the schedule bytes before the first `s0`; the schedule does not
contain its own receipt reference. Epoch B uses 30 open sessions; Epoch A/C use
20. A global authoritative attempt registry is required before these schedules
can be used outside the disconnected validator.

### Frozen CN calendar and session clock

The v3 genesis binds both `v16.open-session-calendar.v1` and
`v16.session-clock.v1`. Their compiler is an offline allowlist, not a fetcher.
The private inventory contains exactly 24 named source files: 22 consumed
official-source captures and two explicit exclusions. The compiler opens only
the 22 consumed files. Six reviewed calendar/clock aliases share a physical
file, so each physical file is opened once and its immutable bytes are then
distributed to the two distinct semantic refs. Extra, missing, renamed, or
substituted files fail closed. There is no glob, `latest`, discovery, network,
browser, provider, or parser fallback.

Each of the 28 source bindings fixes its parser contract, declared official
origin URL, exact retrieval URL/method, raw byte ref, semantic projection or
binary profile, and authority scope. HTML is parsed with the stdlib structured
HTML parser. A semantically equivalent HTML recheck may have different bytes,
but every frozen marker and the complete sealed projection must still match.
Wayback CDX captures must contain one exact selected HTTP 200 row. The four
PDF/DOCX rule files are never passed to an external text extractor: exact byte
SHA-256 selects one of eight code-frozen profiles, and any byte drift fails.
For every binding:

```text
semantic_projection_sha256
  == semantic_projection.semantic_sha256 or selected_profile.semantic_sha256
  == raw_ref.semantic_sha256
```

The 2026 CN calendar contains exactly 19 closed weekdays, seven reopening
dates, and 242 open sessions. SSE, SZSE, and BSE annual and active closure
sources must agree exactly. Rule histories are gapless at `2026-07-06`; the
2023 SSE/SZSE prior-rule effective date is resolved unanimously from three
official first-listing event sources. The listed-equity auction clock is
`09:15-09:25`, `09:30-11:30`, `13:00-14:57`, and `14:57-15:00` in
`Asia/Shanghai`, excluding after-hours fixed-price, block, bond, and fund
trading scopes.

The current macOS materialization may carry only the reviewed
`com.apple.provenance` value `010200b80b586763d5d96f`; any other extended
attribute is outside the reviewed source-materialization contract. This is
separate from the mandatory descriptor ACL and byte checks.

Before each v3 schedule anchor, `v16.calendar-pre-anchor-recheck.v1` compares
exactly nine ordered current sources: three active closure schedules, SSE/SZSE
current notices and clock binaries, and BSE current calendar/clock semantics.
It reopens and recompiles the bound calendar and clock before comparing the
observed refs. Binary rows require `byte_and_semantic_match`; HTML may also use
`semantic_match_with_byte_drift`. A successful local comparison permanently
retains these blockers:

- `calendar_recheck_capture_time_not_independently_evidenced`
- `calendar_recheck_transport_freshness_not_independently_attested`
- `evidence_v2_disconnected_from_authorizing_consumers`

It cannot claim `is_current`, `fresh`, `recheck_completed`, `anchor_ready`, or
independently attested transport freshness.

The v3 schedule adds refs for genesis, session clock, and its schedule-specific
calendar recheck. An `s0` must be an official open session on or after
`2026-07-06`; its UTC times are fixed at `01:15`, `07:00`, and decision cutoff
`07:30`. `s1` opens at `01:15` on the next official session. Epoch A/C target
the exact next 20 open sessions and B the exact next 30, without skips,
overlap, closure dates, weekends, or crossing the 2026 coverage boundary.

`ScheduleEvidenceBundleV3` reopens the complete transitive lineage: v3
genesis; runtime capsule and all eight raw components; transition graph and
every factor-set ref; calendar and clock source bundles; schedule-specific
recheck; and, for B/C, the ordered four frozen model bundles, every typed
transitive model ref, calibration universe, and all lambda-fold artifacts.
`EvidenceLoadLocation` carries only `reference`, explicit `root`, and canonical
`policy`; callers cannot choose a raw/canonical mode. Epoch B and C must retain
identical full model evidence, not merely equal top-level model refs.

Even a valid A/B/C lineage returns `no_new_risk` with exactly the two calendar
recheck blockers plus disconnected-authority, global-attempt-registry, and
external anti-rollback blockers. It is not a production pointer switch.

### Local provisional attempt journal

`evidence_v2/attempt_journal.py` provides only local crash-visible process
coordination. It uses an explicit owner-only root, a fixed lock, exclusive
event creation, a contiguous sequence, predecessor byte hashes, file and
directory `fsync`, head CAS, ordered A/B/C transitions, and absorbing terminal
states. A partial event file is retained and makes later replay fail closed;
there is no overwrite, delete, reset, or retry API.

This journal is deliberately marked `local_provisional_no_anti_rollback`.
Because the same OS user can delete or restore local files, an apparently valid
local prefix cannot prove that a later event never existed. Journal state
therefore always retains
`global_attempt_registry_authority_not_integrated`,
`evidence_v2_disconnected_from_authorizing_consumers`, and
`provisional_journal_head_not_bound_to_external_anti_rollback_authority`.
Initializing a real journal is a separate governed operation and is not part of
local verification. A later migration needs an independently verified external
anti-rollback checkpoint for the exact journal head; the local journal alone
can never remove either readiness migration blocker.

## Canonical wire format

Evidence JSON uses a restricted canonical format:

- UTF-8, Unicode NFC strings, ASCII object keys of at most 128 bytes;
- maximum depth 32, 100,000 members/items, 1 MiB per string, and 16 MiB per
  JSON artifact;
- JSON integers only in `[-(2^53-1), 2^53-1]`;
- no native JSON floats, NaN, infinity, duplicate keys, or negative zero;
- finite binary64 values encoded as `"f64:" + float.hex()`, with both zero
  signs normalized to `f64:0x0.0p+0`;
- sorted keys, compact separators, and exactly one terminal LF.

Every JSON artifact carries a canonical semantic SHA-256 over the object with
`semantic_sha256` removed. Every reference separately binds absolute path,
artifact schema, byte SHA-256, semantic SHA-256, and root policy. Byte identity
and semantic identity are both mandatory.

## Secure read policy

Private evidence and trust material roots are current-user owned mode `0700`;
files are current-user owned mode `0600`, regular, single-link files. All
accepted roots and files must prove extended-ACL absence with the platform ACL
checker. Governed data may use a different owner/mode policy. Traversal
ancestors may retain deny-only platform ACLs, such as the macOS home-directory
delete guard, but any extended allow entry or group/world-write mode is rejected.

Readers use descriptor-relative `O_NOFOLLOW` traversal, enforce size bounds,
compare path and descriptor device/inode identities, and recheck mode, owner,
link count, size, mtime, and ctime after reading. Discovery, globbing,
`latest`, scan fallback, CSV fallback, and mutable-path substitution are
forbidden.

Production evidence intake uses `load_bound_canonical_artifact` or
`load_bound_raw_artifact`. These factories always invoke the built-in Darwin
descriptor ACL verifier and return a byte-bound artifact, not an unbound
decoded mapping. An unsupported platform, ACL lookup error, root/file ACL, or
ancestor allow ACL fails closed. Factories accept only the exact canonical
private, trust-material, or governed-data policy definitions; a caller-created
weakened policy with a reused policy ID is rejected. ACL-injectable lower-level
readers are private, test-only implementation details and are excluded from the
module's public surface.

## Hermetic runtime and model bundles

The runtime capsule binds distinct byte artifacts for the CPython interpreter,
source tree, dependency lock, installed distributions, platform manifest,
PyArrow backend, SciPy backend, and pinned OpenSSL 3 RFC3161 binary. The
OpenSSL runtime path is
`/opt/homebrew/opt/openssl@3/bin/openssl`; the byte SHA and build identity still
must match the capsule. Runtime/provider discovery and network access are
forbidden during recomputation.

Recomputation fixes locale/timezone/hash seed and native numeric thread counts:
`LC_ALL=C`, `LANG=C`, `TZ=UTC`, `PYTHONHASHSEED=0`, and all supported BLAS/OpenMP
thread controls at one.

Each formal branch has one frozen Epoch A bundle bound to its training schedule,
training capture, feature contract, hyperparameters, serialized model, and
deterministic inference entrypoint. The LLM bundle additionally requires a
provider attestation for an immutable model build ID, tokenizer, inference
configuration, and endpoint contract. Aliases such as `latest`, `current`,
`default`, or an unversioned model name are blockers.

## RFC3161 anchoring

Schedule declarations are RFC3161-anchored strictly before the first `s0`
open. Every prediction is separately anchored strictly after `s0` close and
strictly before `s1` open.

The first complete TimeStampResp persisted for an anchor wins through an
exclusive create. Persistence happens before cryptographic validation. A
canonical invalid response permanently fails that anchor and cannot be
replaced, even by identical bytes. Only a transport failure before any response
is persisted permits retry, and the retry must use the exact bound query.
Partial persisted files remain terminal evidence; they are never deleted to
manufacture a retry.

Validation uses the capsule-pinned OpenSSL binary and the exact query, response,
root, untrusted chain, CRL bytes, and anchored artifact bytes. Both
`ts -verify -queryfile` and `ts -verify -data` must succeed against the same
response. A successful attempt transition cannot accept a caller-supplied
receipt: it invokes this verifier over the frozen raw bundle and constructs the
receipt itself. Every later anchor-binding read independently reruns the same
verification and requires byte-for-byte receipt equality. The immutable attempt
envelope binds the anchored artifact, window, policy, OpenSSL identity, trust
material, and revocation material before validation, so a receipt cannot be
reused for a different anchor. It requires:

- exact `Granted` status, query/message imprint/nonce/policy match;
- certificate validity and chain verification at RFC3161 `genTime`;
- strict X.509 verification and root self-signature check;
- one issuer-bound, signature-verified CRL for every non-root certificate;
- CRL `lastUpdate`/`nextUpdate` coverage at the recorded verification time,
  with certificate serial revocation time evaluated relative to `genTime`.

A revocation effective at or before `genTime` is terminal. A revocation whose
effective time is after `genTime` is retained as a warning and does not
invalidate the historical anchor. The verified OpenSSL bytes are copied into
the private verification root and that exact inode is executed; the pinned
source path is rechecked after verification. Recorded verification time cannot
precede `genTime`.

## Research target and benchmark

The only evidence-v2 target is
`CN_20D_MARK_NET_TOTAL_RETURN_EXCESS_VS_CSI300_TRI_V1`. It is explicitly
non-executable and cannot be used as a broker price or order instruction.

Stock marks use `adjusted_close = close * adj_factor`. Benchmark marks use the
official CSI 300 total-return series `H00300.CSI`, code `H00300`, CNY,
gross-pre-tax total return. The price index `000300` is not a substitute.

Stock boundaries are recomputed from one byte-bound strict Parquet table plus
canonical adjustment-factor, PIT-membership, and suspension artifacts. The
stock table has exactly `symbol:string`, `trade_date:date32`, `close:float64`,
`source_observed_at:timestamp(us, UTC)`, and
`source_document_sha256:string`, all non-null and strictly ordered by unique
`(symbol, trade_date)`. Adjustment factors bind that exact table reference;
PIT and suspension evidence bind the schedule calendar. A self-reported entry
or exit mark, or a mark whose source refs are not opened and recomputed, is not
accepted by the target/calibration path.

The H00300 Parquet table has exactly these non-null Arrow fields:

| Field | Arrow type |
| --- | --- |
| `instrument_id` | `string` |
| `trade_date` | `date32` |
| `close_total_return` | `float64` |
| `currency` | `string` |
| `return_type` | `string` |
| `source_system` | `string` |
| `source_observed_at` | `timestamp(us, UTC)` |
| `source_document_sha256` | `string` |

The table is at most 64 MiB and 100,000 rows, strictly ordered by unique trade
date, and bound to a semantic projection of Arrow schema, Parquet key/value
metadata, row groups, encodings, compression, sizes, and column statistics.
The backend is PyArrow and its bytes/version are part of the runtime capsule.
Every one of the 20 schedule-declared target sessions must have an exact H00300
row; boundary-only or price-index evidence is insufficient.

Entry/exit mark precedence is terminal cash settlement, exact adjusted close,
then an authoritative PIT-listed suspension stale mark. A generic stale mark
for a delisted/missing symbol is forbidden. Terminal settlement uses the same
adjusted numeraire:

`terminal_adjusted_mark = raw_cash_per_terminal_share * applicable_adj_factor`

It binds official event evidence, share basis, tax basis, settlement date, and
the applicable adjustment-factor evidence. The target deducts all eight cost
components in fixed order: buy/sell commission, sell stamp duty, buy/sell
transfer fee, buy/sell slippage, and market impact.

The current disconnected source recomputer implements exact-close and
authoritative-suspension routes. It deliberately has no terminal-event source
producer yet; any sample that needs terminal settlement remains blocked until
official-event bytes and adjustment evidence can be parsed and rebound rather
than accepted as caller-declared values.

## Four-branch calibration

Calibration is recomputed separately for `quant`, `fundamental`, `macro`, and
`llm`. Each branch needs at least 300 prospective samples in at least eight
non-overlapping cohorts. Cohorts receive equal weight; samples within a cohort
receive equal weight.

Before the schedule is anchored, the calibration-universe artifact declares
the complete four-branch sample set. Every branch must contain exactly one row
for the same `(slot_id, symbol)` set; duplicate rows are forbidden. Sample IDs
and prediction, outcome, stock, cost, timestamp-attempt, and timestamp-receipt
paths are globally unique. Lambda-fold refs are also complete, distinct, and
fixed at declaration time. Validation requires exact equality with this plan:
missing, extra, reordered lambda, substituted-path, or post-selected samples
fail before metric computation.

Each sample consumes the actual canonical prediction artifact, its independent
RFC3161 binding, and the actual canonical outcome artifact. Predictions do not
embed their later receipt reference, which avoids an impossible hash cycle.
The outcome is then recomputed from its bound schedule, raw stock sources,
stock-mark projection, eight-component cost, H00300 manifest, and H00300
Parquet bytes. Prediction cohort boundaries must equal outcome `s1`/`s20`, and
all four branches use the same prospective cohort windows. Lambda values are
likewise read from bound fold artifacts; reported samples, lambdas, and metrics
cannot be injected by a caller.

Outcomes are positive when realized mark alpha is strictly positive. Brier
delta is `(p-y)^2 - (p0-y)^2`. Log-loss uses the natural logarithm, clips both
model and prior probabilities to `[1e-6, 1-1e-6]`, and reports model minus
prior. ECE is computed within each cohort using five equal-count bins sorted by
`(probability, sample_id)`; the first remainder bins are one sample larger.
Top bucket edge sorts each cohort by `(-predicted_alpha, sample_id)` and takes
the first `ceil(n/5)` realized alphas.

Cluster bootstrap uses 10,000 replicates. For replicate `r`, draw `d`, and `K`
cohorts, the selected cohort is the big-endian integer value of
`HMAC-SHA256(seed, ASCII("r:d")) mod K`. One-sided bounds use nearest ranks at
0.95 for loss deltas and 0.05 for top-bucket edge. `seed` is not a calibration
API input; it is read from the exact pre-`s0` RFC3161-anchored schedule bound by
the sample artifacts, and validation rejects any bootstrap seed drift.

Every branch independently passes all gates:

- Brier-delta bootstrap upper bound `< 0`;
- log-loss-delta bootstrap upper bound `< 0`;
- ECE `<= 0.05`;
- interval coverage in `[0.85, 0.95]`;
- alpha MAE `<` zero-alpha MAE;
- top-bucket edge bootstrap lower bound `> 0`;
- lambda folds in `[0,1]` with range `<= 0.20`.

## Factor B and post-activation C statistics

Epoch B replays the existing v4 chain for exact A/B/C/D factor sets. For
`add`, A/B retain the production set and C/D add the challenger in the empty
slot. For `replace`, A retains the production set, B removes the incumbent, and
C/D replace it with the challenger in the same family/slot. Each arm binds
Eligibility, Quant, Funnel, CodexS1, Bayesian recomputation from the frozen
model bundle, advisory-only RiskAdvisor, CodexIC, and a hypothetical zero-capital
PortfolioConstructor result. Caller-supplied posterior scores are forbidden.

For `M` tested transitions, paired cohort effects use a one-sample Student-t
test with Bonferroni alpha `0.05/M`; constant series fail closed. Family
Benjamini-Hochberg q-values must also be at most `0.10`. Both gates pass
separately. Pearson inputs that are constant and singular/non-finite VIF or
lambda computations are blockers.

Epoch C has exactly 15 predeclared hypotheses: five frozen models times Brier
delta, log-loss delta, and top-bucket edge. Each uses alpha `0.05/15`. C also
requires one continuation receipt for every scheduled `s0`; missing or
post-selected rows are terminal blockers.

## Later producers and authorizations

After A/B/C succeed, a later reviewed producer must build the exact full-union
posterior menu before Codex/IC. It must bind every symbol and all four ordered
formal evidence records; retrieval remains annotation-only and RiskAdvisor
remains advisory-only. Existing external-menu acceptance is not provenance.

The disconnected full-union producer is implemented as a byte-bound facade.
Its public build and validation entrypoints accept the canonical Stage 1
request/response bytes, `PosteriorRuntimeArtifacts`, one canonical eight-part
cost model, and the canonical per-symbol formal/cost artifacts. They do not
accept a prebuilt prior, calibration store, return model, bootstrap object,
correlation matrix, cost aggregate, posterior row, RiskAdvisor result, or
portfolio state.

`PosteriorRuntimeArtifacts` contains the actual canonical bytes for all four
frozen model bundles, base-rate observations, four-branch likelihood
observations, return-model parameters/training manifest, bootstrap
offsets/training manifest, and correlation matrix/training manifest. The
facade validates every byte and semantic SHA, private-root policy, schema,
protocol-attempt identity, training receipt, and cross-reference before it
constructs the internal numeric runtime. Likelihood runtime evidence requires
the same sample/cohort/outcome set for every formal branch, at least 300
samples per branch, and at least eight cohorts.

Q/F/M evidence is recomputed locally from the exact Stage 1 fact package. LLM
evidence must equal the sealed Stage 1 response and its model ID must equal the
immutable provider identity in the bound LLM model bundle. Retrieval cannot
enter scoring. The cost model carries all eight components in canonical order;
fee, slippage, and market-impact aggregates are derived internally and cannot
be supplied by the posterior caller.

The reusable `bayesian/v16` classes remain the internal numeric kernel and
legacy compatibility surface. They are not an evidence or authorization
entrypoint. Evidence-v2 consumers must enter through the byte-bound facade.

The following remain distinct explicit scopes and cannot be inferred from any
other receipt: research training capture, Factor research replay, Factor
production activation, H00300 acquisition, external TSA submission, v16
candidate generation, formal LLM execution, capital/holdings input, live human
new-risk authorization, Dashboard activation, and production/default pointer
switch. Broker, order, and trade authority are outside evidence-v2.

Until all producers, migrations, prospective observations, and authorizations
exist, exact blockers must include at least
`global_attempt_registry_authority_not_integrated` and
`evidence_v2_disconnected_from_authorizing_consumers`.

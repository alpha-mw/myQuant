# Module Map

## Unified authority

### `quant_investor.contracts`

Owns canonical JSON bytes, immutable artifact envelopes, semantic hashing,
compiled contract definitions, and validator identity. It imports no business
stage.

### `quant_investor.system`

Owns content-addressed objects, generation assembly and verification, the
single active pointer, CAS activation, pointer history, and fail-closed
suspension. Candidate builders never call activation.

### `quant_investor.factors.governance`

Owns Factor definitions, the one-time bootstrap evidence, prospective
preregistration and observation, evaluation, admission, deterministic weights,
and the separate active-versus-observed status projection.

### `quant_investor.intelligence`

Owns stable semantic research stages: decision context, Industry, Theme,
Fundamental, advisory review, investment decision, research portfolio,
forward observation, evaluation, evidence compilation, readiness, and
inspection. Its artifacts are candidates until bound by an active generation.

### `quant_investor.mainline`

Reads the active System generation and exposes the only formal result
projection. Uninitialized, blocked, suspended, mismatched, or tampered
generations return a fail-closed result and never scan another directory.

### `quant_investor.cli`

Provides the single `quant-investor` executable. All machine responses are one
canonical JSON line. Read/status commands use exit code 0 when the read itself
succeeds, formal-run unavailability and validation failures use 2, and internal
errors use 3.

## Independent ownership domains

These domains keep their existing ownership:

- `quant_investor.market` and Fundamental maintenance;
- Macro helpers;
- Strategy Record Store;
- Portfolio Cycle;
- Dashboard.

A System generation may bind their immutable resolved inputs but does not
advance or rewrite their independent current pointers.

## Dependency direction

Canonical data adapters and contracts feed Factor governance, then
Intelligence stages, then Mainline, then public readers. System supplies only
storage, verification, status, and activation primitives; it does not import
business stages.

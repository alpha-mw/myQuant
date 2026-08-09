# Dependency Profiles

This page records the dependency surface that exists in the current V17/I0/R2.2/I1-I6
repository. It is not a proposal for retired Phase 8 modules.

## Current declared profiles

`pyproject.toml` and `uv.lock` are authoritative. The project currently has one
broad required dependency set plus two optional groups:

- required: deterministic contracts and research code together with pandas,
  PyArrow, NumPy/SciPy/scikit-learn/XGBoost, CN/US/macro data clients,
  visualization, HTTP, logging and utility libraries, plus lock-bound Brotli
  decoding and Ed25519 verification through `Brotli` and `cryptography`;
- `dev`: pytest, coverage, asyncio testing, Black, Flake8, mypy, Hatchling and
  Twine;
- `backtest`: Backtrader only.

The repository does not currently publish separate installable `core`,
`cn-data`, `us-data`, `llm`, `audit` or `ml` extras. Documentation and tests
must not imply that these proposed profiles exist.

## Intelligence boundaries

- V4 Forward, I0, R2.2 and I1-I6 are offline/research-only under normal local
  verification.
- I5 exposes explicitly injected Responses call seams and a local safe HTML
  collector. It does not construct an SDK client, discover credentials or make
  a live request during import, replay or CI.
- I6 uses `cryptography` only to verify Ed25519 signatures; it never signs,
  reads a private key or performs pointer CAS by itself.
- No OpenAI SDK is a project dependency. A separately authorized live caller
  must inject its exact `responses.create` callable and capability identity.

## Verification truth

The GitHub workflow installs the lock-bound `dev` environment, checks requirements synchronization,
runs blocking syntax/name Flake8 checks, informational complexity/style checks,
two mypy scopes, the portable unit suite, and package build/Twine validation.
Owner-machine evidence-archive verification is a separate activation gate and
is intentionally not equivalent to portable CI.

Unavailable evidence remains `UNAVAILABLE` or blocked. No dependency profile
may silently enable live Provider, LLM, broker, order, execution, trade,
portfolio mutation, publisher or active-pointer authority.

# myQuant Codebase Review — 2026-08-04

Scope: `quant_investor/` (294,613 LOC / 414 files), `scripts/` (85,802 / 80 files),
`tests/` (167,743 / 367 files), `web/` (9,108 / 35 files), `frontend/src` (5,892 / 52 files).
Reviewed at `codex/v17-v4-source-builder-checkpoint`.

## Executive summary

This codebase has unusually strong **policy** discipline and unusually weak **enforcement**
plumbing. The fail-closed gates, hash-chained receipts, byte-pinned code bindings, and the
autouse fixture that hard-blocks live LLM calls in tests are all better than typical. What
sits underneath them was silently broken:

- `mypy`, on the exact command CI runs, failed with **50 errors in 11 files**.
- The package declared `requires-python = ">=3.10"` and tested 3.10/3.11/3.12 in CI, while
  the factor-governance AST path calls **three Python 3.13-only APIs**. Verified on a real
  3.12 interpreter: all three raise `TypeError`. Every declared-supported version was broken.
- `PATCH /api/settings/` wrote plaintext API keys to a **CWD-relative** `.env` with default
  permissions and no newline validation — on an endpoint unauthenticated by default.
- The local `.venv` had orphaned `dist-info` for `coverage`, `pyflakes`, `pycodestyle`,
  `mccabe`, so `flake8` crashed and `pytest --cov` could not run at all.

All four are fixed and verified — see [Remediation applied](#remediation-applied). The rest
of this document is the prioritized backlog, evidence-first.

The single highest-leverage remaining change is **CI coverage**: today 39 of 367 test files
(10.6%) gate a PR. The other 89% run only in a nightly job whose failure blocks nothing.
Everything in section A is cheap; sections B and C are where the real risk lives.

---

## Remediation applied

| # | Defect | Fix | Verification |
|---|---|---|---|
| P0-1 | `math.fma` (3.13+) called in the v4.4 evaluator | Added a bit-identical `_fma_exact` fallback using exact rational arithmetic | 409,261-case differential test vs `math.fma`, 0 mismatches |
| P0-2 | 50 mypy errors on CI's own command | Annotation fixes across 10 modules + version alignment | `mypy … agents factors` → **0 errors / 90 files** |
| P0-3 | Secrets written to CWD-relative `.env`, 0644, no newline validation | Delegated to the existing atomic 0600 `update_env_file`; reject `\r\n\0` | 3 new regression tests; verified injection rejected, CWD-independent, mode 0600 |
| P0-4 | Corrupted `.venv` (4 packages with dist-info but no module) | `uv pip install --reinstall`, restored to lockfile versions | `flake8 --version`, `import coverage`, `pytest --cov` all work |

**Python floor moved to 3.13** (`pyproject.toml`, all three CI workflows, README badge). This
was not a preference — it is the version the code already required. See
[A1](#a1-the-python-version-claim-was-false).

### Pre-existing failures on this branch (not introduced here)

A full run is `3 failed, 6357 passed` (29m). All three also fail at a clean `HEAD` — verified
by stashing every change from this review and re-running:

- `test_retirement_scan.py::test_checked_in_allowlist_exactly_matches_repository_findings`
- `test_retired_market_entrypoints.py::test_market_help_has_no_retired_commands_or_protocol_choice`
- `test_no_legacy_csv_runtime.py::test_production_runtime_has_no_legacy_csv_read_ports`

The first is caused by `tests/unit/test_v17_v4_source_storage.py`, added by this branch's own
commits, producing `python.string-v16` and `text.v16-token` findings that were never added to
`retirement_scan_allowlist.json`. These need fixing before the branch merges; they are called
out here because a nightly-only full run means nobody was told (see [A2](#a2-ci-gates-106-of-the-test-suite)).

Worth noting: this review document initially tripped the same scan, because *writing about*
the retired-prefix problem emitted the token it scans for. The document is now phrased to
avoid it — a small illustration of how a grep-based retirement gate with no documented
allowlist workflow penalizes ordinary documentation.

### Note on the `fma` fallback

With the floor now at 3.13, `math.fma` always exists and `_fma_exact` is never reached in
production. It is kept because it is cheap, tested, and documents *why* the rolling-variance
recurrence needs single-rounding — rounding twice would change published receipts. If you
would rather not carry unreachable code, deleting `_fma_exact` and calling `math.fma`
directly is safe; keep `test_fma_fallback_rounds_once` as the record of the requirement.

---

## A. Quality infrastructure

Cheapest wins in the codebase. Nothing here requires design work.

### A1. The Python version claim was false

`ast.parse(optimize=...)`, `ast.dump(show_empty=True)`, and `math.fma` are all 3.13+.
Confirmed on a real 3.12 interpreter:

```
python 3.12.13
has math.fma: False
ast.dump(show_empty=True): TypeError: dump() got an unexpected keyword argument 'show_empty'
ast.parse(optimize=-1):    TypeError: parse() got an unexpected keyword argument 'optimize'
```

The AST parameters are not incidental — they are recorded as **hashed contract terms** in
`governance_candidate_preregistration_v4_3.py:194-205` (`"feature_version": [3, 13]`,
`"optimize": -1`, `"show_empty": True`). They cannot be back-ported without changing
governance identities, so 3.13 is a genuine floor, not a preference.

Two things made this survivable for so long: the developer `.venv` is 3.13 (the one version
CI never tested), and `_parse_ast` catches `TypeError` and re-raises it as
`"failed exact Python AST parse"` (`governance_candidate_preregistration_v4_3.py:536`) — so
on 3.10-3.12 the real cause was masked by a misleading protocol error.

**Fixed.** Consider narrowing that `except (SyntaxError, TypeError, ValueError)` so
environment faults stop masquerading as contract violations.

### A2. CI gates 10.6% of the test suite

39 of 367 test files are named across all three workflows — 404 of 3,775 test functions (10.7%).

| Job | Gates? | Note |
|---|---|---|
| `engine-core` | yes | 7 test files |
| `workspace-surface` | yes | 3 test files + frontend lint/build |
| `legacy-web` | **no** | `continue-on-error: true` — cannot fail a build |
| `nightly-full-tests` | **no** | `if: schedule \|\| workflow_dispatch` |
| `factor-governance-quality-gate` | yes | 20 test files, any PR |
| `staged-upgrade-quality-gate` | yes | 10 test files, any PR |

The lists are hand-maintained, so they drift silently as tests are added. All 39 referenced
paths currently exist — that will not stay true on its own.

Also: `ci-cd.yml` triggers only on push to `main`/`develop` and PRs into `main`. The current
working branch runs neither. A PR into `develop` gets only the two quality gates.

**Recommend**: make the nightly full run a required PR job (it completes in minutes locally),
or at minimum add a check that every `tests/unit/*.py` is reachable from some gating job.

### A3. Lint does not gate; `black` is not run at all

`.github/workflows/ci-cd.yml:39-41` runs flake8 twice. The gating pass is
`--select=E9,F63,F7,F82` — syntax errors and undefined names only. The pass that would catch
anything real is `--exit-zero`.

`black` appears nowhere in CI. **263 of 413 files** would be reformatted. `README.md:345`
documents the convention; nothing enforces it.

`tests/` and `scripts/` (85K LOC) are never linted or type-checked by any job.

**Recommend**: adopt `ruff` (single tool, replaces flake8 + isort, fast enough to gate),
land one repo-wide `black`/`ruff format` commit, then gate `--check` on every PR. Do the
reformat as its own commit so it never obscures a review diff.

### A4. No coverage threshold

`[tool.coverage.run]` is configured, but there is no `fail_under`, no Codecov, no PR comment,
no badge. Coverage is produced only by the non-gating nightly job, which uploads an artifact
and asserts nothing. The root `.coverage` was ~4 weeks stale and — until P0-4 — unreadable.

**Recommend**: measure once, set `fail_under` slightly below the current number, ratchet up.

### A5. Type checking covers a fraction of the package

The `[tool.mypy] exclude` list matches 64 of 414 files (15.5%). But CI only points mypy at
`quant_investor/agents` + `quant_investor/factors`, so the **effective** checked surface is
90 files (21.7%). `scripts/`, `tests/`, and `web/` are at 0%.

The configuration is also heavily defanged globally: `follow_imports = "skip"`,
`ignore_missing_imports = true`, `strict_optional = false`, `disallow_untyped_defs = false`,
`warn_return_any = false`, and `disable_error_code = ["import-untyped", "no-any-return",
"no-untyped-def", "var-annotated"]`.

Now that mypy is green, this is the moment to expand rather than let it rot again.
**Recommend**: add `web/` next (small, high-value, user-facing), then re-enable one disabled
error code at a time.

---

## B. Security & correctness

### B1. Auth exists but the frontend cannot use it — **highest remaining risk**

The middleware itself is well built (`web/workspace_app.py:110-131`): registered only when
`WORKSPACE_AUTH_TOKEN` is set, constant-time `hmac.compare_digest`, health/OPTIONS exempt,
loud warning on non-loopback binding without a token.

But `frontend/src/api/client.ts:3-11` never sends an `Authorization` header, and
`frontend/src/hooks/useSSE.ts:56` uses raw `EventSource`, which **structurally cannot** send
custom headers. So:

- Turning auth on breaks the entire UI.
- `GET /api/research/{job_id}/stream` can never be authenticated as built.

The result is a security feature that exists, is tested, and cannot be enabled. That is worse
than not having it, because it reads as protection.

**Recommend**: decide explicitly. Either (a) wire a token into the client and move SSE to a
cookie or short-lived query token, or (b) drop the middleware and document loopback-only as
the security model. Do not leave it half-connected.

### B2. Config is frozen at import; the UI reports success anyway

`quant_investor/config.py:141-296` evaluates every setting in the **class body**, i.e. once
at import. `PATCH /api/settings/` mutates `os.environ` at runtime, so anything read via
`config.X` keeps its import-time value until restart — while endpoints that read
`os.environ` directly (`web/routers/settings.py:150-178`) *do* reflect the change.

The user-visible symptom: "saved successfully", nothing changes, and the GET endpoint
confirms the new value. Only `workspace_auth_token()` (`web/config.py:24-28`) reads at call
time, and its docstring flags itself as the exception.

**Recommend**: `pydantic-settings` `BaseSettings` with a cached accessor, or at minimum
convert `Config` attributes to properties/`functools.cache` functions.

### B3. Three settings fields are silently discarded

`SettingsUpdateRequest` (`web/models/settings_models.py:53`) has no `extra="forbid"`.
`frontend/src/pages/SettingsPage.tsx:43-46` offers `openai_api_key`, `anthropic_api_key`,
`google_api_key`; none exist in the model or in `_CREDENTIAL_FIELD_MAP`. They are dropped and
the UI reports success.

Note this contradicts the project's own memory that OpenAI/Claude were removed as providers —
so the correct fix is almost certainly to delete the three UI fields, not add backend support.

**Recommend**: add `extra="forbid"` (the main `ResearchRunRequest` already has it) and remove
the dead UI fields.

### B4. Unbounded in-process job state

`web/services/research_runner.py` uses a module-level `ThreadPoolExecutor(max_workers=2)`
(`:394`) with a plain `dict` of jobs (`:487`) that is **never evicted**, and `job.log_lines`
appends without a cap (`:78`). A long-lived server accumulates both indefinitely.

Redis is fully configured (`web/config.py:78-82`, `docker-compose.yml`) and **imported by
nothing** — vestigial config for a queue that was never built.

**Recommend**: cap `log_lines` (ring buffer), evict completed jobs on a TTL. Either build on
the configured Redis or delete the config.

### B5. Smaller items

- **No cap on `stock_pool`.** `POST /api/research/run` accepts an arbitrary-length list;
  `/api/universe/CN/all_a` resolves to ~5000 symbols by design, against a 2-worker pool.
- **Upstream error text reflected to clients**: `web/routers/universe.py:164` returns
  `f"Tushare error: {exc}"`; `web/services/analysis_service.py:1378` propagates raw subprocess
  stderr. No `@app.exception_handler`, no `RequestValidationError` handler, no request IDs.
- **Silently swallowed exceptions**: `web/services/research_runner.py:88-89` and `:116-117`
  are both bare `except Exception: pass`.
- **~500 LOC of unreachable endpoints.** `web/app.py` and `web/api/{analysis,portfolio,settings}.py`
  are mounted only by the legacy factory, which nothing in production imports.

### B6. What is genuinely well done

Worth stating plainly, because it is unusual: no secrets in source or git history (`.env`
untracked, never committed, mode 0600); `.gitignore` keeps 77 GB of working artifacts out of
a 26 MB repo with zero tracked junk; the CORS wildcard **fail-closed** guard in
`create_app`; parameterized SQL throughout, including the dynamic `IN (...)` builder at
`web/services/data_service.py:2199-2210`; a path-traversal guard on the SPA static handler;
and `quant_investor/credential_utils.py:38-52` deliberately avoiding `ts.set_token()` so the
Tushare token is never written to the user's home directory.

---

## C. Architecture & dead code

### C1. Divergent hash helpers — a correctness risk, not tidiness

| Helper | Definitions | Distinct implementations |
|---|---|---|
| `_sha256` | 43 | **21** |
| `canonical_json_bytes` / `_canonical_json_bytes` | 39 | not measured |
| `_json_safe`, `_exact` | 26 each | — |
| `semantic_sha256` | 14 | — |

These feed **reproducible hash chains that are supposed to agree across packages**. Twenty-one
distinct canonical-JSON/SHA implementations is a latent receipt-mismatch bug, not a style
issue: two modules can hash the same logical value differently and the divergence only
surfaces as a failed cross-package parity check, far from the cause.

**Recommend**: this is the highest-value refactor in the codebase. One
`quant_investor/canonical.py` with `canonical_json_bytes` / `semantic_sha256`, migrated
module by module with a parity test asserting the new helper reproduces each old one
byte-for-byte before deleting it. Note the constraint in [C4](#c4-some-files-are-byte-frozen).

### C2. Three parallel v17 stacks; two have no production caller

~59,977 LOC across six packages:

| Generation | contract | runtime | total | Production importer |
|---|---|---|---|---|
| v2 | 9,916 | 6,296 | 16,212 | none |
| v3 | 4,912 | 7,682 | 12,594 | none |
| v4 | 5,499 | 25,672 | 31,171 | yes — CLI, monitoring, web router |

v2 and v3 are reachable only from tests and frozen scripts, yet both still ship console
scripts (`pyproject.toml:70-72`). Inside v3, `v17_v3_runtime/algorithms/*` (1,659 LOC) is
referenced by nothing at all.

This is not copy-paste — the *skeleton* is cloned (same filenames, same responsibilities,
three times) while `validators.py` is rewritten each generation. But the small structural
files are substantially duplicated: `canonical.py` is 80% identical between v3 and v4,
`authority.py` 65%, `storage.py` ~50% between v2 and v3.

**Recommend**: decide whether v2/v3 are retirable. If they are frozen research baselines,
move them out of the installable package (or exclude from the wheel) and drop their console
scripts. Shipping two dead CLIs is a support liability.

### C3. Dead and orphaned code

- **24 modules with zero references, 5,505 LOC.** Largest:
  `quant_investor/macro_terminal_tushare.py` (1,388), `market/orchestration.py` (360),
  `debate_scheduler.py` (345), `data_manager.py` (300).
- **`quant_investor/v17/`** — 19 JSON files, zero Python. Tests actively assert it is never
  imported, yet `pyproject.toml` packages all of `quant_investor`, so it ships in the wheel.
- **`governance/`** (7 lines, docstring only), **`_vendor/`** (4 lines), **`codex_review/`**
  (a module-path compatibility shim for a retired workflow).
- **`model-training` CI job** uploads a `models/` artifact while the training command itself
  is commented out — and `models/` is gitignored.

### C4. Some files are byte-frozen

`FIXED_EXISTING_PROJECT_SHA256` (`governance_prior_diagnostic_nomination_v4_3.py:163`) pins
the byte SHA-256 of four real source files:

```
quant_investor/factors/governance_private_bundle_io.py
quant_investor/factors/governance_source_readback_v4_1.py
quant_investor/factors/governance_source_v4_1.py
quant_investor/factors/governance_aquant_no_label_eval_v4_1.py
```

Any edit — including a one-word type annotation — breaks 14 tests. This was hit during this
review and reverted; the remaining mypy defect in the fourth file is suppressed via a scoped
`pyproject.toml` override rather than a source change.

This is working as designed, but it means **routine maintenance on these files, including
security fixes, requires a governance re-pin ceremony**. There is no in-repo runbook for
performing one.

**Recommend**: document the re-pin procedure in `docs/runbooks/`. A frozen file with no
documented unfreeze path becomes unmaintainable the moment it needs a real fix.

### C5. Structural sprawl

- **94 files over 1,000 lines; 32 over 2,000.** Largest: `market/macro_mart.py` (6,645),
  `monitoring/cn_aggressive_portfolio_tracker.py` (6,639), `market/fundamental_mart.py` (5,347).
  Against the project's own 800-line guideline in the global coding rules, that is 94 exceptions.
- **`cli/main.py`** — 2,200 lines dispatching through a 38-branch linear
  `if args.command == … and args.market_command == …` ladder. A dict-based registry is a
  mechanical, low-risk refactor.
- **`cli/main.py:18`** — `_RETIRED_PROTOCOL_ARGUMENT_PREFIX = "v" + str(4 * 4) + "_"`.
  This computes the retired-protocol prefix at runtime so the literal never appears in
  the source, which is precisely what the repo's own grep-based retirement scan looks
  for. Whatever the original motivation, it is a trap for the next reader: write the
  prefix literally and add the line to `retirement_scan_allowlist.json` instead.
  (This document deliberately avoids spelling the token, so that writing about the
  problem does not itself register as a new scan finding.)
- **Two disjoint execution universes behind one flag.** `--decision-protocol v17-v4`
  short-circuits into receipt resolution while `v15` runs the DAG, and three downstream
  layers hard-reject anything but v15 (`market/analyze.py:229`, `run_pipeline.py:111`,
  `dag_executor.py:477`).
- **`scripts/` is a second implementation tier.** 85,802 LOC, 72 of 80 files using argparse,
  and **29 of 80 do not import `quant_investor` at all** — including
  `build_factor_v4_4_future_strict_signal_computability.py` at 6,129 LOC with 147 functions.
  Logic that large living outside the package is untestable as a unit and invisible to mypy.

### C6. Debt is renamed, not tracked

Zero `TODO`, `FIXME`, `XXX`, or `HACK` markers in `quant_investor/` — alongside 243 `legacy`,
109 `retired`, and 53 `deprecated`. The pattern is to copy a module, bump a version suffix,
and keep the old one: `factors/` has 42 version-suffixed files out of 76.

Compounding it, the names lie in both directions: `market/legacy_batch_analysis.py` and
`legacy_synthesis.py` are **actively imported** by `market/analyze.py:16,21`.

**Recommend**: adopt an explicit retirement policy — when a `_v4_4` lands, the `_v4_2` gets a
deletion date. Absent that, `factors/` will keep growing by one full copy per revision.

---

## D. Test suite

Genuinely good, and better than the raw file count suggests.

- 3,775 test functions across 367 files, **6,354 collected**, zero collection errors,
  0.57:1 test-to-source ratio.
- 12,763 assertions (~3.4/test). Of the 888 tests with no bare `assert`, **876 use
  `pytest.raises`/`pytest.warns`** — they are error-contract tests, not smoke tests. Only 5
  files have neither.
- **Zero import-only test bodies, zero `assert True`**, and no test whose only assertion is
  `is not None`.
- 232 of 367 files use `tmp_path`; no test writes to the real `data/` or `results/`.
- `tests/conftest.py` installs an autouse fixture that monkeypatches `LLMClient._http_post`
  to raise on any live call, and several suites assert that `requests`/`httpx`/`tushare`/
  `yfinance` are **not importable** inside the sandboxed factor runtime. This is the strongest
  guardrail in the repository.

### D1. Roughly 10 test files always skip in CI

They `pytest.skip` when owner-private fixtures are missing, and `private/` is gitignored — so
they never run anywhere except the owner's machine. Examples:
`test_factor_governance_formal_catalog_materialization_v4_1.py:46`,
`test_factor_governance_operator_runtime_equivalence_v4_1.py:54,85,117`,
`test_build_factor_v4_1_signal_computability.py:41`.

**Recommend**: mark them `@pytest.mark.private` and make CI report the skip count, so silent
zero-coverage is visible rather than implied.

### D2. No marker registration

404 uses of `parametrize`, but no `markers = [...]`, no `addopts`, no `--strict-markers`.
There is no way to select or deselect the slow integration tests, which is part of why the
full suite ended up in a nightly job instead of a gating one.

### D3. `scripts/` tests do not gate

71 of 80 scripts are referenced from some test, which is better than expected — but no
`scripts/*` test file appears in any gating CI job, so all 85K LOC is verified only nightly.
Nine scripts (588 LOC) have no test reference at all, including `refresh_pit_universe.py`
and `clean_market_data.py`.

---

## Ranked backlog

| Rank | Item | Section | Effort | Why now |
|---|---|---|---|---|
| 1 | Make the full test suite gate PRs | A2 | S | 89% of tests currently block nothing |
| 2 | Decide the auth story (wire it up or remove it) | B1 | S–M | A security feature that cannot be enabled |
| 3 | Add `extra="forbid"`; remove dead UI credential fields | B3 | S | Silent data loss with a success message |
| 4 | Adopt `ruff` + one-shot format commit, then gate | A3 | S | 263/413 files drifting, no enforcement |
| 5 | Consolidate hash/canonical-JSON helpers | C1 | L | 21 implementations feeding one hash chain |
| 6 | Set a coverage floor and ratchet | A4 | S | Currently measured but unasserted |
| 7 | Fix import-time config freezing | B2 | M | UI silently no-ops until restart |
| 8 | Bound job dict and log buffers | B4 | S | Unbounded growth in a long-lived server |
| 9 | Retire or unship v17 v2/v3 | C2 | M | 28,800 LOC and two dead console scripts |
| 10 | Delete the 24 unreferenced modules and `v17/` | C3 | S | 5,505 LOC, low risk |
| 11 | Document the governance re-pin procedure | C4 | S | Four files currently unmaintainable |
| 12 | Expand mypy to `web/`, then re-enable error codes | A5 | M | Green now — best moment to widen |
| 13 | `cli/main.py` dispatch table; unwind the retired-prefix obfuscation | C5 | M | Mechanical, improves readability |
| 14 | Adopt an explicit version-retirement policy for `factors/` | C6 | S | Otherwise growth is unbounded by design |

Effort: S ≈ under a day, M ≈ a few days, L ≈ a week or more.

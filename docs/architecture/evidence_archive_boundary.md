# Evidence archive boundary

`quant_investor/factors/governance*.py` is 41 files and 61,616 lines — 32% of
the 192,325-line package — and no runtime entry point reaches any of it. That
combination normally means dead code. Here it does not, and this document
exists so that conclusion is not re-derived, and acted on, every time someone
reads the tree.

## The two layers

**Runtime layer.** What `quant-investor` (`quant_investor/cli/main.py`) and
`daily_runner.py` actually execute: `market/` ingestion, `v17_mainline/` pointer
authority, `v17_v4_contract/` + `v17_v4_runtime/`, `macro/`, `portfolio_cycle/`,
`intelligence/`, and `factors/` modules that are *not* named `governance_*`.

**Evidence archive layer.** `quant_investor/factors/governance*.py` plus the 18
`scripts/build_factor_v4_*.py` builders (23,938 lines). Each builder publishes a
sealed, owner-private evidence bundle for one governance milestone. Once a
milestone is sealed, its builder can never be meaningfully re-run — which is why
several generations (v4.1, v4.2, v4.3, v4.4) sit side by side rather than one
superseding the next.

The archive is reachable only from those builders and from `tests/`.

## Why it cannot be deleted, moved, or edited

`quant_investor/factors/governance_prior_diagnostic_nomination_v4_3.py` defines
`FIXED_EXISTING_PROJECT_SHA256`, a dict **keyed by repository-relative path**
that pins the exact byte hash of:

- `quant_investor/factors/governance_private_bundle_io.py`
- `quant_investor/factors/governance_source_readback_v4_1.py`
- `quant_investor/factors/governance_source_v4_1.py`
- `quant_investor/factors/governance_aquant_no_label_eval_v4_1.py`

`tests/unit/test_factor_governance_prior_diagnostic_nomination_v4_3.py`
re-hashes those paths and compares. Because the keys are paths, **renaming or
moving a file breaks the seal exactly as surely as editing one does.** Any of
those four operations turns a green suite red, and the failure is a governance
failure, not a test-maintenance nuisance.

`pyproject.toml` documents a downstream consequence: a `type-var` defect in
`governance_aquant_no_label_eval_v4_1.py` is suppressed through a mypy override
rather than fixed in place, because fixing it in place would change the bytes.

Changing anything in this layer requires a governance re-pin, not a refactor.

## Consequences for tooling and review

- **The 800-line file cap does not apply here.** Many of these modules are
  thousands of lines. Splitting them is not available.
- **"Unreferenced from any entry point" is not evidence of deadness here.** It
  is the expected state. Apply the usual dead-code reasoning to the runtime
  layer only.
- **Do not "clean up" superseded generations.** v4.1 and v4.2 living beside v4.4
  is the archive working as designed.

## The invariant that is enforced

The runtime layer must never import the archive. This is checked by
`test_runtime_entrypoints_do_not_import_the_evidence_archive` in
`tests/unit/test_v17_active_surface_cleanup.py`, which asserts that neither
`quant_investor/cli/main.py` nor `quant_investor/automation/daily_runner.py`
mentions `governance_`.

If a future feature genuinely needs governance output at runtime, it should read
a published evidence bundle from disk — not import the archive.

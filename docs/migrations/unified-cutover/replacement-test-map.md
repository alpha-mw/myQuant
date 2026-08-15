# Replacement Test Map

This is the acceptance template for the hard cutover. Each legacy surface needs
both a negative removal assertion and a positive stable replacement assertion.
Passing only one side leaves the migration incomplete.

The complete machine-verified disposition ledger is
[`replacement-test-map.json`](replacement-test-map.json). It binds the approved
baseline commit and tree, the selected deleted-test set, and every disposition
with canonical hashes. The resolver must reconstruct that baseline test set,
require exactly one row per deleted test path, reject a row whose baseline test
is still tracked, and AST-resolve each replacement selector in the intended
checkout.

Each machine row contains exactly `baseline_test_path`, `disposition`,
`replacement_test_selectors`, and `behavior_reason`. `REPLACED` requires at
least one sorted exact `tests/...py::test_node` selector;
`BEHAVIOR_INTENTIONALLY_REMOVED` requires an empty selector list and a concrete
reason. Missing, duplicate, stale, unresolved, or invalid rows hard-block the
cutover. The overview below routes stable areas; it is not a substitute for the
complete ledger.

The frozen ledger contains 151 baseline test rows selected by 60 exact seed
patterns: 126 are `REPLACED` and 25 are
`BEHAVIOR_INTENTIONALLY_REMOVED`. Its replaced rows bind 556 selector
occurrences spanning 130 unique exact test nodes, including the real-checkout
absence guard. These counts are
descriptive readback only; the canonical JSON and its hashes remain
authoritative.

| Area | Stable target | Positive contract | Negative legacy contract | Side-effect boundary | Test selector |
|---|---|---|---|---|---|
| Contracts | `quant_investor.contracts` | Canonical serialization, registry, seal, validation, and byte hash agree | Version-named contract import is unavailable | No filesystem writes unless an explicit seal target is supplied | `tests/unit/test_unified_contract*.py` |
| System | `quant_investor.system` | Exact generation verifies, status is deterministic, and activation is exact-target only | Legacy system import/command is unavailable | Status/verify are read-only; `system activate` is the only normal `_active.json` writer | `tests/unit/test_unified_system*.py` |
| Factor governance | `quant_investor.factors.governance` | Bootstrap, prospective evidence, admission, and status enforce their distinct states | Version-named governance import and CLI route are unavailable | Status is read-only; no admission, activation, or CAS | `tests/unit/test_unified_factor*.py` |
| Intelligence | `quant_investor.intelligence` | Stable compile, forward, evaluate, readiness, and inspect routes bind exact evidence | Removed subcommands and executable fail explicitly | No mainline, factor admission, activation, portfolio, or broker mutation | `tests/unit/test_unified_intelligence*.py` |
| Mainline | `quant_investor.mainline` | One exact active generation resolves to a read-only projection | Version-named reader/import and scan-latest behavior are unavailable | Reader performs zero writes | `tests/unit/test_unified_mainline*.py` |
| CLI | `quant-investor` | Every documented stable mapping dispatches with deterministic errors and exit codes | `quant-investor-v17-v4` and removed subcommands are absent | Help and read commands perform zero writes | `tests/unit/test_unified_cli*.py` |
| Migration resolver | `operations/unified_cutover/` and its registered resolver | Every declared source reaches exactly one allowed target | Ambiguous, undeclared, or stale source blocks | No external paths; no protected runtime-state mutation | `tests/unit/test_unified_migration*.py` |
| Codex projections | `operations/codex/` | Skill tree and automation semantic projection match the manifest | Installed state is not treated as repository state | No installation, schedule update, or activation | `tests/unit/test_unified_codex*.py` |

The CI selector must expand `tests/unit/test_unified_*.py` safely and fail when
no matching tests exist. Do not quote a wildcard as a literal pytest path and
do not rely on shell-specific accidental expansion.

## Proposal template

| Legacy identity | Stable identity | Fixture/input | Expected success | Expected legacy failure | Protected pre/post hashes | Test owner | Status |
|---|---|---|---|---|---|---|---|
| `<exact import, CLI route, path, or artifact kind>` | `<exact stable target>` | `<fixture path and hash>` | `<output/status/exit code>` | `<exception/blocker/exit code>` | `<paths and equality rule>` | `<owner>` | `<planned|passing|blocked>` |

This proposal row is for review before regeneration of the canonical ledger; it
is not acceptance evidence. Status may become `passing` only when both
assertions run against the same reviewed checkout. Documentation, a fixture, or
the presence of source files is not execution proof.

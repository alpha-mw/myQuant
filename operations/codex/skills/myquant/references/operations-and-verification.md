# Operations and Verification

Use this reference for commands, current artifacts, capability/source state,
maintenance, or change validation. Keep operations subordinate to the
investment question.

## Source check

Before a capability claim or write:

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse main
git rev-parse origin/main
quant-investor system verify
quant-investor system status
```

Verify the package, entrypoint, test, and artifact in the intended ref. Modified,
untracked, other-branch, fixture-only, or documented material does not establish
merged or active authority. Verify import origins when a virtual environment may
lag the checkout.

## Stable commands

Inspect the exact local help before a run:

```bash
quant-investor system verify --help
quant-investor system status --help
quant-investor system activate --help
quant-investor factor status --help
quant-investor research compile-evidence --help
quant-investor research readiness --help
quant-investor research inspect --help
quant-investor research forward --help
quant-investor research evaluate --help
```

The secondary executable and its old subcommands are removed. Do not create
aliases or wrappers for those removed routes, dynamic-import
fallbacks, or argument pass-throughs. `market download` remains the explicitly
documented compatibility alias for `market maintain`; it is unrelated to this
runtime cutover. The repository migration mapping is in
`docs/migrations/unified-cutover/cli-mapping.md`.

`system verify`, `system status`, `factor status`, `research readiness`, and
`research inspect` are read/verification surfaces. They do not publish,
activate, mutate a portfolio, connect a broker, or trade. `system activate` is
the only normal `results/system/_active.json` writer; it accepts only an exact
validated immutable generation and requires filesystem write permission.
`research forward`, `research evaluate`, and `research compile-evidence` remain
research-only and require their exact registered inputs and write boundaries.

## Public reads

These surfaces read the same already-active generation:

```bash
quant-investor research run \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --strategy-id <canonical-strategy-id>

quant-investor market analyze \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --strategy-id <canonical-strategy-id>

quant-investor market run \
  --workspace-root /Users/maxwell/mySpace/myQuant \
  --strategy-id <canonical-strategy-id>
```

They do not create candidates, portfolios, publications, or activation. A
missing or invalid active generation stays unavailable or blocked; readers do
not scan for the latest run.

The stable system store owns exact content-addressed objects beneath
`results/system/objects/`, immutable manifests beneath
`results/system/generations/`, retained pointer bytes beneath
`results/system/pointer_history/`, and only `results/system/_active.json` as the
current pointer. Status may be `UNINITIALIZED`, `PARTIAL`, `ACTIVE`, `BLOCKED`,
or `SUSPENDED`; its exact public keys are `state`, `verified`,
`active_pointer_sha256`, `generation_id`, `readiness`, `blockers`, and
`external_routing_state`. External routing is independently `ACTIVE`,
`UNCONFIRMED`, `SYSTEM_ACTIVE_AUTOMATION_DISABLED`, or
`SYSTEM_EXTERNAL_ROUTING_DRIFT`. Do not collapse either state into a boolean.

## Factor health

Use `quant-investor factor status` with exact explicit inputs. Verify strict CN
Parquet/PIT coverage, the admitted factor-set identity, prospective evidence,
runtime contracts, statistics, costs, maturity, factor/family identity, weights,
and fresh health required by the registered contract.

Keep bootstrap, prospective research, admission, health, stock candidates, and
portfolio state separate. Data-blocked is not an alpha failure, but it blocks a
healthy/readiness claim. Status is never active-generation publication or
compare-and-swap authority. Retired factor-state surfaces must remain absent;
their presence is a cutover blocker.

The status projection contains exactly `status_id`, `active`, `observed`,
`readiness`, `blockers`, and `activation_mutation_authorized`. Preserve the
separate active lane/route and observed state; do not collapse them into one
health label.

## Minimum data check

For ordinary investment work, check only what the conclusion needs:

1. market/PIT date, universe coverage, and tradability;
2. Fundamental freshness for earnings, valuation, or quality claims;
3. Industry/Theme identity evidence when used for admission or concentration;
4. Macro/release evidence only when used for thesis or portfolio risk;
5. no CSV, stale, mock, inferred, cached, or latest-by-mtime fallback.

Read-only market health check:

```bash
quant-investor market storage-validate --market CN
```

Do not infer suspension from a missing bar. For explicit maintenance, inspect
`market maintain --help` and obtain separate authority before provider calls,
promotion, recovery writes, or materialization.

## Verification

Run the narrow stable tests for the changed responsibility from
`tests/unit/test_unified_*.py`. For broad changes, run:

```bash
uv run pytest tests/unit -q
uv run flake8 quant_investor --count --select=E9,F63,F7,F82 --show-source --statistics
uv run mypy quant_investor/contracts quant_investor/system \
  quant_investor/factors/governance quant_investor/intelligence \
  quant_investor/mainline quant_investor/cli --ignore-missing-imports
```

A passing check proves only the exact source basis tested. It does not admit a
factor, activate a result, authorize new risk, or grant execution authority.

## External deployment boundary

Repository files under `operations/codex/` are inert deployment copies. Verify
their manifest, YAML/TOML/JSON syntax, semantic projection, and byte mapping,
but do not write `~/.codex`, update a schedule, or activate an automation unless
the user separately authorizes that exact external deployment.

Never call live providers or models, mutate a paper ledger, publish, invoke
`system activate`, connect a broker, create an order, execute, or trade merely
because a builder, test, or deployment copy exists.

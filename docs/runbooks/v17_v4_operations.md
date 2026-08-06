# V17 v4 Mainline Operations

## 1. Preconditions

- Work from a clean, reviewed build of the V17 v4 contract and runtime.
- Use strict CN Parquet canonical data; validate the active market pointer,
  manifest, PIT membership, Fundamental and Macro generations, and cutoff.
- Require the complete Factor, risk, holdings, and portfolio closure expected
  by the mainline run contract.
- Keep provider acquisition or data maintenance separate from a decision run.

Do not proceed on missing data, missing receipts, stale pointers, schema/hash
drift, or unresolved readiness blockers.

## 2. Build an immutable run

The governed mainline writer produces one closed
`myquant.v17.v4.mainline-run.v1` at:

```text
results/v17_mainline/strategies/<strategy-id>/runs/<run-id>/run.json
```

Validate the exact run bytes and every transitive reference after persistence.
Creating this run does not make it public.

## 3. Activate operationally

Operational activation is a separate act from installing or deploying code.
Advance only:

```text
results/v17_mainline/strategies/<strategy-id>/_active.json
```

The activation writer must receive the expected pointer prevalue, revalidate
the immutable run, perform an atomic CAS replacement, and read back the exact
proposed bytes. Any mismatch stops with no pointer change. Never activate by
renaming a run, editing the pointer manually, or scanning for the latest run.

## 4. Read the public run

All commands below read the same pointer and return the same
`myquant.v17.v4.mainline-public-run.v1` authority chain:

```bash
quant-investor research run \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>

quant-investor market analyze \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>

quant-investor market run \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>
```

Web readers use `GET /api/research/{strategy_id}` and may pin
`expected_pointer_sha256`. A stale expected SHA is a blocker, not permission to
read another generation.

## 5. Expected fail-closed states

| Condition | Result | Writes |
|---|---|---:|
| Active pointer absent | `V17_MAINLINE_UNINITIALIZED` | 0 |
| Pointer/run/closure invalid | `V17_MAINLINE_BLOCKED:<blocker>` | 0 |
| Market is not CN | `V17_MARKET_UNSUPPORTED` | 0 |
| `market backtest` | `V17_BACKTEST_UNAVAILABLE` | 0 |

After any failure, inspect the exact pointer and run reference. Do not create a
placeholder pointer, substitute a Shadow session, scan an older result, or
infer success from a run directory.

## 6. Forward Shadow research

Forward evidence is deliberately separate:

```bash
quant-investor-v17-v4 run-forward \
  --workspace-root /absolute/path/to/myQuant \
  --request-path data/private/v17_v4_runs/forward_requests/<request_id>.json \
  --request-sha256 <sha256>
```

Its final session reference proves only a completed Shadow observation. It
does not activate the mainline, publish a public run, or authorize new risk,
broker, order, execution, or trade activity.

## 7. Verification

Run the narrow contract and CLI tests for the changed surface, then run the
repository's staged upgrade quality gate for a broad release. Verification
must include missing-pointer and invalid-pointer no-write assertions, CN-only
routing, public DTO equality across surfaces, unsupported-backtest no-write,
and Shadow/mainline isolation.

## 8. CN Macro and release-calendar maintenance

`quant-investor market macro-maintain` is the only registered writer for the
daily CN Macro observation roll and official release-calendar coverage
extension. It is dry-run by default. A committing run requires both
`--allow-live` and `--commit`, exact expected-before SHA-256 values for the
release-calendar and Macro-observation pointers, and exact path+SHA bindings
for the immutable market snapshot manifest, its target-date coverage manifest,
and the full-A scope artifact.

The command calls only the NBS and PBC official coverage URLs. It stores each
complete HTTP response entity as hash-bound `coverage_response` evidence and a
v2 coverage receipt that binds that response. The common cutoff is frozen after
both response entities have completed and before either pointer write. The
release calendar is published first by CAS; the local breadth observation then
rolls against the same cutoff, target, market-open-days evidence and canonical
market inputs. If the second stage fails, the release child may remain as an
independently healthy promotion and the receipt reports `PARTIAL`. A failed
stage never substitutes stale, inferred, public-fallback or manual values.

The command must not import or restore the retired
`quant_investor.market.macro_mart` producer. It does not run analysis, create
candidates or portfolios, activate Factor or Mainline state, render Dashboard
output, connect a broker, create an order, or trade.

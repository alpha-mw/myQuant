# Factor Production

This runbook covers the isolated Factor production authority only. It never
activates System, Mainline, Investment, portfolio, Strategy Record, broker,
order, trade, or funds-transfer authority.

## Read-only commands

`factor status` remains the prospective governance-status builder and requires
an exact request file. It is not a production pointer read.

Use these commands for production reads:

```bash
quant-investor factor production-status --workspace-root <workspace>
quant-investor factor production-verify --workspace-root <workspace>
quant-investor factor production-signal \
  --workspace-root <workspace> \
  --factor-id pv_low_dollar_volume_5d
```

The production readers must report:

```text
authority_domain = FACTOR_PRODUCTION_ONLY
system_runtime_state = NOT_EVALUATED
grants_system_authority = false
grants_trading_authority = false
```

Signal reads accept only LOW or W80 and return values sealed in the verified
active Factor generation. W75 remains control-only and cannot be read as an
active production signal.

## First activation

The only public first-activation command is:

```bash
quant-investor factor production-activate \
  --workspace-root <workspace> \
  --market-data-root <strict-market-root> \
  --calendar-capture-root <published-calendar-capture-root> \
  --expected-calendar-success-sha256 <sha256> \
  --release-repository-root <exact-release-repository-root> \
  --activation-inputs factor-activation-inputs.json \
  --expected-activation-inputs-sha256 <sha256> \
  --expected-empty
```

`factor-activation-inputs.json` is canonical owner-controlled JSON with exactly:

```text
as_of
deployed_release_ref
factor_policy_ref
factor_active_set_ref
factor_validation_attestation_ref
factor_implementation_refs
final_commit
final_tree
```

The command internally performs strict source preparation, immutable Factor
generation construction, current release-install and legacy-zero-call replay,
activation-byte preparation, the sole expected-EMPTY atomic no-replace CAS,
permanent marker publication/readback, and final Factor verification. It does
not accept caller-created receipts, bundles, pointers, markers, scanner
callbacks, counts, or authority flags.

Stop before activation if `_active.json` or `_production_complete.json` already
exists, any source/release/hash/recomputation gate fails, or `--expected-empty`
is absent. Never delete or overwrite an existing pointer to retry.

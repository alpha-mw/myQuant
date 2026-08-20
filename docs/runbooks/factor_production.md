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

Prepare the frozen release first from a clean detached checkout. The command
returns `installed_python`, the workspace-relative release-input path, and its
exact SHA. Calendar capture and Factor activation must then run through that
same installed interpreter, with an empty `PYTHONPATH` and a working directory
outside the source checkout:

```bash
quant-investor system release-prepare \
  --workspace-root <workspace> \
  --release-root <owner-only-release-root> \
  --release-repository-root <clean-detached-checkout> \
  --final-commit <commit> \
  --final-tree <tree>

PYTHONPATH= <installed-python> -I -m quant_investor system calendar-capture \
  --workspace-root <workspace> \
  --capture-parent <owner-only-capture-parent> \
  --release-repository-root <clean-detached-checkout> \
  --capture-root-name <unique-root-name> \
  --cutoff-date <YYYYMMDD> \
  --release-install-input <workspace-relative-input-path> \
  --expected-release-install-input-sha256 <sha256>
```

The only public first-activation command is likewise executed by the installed
interpreter:

```bash
PYTHONPATH= <installed-python> -I -m quant_investor factor production-activate \
  --workspace-root <workspace> \
  --market-data-root <strict-market-root> \
  --calendar-capture-root <published-calendar-capture-root> \
  --expected-calendar-success-sha256 <sha256> \
  --expected-empty
```

The command internally performs strict source preparation, immutable Factor
generation construction, installed implementation discovery, Bootstrap policy,
active-set and validation-receipt construction, current release-install and
legacy-zero-call replay,
activation-byte preparation, the sole expected-EMPTY atomic no-replace CAS,
permanent marker publication/readback, and final Factor verification. It does
not accept caller-created release, policy, active-set, implementation, receipt,
bundle, pointer, marker, scanner, count, commit/tree, as-of, or authority inputs.
Those identities are derived from the exact installed release, the published
Calendar capture and the strict Market-bound PIT snapshot.

Stop before activation if `_active.json` or `_production_complete.json` already
exists, any source/release/hash/recomputation gate fails, or `--expected-empty`
is absent. Never delete or overwrite an existing pointer to retry.

An interruption before release install or release-input publication leaves no
accepted deterministic target; rerun the same command and exact commit/tree.
An interruption after atomic publication reuses the exact verified winner. A
concurrent conflicting winner is rejected. After a Factor pointer CAS but
before marker publication, rerunning the same installed command performs only
exact marker recovery; it never performs a second CAS.

# Filling a Paper session at its open

The Paper account (`aggressive-tech-manufacturing-paper-v1`) acts on orders the
owner policy produced after one session's close and fills them at the next
session's open. This is the exact sequence for a session `D`.

Everything runs from the **installed release** (the account's writer verifies the
release input), not from the repository:

```bash
INSTALL=/Users/maxwell/mySpace/myQuant-release-authority/2259530d7d5dfd11215f9f804306c06a32444e7e-unified-runtime/installs/2259530d7d5dfd11215f9f804306c06a32444e7e-d8748d0bf32228e78cd9070568a3bd20ac84e226c68e3c62936232686a6ea2ef
INPUT=results/releases/7b88feb4beb0f7c389e4da9359bc2aed29f724a17c0f7d612090a849493d62ce/release-install-input.json
INPUT_SHA=7b88feb4beb0f7c389e4da9359bc2aed29f724a17c0f7d612090a849493d62ce
ACCOUNT=aggressive-tech-manufacturing-paper-v1
CHECKOUT=/Users/maxwell/mySpace/myQuant-release-checkouts/2259530d7d5dfd11215f9f804306c06a32444e7e-unified-runtime
```

## 0. Preconditions

- The session's strict snapshot is published (`data/parquet/cn/_latest.json`).
- The plans for this session exist: `paper_shadow_orders.py --write` was run
  after the **previous** session's close, and `plans.json` carries
  `eligible_from_trade_date = D`.
- The account is `READY`:

  ```bash
  PYTHONPATH= $INSTALL/bin/python -I -m quant_investor paper account-status \
    --workspace-root /Users/maxwell/mySpace/myQuant --account-id $ACCOUNT
  ```

## 1. Capture the session's price limits

```bash
cd /Users/maxwell/mySpace/myQuant
.venv/bin/python scripts/operations/paper_capture_session_limits.py --trade-date D --write
```

Output names `evidence_path` and `evidence_sha256` — keep both.

## 2. Emit intent + eligibility files

```bash
.venv/bin/python scripts/operations/paper_session_eligibility.py \
  --account-id $ACCOUNT --trade-date D \
  --limits <evidence_path> --expected-limits-sha256 <evidence_sha256> \
  --plans <private path>/plans.json --write
```

`skipped` must be empty; each entry lists the symbol's `action`, `shares` and the
two refs. A skipped symbol means evidence is missing — fix the evidence, do not
hand-edit a file.

## 3. Preview, then fill

```bash
PYTHONPATH= $INSTALL/bin/python -I -m quant_investor paper risk-exit-preview \
  --workspace-root /Users/maxwell/mySpace/myQuant --account-id $ACCOUNT \
  --intent <intent_ref.path> --expected-intent-sha256 <intent_ref.sha256> \
  --eligibility <eligibility_ref.path> --expected-eligibility-sha256 <eligibility_ref.sha256>
```

The preview must report `FILLED` (or a named pending) and prints
`expected_current_pointer_sha256`. Then commit it:

```bash
PYTHONPATH= $INSTALL/bin/python -I -m quant_investor paper risk-exit-run \
  --workspace-root /Users/maxwell/mySpace/myQuant --account-id $ACCOUNT \
  --intent <intent_ref.path> --expected-intent-sha256 <intent_ref.sha256> \
  --eligibility <eligibility_ref.path> --expected-eligibility-sha256 <eligibility_ref.sha256> \
  --expected-current-pointer-sha256 <preview pointer> \
  --release-install-input $INPUT \
  --expected-release-install-input-sha256 $INPUT_SHA \
  --release-repository-root $CHECKOUT --allow-write
```

Repeat per symbol; each fill advances the account pointer, so re-read it from the
previous command's output. Exact replay of the same intent returns
`NO_ACTION_ALREADY_APPLIED`; a conflict fails closed.

## 4. Confirm

```bash
PYTHONPATH= $INSTALL/bin/python -I -m quant_investor paper verify \
  --workspace-root /Users/maxwell/mySpace/myQuant --account-id $ACCOUNT
```

Position count, cash and the ledger's `acquisition_lots_json` must show the new
shares as unsettled until the next session.

## Notes

- A limit-up open pends a buy; suspension, a pending corporate action, or an open
  at/below limit-down pends a sell. An intent expires after three open sessions
  (`CANCEL_AND_REQUIRE_NEW_REVIEW`) and a buy that cannot afford one lot is
  terminally `SKIPPED`, never partially filled.
- The account is a Paper account: `broker=false`, `real_order=false`,
  `actual_holdings_mutation=false` on every record.

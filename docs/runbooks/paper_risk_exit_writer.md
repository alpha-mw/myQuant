# Paper Risk Exit Writer v1

Release A provides a registered, sell-only Paper execution domain without
activating a live Paper account. It is independent from actual Store, Factor,
System, broker, real orders, and actual holdings.

## Authority

- Writer: `cn-paper-risk-exit-writer.v1`
- Policy: `owner-paper-risk-execution-policy-20260901-v1`
- Policy SHA: `d3f86f3ba26556d084eebc48136864a5ba858efe75c9c9d139fb99627d746961`
- Actions: `REDUCE_25`, `REDUCE_50`, `EXIT_100`
- Broker/real order/live execution/actual holdings: always false

The writer consumes only explicit SHA-bound `paper-risk-intent.v1` and
`paper-input-eligibility.v1` files. It never parses Markdown, scans latest
results, reads automation memory as authority, or clones actual holdings.

## CLI

Read-only surfaces:

```bash
python -m quant_investor paper writer-status --workspace-root <root>
python -m quant_investor paper account-status --workspace-root <root> --account-id <id>
python -m quant_investor paper risk-exit-preview \
  --workspace-root <root> --account-id <id> \
  --intent <path> --expected-intent-sha256 <sha> \
  --eligibility <path> --expected-eligibility-sha256 <sha>
python -m quant_investor paper verify --workspace-root <root> --account-id <id>
```

Write surfaces require `--allow-write`, exact expected pointer state, and a
deeply replayed installed release input:

```bash
python -m quant_investor paper account-register ... --allow-write
python -m quant_investor paper risk-exit-run ... --allow-write
```

Release A must not invoke either write command in the live workspace because no
owner-registered Paper account exists. Expected status is
`PAPER_ACCOUNT_NOT_REGISTERED` and no `results/paper/accounts` directory.

## Execution

- First valid Calendar OPEN session on/after `eligible_from_trade_date`
- Raw, unadjusted next open
- Sell price `floor_to_CNY_0.01(open × 0.95)`
- Limit/suspension/corporate-action evidence missing: pending, never guessed
- `REDUCE_25/50` floor to 100-share lots; `EXIT_100` may clear an odd lot
- Settled acquisition lots enforce T+1
- Third evaluated OPEN session expires the intent; suspension does not extend it
- Partial fills are disabled
- Commission 0.01% minimum CNY5; transfer 0.001%; stamp duty 0.05% sell-only

## Storage

Paper owns only:

```text
results/paper/accounts/<account_id>/
  registration.v1.json
  .writer.lock
  _record_store/current.v1.json
  _record_store/pointer_history/
  records/<sequence>-<identity>/
```

Every transaction writes immutable intent, eligibility, order/fill or pending,
Parquet ledger, account state, receipt, and closure before expected-preimage
pointer CAS. Exact replay writes zero bytes; conflicts fail closed. Orphan
records are never auto-adopted.

The complete schema and atomicity contract is in
`docs/plans/paper_risk_exit_writer_v1.md`.

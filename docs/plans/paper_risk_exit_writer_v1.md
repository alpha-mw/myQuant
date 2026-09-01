# Paper Risk Exit Writer v1 — Release A Contract

Status: design gate only; no live Paper account or write authority.

## Scope

Release A implements an independent `quant_investor.paper` domain for sell-only
Paper risk exits. It does not register a live account, clone actual holdings,
connect a broker, create a real order, modify actual Store/System/Factor, or
activate an automation. All write-path tests use synthetic roots below
`/private/tmp`.

The live workspace completion state remains `PAPER_ACCOUNT_NOT_REGISTERED`.

## Exact schemas

Every JSON document is canonical UTF-8 with sorted keys, compact separators,
no terminal newline, no duplicate keys, no non-finite numbers, and exact fields.
Every document contains `schema_version` and `semantic_sha256`; semantic SHA is
computed over the exact canonical document without `semantic_sha256`.

### `paper-writer-registration.v1`

Exact fields:

```text
schema_version, semantic_sha256, writer_id, writer_version,
allowed_account_type, allowed_actions, allow_new_risk,
paper_order, paper_fill, paper_ledger_mutation,
broker, real_order, live_execution, actual_holdings_mutation
```

Fixed writer is `cn-paper-risk-exit-writer.v1`; actions are exactly
`REDUCE_25, REDUCE_50, EXIT_100`; all real-authority flags are false.

### `paper-account-registration.v1`

Exact fields:

```text
schema_version, semantic_sha256, account_id, account_type, strategy_id,
currency, allowed_writer_id, policy_ref, genesis_source_ref,
initial_cash, initial_positions, all_initial_shares_settled,
broker, real_order, actual_holdings_mutation
```

Each `initial_positions` row has exact fields:

```text
symbol, name, shares, settled_shares, avg_cost, cost_basis,
realized_pnl, cumulative_fees, acquisition_lots
```

Each acquisition lot has exactly `shares, acquisition_date, settlement_date`.
Initial shares must equal settled shares and the sum of settled acquisition
lots. Account registration is a separate owner-authorized expected-empty
ceremony; writer never calls it automatically.

### `paper-risk-intent.v1`

Exact fields:

```text
schema_version, semantic_sha256, source_intent_id, idempotency_key_sha256,
economic_action_key_sha256, account_id, strategy_id, signal_date,
eligible_from_trade_date, symbol, action, requested_ratio, requested_shares,
reason_codes, policy_ref, expected_account_pointer_sha256,
expected_position, evidence_refs, broker, real_order,
actual_holdings_mutation
```

`expected_position` has exactly `shares, settled_shares, avg_cost`.
Intent is supplied only by explicit workspace-relative path plus byte SHA.
Writer never parses Markdown, memory, latest paths, or actual Store.

### `paper-input-eligibility.v1`

Exact fields:

```text
schema_version, semantic_sha256, account_id, source_intent_ref,
symbol, signal_date, eligible_trade_date, evaluated_trade_date,
open_price, previous_close, limit_up, limit_down, suspended,
corporate_action_state, open_session_ordinal, expiry_session_ordinal,
calendar_ref, raw_bar_ref, price_limit_ref, suspension_ref,
corporate_action_ref, evidence_status
```

All refs have exactly `path, sha256`. Symbol and eligible date must agree across
every referenced artifact. `evidence_status=READY` requires valid raw unadjusted
open/pre-close, exact provider price limits, exact suspension state, Calendar
OPEN, and `corporate_action_state=CLEAR`. Missing evidence yields typed pending;
writer never guesses board/ST limits or corporate-action adjustments.
Only genuinely absent or not-yet-available evidence may yield pending. A
supplied path, SHA, content, symbol, or session mismatch is a hard integrity
conflict with zero writes.

### `paper-order.v1`

Exact fields:

```text
schema_version, semantic_sha256, order_id, account_id, source_intent_ref,
policy_ref, symbol, side, action, shares, trade_date, price_type,
reference_open, adverse_slippage_fraction, simulated_price,
status, broker, real_order
```

### `paper-fill.v1`

Exact fields:

```text
schema_version, semantic_sha256, fill_id, order_ref, account_id, symbol,
side, shares, trade_date, simulated_price, gross_proceeds,
commission, transfer_fee, stamp_duty, total_fees, net_cash_proceeds,
realized_pnl_delta, broker, real_order, actual_holdings_mutation
```

### `paper-pending.v1`

Exact fields:

```text
schema_version, semantic_sha256, pending_id, source_intent_ref,
account_id, symbol, status, first_eligible_trade_date,
last_evaluated_trade_date, evaluated_open_session_count,
expiry_sessions, blocker_codes
```

Allowed status is exactly one of:

```text
PENDING_NEXT_SESSION, PENDING_T1, PENDING_SUSPENDED,
PENDING_LIMIT_BLOCKED, PENDING_CORPORATE_ACTION,
PAPER_PRICE_LIMIT_EVIDENCE_MISSING,
EXPIRED_REEVALUATION_REQUIRED, NO_ACTION_BELOW_MINIMUM_LOT,
REVIEW_ONLY
```

### `paper-account-state.v1`

Exact fields:

```text
schema_version, semantic_sha256, account_id, sequence, as_of_trade_date,
cash, realized_pnl, cumulative_fees, positions,
applied_source_intents, applied_economic_actions, pending_intents,
broker, real_order, actual_holdings_mutation
```

Position rows use the account-registration position schema plus
`last_trade_date, last_fill_id`. Applied source/economic maps bind key to exact
intent SHA and terminal outcome ref. Pending rows bind source intent ID to exact
pending ref and allow at most one later terminal transition.

### `paper-transaction-receipt.v1`

Exact fields:

```text
schema_version, semantic_sha256, transaction_id, account_id, sequence,
previous_pointer_sha256, registration_ref, writer_registration_ref,
policy_ref, intent_ref, eligibility_ref, order_ref, fill_ref,
pending_ref, ledger_ref, account_state_ref, write_set,
command_status, broker, real_order, actual_holdings_mutation
```

Exactly one of `fill_ref` or `pending_ref` is non-null.

### `paper-closure.v1`

Exact fields:

```text
schema_version, semantic_sha256, closure_id, account_id, sequence,
predecessor_closure_ref, transaction_receipt_ref, account_state_ref,
ledger_ref, source_intent_ref, eligibility_ref, policy_ref,
registration_ref, writer_registration_ref
```

### `paper-current-pointer.v1`

Exact fields:

```text
schema_version, semantic_sha256, account_id, sequence,
active_closure_ref, previous_pointer_sha256
```

### Parquet ledger

Exact ordered columns and types:

```text
account_id string
symbol string
name string
shares int64
settled_shares int64
avg_cost decimal128(20,4)
cost_basis decimal128(20,4)
realized_pnl decimal128(20,4)
cumulative_fees decimal128(20,4)
last_trade_date string
last_fill_id string
acquisition_lots_json string
```

Rows are unique and ASCII-sorted by symbol. Parquet metadata is deterministic.

## Release readiness

`account-register --allow-write` and `risk-exit-run --allow-write` require an
exact `release-install-input.json` path and SHA. The command replays the nested
system release/install evidence and requires:

```text
final_commit == git commit embedded in release evidence
final_tree == release tree
python_executable == current sys.executable
import_origin == resolved quant_investor.__file__
import_origin is below exact install_root
release verification state == PASS
```

Any mismatch returns `PAPER_WRITER_RELEASE_NOT_READY` before lock or directory
creation. Source-checkout preview/status/verify remain read-only.

## Execution semantics

- Execution date is the first exact Calendar OPEN session on or after
  `eligible_from_trade_date`; only that session's raw unadjusted open is valid.
- `REDUCE_25/50`: floor `settled_shares × ratio` to a 100-share lot; never round
  up. Zero becomes `NO_ACTION_BELOW_MINIMUM_LOT`.
- `EXIT_100`: sell every settled share; the final odd lot may be liquidated.
- Acquisition lots whose settlement date exceeds execution date are excluded;
  requested shares above settled shares become `PENDING_T1`.
- Sell price is `floor_to_CNY_0.01(next_open × 0.95)`.
- If open is at/below limit down, suspended, the calculated price is below
  limit down, or exact limit evidence is missing, no fill occurs and price is
  never clamped.
- Corporate-action state other than `CLEAR` yields
  `PENDING_CORPORATE_ACTION`.
- Expiry counts exact Calendar OPEN sessions inclusively: `eligible_from` is
  session 1, and the intent becomes `EXPIRED_REEVALUATION_REQUIRED` after the
  close of session 3 if still unfilled. Weekends/closed dates do not count;
  symbol suspension does not extend expiry.

## Fees and accounting

Policy effective date is 2026-09-01. Each fee is calculated independently with
Decimal and rounded `ROUND_HALF_UP` to CNY 0.01 before summing:

```text
commission = max(gross × 0.0001, 5.00)
transfer_fee = gross × 0.00001
stamp_duty = gross × 0.0005
total_fees = commission + transfer_fee + stamp_duty
gross - total_fees > 0
cash_after = cash_before + gross - total_fees
realized_delta = gross - total_fees - avg_cost_before × sold_shares
shares_after = shares_before - sold_shares
remaining_avg_cost = avg_cost_before
cost_basis_after = remaining_avg_cost × shares_after
```

Golden vectors cover commission below/above minimum and cent rounding.

## Idempotency

- Source key is `(account_id, source_intent_id)` and binds exact intent SHA.
- Economic key is SHA of
  `account|policy|signal_date|symbol|action|requested_shares`.
- Same source/economic key and identical payload against pointer-selected state
  returns `NO_ACTION_ALREADY_APPLIED` with zero writes.
- Same source key or economic key with changed payload/policy is
  `PAPER_IDEMPOTENCY_CONFLICT`.
- One pending intent may transition once to filled or expired. No intent can
  create more than one fill.

## Storage and atomicity

Paper root is only `results/paper/accounts/<account_id>`. Account IDs are strict
lowercase hyphenated identifiers. Directories are owner-only 0700; files are
owner-only regular one-link 0600. Every path component rejects symlinks,
case-fold aliases, traversal, hardlinks, and out-of-root resolution.

The account lock is acquired before reading registration/current state and held
through staging, file fsync, staged-directory fsync, final record rename,
pointer-history exact write, expected-preimage CAS, parent fsync, pointer
readback, and full closure verification.

Fault hooks exist at each durable boundary. Before pointer CAS, any failure
keeps the old pointer or expected absence. A final record without pointer is an
orphan and is never auto-adopted; exact replay may verify the same orphan and
finish only when every byte matches, otherwise conflict. Pointer history is
retained before replacement.

## Release A commands

```text
paper writer-status
paper account-status
paper risk-exit-preview
paper risk-exit-run --allow-write
paper verify
paper account-register --allow-write
```

Writer/status/account-status/preview/verify must create no directories, locks,
records, or pointers. In the live workspace Release A must return
`PAPER_ACCOUNT_NOT_REGISTERED`; account-register and risk-exit-run are not
invoked. All write-path tests and synthetic genesis run only under
`/private/tmp` and assert unchanged actual Store, Factor, and System bytes.

## Stop conditions

Stop before any write on release, expected-pointer, account, registration,
policy, intent, eligibility, date, T+1, price-limit, corporate-action,
idempotency, accounting, lock, path, ownership, mode, durable-write, CAS, or
readback mismatch. No auto-recovery may adopt an unknown orphan.

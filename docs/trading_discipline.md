# CN Aggressive Tech Manufacturing Trading Discipline

Status: hard gate
Effective date: 2026-06-11
Strategy-record mirror:
`results/strategy_records/CN/aggressive_tech_manufacturing/trading_discipline.md`

This is the canonical trading-discipline entrypoint for
`CN/aggressive_tech_manufacturing`. The strategy-record mirror keeps the full
operational text for daily review artifacts; this file anchors the project docs
path referenced by the myQuant workflow.

## Core Principle

Separate new-risk actions from risk-reduction sell actions.

New-risk actions are:

- `buy_now`
- `add_now`
- `switch_now`
- any paired replacement buy

Risk-reduction actions are:

- `reduce_risk`: sell part of an existing effective manual-ledger position.
- `clear_risk`: clear an existing effective manual-ledger position.

Incomplete same-day daily bars, missing buy-side DAG branches, weak candidate
persistence, or `prepare_switch` state must keep blocking new-risk actions.
They must not trap an existing position when the sell leg only reduces risk.

## New-Risk Gate

Before any new-risk action can be filled, all of the following must pass:

- latest valid local/manual `ledger_after_manual_switch.parquet` is the effective
  baseline;
- candidate state progression and persistence gates pass;
- score-gap and hysteresis gates pass;
- portfolio count, turnover, and cooldown gates pass;
- Parquet, v13 DAG, factor, and evidence gates pass;
- realtime quote gate passes for every buy and sell leg;
- artifacts are written and the manifest marks the trade as filled.

If any gate fails, the action remains `watch_only`, `tracking`,
`prepare_switch`, `pending`, or `rejected`.

## Risk-Reduction Sell Gate

Risk-reduction sell actions may be filled when all of the following pass:

- the symbol is present in the latest valid local/manual
  `ledger_after_manual_switch.parquet`;
- the action only decreases shares and cannot create a buy, add, or switch-in
  leg;
- formal holdings review or deterministic tracker evidence marks the position
  as below its stage stop, broken stop-loss, hard data failure, realized risk
  event, or thesis invalidation;
- sell quantity is no larger than effective manual-ledger shares and is rounded
  to a valid A-share lot;
- turnover and cooldown gates pass, or the review records an explicit
  risk-reduction override reason;
- a fresh realtime quote passes validation.

For `reduce_risk` and `clear_risk`, incomplete same-day bars, missing buy-side
candidate DAG branches, weak candidate persistence, or `prepare_switch` state
must not block the sell leg. Those limitations block only new-risk actions such
as `buy_now`, `add_now`, `switch_now`, or any paired replacement buy.

## Realtime Quote Gate

Static report prices, daily `close`, `prev_close`, or candidate-pool
`latest_close` are never valid fill prices.

Required quote fields:

- source
- quote timestamp
- realtime execution price field, such as `current`, `last`,
  `last_price`, `trade_price`, `bid`, `bid_price`, `ask`, `ask_price`, or
  another field explicitly supplied by the realtime quote payload
- open
- high
- low
- prev_close

Reject or keep pending if:

- quote retrieval fails;
- quote timestamp is missing or stale;
- no realtime execution price field is present or the price is non-positive;
- the realtime execution price is outside the reported open-high-low range;
- buy price equals only a static daily close, previous close, or report price;
- a sell order would sell more shares than the effective ledger holds;
- a buy order would violate A-share board-lot rounding after cash checks.

## Artifact Rule

Filled, pending, and rejected local/manual actions must be written to the
timestamped strategy record and `raw_exports/`:

- `manual_switch_and_take_profit_orders.csv`
- `manual_execution_manifest.json`
- `ledger_after_manual_switch.parquet`
- `daily_execution_review.md`
- updated `latest_notes_payload.md`

The full operational mirror is:
`results/strategy_records/CN/aggressive_tech_manufacturing/trading_discipline.md`.

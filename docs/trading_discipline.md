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
- Parquet, v14 DAG, factor, and evidence gates pass;
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

## Sell-Point Return Impact Discipline

Sell timing is an independent portfolio-return decision. A sell does not need a
replacement candidate when the action only reduces risk; cash is a valid
position when no v14-complete candidate passes the new-risk gates.

Trailing profit protection is a primary review tool for profitable holdings.
Every formal review must compute a moving take-profit status when enough local
price history exists. Use the highest valid close or execution-time realtime
price since the effective manual-ledger buy as the profit peak, then compare the
current verified price with that peak:

- `peak_unrealized_profit = max(peak_price - buy_price, 0) * shares`
- `current_unrealized_profit = max(current_price - buy_price, 0) * shares`
- `profit_giveback_ratio = (peak_unrealized_profit - current_unrealized_profit)
  / peak_unrealized_profit`

If the profit peak or current realtime price is unavailable, mark the trailing
take-profit status `unconfirmed` and do not infer a sell. A
`profit_giveback_ratio >= 20%` is a mandatory review trigger, not an automatic
fill. It should at least move the holding to `hold_with_trailing_stop` and
force an explicit reason in the daily review. It becomes a `reduce_risk`
candidate when the 20% giveback is accompanied by a falling Codex score,
weakening trend breadth, oversized position weight, theme crowding, or Markov /
RiskGuard / Theme risk tightening. A giveback above roughly 35%, or any
giveback combined with a broken stage stop, thesis break, or score below 60,
can justify reducing 50% or more, or `clear_risk` when the risk case is broken.
All quantities still need A-share lot rounding and a fresh realtime quote gate.

Every formal review must classify weak, over-target, stopped, or oversized
holdings into one of these sell states:

- `hold_with_trailing_stop`: keep the holding, but raise the explicit trailing
  stop or review trigger.
- `reduce_risk`: sell a valid A-share lot to lower position risk or protect
  accumulated profit.
- `clear_risk`: exit the holding when the stop, thesis, data, or portfolio
  risk case is broken.
- `cash_hold`: sell without replacement and hold cash when no buy/switch
  candidate passes all new-risk gates.
- `no_sell_signal`: keep the position because the sell trigger is not present.

The review must not treat "no suitable candidate" as a reason to keep a broken
or oversized holding. It must separately ask whether holding the existing
position still improves expected portfolio return after drawdown risk, Markov
or RiskGuard caps, theme concentration, and opportunity cost are considered.

Sell triggers:

- Hard risk sell: broken stage stop plus weak score, or explicit stop-loss,
  risk event, hard data failure, or thesis invalidation.
- Weak-holding sell: score below 60 in two of the last three valid reviews, or
  below 50 with no complete three-branch support.
- Profit-protection trim: materially above target or cost basis, or trailing
  profit giveback reaches the review threshold, while score is falling, trend
  breadth weakens, position weight is oversized, or Markov / RiskGuard / theme
  caps tighten.
- Concentration sell: a single position or theme exceeds the active risk budget
  even if the name remains profitable.
- No-candidate cash sell: any of the sell triggers above fires, but replacement
  candidates remain `watch_only`, `tracking`, or `prepare_switch`.

Concentration calibration for this strategy:

- The target portfolio shape is concentrated: ideally 3 to 5 effective
  holdings, with no more than 8 effective holdings.
- A 25% to 35% single-name weight is not a sell trigger by itself when the
  holding still has valid thesis support, complete review evidence, acceptable
  trend health, and no stop-loss or liquidity risk trigger.
- Concentration sell requires concentration plus deterioration: weakening score
  or trend, broken stop or thesis, theme crowding, liquidity/fill risk, a need
  to reduce gross exposure, or an explicit active-risk-budget override.
- Markov, RiskGuard, and Theme caps must still be disclosed. When those caps are
  tighter than the concentrated target shape, the report must separate model
  risk-cap pressure from the human strategy target instead of using
  concentration alone as a sell reason.

Every sell, rejected sell, and missed sell must receive a return-impact audit in
`trade_learning_review.md` or the daily execution review. Use the effective
manual ledger and strict Parquet / realtime quote fields to compare:

- realized PNL and released cash;
- sell price versus later 1, 3, 5, and 10 trading-day closes when available;
- avoided drawdown after the sell;
- opportunity cost if the sold shares outperformed cash or the replacement;
- cash drag while replacement candidates remain unqualified;
- replacement alpha if a paired buy was filled;
- whether a partial trim would have beaten a full clear or full hold.

Process quality is judged at the portfolio level. A disciplined sell that later
underperforms a full hold can still be acceptable when it reduced concentration,
stop-loss, data, or regime risk. An undisciplined hold is negative process
quality when a stop or thesis break was present, even if a later rebound hides
the error.

Local audit note as of 2026-06-29: reviewing valid local/manual sell records
against strict Parquet closes through 2026-06-26 showed both effects. Weak-stock
sales in symbols such as `601179.SH`, `600903.SH`, `600578.SH`, `002608.SZ`,
and `688301.SH` avoided later drawdown, while early trims in strong continuing
trends such as `600487.SH`, `002008.SZ`, `301377.SZ`, and `600888.SH` created
opportunity cost. The rule is therefore asymmetric: sell broken or oversized
risk without waiting for a buy candidate, but do not mechanically take profit
from a strong winner without a trend, score, concentration, or risk-cap trigger.

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

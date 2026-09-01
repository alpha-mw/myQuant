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

- latest valid `manual_execution_manifest.json` declares the exact contained
  `next_ledger_path`; that CSV or Parquet sidecar is the effective baseline only
  after stable readback, schema validation, declared SHA256 validation, and—for
  v3 manifests—ledger market-value, cash-plus-market total-value, and
  capital/PNL/return identity reconciliation; v3 `capital_cny` must equal the
  selected formal record's capital base, and these financial fields plus the
  ledger SHA must match `financial_state_sha256`;
- candidate state progression and persistence gates pass;
- score-gap and hysteresis gates pass;
- portfolio count, turnover, and cooldown gates pass;
- strict Parquet, active generation, Factor, and evidence gates pass;
- realtime quote gate passes for every buy and sell leg;
- artifacts are written and the manifest marks the trade as filled.

If any gate fails, the action remains `watch_only`, `tracking`,
`prepare_switch`, `pending`, or `rejected`.

## Risk-Reduction Sell Gate

Risk-reduction sell actions may be filled when all of the following pass:

- the symbol is present in the latest valid manifest-declared local/manual
  `ledger_after_manual_switch.csv` or `.parquet`;
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
- local/manual paper fill was explicitly enabled for that invocation; the
  default is advisory-only and must carry the ledger forward unchanged.
- the existing decision log contains a same-trade-date, same-symbol advisory
  and explicit Maxwell `human_action` pair linked by
  `paired_advisory_event_id`; advisory action and `metadata.shares` must match
  the order, while human_action must carry an exact compatible action,
  `metadata.authorized=true`, `approved_by=maxwell`, and the same integer
  share count. Free text and the CLI flag alone never authorize a fill.

Owner-approved `owner-paper-risk-execution-policy-20260901-v1` 是一个窄的 successor：
它允许所有 registered Paper accounts 自动执行 risk-reducing sells，而无需逐笔
`human_action`，但必须有 registered Paper writer 并闭合 policy/account/calendar/T+1/lot/
price-limit/suspension/next-open/5% adverse slippage/fees/tax/corporate-action/expiry/
idempotency。它不允许新风险买入、Store/actual holdings mutation、broker 或真实订单。

For `reduce_risk` and `clear_risk`, incomplete same-day bars, missing buy-side
candidate DAG branches, weak candidate persistence, or `prepare_switch` state
must not block the sell leg. Those limitations block only new-risk actions such
as `buy_now`, `add_now`, `switch_now`, or any paired replacement buy.

## Sell-Point Return Impact Discipline

Sell timing is an independent portfolio-return decision. A sell does not need a
replacement candidate when the action only reduces risk; cash is a valid
position when no active-generation candidate passes the new-risk gates.

Trailing profit protection is a primary review tool for profitable holdings.
Every formal review must compute a moving take-profit status when enough local
price history exists. Use the highest valid close or execution-time realtime
price since the effective manual-ledger buy as the profit peak, then compare the
current verified price with that peak:

- `peak_unrealized_profit = max(peak_price - buy_price, 0) * shares`
- `current_unrealized_profit = max(current_price - buy_price, 0) * shares`
- `profit_giveback_ratio = (peak_unrealized_profit - current_unrealized_profit)
  / peak_unrealized_profit`

当 exact owner fill、fee-inclusive avg cost、shares、entry/add trade date 与 strict-close
history 全部闭合时，日度研究自动计算：

- `moving_take_profit_review_price = avg_cost + 80% × (peak_price - avg_cost)`；
- `moving_take_profit_reduce_price = avg_cost + 65% × (peak_price - avg_cost)`；
- `moving_stop_price = moving_take_profit_review_price`，但它只是复核触发，不是自动订单。

最近一次 add 改变整仓成本时，默认把 `trailing_profit_tracking_start_date` 重置为该 add 的
exact trade date，并从该日重新计算整仓 peak；若 add 后 `peak_price <= avg_cost`，状态固定为
`NOT_APPLICABLE_UNTIL_POSITIVE_PROFIT_PEAK`。缺 owner fill/date/fee/quantity 任一项时保持
`UNCONFIRMED/NON_EXECUTABLE`。这一自动计算不改 Store ledger，不创建订单或成交。

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
- quote timestamp is not on the current local trade date or is older than the
  configured TTL (default `300` seconds);
- no realtime execution price field is present or the price is non-positive;
- the realtime execution price is outside the reported open-high-low range;
- buy price equals only a static daily close, previous close, or report price;
- a sell order would sell more shares than the effective ledger holds;
- a buy order would violate A-share board-lot rounding after cash checks.

Provider quotes are disabled by default. The formal-review entrypoint accepts
either an explicit live-provider opt-in or a local quote JSON under ignored
`results/`; the local artifact must be a regular non-symlink file with `0600`
permissions, a supported schema, stable double-read bytes, and a recorded
SHA256. Local quote input never falls back to an external provider.

The default `--advisory-only` mode may write a validated risk-reducing sell as
`pending_authorization`, but it cannot call the ledger mutation path. Only an
explicit `--allow-local-manual-fills` invocation may write a local/manual paper
fill only after every data, action, quote, lot-size, and same-day paired
decision-log authorization gate passes. Neither mode may call a broker.

## Artifact Rule

Filled, pending, and rejected local/manual actions must be written to the
timestamped strategy record and `raw_exports/`:

- `manual_switch_and_take_profit_orders.csv`
- `manual_execution_manifest.json`
- manifest-declared `next_ledger_path`, CSV/Parquet hashes, quote provenance,
  and advisory/manual-fill mode
- `ledger_after_manual_switch.parquet`
- `daily_execution_review.md`
- updated `latest_notes_payload.md`

The full operational mirror is:
`results/strategy_records/CN/aggressive_tech_manufacturing/trading_discipline.md`.
# 每日研究风险监测合同

本节是稳定的监测政策，放在交易纪律而不是交易笔记中。它不随每周数据更新而改写；只有
监测项、来源优先级、状态语法、阈值或 authority 合同本身变化时，才升级本节和 automation
prompt。普通日度/周度数据变化只生成本轮观察，不修改纪律或调度配置。

交易笔记和 automation task 只记录当天观察和结果。每日 automation 每轮必须重新读取本节，
并从 exact Store、Factor、官方发布或明确绑定的公开行情证据计算当日状态。周度 automation
聚合本周 observation，并在当期报告中生成下一周事件清单；它不把临时事件日期写回本节。
缺数据时保持 `MISSING`，不得用旧值、聊天记忆或网页摘要补齐。

每日输出固定监测矩阵：

```text
monitor | cadence | as_of | observed_at | value | prior_value | state |
trigger_state | evidence_ref | investment_authority
```

`state` 只允许 `UPDATED | NO_NEW_OFFICIAL_RELEASE | MISSING | CONFLICT`；
`trigger_state` 只允许 `CLEAR | WATCH | BREACH | NOT_CONFIGURED`。监测结果本身没有
broker、order、execution、trade、持仓或正式个股动作 authority。

## 宏观与政策

- 美国就业：非农新增、失业率、平均时薪；只消费 BLS 官方 release，月度或发布日更新。
- 美国劳动力成本：生产率、单位劳动力成本；只消费 BLS 官方 release，季度更新。
- 美国长端利率：10 年和 30 年期国债收益率；只使用带 source/date 的官方或受信公开证据，
  每日记录；无法闭合时为 `MISSING`。
- 中国稳投资：政策性金融工具实际投放、专项债支付、重大项目开工、设备订单、民间投资
  参与度；只消费国务院、国家发改委、财政部、人民银行、国家统计局等官方发布。没有新
  发布时为 `NO_NEW_OFFICIAL_RELEASE`，不得把计划额度当实际投放。

宏观 trigger 仅能收紧或提示研究风险方向；没有 owner-approved numeric threshold 时统一
为 `NOT_CONFIGURED`，不得由 automation 临时发明阈值。

## AI 硬件与现金流兑现

- 分开记录订单/收入可见度、毛利率、自由现金流、资本开支、融资与收入兑现时间。
- 只接受公司公告、交易所文件、财报及其他 source-bound official evidence。
- “订单强但现金流更晚、资本需求更高或毛利率下降”标记为 `WATCH`，只影响研究估值风险，
  不自动映射到任何 A 股公司或持仓动作。

## 组合与纪律

- 每日从 pointer-selected Store-v3 与 strict Market 证据报告 cash、gross、Top1、Top3、
  equity HHI、PCB/AI 硬件权益占比、60-session 相关性、流动性和 turnover。
- 每只当前持仓固定报告 `stage_target_price`、`moving_take_profit_review_price`、
  `moving_take_profit_reduce_price`、`stage_stop_price`、`moving_stop_price`、
  `owner_stop_price`、`threshold_state` 和 exact `evidence_ref`。缺少正式阈值时必须写
  `NOT_CONFIGURED`；`unconfirmed`、缺 entry anchor 或 retired 阈值必须保留原状态并标明
  `NON_EXECUTABLE`，不得从历史峰值、成本、聊天记忆或通用百分比临时推导。
- 正式 Store valuation 相对最新注册交易日落后超过一个交易日时，freshness 标记 `WATCH`；
  缺 continuity 或 pointer/catalog closure 时为 `BREACH`，并阻断持仓结论。
- Owner stop 只读取 exact owner policy；仅 `STRICT_CN_DAILY_CLOSE <= stop` 为 `BREACH`。
  盘中触碰保持 `WATCH`，不得自动创建订单或成交。
- Owner stop strict-close breach 后，automation 自动生成 evidence-bound EXIT_REVIEW card，
  包含 symbol、shares、strict close、stop、buffer、建议 `reduce_risk/clear_risk` 数量、quote
  freshness、Store/Factor/Decision 状态和全部 blocker。现有 policy 的
  `OWNER_CONFIRMATION_REQUIRED_EXIT_REVIEW` 下，人工只需选择 `CONFIRM / HOLD / REVISE`；
  automation 不得连接 broker 或写 actual holdings。取消最终人工确认需要一个新的、明确
  owner-approved successor policy，不能由本合同推断。
- Formal action freshness 只来自 unified System/Mainline 的 exact active decision closure；
  缺失时为 `NOT_CONFIGURED`，Decision Log 为 `NOT_APPLICABLE`，不得写伪 no-action。

## 中国市场因子与条件

每日矩阵增加以下中国市场域；只消费 strict CN Market/PIT、registered Factor、Store、交易所、
人民银行、国家统计局或其他明确 source-bound 官方证据：

- **数据与交易日闭环**：latest complete Market、PIT membership、Calendar、Factor head 必须
  对齐目标前一交易日。日期不一致、coverage/lineage blocker、LOW/W80 observation 缺失或
  非 `OPEN/NON_AUTHORIZING` 时为 `BREACH`，阻断当日新风险和 Top100/Decision 结论。
- **Factor 生产链**：分别报告 LOW、W80 active signal，W75 control-only 状态、Factor pointer、
  generation、as-of、LOW/W80 observations、Top100、Theme replay 与 Decision closure。不得把
  W75、旧 observation 或旧 Top100 当作当前候选。
- **Factor cadence**：LOW/W80 数值、排名、production head 与 observations 必须在每个交易日
  收盘后更新，并在下一交易日 MORNING_STRATEGY 前对齐前一交易日；Factor 定义、方向、权重、
  准入、淘汰、成熟度与健康评审才按周度或更低频治理。周度信号更新不能替代日度生产 rollover。
- **指数趋势条件**：在 exact index close 可用时，分别报告沪深300、中证500、中证1000、
  创业板指、科创50的 close/MA20/MA60，并分类
  `TREND_SUPPORTIVE (close >= MA20 >= MA60)`、
  `TREND_DEFENSIVE (close < MA20 < MA60)` 或 `TREND_MIXED`。该分类只影响研究风险预算，
  不直接生成个股动作。
- **市场广度**：报告上涨/下跌家数、advance ratio、全市场高于 MA20/MA60 比例、涨跌停家数、
  新高/新低数量及中位数收益。没有 owner-approved numeric threshold 时只报原值，
  `trigger_state=NOT_CONFIGURED`。
- **流动性与拥挤**：报告全市场成交额、5日/20日成交额比、主要指数换手、持仓成交额占比、
  PCB/AI 硬件权益占比与60-session相关性。缺 exact evidence 为 `MISSING`；无 owner 阈值
  不得自动把高换手、缩量或高相关性升级为交易指令。
- **波动与尾部风险**：报告主要指数20日实现波动率、20/60日回撤、当日高低振幅、跌停数量、
  停牌/不可交易覆盖。Market/PIT/可交易性 closure 失败为 `BREACH`；其余阈值未配置时为
  `NOT_CONFIGURED`。
- **风格、行业与估值**：报告大/小盘、成长/价值、科技制造/非科技相对强弱，SW2021 Industry、
  Theme membership 与 source-bound economic exposure；指数 PE/PB、盈利预测修正仅在 exact
  registered source 存在时报告，否则为 `MISSING`。
- **中国宏观流动性条件**：报告人民币汇率、中国10年期国债收益率、人民银行公开市场操作与
  政策利率、社会融资/信贷及官方政策实际落地；没有当日/最新官方 release 时按证据状态写
  `NO_NEW_OFFICIAL_RELEASE` 或 `MISSING`，不得用网页摘要补值。

Factor 解阻顺序固定为：registered maintenance 完成 Market/PIT/History/Calendar closure
→ `factor production-rollover` 以 expected pointer 推进到前一交易日
→ `factor production-observe` 注册同日 LOW/W80 OPEN observations
→ `research pool-publish` 生成同日 Top100
→ DC primary / registered TDX fallback Theme replay
→ Fundamental/Macro/Decision compile。任一上游未闭合时，下游保持 `MISSING/BLOCKED`，
不得运行 maintenance fallback、复用旧结果或跳级生成策略。

每日 automation 必须在固定输出中单列该矩阵，并在 automation memory 只保存本轮 receipt、
日期和 blocker 摘要；memory 不得成为下一轮数值或 authority 的 fallback。

因此，同一版本的监测合同只需同步一次。后续数据发布触发 observation 更新，不要求再次
更新日度或周度 automation；只有合同版本升级时，才同时更新两个 consumer 并做配置 readback。

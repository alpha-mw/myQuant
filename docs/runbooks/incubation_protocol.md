# v13 冻结与孵化协议（已退役）

> 状态：`retired`，自 2026-07-15 起不再约束当前 `main`、PR 合并或
> schedule 运行。以下内容仅用于解释 `v13-frozen-20260707` 历史绩效与
> 证据，不得作为 v14 的激活、冻结或回滚授权。历史 tag、账本和报告保持
> 不变；当前运行契约见 `docs/runbooks/v14_operations.md`。

## 1. 冻结对象

- 冻结版本：`v13-frozen-20260707`
- 策略对象：`CN/aggressive_tech_manufacturing`
- `freeze_start_date = 2026-07-07`
- `incubation_length_trading_days = 90`
- `weekly_tracking_time = 周五 17:30`

## 2. 历史冻结期代码规则

以下规则只描述已结束的 v13 冻结期，不再是当前合并门：

1. 通过 PR 提交，不直接绕过代码审查。
2. PR 必须打 `freeze-exception` 标签。
3. PR 描述必须说明 bug、影响面、为何不改变策略主张、验证命令与结果。
4. 不允许引入新 alpha、风险偏好放宽、参数主动优化、数据口径替换、择时规则重写或人工事后调参。
5. `results/strategy_records/`、`results/track_record_audit/`、benchmark CSV、审计报告/JSON/PNG 继续不得 commit。
6. `freeze_exception_approver = tests_green_plus_codex_review`
7. `freeze_exception_merge_gate = maxwell_explicit_confirmation`。任何 freeze-exception 的合并进 main 动作，须 Maxwell 在会话中原文确认后方可执行。冻结期边界不应由被冻结方（Codex）自行把守合并动作；审批可自动化，合并须人肉。
8. `github_branch_protection = disabled`。这是 Maxwell 选择；上述人肉合并门为其补偿性控制。

## 3. 周度预登记追踪指标

每周收盘后固定复跑 Phase 12 审计结构，至少记录：

1. 对科创50 `star50_nav` 的周超额收益。
2. 对科创50 `star50_nav` 的滚动 8 週累计超额收益。
3. 滑点漂移：按买/卖方向统计提案价 vs 实际成交价 bps 均值、中位数、P90。
4. 周换手率与滚动 4 周换手率。
5. 出场后 5/10/20 日收益滚动均值、中位数与为负占比。
6. Markov regime 触发次数、`ThemeGatePolicy.from_markov` 档位变化次数、相位闸门触发次数。
7. 单票权重超过 `RISK_GUARD_SINGLE_NAME_WEIGHT_CAP` 的存量提示数量与最大超限幅度。
8. 当前 peak-to-trough NAV drawdown 档位。
9. 相位择时 alpha、组内选股 alpha、配置 beta 的滚动对账。
10. 60 日相位择时 alpha。
11. 对科创50 `star50_nav` 的回归 beta、年化 alpha、alpha t 值与 R²。
12. 信息比率 IR（日频，同时留存年化换算）。
13. 暴露合规率：实际总暴露 ≤ Markov/系统建议总暴露上限的天数占比。
14. 选择 alpha：实际所选标的收益 − 当日 shortlist 菜单等权收益。
15. 毛/净双口径全窗口收益与周度成本拖累估计。
16. 两条影子账本差值：`shadow_nav_cap050` vs actual、`shadow_nav_machine_exit` vs actual。
17. 决策日志完整率：`advisory` 与 `human_action` 有同日配对记录的比例。

## 4. 历史 Kill / 加仓规则

以下阈值是 v13 的预登记风险偏好输入，只能用于复现和解释对应历史窗口；
它们不再冻结 v14 的代码或调度面，也不自动授权交易动作。

### 回撤阶梯

峰值口径 = 从历史最高 NAV 起算的回撤；口径为 peak-to-trough NAV drawdown。四档为暴露上限动作，非强制平仓；结合 `overweight_handling = notify_only_no_forced_sell`，减半/减 1/4 指下调目标暴露上限并提示，不强制卖出既有超限持仓，除 -40% 清仓档为逐票人工确认清仓。

- `drawdown_tier_1_review = -12%`：触发强制人工复核，附衰减瀑布，不动仓（软哨兵）。
- `drawdown_tier_2_half = -20%`：目标暴露减半。
- `drawdown_tier_3_quarter = -30%`：目标暴露减至 ≤ 1/4。
- `drawdown_tier_4_clear = -40%`：清仓并强制 30 日复盘。

### Kill 规则

任一触发 → 进入 kill 评审，非自动清仓；评审动作 = 逐票人工复核。

- `kill_excess_window_weeks = 8`；`kill_excess_cumulative_threshold = -10%`：滚动 8 週累计超额 vs 科创50（`star50_nav`）低于 -10%。
- `kill_phase_alpha_window_days = 60`；`kill_phase_alpha_threshold = -5%`：60 日相位择时 α 低于 -5%。
- `kill_slippage_p90_bps = 50`；`kill_slippage_sustain_weeks = 2`：滑点 P90 > 50bps 持续 2 週。

### Add 规则

放宽暴露/加倉的前置条件，与 kill 共用 8 週窗口。

- `add_excess_window_weeks = 8`；`add_excess_cumulative_threshold = +5%`：滚动 8 週累计超额 vs 科创50 > +5%。
- `add_phase_alpha_threshold = +5%`：相位择时 α > +5%。
- `add_size_limit_pct_nav = 10%`：单次加倉不超过 NAV 10%。

### 其余生效值

- `markov_drawdown_advantage_pct = 5`：回撤期组合相对基准占优的期望验证值，仅追踪不作动作。
- `post_exit_positive_threshold = 30%`：出场后 5/10/20 日为负占比的监控上界，仅追踪。
- `funding_nature = mixed_per_trade_labeled`
- `overweight_handling = notify_only_no_forced_sell`
- `add_new_capital = manual_review_required`
- `freeze_new_trades = manual_override_required`
- `machine_daily_advice = risk_reducing_sell_allowed`
- `Codex` 仅在 `maxwell` 确认后记录人工动作；`advisory` 与 `human_action` 需要同日配对。
- Phase 5b/9b：冻结后再议，冻结期内不作为加仓或策略改写理由。

## 5. 设计说明

kill/add 均采用「长窗口(8週) + 累计 + 有符号阈值」，与 60 日相位 α 哲学统一；不用逐週 0% 阈值。原因：本策略为集中型（审计实测 Top3 贡献 80.8%、命中率 30.77%、年化波动 56%），沉闷週跑输指数是形态特征而非病态，逐週阈值会误杀健康策略。回撤四档间距按 56% 年化波动校准，避免「满仓→半仓→清仓」三段跳在快速下行中脱节。

## 6. 当前持仓人工输入附录

产业实质判断（真订单/真产能 vs 纯故事）由 `maxwell` 人工完成；本表只记录冻结起点的人为输入与披露日缺口。

| 标的 | 公司 | 产业实质判断 | H1 披露日 |
| --- | --- | --- | --- |
| `002008.SZ` | 大族激光 | 看不清 | 披露日需人工补充 |
| `002384.SZ` | 东山精密 | 真订单 | 披露日需人工补充 |
| `002463.SZ` | 沪电股份 | 真订单 | 披露日需人工补充 |
| `002851.SZ` | 麦格米特 | 看不清 | 披露日需人工补充 |
| `605358.SH` | 立昂微 | 真订单 | 披露日需人工补充 |

## 7. 期满评估模板

期满后按 Phase 12.2 报告结构复跑并对比冻结起点：

1. A 基础绩效：累计收益、年化波动、最大回撤、Calmar、Sharpe、周换手、持仓期、逐笔命中率与盈亏比。
2. B 基准与超额：科创50为主考官，同时列 `chinext_nav`、`csi300_nav`、`industry_ew_nav`。
3. C 三分解：配置 beta、组内选股 alpha、相位择时 alpha 与总超额对账。
4. D 集中度与稳健性：Top3 贡献占比与剔除 Top3 后收益。
5. E 对手盘验证：入场/出场后 5/10/20 日收益分布。
6. F 执行质量：滑点、门禁拒单、总成本拖累。
7. G 归因切分：版本切段、人工 vs 系统、资金性质。

若全窗口超额 vs 科创50 ≤ 0 且相位择时 alpha ≤ 0，复评结论必须沿用 Phase 12 预登记失败规则。

## 8. v14 Fundamental Research 历史过渡分段

本节记录从 v13 向 Fundamental v14 迁移时采用的历史边界。当前权威运行
契约已经迁移到 `docs/runbooks/fundamental_research_v14.md` 和
`docs/runbooks/v14_operations.md`；下列 freeze-exception 合并要求不再生效。

1. `v13-frozen-20260707` 的历史决策、成交和绩效序列保持不可变，不回填、
   不重算，也不与新版本拼接成同一冻结策略绩效。
2. 当前策略命名为 `v14-fundamental-research`，只能从 Maxwell 明确批准的
   生效交易日和时间戳开始独立记录。
3. 功能实现、PR 合并和生产激活是三个不同事件。每个 freeze-exception
   PR 仍需 Maxwell 原文确认合并；从 `shadow`/`limited` 切入更高应用档位
   还需独立的 hash-bound activation confirmation。
4. `off` 和 `shadow` 不改变生产 fundamental 分数。`limited` 先以
   `±0.03` 运行 5 个交易日，再在无 critical error 时以 `±0.05` 运行
   5 个交易日；`production` 上限为 `±0.10`。
5. 进入 limited 前至少需要 10 个不同交易日、30 份 validated dossiers、
   10 家公司、3 个行业并覆盖全部持仓；进入 production 前累计至少
   20 个不同交易日、60 份 validated dossiers、20 家公司、3 个行业，
   且 PIT/hash/source/cap/exactly-once/control-chain critical error 为零。
6. 新版本持续保存同输入、无 dossier 的 counterfactual，分别报告 shortlist、
   Bayesian rank、目标权重和 NAV 归因差异。
7. 接受未来信息、identity/hash 错配、secondary-only 非零贡献、分数叠加、
   越过 cap、控制链绕过或私有数据泄漏时立即切回 `off`；最近 20 个
   received jobs（样本至少 10）验证成功率低于 80% 时切回 `shadow`。
8. 激活计数只能由 `fundamental-research-activation-evidence.v2` 从
   job/application/longitudinal 三条 hash-chain ledger 重算；人工填写的 v1
   count sheet 不再可用于激活。limited 至少需要 10 个目标权重反事实日期，
   production 至少需要 20 个目标权重日期和 10 个 realized NAV 归因日期。

详细协议、证据资格和离线操作流程见
`docs/runbooks/fundamental_research_v14.md`。

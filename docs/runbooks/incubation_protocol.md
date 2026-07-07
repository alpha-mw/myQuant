# v13 冻结与孵化协议

## 1. 冻结对象

- 冻结版本：`v13-frozen-20260707`
- 策略对象：`CN/aggressive_tech_manufacturing`
- 起始条件：Phase 12 的 12.1-12.3 提交、验收命令与本协议提交完成，并由 `maxwell` 在冻结起始前填写第 4 节 `TODO(maxwell)` 数值阈值后再次 commit。
- 冻结期：60-90 个交易日，以 A 股交易日历为准。

## 2. 冻结期代码规则

冻结期内唯一允许的代码变更是 bug 修复。每次例外必须满足：

1. 通过 PR 提交，不直接绕过代码审查。
2. PR 必须打 `freeze-exception` 标签。
3. PR 描述必须说明 bug、影响面、为何不改变策略主张、验证命令与结果。
4. 不允许引入新 alpha、风险偏好放宽、参数主动优化、数据口径替换、择时规则重写或人工事后调参。
5. `results/strategy_records/`、`results/track_record_audit/`、benchmark CSV、审计报告/JSON/PNG 继续不得 commit。

## 3. 周度预登记追踪指标

每周收盘后固定复跑 Phase 12 审计结构，至少记录：

1. 对科创50 `star50_nav` 的周超额收益。
2. 滑点漂移：按买/卖方向统计提案价 vs 实际成交价 bps 均值、中位数、P90。
3. 周换手率与滚动 4 周换手率。
4. 出场后 5/10/20 日收益滚动均值与中位数。
5. Markov regime 触发次数、`ThemeGatePolicy.from_markov` 档位变化次数、相位闸门触发次数。
6. 单票权重超过 `RISK_GUARD_SINGLE_NAME_WEIGHT_CAP` 的存量提示数量与最大超限幅度。
7. 相位择时 alpha、组内选股 alpha、配置 beta 的滚动对账。
8. 对科创50 `star50_nav` 的回归 beta、年化 alpha、alpha t 值与 R²。
9. 信息比率 IR（日频，同时留存年化换算）。
10. 暴露合规率：实际总暴露 ≤ Markov/系统建议总暴露上限的天数占比。

## 4. Kill / 加仓规则占位

以下阈值必须由 `maxwell` 在冻结正式起始前填写并 commit。未填写前，本协议只完成登记，不视为生效。

- Kill rule：若连续 `TODO(maxwell)` 周对科创50周超额低于 `TODO(maxwell)`，则进入冻结期复核/停止新增资金。
- Kill rule：若滚动 `TODO(maxwell)` 日相位择时 alpha 低于 `TODO(maxwell)`，则停止机器日频调仓主张。
- Kill rule：若滑点漂移买入或卖出方向 P90 超过 `TODO(maxwell)` bps，且持续 `TODO(maxwell)` 周，则冻结新增交易。
- Kill rule：若出场后 10/20 日收益滚动均值显著为正并超过 `TODO(maxwell)`，则判定卖出偏早，暂停相位出场规则扩展。
- Add rule：若连续 `TODO(maxwell)` 周对科创50周超额高于 `TODO(maxwell)`，且相位择时 alpha 高于 `TODO(maxwell)`，可进入小额加仓评估。
- Add rule：若 Markov/相位闸门在回撤期触发并使最大回撤低于科创50 `TODO(maxwell)`，可评估提高资金上限。

<!-- PROPOSAL (Claude, 2026-07-07) — 示例值，未生效，须 Maxwell 改定后移入正式字段并提交
- 回撤阶梯示例：峰值回撤 -20% -> 目标暴露减半；峰值回撤 -30% -> 清仓并强制 30 日复盘。
- 评审触发示例：滚动 60 日超额 vs 科创50 < 0 -> 触发策略评审。
- 执行审查示例：周均滑点 > 冻结起始基线 2 倍 -> 触发执行审查。
-->

## 5. 期满评估模板

期满后按 Phase 12.2 报告结构复跑并对比冻结起点：

1. A 基础绩效：累计收益、年化波动、最大回撤、Calmar、Sharpe、周换手、持仓期、逐笔命中率与盈亏比。
2. B 基准与超额：科创50为主考官，同时列 `chinext_nav`、`csi300_nav`、`industry_ew_nav`。
3. C 三分解：配置 beta、组内选股 alpha、相位择时 alpha 与总超额对账。
4. D 集中度与稳健性：Top3 贡献占比与剔除 Top3 后收益。
5. E 对手盘验证：入场/出场后 5/10/20 日收益分布。
6. F 执行质量：滑点、门禁拒单、总成本拖累。
7. G 归因切分：版本切段、人工 vs 系统、资金性质。

若全窗口超额 vs 科创50 ≤ 0 且相位择时 alpha ≤ 0，复评结论必须沿用 Phase 12 预登记失败规则。

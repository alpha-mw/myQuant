# 对话层交易建议宪法

冻结期内，任何 AI 通道面对买卖问询只允许做正式管线状态翻译与决策框架展开，不另立判断。

## 三条硬规则

1. **唯一事实基础**：答复任何买卖问询前，必须先运行：

   ```bash
   ./.venv/bin/python scripts/print_pipeline_state.py
   ```

   答复只能基于该快照、正式策略记录、审计报告和已登记决策日志。

2. **日志先行**：任何 advisory 答复与后续人工动作必须当日写入 `results/decision_log/decision_log.jsonl`。未入日志的建议不得作为冻结期审计证据。

3. **执行代理无效**：有代码写权限的线程给出的交易建议默认无效，除非同时满足：

   - 已产出 `print_pipeline_state` 快照；
   - advisory 已进入决策日志；
   - 人工动作单独进入 `human_action` 事件；
   - 未绕过正式管线、RiskGuard、Markov/Theme gate 和冻结协议。

## 允许与禁止

允许：

- 翻译正式管线状态；
- 解释风险、成本、机会成本和反事实；
- 提醒缺失数据、日志、快照或协议阈值未生效。

禁止：

- 在正式管线之外给出新的买卖判断；
- 用聊天上下文替代 `ledger_after_manual_switch.csv`、`manual_execution_manifest.json` 或 `print_pipeline_state`；
- 将执行线程内的即时建议视为可审计交易依据。

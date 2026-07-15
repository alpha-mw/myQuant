# 对话层交易建议宪法

在当前治理下，任何 AI 通道面对买卖问询只允许做正式管线状态翻译与决策框架展开，不另立判断。

## 三条硬规则

1. **唯一事实基础**：答复任何买卖问询前，必须先运行：

   ```bash
   ./.venv/bin/python scripts/print_pipeline_state.py
   ```

   答复只能基于该快照、正式策略记录、审计报告和已登记决策日志。

2. **日志先行**：任何 advisory 答复与后续人工动作必须当日写入 `results/decision_log/decision_log.jsonl`。未入日志的建议不得作为治理审计证据。

3. **执行代理无效**：有代码写权限的线程给出的交易建议默认无效，除非同时满足：

   - 已产出 `print_pipeline_state` 快照；
   - advisory 已进入决策日志；
   - 人工动作单独进入 `human_action` 事件；
   - 未绕过正式管线、RiskGuard、Markov/Theme gate 和当前治理协议。

4. **默认只建议**：A 股日度正式复盘默认 `advisory_only=true`。即使风险减仓建议和
   fresh quote 均通过，也只能记录为 `pending_authorization` 并原样结转 ledger；只有
   Maxwell 明确授权的 `--allow-local-manual-fills` 运行，且既有 decision log 中存在
   同交易日、同标的、通过 `paired_advisory_event_id` 绑定的 advisory 与显式
   `human_action`，才可写本地/manual paper fill。CLI flag 不能替代该配对证据；仍禁止
   券商或真实下单接口。可执行配对必须使用结构化字段：advisory 的精确动作与
   `metadata.shares`；human_action 的精确动作、`metadata.authorized=true`、
   `approved_by=maxwell`、完全相等的 `shares`。

## 允许与禁止

允许：

- 翻译正式管线状态；
- 解释风险、成本、机会成本和反事实；
- 提醒缺失数据、日志、快照或协议阈值未生效。

禁止：

- 在正式管线之外给出新的买卖判断；
- 用聊天上下文替代 `ledger_after_manual_switch.csv`、`manual_execution_manifest.json` 或 `print_pipeline_state`；
- 将执行线程内的即时建议视为可审计交易依据。

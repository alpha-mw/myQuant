# 对话层交易建议宪法

在当前治理下，任何 AI 通道面对买卖问询只允许做正式管线状态翻译与决策框架展开，不另立判断。

## 三条硬规则

1. **唯一事实基础**：答复任何买卖问询前，必须先解析该策略的活跃主线指针：

   ```bash
   ./.venv/bin/quant-investor research run \
     --workspace-root /absolute/path/to/myQuant \
     --strategy-id <strategy-id>
   ```

   答复只能基于该只读结果、正式策略记录、审计报告和已登记决策日志。

   该命令解析唯一一个指针，不扫描目录、不回退、不自举，因此它的**失败也是状态**：
   返回 `mainline_state=UNINITIALIZED` 或 `investment_state=BLOCKED` 时，不存在
   可翻译的正式管线状态，本条即未满足，不得给出任何买卖建议 —— 只能如实说明主线
   未初始化或被哪个 blocker 挡住。缺失的快照不是「用别的材料凑」的理由。

   > 历史注记：本条曾指向 `scripts/print_pipeline_state.py`。该脚本已于
   > `389562a`（2026-08-05）随一批过时脚本删除，且不应重建：它按目录名排序扫描
   > **最近一次** strategy record，并在文件缺失或损坏时静默返回空对象 —— 正是当前
   > 架构明令禁止的「扫描最近运行」与「静默替换」。活跃指针读取是它的正确替代。

2. **日志先行**：任何 advisory 答复与后续人工动作必须当日写入 `results/decision_log/decision_log.jsonl`。未入日志的建议不得作为治理审计证据。

3. **执行代理无效**：有代码写权限的线程给出的交易建议默认无效，除非同时满足：

   - 已解析出活跃主线指针结果（非 `UNINITIALIZED` / `BLOCKED`）；
   - advisory 已进入决策日志；
   - 人工动作单独进入 `human_action` 事件；
   - 未绕过正式管线、RiskGuard、Markov gate 和当前治理协议。

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
- 用聊天上下文替代 `ledger_after_manual_switch.csv`、`manual_execution_manifest.json` 或活跃主线指针读取结果；
- 将执行线程内的即时建议视为可审计交易依据。

# 决策日志规范

`results/decision_log/decision_log.jsonl` 是冻结期买卖问询与人工动作的本地治理账本。该目录被 `.gitignore` 覆盖，不得 commit。

## 事件类型

每行一个 JSON object，公共字段：

- `schema_version`: 固定 `decision_log.v1`
- `event_id`: 稳定哈希 ID
- `recorded_at`: 记录时间，UTC ISO-8601
- `trade_date`: 事件对应交易日
- `metadata`: 额外结构化字段

### `pipeline_proposal`

来自正式管线的提案或状态：

- `proposal_summary`: 管线提案摘要
- `symbol`: 可选，关联标的
- `machine_suggestion`: 可选，机器建议
- `regime_state`: 可选，当时 Markov regime 状态

### `advisory`

来自对话/工作台/其他 AI 通道的咨询答复：

- `channel`: 问询渠道
- `question`: 用户原始问题摘要
- `answer_summary`: 答复摘要，不保存长原文
- `answer_source`: `codex_thread` / `claude` / `workbench` / `other`
- `symbol`: 可选，关联标的

### `human_action`

人工最终动作：

- `action`: 例如 `buy` / `sell` / `hold` / `reduce` / `clear`
- `symbol`: 可选，关联标的
- `rejected_options`: 当时被拒的候选/选项集
- `regime_state`: 当时 Markov regime 状态
- `machine_suggestion`: 当时正式管线建议

## 操作入口

追加单条记录：

```bash
./.venv/bin/python scripts/log_decision.py \
  --event-type advisory \
  --trade-date 2026-07-07 \
  --channel codex \
  --question "是否卖出 688519" \
  --answer-summary "先运行 print_pipeline_state，仅翻译正式管线状态" \
  --answer-source codex_thread
```

查看最近记录：

```bash
./.venv/bin/python scripts/log_decision.py --list 10
```

任何买卖问询若没有对应 `advisory` 与后续 `human_action` 配对，周度追踪中的 `decision_log_completeness` 必须如实下降。

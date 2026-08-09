---
name: myquant-backend-ops
description: Use for repository-local myQuant backend inspection and operation: exact V17 mainline readers, CN market maintenance, V4 Forward/Shadow evidence, I0/R2.2 research evaluation, canonical local data, and fail-closed diagnostics. Do not use for generic Python tasks, live brokerage/execution, or for treating the standalone legacy automation package as an active V17 workflow.
---

# myQuant Backend Ops

## Overview

把这个 skill 当作仓库内后端工作的路由入口。先确认用户要处理的是主线读取、
数据维护、Forward/Shadow、I0/R2.2/I1、I2-I6 v2，还是独立旧 automation；这些 lane 不能
互相替代。

## 核心边界

- V17 v4 是唯一 decision mainline；公开主线目前是只读的。
- `research run`、`market analyze`、`market run` 读取同一个精确 active
  pointer，不生成候选、组合或新 run。
- 仓库没有公开 production publisher 或 activation command。
- V4 `run-forward` 是 research-only Shadow evidence；不推进 mainline。
- I0 的 Markov 模型是 Market/Industry/Theme 三层、严格单步、因果过滤。
- R2.2 `research-evaluate` 离线、stdout-only；memory 只是 append proposal。
- I1 是完整复放 I0/可选 R2.2 的 library-only 决策审查层；没有公开 CLI、Web、
  Provider、模型调用、持久化 writer 或组合权限。
- I2-I6 位于 `quant_investor/intelligence_v2`，仍是 library-only。I5 公开搜索
  不设域名白名单，但本地安全固证和确定性准入保持权威；I6 只提供研究组合、Paper
  和签名发布契约，普通任务不得调用 pointer CAS。
- `market maintain` 等维护命令和主线读取分开；严格 Parquet canonical 数据
  不能被 CSV 静默替代。
- LLM 只允许 advisory；deterministic gate 和风险 veto 保持权威。
- `quant_investor/automation` 是未注册、依赖未闭合的独立旧实现，不是 V17
  daily loop。只有用户明确要求 legacy automation 且依赖验证通过时才进入。

## 工作流

### 1. 分类

- `mainline-read`：读取一个 strategy 的活动 V17 结果。
- `maintenance`：维护、验证或物化本地 CN canonical 数据。
- `forward-research`：运行或检查 V4 Forward/Shadow evidence。
- `intelligence-evaluation`：运行 I0/R2.2 的显式、内容绑定研究评价。
- `decision-review`：构建或验证 library-only I1 决策、memo、discipline 或
  external paper-review proposal。
- `intelligence-v2`：验证 B0/I2-I6 闭包；默认离线，不调用 live Web/model，
  不生成真实候选，不激活 pointer。
- `legacy-diagnostic`：用户明确指定旧 API 或 standalone automation 后才进入。

### 2. 先读公开面

优先顺序：

1. `README.md` 和 `docs/README.md`
2. `quant_investor/cli/main.py`
3. `quant_investor/v17_v4_runtime/cli.py`
4. `quant_investor/intelligence/evaluator/cli.py`
5. 对应 contract、service 或底层实现

### 3. 按需读取 references

- 命令：`references/entrypoints-and-commands.md`
- 路径和产物：`references/runtime-paths-and-artifacts.md`

### 4. Fail closed

先确认精确路径、SHA、pointer、manifest、request 和 cutoff。缺失或失配时返回
明确 blocker；不要扫描 `latest`、猜测旧结果、用 Shadow 替 mainline、用 CSV
替 Parquet，或把可选输入的无效值当成“未提供”。

## 执行与排障原则

- 查询和诊断先做只读检查；用户明确要求运行时才执行相应 lane。
- 不在本地验证中调用 live Tushare、LLM、broker 或 execution API，除非用户
  明确要求 live run。
- 长任务以最终 receipt、result JSON 和 blocker 为准，不以安静 stdout 推断
  成败。
- 展示面是 `portfolio_dashboard/` 静态页，由 `scripts/export_cn_aggressive_
  dashboard_data.py` 生成的 JSON bundle 驱动，没有运行中的服务。

## References

- `references/entrypoints-and-commands.md`
- `references/runtime-paths-and-artifacts.md`

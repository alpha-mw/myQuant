---
name: myquant-backend-ops
description: Use when Codex is working on the myQuant or QuantInvestor backend and needs to inspect or operate `quant-investor` CLI flows, market pipelines, workspace or web APIs, SQLite or local data stores, results directories, runtime env vars, or backend diagnostics. Trigger on requests about research runs, `quant-investor web`, `/api/research`, presets, universe, settings, data endpoints, `stock_database.db`, `web_runs.db`, `app.db`, `results/web_analysis`, or review-layer behavior. Do not use for pure frontend UI work or generic Python tasks unrelated to myQuant.
---

# myQuant Backend Ops

## Overview

把这个 skill 当作 `myQuant` 后端任务的默认入口。先判断用户意图，再优先走公开入口、公开 API 和运行时产物，不要一上来钻进内部实现。

## 适用与不适用

适用：
- `myQuant`、`QuantInvestor`、`quant-investor` 相关后端问题
- CLI 运行、市场流水线、workspace 运行、API 定位、结果与数据库排查
- `.env`、`data/`、`results/`、LLM review layer 降级与可用性判断

不适用：
- 纯前端页面样式、React 组件重构、Vite/Tailwind 视觉问题
- 与 `myQuant` 无关的通用 Python、FastAPI、SQLite 教学问题

## 工作流

### 1. 先分类请求

- `inspect`：解释能力、路径、数据库职责、结果产物、已有 API 或运行方式
- `execute`：用户明确要求“启动 / 运行 / 跑 / 下载 / 分析 / 回测”
- `api-location`：定位 endpoint、请求模型、service、store、数据库归属
- `debugging`：分析某次运行、history、workspace、analysis、数据库写入为什么异常

默认规则：
- 查询类和排障类先 `inspect`，确认入口、路径、数据层和现有结果
- 明确运行类可以直接 `execute`，但必要时先做轻量检查（命令是否存在、关键 env、目标路径）

### 2. 公开入口优先

优先顺序：
1. `README.md`
2. `quant_investor/cli/main.py`
3. `web/main.py`
4. `web/workspace_app.py`
5. `web/routers/*.py` 与 `web/api/*.py`
6. `web/services/*.py`
7. 更深层的 `quant_investor/*`

除非用户明确问内部实现，否则优先给出：
- 该做什么命令
- 该看哪个公开 endpoint
- 结果会落到哪里
- 哪个 service 或 store 负责持久化

### 3. 按任务类型分流

#### `inspect`

- CLI / 运行入口：先读 `references/entrypoints-and-commands.md`
- 路径 / 数据库 / 结果产物：先读 `references/runtime-paths-and-artifacts.md`
- API / service 归属：先读 `references/workspace-api-surface.md`

#### `execute`

优先使用这些公开入口：
- `quant-investor research run`：单一主线研究
- `quant-investor market download|analyze|run|backtest`：全市场工作流
- `quant-investor web`：workspace 运行时

不要默认从内部模块直接启动，除非是在定位 legacy 路由或调试框架装配。

#### `api-location`

先回答这四件事：
1. 路由文件在哪
2. 请求 / 响应模型在哪
3. service 或 store 在哪
4. 落到哪个数据库或结果目录

如果用户只问“接口在哪”，先给公开路由，再补 service 和持久化归属。

#### `debugging`

先确认故障发生在哪个公开面：
- `quant-investor research run`
- `quant-investor market ...`
- `quant-investor web`
- `/api/research`
- `/api/presets`
- `/api/universe`
- `/api/settings`
- `/api/data`
- legacy `/api/v1/*` 或 `/analysis`

排障时先看：
- 请求模型或参数
- 对应 router 与 service
- 相关 SQLite / `results/` / `data/` 产物
- 是否只是 LLM review layer 降级，而不是硬失败

## myQuant 特定规则

- `quant-investor web` 的真实入口是 `web.main:app`，它导向 `web.workspace_app`
- 当前 workspace 公共 API 面主要在 `web/routers/*.py`，并额外挂了 `web/api/data.py`
- `web/app.py` 与 `web/api/analysis.py`、`web/api/portfolio.py`、`web/api/settings.py` 更像 legacy `/api/v1` 或旧分析面；只有在用户明确提到这些接口时再深挖
- LLM review layer 是 advisory-only。缺少 `KIMI_API_KEY`、`DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY` 等 key 时，优先说明会安全降级，不要误判为整个主线不可运行

## References

- `references/entrypoints-and-commands.md`
- `references/runtime-paths-and-artifacts.md`
- `references/workspace-api-surface.md`

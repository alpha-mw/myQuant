# myQuant Entrypoints and Commands

先用公开入口回答问题，再决定是否需要读内部模块。

## Public Entrypoints

| 场景 | 首选入口 | Source of truth |
|---|---|---|
| 单股或小池研究 | `quant-investor research run` | `quant_investor/cli/main.py`, `quant_investor/pipeline/mainline.py` |
| 全市场下载 | `quant-investor market download` | `quant_investor/cli/main.py`, `quant_investor/market/download*.py` |
| 全市场分析 | `quant-investor market analyze` | `quant_investor/cli/main.py`, `quant_investor/market/analyze.py` |
| 全市场 daily pipeline | `quant-investor market run` | `quant_investor/cli/main.py`, `quant_investor/market/run_pipeline.py`, `quant_investor/market/dag_executor.py` |
| 本地回测 | `quant-investor market backtest` | `quant_investor/cli/main.py`, `quant_investor/market/backtest.py` |
| Workspace 运行时 | `quant-investor web` | `quant_investor/cli/main.py`, `web/main.py`, `web/workspace_app.py` |
| 前端开发 | `./run_web.sh` | `run_web.sh`, `frontend/` |

## High-Frequency Command Patterns

### Single-mainline research

```bash
quant-investor research run \
  --stocks 000001.SZ 600519.SH \
  --market CN \
  --capital 1000000 \
  --risk 中等
```

Use when:
- 用户要跑单股或少量标的研究
- 用户问“单一主线研究怎么启动”
- 用户问 `QuantInvestor` 主流程对应哪个 CLI

### Full-market download

```bash
quant-investor market download --market CN --years 3 --workers 4
```

Use when:
- 用户要补全本地市场数据
- 用户问下载阶段或 `data/` 来源

### Full-market analysis

```bash
quant-investor market analyze --market CN --mode batch --top-k 12
```

Use when:
- 用户要基于已有本地数据做全市场研究
- 用户问 shortlist / candidate / portfolio 建议从哪里来

### Full-market pipeline

```bash
quant-investor market run --market CN --mode batch --top-k 12
```

Use when:
- 用户要完整 daily path
- 用户说“下载 + 分析一起跑”

### Workspace runtime

```bash
quant-investor web --reload
```

Use when:
- 用户要启动工作台
- 用户要检查 `/api/health`
- 用户要定位 workspace 路由与服务层

Current runtime entry:
- `web.main:app` -> `web.api:app` -> `web.workspace_app:app`

## Request Routing Examples

### “启动 myQuant 工作台并检查后端健康状态”

优先路径：
1. `quant-investor web`
2. `GET /api/health`
3. 如需补充，再看 `GET /api/settings/`

### “跑一个 CN 单股研究，并告诉我结果去哪看”

优先路径：
1. `quant-investor research run --stocks <symbol> --market CN`
2. 结果与历史优先看 `data/web_runs.db`、`data/workspace_learning/`
3. 如果是旧分析面，再看 `results/web_analysis/`

### “帮我定位 preset / universe / research job 的后端接口和服务层”

优先路径：
1. `web/routers/presets.py`
2. `web/routers/universe.py`
3. `web/routers/research.py`
4. 对应 `web/services/*`

## Notes

- 查询或解释类请求，先说明应该走哪个入口，不要直接跳内部实现
- 用户明确要求运行命令时，可以直接执行；但如果涉及长耗时或外部依赖，先做轻量检查
- `quant-investor web` 是 workspace 正统入口；不要默认从 `uvicorn web.app:app` 之类 legacy 工厂起服务

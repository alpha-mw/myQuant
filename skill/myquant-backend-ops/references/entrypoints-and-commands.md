# myQuant Entrypoints and Commands

先用当前公开入口回答；不要把内部 helper、旧 DAG 或规范中的未来 writer 当成
可运行命令。

## Public entrypoints

| 场景 | 当前入口 | 行为 |
|---|---|---|
| 读取活动主线 | `quant-investor research run` | 精确读取一个 strategy pointer |
| 兼容读取面 | `quant-investor market analyze` | 与 `research run` 同一只读结果 |
| 兼容读取面 | `quant-investor market run` | 与 `research run` 同一只读结果 |
| 市场维护 | `quant-investor market maintain` | 独立的本地数据维护 lane |
| 旧下载别名 | `quant-investor market download` | compatibility alias |
| 主线回测 | `quant-investor market backtest` | 固定 fail closed：不可用 |
| V4 Forward | `quant-investor-v17-v4 run-forward` | 显式 request 的 Shadow evidence |
| R2.2 评价 | `quant-investor-v17-v4 research-evaluate` | 离线、stdout-only |
| Workspace | `quant-investor web` | 启动当前 FastAPI workspace |

仓库当前没有 production mainline publisher 或 activation command。

## Exact mainline read

三个公开命令必须提供相同的参数：

```bash
quant-investor research run \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>

quant-investor market analyze \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>

quant-investor market run \
  --workspace-root /absolute/path/to/myQuant \
  --strategy-id <strategy-id>
```

它们都只读取：

```text
results/v17_mainline/strategies/<strategy-id>/_active.json
```

不接受旧的 `--stocks`、`--capital`、`--risk`、`--mode` 或 `--top-k` 运行
语义，不扫描最新 run，也不写结果。

## Market maintenance

```bash
quant-investor market maintain \
  --market CN \
  --staged \
  --resume \
  --batch-size 200 \
  --max-batches-per-run 1
```

这是数据维护，不是 V17 decision run。执行前检查本地配置、canonical
pointer、manifest 和任务是否允许 provider acquisition。正式验证时不得用 CSV
替代严格 Parquet canonical 数据。

## V4 Forward/Shadow

```bash
quant-investor-v17-v4 run-forward \
  --workspace-root /absolute/path/to/myQuant \
  --request-path data/private/v17_v4_runs/forward_requests/<request-id>.json \
  --request-sha256 <exact-byte-sha256>
```

request path 和 byte SHA 必须精确。Shadow receipt 不会创建或推进 mainline
pointer。

## R2.2 Forward Research Evaluator

```bash
quant-investor-v17-v4 research-evaluate \
  --workspace-root /absolute/path/to/myQuant \
  --request-path data/private/research_intelligence/evaluation_requests/forward-evaluation-request-<sha256>.json \
  --request-sha256 <exact-byte-sha256>
```

该命令只输出一行 canonical JSON envelope。它不写 result、不 append memory、
不改 factor weight、不选组合、不读写 active pointer。

## Workspace runtime

```bash
quant-investor web --reload
```

装配链：

```text
web.main:app -> web.api:app -> web.workspace_app:app
```

健康检查是 `GET /api/health`。不要默认启动 `web.app:app`；它是显式旧 API
场景才检查的 legacy factory。

## Routing reminders

- “跑一个股票研究”在当前 public mainline 不是可用写入工作流；先要求明确
  strategy id，然后读取已有活动结果。
- “全市场 daily pipeline”不能被 `market run` 这个兼容名称误导：当前实现仍
  是只读 active pointer。
- “回测”应返回 `V17_BACKTEST_UNAVAILABLE`，不能偷偷调用旧 backtest。
- “每天自动跑 I0/R2.2”目前没有正式 scheduler；不能以 standalone legacy
  automation 代替。

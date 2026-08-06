# myQuant Workspace API Surface

以 `web.workspace_app` 实际挂载的 router 为准，不复用旧 endpoint 清单。

## Runtime entry

```text
quant-investor web
-> web.main:app
-> web.api:app
-> web.workspace_app:app
```

当前装配文件：

- `web/main.py`
- `web/api/__init__.py`
- `web/workspace_app.py`

健康检查：`GET /api/health`。

## Current public routers

### Research mainline read

```http
GET /api/research/{strategy_id}
```

可选 query：`expected_pointer_sha256`。

- Router: `web/routers/research.py`
- Response model: `web/models/research_models.py`
- Authority reader: `quant_investor.v17_mainline.read_public_run`
- Source: `results/v17_mainline/strategies/<strategy-id>/_active.json`

这是精确、只读 endpoint。没有 POST run、history list、job stream、report 或
delete endpoint，也不写 `web_runs.db` 或 learning case。

### Settings

- `GET /api/settings/`
- `GET /api/settings/models`
- `PATCH /api/settings/`

Router: `web/routers/settings.py`。Models: `web/models/settings_models.py` 与
`web/models/research_models.py`。设置写入是显式外部副作用；修改前必须确认
用户授权，并避免暴露 credentials。

### Universe

- `GET /api/universe/{market}/presets`
- `GET /api/universe/{market}/{key}/symbols`
- `POST /api/universe/{market}/resolve`

Router: `web/routers/universe.py`。Universe 辅助 endpoint 的市场能力不扩大
V17 public mainline 的 CN-only authority boundary。

### Data

- `GET /api/data/statistics`
- `GET /api/data/market/overview`
- `GET /api/data/stocks`
- `GET /api/data/stocks/{ts_code}`
- `GET /api/data/stocks/{ts_code}/dossier`
- `GET /api/data/stocks/{ts_code}/overview`
- `GET /api/data/stocks/{ts_code}/ohlcv`
- `GET /api/data/stocks/{ts_code}/competitors`
- `POST /api/data/import`

Router: `web/api/data.py`。Service: `web/services/data_service.py`。`import` 是
显式写入；只在用户要求且输入来源清楚时调用。

## Not mounted as current workspace API

以下不是当前 `web.workspace_app` 的公开 router：

- `/api/presets/*`
- `POST /api/research/run`
- `/api/research/history/*`
- research job stream/report/delete routes
- legacy `/api/v1/*` analysis/portfolio surface

`web/app.py` 是旧 factory。只有用户明确提到 legacy `/api/v1` 或旧 analysis/
portfolio 行为时才检查，并先验证它当前实际挂载了什么；不要把它的 route、
service 或数据库归属写成 workspace 的现行能力。

## Ownership answer order

用户问“接口在哪 / 谁写数据 / 为什么没落盘”时按以下顺序回答：

1. 当前 runtime 是否实际挂载该 route。
2. Router 和 request/response model。
3. Service 或 exact authority reader。
4. 实际读取/写入的 pointer、数据库或 artifact。
5. 是否需要外部副作用授权。

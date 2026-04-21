# myQuant Workspace API Surface

先回答公开 API 在哪，再回答 service 和持久化层在哪。

## Runtime Entry

Current workspace runtime:

```text
quant-investor web
-> web.main:app
-> web.api:app
-> web.workspace_app:app
```

Primary assembly files:
- `web/main.py`
- `web/api/__init__.py`
- `web/workspace_app.py`

Health endpoint:
- `GET /api/health`

## Current Public Routers

### `/api/research`

Router:
- `web/routers/research.py`

Models:
- `web/models/research_models.py`

Services and stores:
- `web/services/research_runner.py`
- `web/services/run_history_store.py`

Important endpoints:
- `POST /api/research/run`
- `GET /api/research/history/list`
- `GET /api/research/startup-context`
- `GET /api/research/{job_id}`
- `GET /api/research/{job_id}/stream`
- `GET /api/research/{job_id}/report`
- `DELETE /api/research/{job_id}`

Persistence:
- `data/web_runs.db`
- `data/workspace_learning/`

### `/api/presets`

Router:
- `web/routers/presets.py`

Models:
- `web/models/research_models.py`

Store:
- `web/services/preset_store.py`
- `web/services/run_history_store.py`

Important endpoints:
- `GET /api/presets/`
- `GET /api/presets/{preset_id}`
- `POST /api/presets/`
- `PUT /api/presets/{preset_id}`
- `DELETE /api/presets/{preset_id}`

Persistence:
- `data/web_runs.db`

### `/api/universe`

Router:
- `web/routers/universe.py`

Supporting data providers:
- `quant_investor.data.universe.cn_universe`
- `quant_investor.data.universe.us_universe`

Important endpoints:
- `GET /api/universe/{market}/presets`
- `GET /api/universe/{market}/{key}/symbols`
- `POST /api/universe/{market}/resolve`

Notes:
- `market` is `CN` or `US`
- 这是公开的 universe 解析面，不要先跳到市场下载模块

### `/api/settings`

Router:
- `web/routers/settings.py`

Models:
- `web/models/settings_models.py`
- `web/models/research_models.py`

Supporting store:
- `web/services/run_history_store.py`

Important endpoints:
- `GET /api/settings/`
- `GET /api/settings/models`
- `PATCH /api/settings/`

Persistence:
- `.env`
- `data/web_runs.db` summary reads

### `/api/data`

Router:
- `web/api/data.py`

Service:
- `web/services/data_service.py`

Important endpoints:
- `GET /api/data/statistics`
- `GET /api/data/market/overview`
- `GET /api/data/stocks`
- `GET /api/data/stocks/{ts_code}`
- `GET /api/data/stocks/{ts_code}/dossier`
- `GET /api/data/stocks/{ts_code}/overview`
- `GET /api/data/stocks/{ts_code}/ohlcv`
- `GET /api/data/stocks/{ts_code}/competitors`
- `POST /api/data/import`

Persistence:
- `data/stock_database.db`
- some derived reads from `data/app.db` and `results/`

## Legacy API Surface

Only go here when the user explicitly mentions `/api/v1`, `analysis`, legacy `portfolio`, or old health endpoints.

Legacy factory:
- `web/app.py`

Mounted legacy routers:
- `web/api/analysis.py` -> `/api/v1/analysis`
- `web/api/data.py` -> `/api/v1/api/data` via the legacy factory prefixing behavior
- `web/api/portfolio.py` -> `/api/v1/portfolio`
- `web/api/settings.py` -> `/api/v1/settings`

Legacy health:
- `GET /api/v1/health`

Important caution:
- `web/api/data.py` is currently shared between current workspace and legacy factory, so always state which runtime you are talking about

## Ownership Pattern

When the user asks “接口在哪 / 谁在写库 / 为什么没落盘”, answer in this order:
1. Router file
2. Request or response model file
3. Service or store file
4. SQLite file or results directory

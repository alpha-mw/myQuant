# myQuant 综合整改 Master Runbook（供 Codex 执行）

---

## 使用说明

1. 把本文件放入 `docs/runbooks/` 并随首个 commit 入库，作为断点续跑的锚。
2. **断点续跑协议**：每次新会话，先读本文件 + `git log --oneline -30`，凭各 Phase 的 commit message 判断进度，从未完成的最早 Phase 继续。建议每完成 2–3 个 Phase 开新会话。
3. **每个 Phase 一个独立 commit**，使用该 Phase 指定的 message。验收命令全绿才允许 commit；红灯连续两次修复失败 → 停止并向我报告，不要继续。
4. **两个 STOP 门**：Phase 5a 与 Phase 9a 只产出设计文档，产出后**必须停下等我批准**，未批准不得进入 5b / 9b。
5. Phase 顺序即执行顺序（0 → 11）。Phase 4 依赖 Phase 2（拥挤度权重进扫描）、Phase 6 依赖 Phase 4（净成本列加进证据表）。

---

## 全局硬约束（适用于所有 Phase，违反即失败）

1. 先读 `AGENTS.md` 并全程遵守：offline、确定性、测试中不调用任何外部 API（Tushare / yfinance / LLM / broker），保持 `research run` / `market maintain|analyze|run|backtest` 及 `python daily_runner.py` 等公共契约。
2. **契约测试永绿清单**（每个 Phase 验收都要跑）：
   `tests/unit/test_theme_default_off_contract.py`、`test_theme_fail_safe_contract.py`、`test_theme_no_production_wiring.py`、`test_code_retirement_audit.py`，以及 CI 引擎子集（`ci-cd.yml` engine-core 列出的 7 个测试文件）。
3. **default-off 教义**：一切行为变更路径置于默认关闭的开关之后；契约测试若枚举开关清单，新开关必须登记。
4. 不新增第三方依赖；不重构与当前 Phase 无关的代码；数据/快照缺失一律 fail-open（记诊断，不中断）。
5. 有歧义：先循仓库既有模式；仍无法裁决 → 停下问我，不自作主张扩大范围。
6. **全局回归命令**（每 Phase 结束追加执行）：

```bash
pytest tests/unit/test_market_cli_pipeline.py tests/unit/test_mainline_dag_cutover.py \
       tests/unit/test_llm_model_routing.py tests/unit/test_llm_transport_policy.py \
       tests/unit/test_llm_usage_tracking.py tests/unit/test_review_layer_budgeting.py \
       tests/unit/test_single_mainline.py \
       tests/unit/test_theme_default_off_contract.py tests/unit/test_theme_fail_safe_contract.py \
       tests/unit/test_theme_no_production_wiring.py tests/unit/test_code_retirement_audit.py \
       --override-ini addopts='' -q
flake8 quant_investor web scripts daily_runner.py --count --select=E9,F63,F7,F82
```

---

## Phase 0 — 基线修复（必须最先执行）

**目标**：让仓库恢复"可克隆即可运行"的状态，并应用已交付的三个外部补丁。

步骤：
1. 检查 `git log`：若尚无对应 commit，依次 `git am` 三个补丁（在仓库外我已提供文件）：
   - `0001-fix-risk-bilingual-RiskGuard-hard-veto-keywords-for-.patch`（RiskGuard 中英双语一票否决关键词 + 回归测试）
   - `0002-fix-web-loopback-default-binding-optional-Bearer-aut.patch`（回环默认绑定 + 可选 Bearer 认证 + CORS 通配符禁止）
   - `0003-fix-gitignore-root-anchor-models-so-web-models-sourc.patch`（`.gitignore` 的 `models/` 根锚定，解除对 `web/models/` 的误伤）
2. 提交只存在于本机、从未入库的模块：`git add web/models/ quant_investor/data/universe/us_universe.py && git commit`。
3. 验收：
   - `pytest tests/unit/test_workspace_api_contract.py tests/unit/test_risk_guard_cn_veto.py tests/unit/test_workspace_auth.py --override-ini addopts='' -q` 全绿（真实 schema 恢复后，此前因缺 `web.models` 无法收集的测试应全部通过）；
   - 全局回归命令全绿。

**Commit**：`chore(repo): apply remediation patches and commit missing web/models + us_universe`

---

## Phase 1 — Theme Holding Guard（主题阶段 → 持仓守卫）

**目标**：已持仓标的的主题转入 `distribution` / `overextended` 时，`cn_aggressive` 日度 review 必须收到守卫信号并（可选）收紧阶段止损展示。当前该信息只在入池侧生效，持仓侧完全没有接线。

**必读**：`docs/theme_rotation_system.md`；`quant_investor/themes/storage.py`（`ThemeSnapshotStore.load_latest / load_latest_with_path / load_recent`，默认根 `results/theme_snapshots`——复用它，不新造存储）；`quant_investor/market/dag/theme_context.py`（`theme_rotation.v1` 的 `symbol_primary_theme / symbol_phase / symbol_risk_flags / theme_scores` 字段名）；`quant_investor/themes/types.py`（`ThemePhase` 枚举）；`quant_investor/monitoring/cn_aggressive_portfolio_tracker.py` 中 `stage_stop_price` 与止损缓冲的产生/展示方式，及 `cn_aggressive_daily_review.py` 的报告拼装位置（tracker 风格是"只读 + 建议式文案"，收紧只影响 review 展示，不写回持久化字段）；`tests/unit/test_cn_aggressive_daily_review.py`、`test_theme_storage.py` 的测试风格。

**实施**：
1. 新增纯函数模块 `quant_investor/monitoring/theme_holding_guard.py`：
   - `@dataclass ThemeHoldingSignal`：`symbol / primary_theme_id / primary_theme_name / phase / risk_flags / guard_level(none|watch|tighten) / reasons`。
   - `evaluate_holding_theme_guard(holding_symbols, theme_payload) -> dict[str, ThemeHoldingSignal]`，纯函数无 I/O；规则表（首个命中）：
     - `phase=="distribution"` 或 flags 含 `theme_distribution_risk` → `tighten`
     - `phase=="overextended"` 或 flags 含 `theme_overextended` / `theme_overextended_no_chase` / `theme_fake_breakout_risk` → `watch`
     - 快照缺失 / 无主题映射 / phase 空或 `unclassified` → `none` + 诊断 reason
   - 快照读取放独立薄封装（内部用 `ThemeSnapshotStore`），主函数可用合成 payload 直测。
2. 接线 daily review：主开关 `THEME_HOLDING_GUARD_ENABLED`（默认 `0`，读取仿照 `themes/scanner.py` 里 `THEME_POLICY_CATALYST_ENABLED` 的模式）。关闭时零行为变化。开启时：报告新增「主题状态守卫」小节（watch/tighten 持仓 + 原因；快照不可用输出单行 `theme snapshot unavailable`）；对 `tighten` 持仓将**展示层**止损缓冲乘 `THEME_HOLDING_GUARD_TIGHTEN_RATIO`（默认 `0.5`），行内追加「主题转弱（distribution），止损缓冲收紧」。
3. 文档：`docs/theme_rotation_system.md` 增 **Holding-side Guard** 小节（组件、数据流、两个环境变量、fail-open、default-off 声明）。

**测试** `tests/unit/test_theme_holding_guard.py`：distribution→tighten、overextended→watch、confirmed_rotation→none；仅 risk_flags 触发；快照缺失 fail-open 不抛；主开关关 → review 输出与基线逐行一致；开 + tighten → 缓冲比例正确、报告行存在、自定义 RATIO 生效；monkeypatch 快照加载做端到端集成断言。

**验收**：
```bash
pytest tests/unit/test_theme_holding_guard.py tests/unit/test_cn_aggressive_daily_review.py \
       tests/unit/test_cn_aggressive_portfolio_tracker.py --override-ini addopts='' -q
```
外加全局回归命令。

**Commit**：`feat(theme): wire theme phase guard into holding daily review`

---

## Phase 2 — Crowding Diagnostics（拥挤度三件套）

**目标**：`ThemeScanner` 当前过热判断只有本主题自身价量代理，缺全市场相对维度。新增三个全部可由本地行情算出的拥挤度指标并接入既有 gate/boost 路径，不引入外部数据。

**指标定义**（universe = 本次扫描全部标的，M = 主题成员）：
1. `theme_turnover_share = sum(amount, M 最新日) / sum(amount, universe 最新日)`；`amount` 缺列用 `close×vol` 近似并记 `amount_approximated=true`。历史基线复用 `themes/smoothing.py` 的 `smooth_numeric_series` 对快照历史（`theme_context.py` 已经通过 `ThemeSnapshotStore.load_recent` 传入 `snapshot_history`，limit 10）做 SMA10/Δ5d/trend——零新增存储；`turnover_share_stretch = clamp((share_today/share_sma10 − 1)/1.0)`，历史不足 stretch=0 并记 smoothing_status。
2. `theme_limitup_ratio` = 当日涨停成员数 / member_count。涨停近似判定：阈值按 `ts_code` 前缀（`688/689`、`300/301`→19.5%；`8`/`4` 开头北交所→29.5%；其余→9.5%）；若调用方提供可靠 ST 名称/标识，则按交易日分段（2026-07-06 前 5% 制度、当日及以后 10% 制度），仅有 `ts_code` 时不得推断历史 ST 状态；条件 `pct_chg ≥ 阈值` 且 `close ≥ high×(1−0.002)`；**量纲探测**：Tushare `pct_chg` 是百分数（9.98 非 0.0998），序列绝对值最大 >1 视为百分数；`pct_chg` 缺列由 close 序列自算。`limitup_norm = clamp(ratio/0.30)`。
3. `member_turnover_concentration` = 主题内当日成交额 Top3 之和 / 主题成交额（member_count<4 记 1.0 并打诊断）。

组合分：`crowding_risk = clamp(0.45×stretch + 0.35×limitup_norm + 0.20×concentration)`；权重放模块级 dict（与 `_THEME_PHASE_ADJUSTMENTS` 同风格），供 Phase 4 扫描。

**必读**：`themes/scanner.py`（`_symbol_metrics / _score_theme / _theme_overextension_risk / _infer_phase / _risk_flags`）；`themes/types.py`（`ThemeScore.to_dict`）；`themes/smoothing.py`；`theme_context.py` 约 250–290 行 snapshot_history 装配；**关键前置**：canonical parquet 含 `amount/pct_chg`（`tushare_data_cleaning.py` required_columns），但 DAG 运行时投影 `DAG_RUNTIME_PRICE_VOLUME_COLUMNS`（`market/dag/context.py` 约 52 行）可能未包含——先查投影，缺列则二选一并在 commit 说明理由：(a) 扩投影并评估全市场批读内存影响；(b) scanner 侧降级近似 + diagnostic。

**接线**：主开关 `THEME_CROWDING_ENABLED`（默认 `0`）。关闭 = 新字段中性默认、无新 flag、不进 phase/overextension、funnel 无惩罚。**Universe 合格性 gate**（仿 Markov production-eligibility）：扫描标的总数低于 `THEME_CROWDING_MIN_UNIVERSE`（量级与读取模式参考 `MARKOV_REGIME_MIN_MARKET_SAMPLE`）时 `crowding_status="insufficient_universe"`，只记诊断。开启且合格时：`_theme_overextension_risk` 加 `+0.30×crowding_risk`（仍 clamp [0,1]，不改既有基础公式）；`_risk_flags`：`crowding_risk≥0.70`→`theme_crowded`，`limitup_ratio≥0.20 且 breadth<0.40`→`theme_narrow_leadership`；`deterministic_funnel._THEME_RISK_FLAG_PENALTIES` 增 `theme_crowded:-0.03`、`theme_narrow_leadership:-0.02`（该表只在已开启的 `theme_boost_enabled` 路径生效，属既有门控内增量）。`ThemeScore` 增字段并补 `to_dict`；**旧快照兼容**：无新字段的历史 JSON 加载必须默认值兜底不抛；`schema_version` 是否 bump 查 `versioning.py` 与既有 additive 先例。

**测试** `tests/unit/test_theme_crowding.py`：三指标手算小例断言；两档涨停阈值 + 量纲两路径；amount 缺列近似 + 诊断；`theme_crowded` flag 与 overextension 加法项（≤1.0）；insufficient_universe fail-closed；主开关关 → `to_dict` 既有键逐键与基线一致；旧快照兼容；funnel −0.03 惩罚生效。

**验收**：
```bash
pytest tests/unit/test_theme_crowding.py tests/unit/test_theme_scanner.py \
       tests/unit/test_theme_smoothing.py tests/unit/test_theme_snapshot_persistence.py \
       tests/unit/test_deterministic_funnel_theme_boost.py --override-ini addopts='' -q
```
外加全局回归命令。

**Commit**：`feat(theme): add local crowding diagnostics (turnover share, limit-up, concentration)`

---

## Phase 3 — industry_map 的 PIT 注记（小）

**目标**：replay/calibration 用的是**当天的**行业标签回放历史（公司改分类会产生轻微前视）。不改数据，只把局限显性化。

**实施**：`themes/replay.py` 与 `themes/calibration.py` 的报告/结果组装处加固定注记「industry labels are as-of run date, not point-in-time; replay carries mild reclassification look-ahead」+ 结果 metadata 增 `pit_industry_labels=false`；`docs/theme_rotation_system.md` 同步一段。测试断言注记与字段存在（扩展 `test_theme_replay.py` / `test_theme_calibration_report.py`）。

**Commit**：`docs(theme): annotate non-PIT industry labels in replay/calibration outputs`

---

## Phase 4 — Calibration 阈值扫描落地（把 magic numbers 变成有出处）

**目标**：`themes/calibration.py` 已有 `evaluate_threshold / build_threshold_diagnostics / ThresholdDiagnostic(thresholds_dataframe)` 框架但没有被系统地跑过。落一个离线扫描器，让 `(r+0.10)/0.30`、过热起点 `0.08`、phase 门槛 `35/55/70`、Phase 2 的 crowding 权重这些常量拿到 5/10/20 日 forward alpha 证据。

**实施**：
1. 新 `scripts/run_theme_threshold_sweep.py`：输入本地快照历史（`ThemeSnapshotStore.load_recent`）+ 本地行情 forward returns；对上述常量做网格扫描，调用既有 calibration API；输出 `results/theme_calibration/threshold_sweep_<date>.{json,md}`（阈值 × forward alpha / hit rate 表）。纯离线、确定性（固定输入 → 固定输出）。
2. `docs/theme_rotation_system.md` 增 **Evidence** 小节：登记当前每个 magic number 与其最近一次 sweep 证据的对照表（值可先留占位，由我跑真数据后填）。
3. 测试：合成快照史 + 合成行情跑通 sweep 骨架，断言输出 schema 与确定性（同输入两跑逐字节一致）。

**Commit**：`feat(theme): add offline threshold sweep runner with forward-alpha evidence output`

---

## Phase 5 — 概念级主题（跨行业热点）【STOP 门】

**背景**：`industry_map` 来自公司档案行业标签，抓不住「低空经济」这类跨行业概念。**硬红线：禁止用当前 Tushare 概念成分回放历史（事后归类是主题回测第一大前视）。**

### 5a 设计文档（做完必须停，等我批准）

产出 `docs/architecture/<日期>-concept-themes.md`，内容必须含：
1. 两条路线的完整对比：(A) 本地收益率相关性聚类出统计主题（零外部数据，纯教义内）；(B) point-in-time JSONL 概念成分表——复用 `themes/policy.py` 的本地 JSONL 解析模式，membership 记录带 `effective_from/effective_to`，由人工/半自动维护。
2. `theme_membership.v1` schema 提案；与现有 industry_map 的合并语义（概念与行业并存时的主主题裁决）。
3. replay / calibration / smoothing 历史兼容；开关设计与契约测试影响面。
4. 决策矩阵（维护成本、前视风险、覆盖度、确定性）+ 你的推荐。

**Commit（仅设计）**：`docs(theme): concept-level theme design proposal` → **STOP，等批准**。

### 5b 实施
仅在我书面批准 5a 后，按批准稿执行。默认不启动。

---

## Phase 6 — 成本入环（execution cost 接入回测与主题证据）

**目标**：`portfolio_backtest.py` 现在是 flat bps（佣金 3bp + 印花 + 滑点 5bp）；`factors/execution_cost.py` 里已有 `FactorExecutionCostConfig / DailyExecutionCostRecord / SymbolExecutionCostRecord` 的 participation-based 成本体系但没接进来。热点票的冲击成本恰在最想买时最贵，theme 证据必须看净成本。

**关键契约**：仓库存在 `tests/unit/test_factor_execution_cost_no_runtime_effect.py`——先读它，大概率要求 execution_cost 不得影响运行时决策路径。实现必须满足：新增能力全部置于 `EXECUTION_COST_MODEL_ENABLED`（默认 `0`）之后，关闭时 `portfolio_backtest` 数值输出与现行 flat bps **逐数值一致**（写基线回归测试），契约测试保持绿。若契约与设计冲突，以"独立回测入口/参数注入"满足契约并在 commit 正文说明。

**实施**：
1. `portfolio_backtest.py` 的 `TransactionCostModel` 增可选注入点：由 execution_cost 的记录/配置驱动 participation-based 成本（成交额占比 → 冲击），flag 关闭时行为不变。
2. Phase 4 的 sweep 输出增 net-of-cost forward alpha 列（毛 / 净并列）。
3. 测试：flag 关 = 基线逐数值一致；flag 开 = 高参与率场景成本单调上升的合成用例；契约测试绿。

**Commit**：`feat(backtest): optional participation-based execution cost model (default-off)`

---

## Phase 7 — 死模块退役（第一轮 P0 #3）

**目标**：9 个零引用孤儿模块与"单一主线"教义冲突，且部分含方法论错误（未播种的 MC VaR、把随机噪声当压力测试）。用仓库既有的退役机制处理，不发明新机制。

**机制锚点**：`scripts/workspace_layout.py` 中已有 `code_retirement_candidate` 三元组注册表（9 条 Kronos 先例同款：`(Path, "code_retirement_candidate", 说明)`），`tests/unit/test_code_retirement_audit.py` 审计 `exists=False` 与 `reference_count==0`。

**清单**（我核查引用为 0，以现场重扫为准）：`quant_investor/{advanced_risk_metrics, factor_analyzer, news_analysis, sentiment_analysis, signal_calibration, stress_tester, var_calculator, financial_analysis, risk_management_layer}.py`。

**实施**：逐个：全仓 grep 引用（含 tests/docs）→ 确认 0 → 注册进三元组表（说明里写明退役原因；`var_calculator` 的说明须注明"如恢复需重写：seeded rng / ddof=1 / 最小样本护栏"）→ 删除文件 → 扩 `test_code_retirement_audit.py` 断言。**发现任何真实引用 → 停下报告，不强删。**

**Commit**：`refactor(core): retire 9 orphaned risk/analysis modules via retirement audit`

---

## Phase 8 — 静默异常入诊断（第一轮 P1 #5）

**目标**：`market/download_cn.py` 内约十余处 `except Exception: pass/continue`（286/315/890/946/976/996 等）在数据管道里静默吞错——停牌检测失败与真实缺数不可区分。

**实施**：本 Phase **只改 `download_cn.py`**。模式：捕获 → 构造 `DataQualityIssue`（`quant_investor/data_quality_contract.py`，字段 `issue_id/symbol/issue_type/severity=warning/message=str(exc)/source="download_cn.<函数名>"`）→ 挂入该文件既有 issues 聚合路径（先读它如何汇集；确无聚合路径的场景至少 `logger.debug`）。**不改变控制流**（原 continue 仍 continue）。全 repo 其余同类点只产出清单登记到 `docs/`，本 Phase 不动。测试：monkeypatch 制造异常，断言 issue 产生且下载流程不中断。

**Commit**：`fix(data): route silent download_cn exception swallows into data-quality diagnostics`

---

## Phase 9 — 退市股 Universe / 幸存者偏差（第一轮 P0 #2）【STOP 门】

**背景锚点**：`market/download_cn.py:227` 与 `stock_database.py:362` 均 `stock_basic(list_status="L")`——只下现存上市股，所有基于本地 mart 的因子回测系统性高估，且对激进集中组合的伤害最大。

### 9a 设计文档（做完必须停，等我批准）

产出 `docs/architecture/<日期>-pit-universe.md`，必须含：拉取 `L,D,P` + `delist_date` 的落库方案；`is_listed(symbol, date)` PIT 函数与 tradability/funnel 门控接入点；历史行情回补策略与 Tushare 积分/QPS 成本估算；replay 与 `factors/backtest.py` 的影响面；**回补暂缓时的替代方案**（对现有回测结果按退市缺口做敏感性折减的方法论）；迁移与回滚步骤。

**Commit（仅设计）**：`docs(data): point-in-time universe design for delisting survivorship` → **STOP，等批准**。

### 9b 实施
需要我两个显式授权：(1) 设计批准；(2) **一次性在线回补窗口授权**（实施本身要调 Tushare，突破 offline 默认，必须单独点头）。

---

## Phase 10 — 工程质量批（第一轮 P2 #6/#8/#9/#10 + P3 #12）

**10.1 CI 与测试基建**：`ci-cd.yml` 增 nightly 全量 job（`schedule` cron + `workflow_dispatch`；`pytest tests/ --cov=quant_investor` 上传覆盖率工件；PR 仍跑现有快子集）；`pyproject.toml` 的 `[tool.pytest.ini_options]` **去掉默认 `--cov`**（本地/PR 变快，CI nightly 显式加，顺带删除各处 `--override-ini addopts=''` 的需要）；mypy 从 `quant_investor/agents/`、`quant_investor/factors/` 起步收紧（这两个包去掉 `--no-strict-optional`，用 per-module override 表，其余暂维持）；新增 `scripts/check_requirements_sync.py`（pyproject 与 requirements.txt 漂移检查）并入 CI。

**10.2 `.env` 加载去重**：五处手写解析（`quant_investor/config.py`、`quant_investor/llm_policy.py`、`web/config.py`、`web/workspace_app.py` lifespan、`web/services/settings_service.py`）合并为单一 `quant_investor/env_loading.py`。**精确保持现行优先级：`os.environ` 已有键不覆盖**。逐 key 行为回归测试。

**10.3 命名冲突**：`quant_investor/agent_orchestrator.py` 重命名为 `quant_investor/control_chain.py`；保留旧模块名 re-export shim + `DeprecationWarning`；更新 `tests/unit/test_control_chain_orchestrator.py` / `test_single_mainline.py` 的 import。

**10.4 仓库卫生**：`the-security-guide.md` → `docs/notes/`；`run_us_aggressive_analysis.py` → `scripts/`（先 grep 全部引用，必要时根目录留一行 shim 保契约）；`portfolio_dashboard/generated/` 入 `.gitignore` 并 `git rm --cached`；`docs/` 增一段 injection-surface 设计说明（LLM/新闻文本进入 risk_texts 至多引发**保守方向**的误否决，是有意的失效极性）。

**Commit**（可按 10.1–10.4 拆四个 commit，message 前缀分别 `ci:` / `refactor(config):` / `refactor(core):` / `chore(repo):`）

---

## Phase 11 — daily_runner 拆分（第一轮 P2 #7 一期）

**目标**：`daily_runner.py`（1,488 行，根目录）内含 `HistoryLoader / AnalysisRunner / ReportBuilder / PersistenceManager`，与包内职责重复。迁入 `quant_investor/automation/`，根目录留薄 shim。

**硬序（不可颠倒）**：daily_runner **当前没有任何测试** → 第一步先写特征化测试：`load_config`、`dry_run`、`print_last_report` 的行为快照（monkeypatch 掉一切外部路径），绿了才允许动代码。第二步迁移四个类到 `quant_investor/automation/`（按类拆文件），`daily_runner.py` 变成 import + 转发的薄 shim——**`python daily_runner.py` 的调用方式是 AGENTS.md 契约，必须原样可用**。第三步特征化测试在迁移后不改一字仍绿。

`cn_aggressive_portfolio_tracker.py`（4,068 行）与 `web/services/data_service.py`（3,008 行）本 Phase **只产出拆分设计 notes** 到 `docs/`，不动代码。

**Commit**：`refactor(automation): extract daily_runner internals into quant_investor.automation`

---

## 收尾 — 人工清单（CC 不执行，完成 Phase 11 后打印提醒我）

1. GitHub 仓库设置：启用分支保护 + 必需 status checks；**确认 Actions 已启用**（外部审计时 head commit 上可见的 check runs 为 0，三个 workflow 可能从未真正跑过）。
2. 若需对局域网/远程暴露工作台：设置 `WORKSPACE_AUTH_TOKEN` 并显式改 `API_HOST`。
3. Phase 4 的 sweep 用真实快照历史跑一轮，把 Evidence 表的占位填实。
4. 待我批准的两个 STOP：Phase 5b（概念级主题实施）、Phase 9b（PIT universe 实施 + 在线回补授权）。

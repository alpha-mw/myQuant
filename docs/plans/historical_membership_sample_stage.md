# 历史成员样本端到端 — 本阶段新增契约与架构变化（第 3 版，已闭合 Architect 与 Critic 意见）

范围：仅列出**本阶段新增**的契约与架构改动。可行性阶段已确认的事实见
`reports/operations/historical_membership/20260908/feasibility_stage_report.md`，不再重复调查。

**不变量**：生产 `full_a`、生产指针（`5cf35e39…` / `cn_fundamental_primary_20260904_restated_v3`）、
当前调度、实际持仓与交易状态一律不动；不晋升历史产物；不启动 5,751 只全量重建。

---

## 样本（4 只，尽量一只覆盖多场景）

| 标的 | 场景 | 选择理由（证据见归档目录） |
|---|---|---|
| `600311.SH` | ①真实漏抓 + ③退市后仍有披露 | 窗口内供应商 357 行、本地 0 行；`effective_to=20230327`，而 income/fina_indicator 到 `20260630`——晚于摘牌三年多 |
| `600068.SH` | ②有停牌证据、窗口内无成交 | `suspend_d` 20210906–20210910 共 5 行 `S`，覆盖窗口∩区间全部交易日；`effective_to=20210913`。⚠️ **预期会被 `_canonical_bar_history_bounds`（`mart:2835`）硬拒**：它要求每个身份在资格窗口内至少有一根行情，且 `non_blocking_absent` 在此处**无豁免**。该股窗口内确无成交，供应商也补不出来 → **按阻塞交付并指名所需改动，不强行放行** |
| `000004.SZ` | ④退市但本地**有**窗口行情（对照 142 组） | 与 `600311.SH` 构成"同为退市、漏抓与否"的对照 |
| `000001.SZ` | ⑤在市对照 | 开区间 `effective_to` 为空 |
| `600000.SH` | ⑤第二在市对照（**Critic 促成的新增**） | 保证全窗口合格数 ≥ 2，见下方分位可达性 |

**提交给重建的批次 = 4 只**：`600311.SH`、`000004.SZ`、`000001.SZ`、`600000.SH`。
**`600068.SH` 不进入重建批次**——它必然在 `mart:2835` 硬拒，明知会失败仍提交等于故意让整批失败。
它作为**独立阻塞项**交付：停牌证据已完备（只读），另附所需改动说明。

---

## 新增契约（3 个）

> **命名与治理更正（Architect 指出）**：仓库对内联 JSON `schema_version` 的既有约定是
> `cn-*.vN`（`daily-basic-coverage-intervals.v2`、`cn-full-a-coverage.v4`、`cn-fundamental-pointer.v1`），
> `myquant-*.vN` 只出现在 `market_data_store.py`。三个契约改用 `cn-research-*.v1` 前缀。
> 另有既有研究约定 `_RESEARCH_COMMON_FIELDS = {authority, production, research_only, run_state}`
> （`quant_investor/contracts/builtins.py:1025-1027`），C1 复用 `research_only` 语义，
> **不再自造** `tradable_universe: false`。

### C1. `cn-research-historical-scope.v1`
研究用历史范围产物，**显式非生产、非可交易**。字段：
`symbols`（区间相交推导，非按 `delist_date`）、`window_start`/`as_of`、
`membership_ref{path,sha256,generation_id}`、`intervals`（逐身份 `effective_from`/`effective_to`/`status`）、
`derivation`（谓词原文）、`symbol_set_sha256`（**独立计算，禁止复用 `full_a` 的数量或 SHA**）、
`tradable_universe: false`、`research_only: true`。

### C2. 期望覆盖——**改为复用既有能力，不新建第三套模型**
`quant_investor/market/cn_history_audit.py` 已实现"`trade_cal` 推导的开市日
（`:457-499`，源 `tushare.trade_cal`）∩ 成员区间（`:166`）− `suspend_d` 证据（`:509-1004`）"，
入口 `build_cn_history_audit`（`:212`）/ `run_cn_history_audit`（`:1164`），
精确证据需 `--allow-online`（`:1233`）。
本阶段**以受限 scope 调用它**，只新增 `cn-research-coverage-expectation.v1` 作为其**输出封装**：
逐身份 `expected` / `observed` / `legitimately_excluded` / `unexplained`；
`unexplained > 0` 即**阻塞**。**禁止**用本地 `bar_first`/`bar_last` 缩短期望窗口。

### C3. `cn-research-validation.v1`
受控研究验收合同。**不冒充生产全量验收**：显式声明样本规模、
不满足生产 `expected_scope_count` 闭合恒等式、不产生可晋升产物。

---

## 架构变化（4 处，范围明确）

### A1. 研究根隔离
产物写入 **`data/private/research/hist_sample_20260908/`**——按仓库既有私有根约定
`data/private/…`（`scope_transition.py:73-74`），非我第 1 版写的 `data/research/…`。
自带 market 指针、覆盖声明、fundamental 指针；生产根只读。

⚠️ **治理前提（Architect 指出，必须显式记录而非默默绕开）**：
`FUNDAMENTAL_RESEARCH_ROOT` / `FUNDAMENTAL_RESEARCH_OVERLAY_MODE` /
`FUNDAMENTAL_RESEARCH_ACTIVATION_PATH` 均在 `RETIRED_ENV_KEYS`（`config.py:70-87`），
`_reject_retired_env_keys`（`:92-99`）在 import 时即抛 `RuntimeError`。
即：**本仓库曾刻意废弃过 fundamental 研究根 overlay**。
本阶段（a）**不使用**这些环境变量名传递研究根，改为显式函数参数；
（b）在交付中说明本次隔离与被废弃 overlay 的差异（一次性、只读生产、不产出可晋升物）。

⚠️ **锁可见性**：`assert_scope_readable` 以 `data_root.parent` 找
`<workspace>/data/private/cn_daily_maintenance/scope_transition_active.json`。
研究根须置于能观察到该标记的位置，或在研究运行中**显式断言生产标记**，
否则可能在生产 scope transition 进行中读取生产产物。

### A2. 三处硬编码解绑

**A2-1 `mart:5919`** `universes == ["full_a"]` → 具名允许表，新增键必须携带 C1 产物。
⚠️ **顺序缺陷（Architect 指出）**：该检查在 `:5919`，而隔离证据 `staging_base` 在 `:5933-5936`
才算出。允许表闸门**必须下移到 `:5936` 之后**，否则具名键会在拿到隔离证据前就被拒。
生产不变由构造保证：唯一生产调用方传 `["full_a"]`，默认允许表即 `{"full_a"}`。

**A2-2 `mart:5941`** ——**我第 1 版写错了，会破坏生产，已更正**。
原写"改用传入的 `data_root`"：但 `data_root` 默认 `DEFAULT_FUNDAMENTAL_ROOT = data/parquet/cn`，
而 `_resolve_symbols_from_parquet_universe` 要的是 **market 根**
（`DEFAULT_MARKET_DATA_ROOT = data`，内部拼 `<root>/cn_universe/cn_index_components.json`
与 `MarketDataReader(data_root=…)`）。改用 `data_root` 会去找 `data/parquet/cn/cn_universe/…`。
**正确做法**：给 `run_cn_fundamental_maintenance` 新增
`market_data_root: str | Path = DEFAULT_MARKET_DATA_ROOT`，默认字面量即当前值。
另需注意 `MarketDataReader._load_components` 有 `Path.cwd()/data/cn_universe/…` 回退
（`market_data_reader.py:1369-1372`），研究运行中其他 reader 调用可能静默读到生产成分股。
**第 2 版只指出风险未给对策，现补**：研究根**必须**自带 `cn_universe/cn_index_components.json`，
运行入口启动时断言该文件存在且其 `full_a` 恰为样本集合；
若缺失即**报错退出**，不允许落入 CWD 回退。

**A2-3 `mart:2358-2360`** `symbols.extend(scoped or sorted(serving_symbols))`
**fail open 到生产全集** → 空交集必须**报错**。
**不可按根隔离**（该函数只有一个活调用方），是**全局修缺陷**，不是放宽。
另记：`:2351` 无论请求哪个 universe 都硬取 `universe_key="full_a"` 作 `serving_symbols`，
故具名键不能扩出研究 serving 根之外。
⚠️ **安全性是数据状态而非代码不变量（Critic 更正我第 2 版的表述）**：改后生产不失败，
**仅因**当前 `cn_index_components.json` 的 `full_a`（5,556 项）与 `serving_symbols` 交集非空。
成分股重生成、universe 键改名或迁移窗口期内该交集可能为空——
届时现状会静默回退到全集继续跑，改后则**硬失败**。此风险**显式接受并记录**，不假称由构造保证。

### A3. 覆盖声明路径参数化（**必须做，非可选**）
`DAILY_BASIC_COVERAGE_BOUNDARY_PATH`（`mart:2625-2627`）唯一读者是 `_declared_coverage_intervals`
（`:2659`）。`scope_transition.py:436` 自建同名字面量、互不相干，故**可安全分离**。
实现须为**可选参数 + 模块全局回退**（`boundary_path or DAILY_BASIC_COVERAGE_BOUNDARY_PATH`），
因为三处测试 monkeypatch 的是模块属性
（`test_fundamental_suspension_aware_history.py:227,261`、`test_fundamental_generation_promotion.py:54`）。

**为什么必须做**：生产覆盖声明现有 61 个区间、`cutoff=20260904`，**4 只样本无一在内**。
不参数化则 scope 过滤得零行、`_validate_daily_history_coverage_intervals` 空集短路不报错，
而 `:2720-2722` 会把**生产文件的绝对路径与 SHA256** 盖进研究 scope evidence——
provenance 静默错误且无任何报错。
**禁止**采用 `scripts/generate_daily_basic_coverage_intervals.py:77-81` 的"临时改名生产文件"办法
（那是改动生产输入，且与 `scope_transition._retire_stale_coverage_declaration` 竞态）。
`pointer_membership_path`（`:2932-2935`）**无需**参数化——写绝对路径即根正确。
**schema 保持 v2**。

### A4. 最小研究消费适配（mart → `MatrixDataBundle`）
**显式仅供研究**，不建通用回测平台。

**必备字段（`matrix.py`）**：`contract_id`/`universe` 非空；`symbols` 非空、无重复、
**默认要求已排序**（掩码行序须按排序后的顺序构造）；`dates` 非空、ISO `YYYY-MM-DD`、
**严格升序**（mart 侧是 `YYYYMMDD`，需转换）；掩码形状 `len(symbols) × len(dates)`。
**能跑起来的最小字段集**：`execution_price` 默认解析为 vwap，而 `vwap` 仅在
`amount` 与 `volume` 同时存在时才派生（`matrix.py:686-687`），否则 `backtest.py:858-859` 抛错
→ 需 `close`+`amount`+`volume`，或显式 `execution_price="close"` 只给 `close`。
日期数须 ≥ `delay_days + holding_period_days + 1`。

⚠️ **静默空跑——第 2 版我给的修法是错的，Critic 指出后已更正。**

一般规律（实测 `_quantile_for_rank`，`backtest.py:912-913`）：
**`long_quantile = quantile_count` 可达，当且仅当 `quantile_count ≤ 当日合格数`。**

| 合格数 | `qc=4` | `qc=2` |
|---|---|---|
| 4 | 可达 | 可达 |
| 3 | **不可达→权重恒 0** | 可达 |
| 2 | **不可达→权重恒 0** | 可达 |
| 1 | 不可达 | 不可达（结构使然，非缺陷） |

我第 2 版定的 `quantile_count=4` 在合格数为 3 时不可达，
而 `600068.SH` 被阻塞后合格数**正是 3**——等于把刚修掉的空跑原样复现。
**已改为 `quantile_count=2`、`long_quantile=2`、`long_short=False`**
（`quantile_count ≥ 2` 是硬下限，`schema.py:423`；`long_short` 关闭以免 `short_quantile` 必填）。

**合格数时间线**（据样本 `effective_to`）：
2021-09~2023-03 为 3（`600311`+`000004`+`000001`+`600000` 中 600311 在市，故实为 4）→
`600311` 摘牌后为 3 → `000004` 摘牌（20260714）后为 2。加入 `600000.SH` 后**全窗口 ≥ 2**。

**验收标准改为逐日、结构化**（"权重非全零"太松，会掩盖正要证明的失败）：
- 合格数 ≥ 2 的每一日，`long_rows` 必须非空
- `600311.SH` 在其末次可交易日**及之前**合格、之后**不合格**
- `000004.SZ` 同理
- 任一标的在其 `effective_to` **之后无任何面板行**

**逐格 `reason` 保留**：`universe_mask` 内**不能**携带 reason
（`_coerce_bool_matrix`，`matrix.py:218-231`，非 bool 元素直接拒绝），须放 `bundle.metadata`。
实现为**兄弟函数** `build_pit_universe_mask_with_reasons(...) -> (mask, reasons)`，
**不原地改** `pit_universe.py:751` 的返回类型（会破坏
`tests/unit/test_pit_universe.py:189,196` 与 `test_factor_backtest.py:295`）。

---

## 成员资格 / 可交易性 / 目标权重 / 实际持仓 的边界（按已确认证据）

- **成员资格**：`effective_to` **闭区间**，依据既有成员合同与 `mart:3068`。
  **不改已发布派生合同语义**——可行性阶段已证：单独处理可交易性即可得到正确行为。
- **可交易性**：`effective_to` 当日为 false（20/20 样本在摘牌日无成交）。
  该修正落在 `pit_universe.py:656`——它当前用一条对可交易性正确的规则返回了**成员资格**判定。
  **本阶段做它是因为消费端需要，不是因为它便宜。**
  ⚠️ **不可按根隔离**：`evaluate_listing_status` 是无 root/config 参数的纯函数，
  改动会波及 `build_pit_universe_mask:751`、`build_pit_delisted_field:773`、
  `filter_symbols_by_pit_status:806-811` → `MarketDataReader.list_symbols`、`download_cn.py:425`。
  且会翻转 `tests/unit/test_pit_universe.py:190-194`（该测试当前断言的正是与本裁定相反的边界）。
  当前波及面小**仅因 `PIT_UNIVERSE_ENABLED=0` 这一开关的偶然**，
  **不能表述为"由构造保证生产不变"**。
  另注：`evaluate_listing_status` **根本不读 `effective_to`**；
  `PITUniverseRecord.__post_init__:372` 用 `effective_to = compact_date(effective_to) or delist_date`，
  两者可能不同（如 `600311.SH`），适配层不得假定相等。
- **目标权重为 0 本身可以正确**（`backtest.py` 无成交模型、无持仓账），
  **不据此认定既有测试有误**。
- **实际持仓结转与退出损失**：现有模型不支持。本阶段**不伪称已解决**——
  若样本验证需要真实结转，明确标记为**阻塞**并说明所需范围。
  缺失收益**不默认为零、不静默删日、不假定成交**；无可靠退出价或回收证据时**不给出确定收益**。

---

## 执行顺序（Architect 指出的真实约束）

1. **先建研究行情表**：`_canonical_bar_history_bounds`（`mart:2729-2846`）对**所有**请求身份
   要求资格窗口内至少一根行情，`non_blocking_absent` 在此**无豁免**。
   故研究 bar 表必须先从供应商补齐（`600311.SH` 等），**再**建 scope evidence。
   另需满足：`_canonical_bar_paths`（`:2555-2622`）要求至少一个 parquet、无 symlink；
   列必须恰为 `{ts_code, trade_date}`（检查在调用方 `:2793`）、重读一致（`:2841-2842`）。
2. **循环依赖可解**：`non_blocking_absent_symbols` 直接取自研究指针的 `coverage`，
   它有**两处**用途，不是一处（第 2 版写"唯一"，Critic 更正）：
   `:3073-3074` 放行非活跃区间；**以及 `:3093`** 抑制"活跃成员区间与退市日矛盾"这一独立完整性检查。
   手工编写研究指针时必须知道授权的是两个爆炸半径。
   且 `build_canonical_scope_evidence` **内部并不校验**
   `observed + non_blocking_absent == complete_count` 闭合恒等式
   （该式在 `download.py:687` 产出、由本阶段不调用的治理消费者断言）。
3. **本阶段仍走权威重建代码路径**：`build_canonical_scope_evidence` 只经
   `fetch_tushare_fundamental_full_rebuild` 到达，需 `allow_live=True` 与 `checkpoint_root`。
   "不启动全量重建"这条不变量指的是**规模**（4 只 vs 5,751 只），**不是**避开该代码路径。
4. **两道第 2 版完全没写的硬闸门（Critic 指出）**——它们位于
   `build_canonical_scope_evidence` **之后**、写出 generation **之前**，且都是全有或全无：
   - **端点审计零容忍**：`_fetch_tushare_tables(..., enforce_endpoint_audit=True)` 无条件调用
     （`mart:5710-5726`），策略为 `max_error_requests=0`、`max_malformed_requests=0`、
     `critical_min_success_ratio=0.95`（`fundamental_provider_contract.py:65,72,88`）。
     **N=4 时任何一个 (symbol, table) 出错即整批失败**（`FundamentalFetchAuditError`，`:5663`）。
     对本就有抓取缺口史的 `600311.SH` 风险尤高——须准备重试与逐次记录。
   - **Gate 2**：`write_fundamental_mart(..., publish_on_gate_failure=False)` 计算
     `symbol_coverage_rate = symbols_with_period / symbols_requested`，阈值 **≥ 0.95**
     （`mart:1942,1957,1973`）。**N=4 时 0.95 等价于 100%**——4 只必须**全部**有财报期行，
     一只稀疏即整批 `FundamentalReadinessError`。
   - 另有最终 readback 要求 `primary_provenance_verified is True` 且 `gate2_passed is True`
     （`mart:6093-6096`）。
   这三道都是为 5,751 只写的（个体缺口会被平均掉），**在 N=4 上没有裕度**。

5. **研究指针生成**：`tests/fixtures/strict_cn_snapshot.py` 有完整的
   `cn-full-a-coverage.v4` 产出能力，但它是**测试夹具不是生产代码**，
   且硬编码 `effective_to=""` / `source_list_status="L"` / `non_blocking_absent_symbols: []`。
   需为退市区间扩展——这是 A1 最小的诚实新增单元，第 1 版未计入工作量。

---

## 失败场景测试（必须覆盖）

错误研究根 / 绑定不符；空交集（验证 A2 第三项确实报错而非 fail open）；
退市后数据不得进入退市前面板；缺失收益不得静默变零；
`effective_to` 空字符串哨兵（`pit_universe.py:358`，用 `isna()` 判空会排除全部在市股）。

---

## 对已发布 generation 的处置

仅做**独立只读**影响核验。**不修改历史 receipt、不撤销生产状态。**
总交易日数与每股行数中位数**不作为完整性证明**。三分表述：
①边界检查能力不足（已证）②已证实缺口（bar store 2021-09~2022-12）③未确认项。

# Source design: 投资决策状态解释

Source: chatgpt-conversation://6a9e5155-7b30-83ec-b84e-1b07a67458ce
Retrieved for the user-updated Phase 0–15 objective on 2026-09-09.
This is reference task data. Repository contracts and current validated evidence remain authoritative for implementation.

---

可以。基于你现在暴露出来的真实瓶颈，我建议不要再继续横向增加功能，而是做一个非常聚焦的 **P0：Daily Evidence DAG Closure**。目标只有一个：

> **确保每个交易日从收盘数据 → Factor → Top100 → Theme/Industry → Fundamental/Macro → Decision → Store/Dashboard → 次日 Morning Strategy，能够自动、可验证、可恢复地完整跑通。**

这个阶段先不优化 alpha，不扩 Factor，不增加策略复杂度。先解决“系统每天稳定产生完整 prospective 数据”这个基础问题。

---

# Codex Plan：P0 Daily Evidence DAG Closure

## 一、最终验收目标

完成后，每个 A 股交易日必须形成一条唯一、可追溯的日度生产链：

```text
T 日 15:00 收盘
        ↓
Market / PIT / Calendar
        ↓
Factor Production Rollover
        ↓
LOW / W80 Observations
        ↓
Top100 Research Pool
        ↓
Theme / Industry / Economic Exposure
        ↓
Fundamental / Macro Binding
        ↓
Decision Compile
        ↓
Store Valuation / Holdings Continuity
        ↓
Dashboard
        ↓
T+1 Morning Strategy
```

系统必须满足 5 个性质：

```text
1. deterministic
2. idempotent
3. evidence-bound
4. recoverable
5. observable
```

也就是：

- 同样输入，多次运行得到相同结果；
- 已成功节点不会重复污染；
- 每一个下游输出都知道自己依赖哪个上游 SHA；
- 某个节点失败后可以从断点继续；
- 不需要人工翻日志才能知道哪里断了。

---

# 二、P0 核心原则

## 原则 1：Daily DAG 成为唯一生产入口

目前系统存在一个隐性问题：

很多模块本身都能运行，缺少一个真正的 **production orchestrator**。

因此 Codex 第一件事不是修改 Factor，而是建立：

```text
quant-investor daily-production
```

或者等价入口：

```text
quant-investor production daily-close
```

这个命令成为：

> **日度投资证据链唯一标准生产入口。**

Automation 不再分别“猜应该运行哪些脚本”。

Automation 只调用：

```bash
quant-investor production daily-close \
  --market CN \
  --strategy aggressive_tech_manufacturing \
  --trade-date YYYYMMDD
```

内部 orchestrator 决定哪些节点应该跑。

---

# 三、Phase 0：先冻结现状，建立真实基线

### 目标

在改代码前，先回答一个问题：

> 当前每个 DAG 节点到底有没有 production contract？

Codex 必须先 inventory 以下节点：

```text
Calendar
Market
PIT
Factor
Observation LOW
Observation W80
Top100
Theme
Industry
Economic Exposure
Fundamental
Macro
Decision
Store
Dashboard
Morning Strategy
```

对每个节点记录：

```yaml
node:
  canonical_command:
  input_authorities:
  output_authority:
  pointer:
  manifest:
  idempotency:
  validation_command:
  success_status:
  downstream_consumers:
```

产物：

```text
docs/architecture/daily_evidence_dag.md
results/system/daily_dag_baseline.json
```

这里不要改行为。

先把真实系统画出来。

### Codex 必须特别确认

现在已经知道：

```text
Factor 20260904 = VERIFIED / ACTIVE / READY
LOW/W80 = OPEN
Top100 = unavailable
Store official = 20260903
Dashboard = 20260903
```

要准确查出：

> Top100 为什么没有自动从 Factor 后面继续执行？

必须定位成明确的工程原因，例如：

```text
NO_TRIGGER
ORCHESTRATOR_MISSING
COMMAND_FAILURE
INPUT_CONTRACT_FAILURE
AUTHORIZATION_GATE
SCHEDULER_ORDERING
IDEMPOTENCY_CONFLICT
```

不能只写：

```text
Top100 missing
```

---

# 四、Phase 1：建立统一 DAG 状态模型

这是这轮最重要的基础设施。

每个节点统一只有以下状态：

```text
NOT_STARTED
READY
RUNNING
SUCCEEDED
PARTIAL
BLOCKED
FAILED
SKIPPED
STALE
```

同时增加：

```text
trigger_reason
blocking_reason
retryable
upstream_refs
output_refs
started_at
finished_at
attempt
```

每日生成：

```text
results/operations/daily_production/CN/YYYYMMDD/
    dag-status.v1.json
```

示意：

```json
{
  "trade_date": "20260904",
  "nodes": {
    "market": {
      "status": "SUCCEEDED"
    },
    "factor": {
      "status": "SUCCEEDED"
    },
    "observations": {
      "status": "SUCCEEDED"
    },
    "top100": {
      "status": "BLOCKED",
      "blocking_reason": "MISSING_PRODUCTION_TRIGGER"
    }
  }
}
```

这样第二天不需要 Morning Strategy 才发现 Top100 没有。

---

# 五、Phase 2：建立 dependency resolver

不要把流水线简单写成：

```python
run_market()
run_factor()
run_top100()
...
```

而是声明 dependency graph：

```text
Market
 ├── PIT
 └── Calendar
       ↓
Factor
       ↓
Observation
       ↓
Top100
       ↓
Theme
       ↓
Decision
```

节点只有在：

```text
all(required_upstream.status == SUCCEEDED)
```

时进入 READY。

例如：

```text
Top100 requires:
- Factor ACTIVE as_of trade_date
- LOW observation trade_date
- W80 observation trade_date
- Market/PIT SHA matching Factor bindings
```

如果 Factor 有，但 observations 不匹配：

```text
Top100 = BLOCKED
reason = OBSERVATION_DATE_MISMATCH
```

不能静默跳过。

---

# 六、Phase 3：先彻底修复 Factor → Top100

这是当前第一个实际断点。

## 3.1 明确 Top100 production contract

Codex 检查并统一 Top100 输入：

```text
trade_date
factor_generation_id
factor_pointer_sha
LOW observation ref
W80 observation ref
Market/PIT refs
strategy policy
```

输出：

```text
results/intelligence/research_pool/
  aggressive_tech_manufacturing/
  YYYY-MM-DD/
    top100.parquet
    manifest.json
```

manifest 必须包含：

```text
trade_date
generated_at
factor_generation_id
factor_pointer_sha
low_observation_sha
w80_observation_sha
market_pointer_sha
pit_pointer_sha
policy_sha
row_count
top100_sha
```

---

## 3.2 production publish 必须 idempotent

重复运行：

```bash
quant-investor research pool-publish ...
```

如果所有 input SHA 相同：

```text
ALREADY_SUCCEEDED
```

不能：

```text
overwrite
```

如果同日产物已存在但 input SHA 不同：

```text
CONFLICT
```

fail closed。

---

## 3.3 自动触发

当：

```text
Factor = SUCCEEDED
LOW = SUCCEEDED
W80 = SUCCEEDED
```

Top100 自动 READY。

Orchestrator 自动调用。

不再依赖另一个 automation 人工安排。

---

# 七、Phase 4：Top100 → Theme / Industry / Exposure

这里继续把目前的“研究组件”变成日度生产节点。

固定：

```text
Theme primary = DC
TDX = registered fallback only
```

禁止重新引入双源投票。

输出必须区分：

```text
membership
economic_exposure
confidence
source_evidence
```

尤其你一直强调的：

```text
002463.SZ 沪电股份
002384.SZ 东山精密
```

必须出现在 PCB / AI hardware 专项 evidence 中。

Theme 节点只有两种可接受完成状态：

```text
SUCCEEDED
PARTIAL_WITH_EXPLICIT_MISSING
```

禁止：

```text
silent incomplete
```

---

# 八、Phase 5：Fundamental / Macro 采用“可降级闭环”

这里不要重犯之前 Fundamental lag 太严格的问题。

Fundamental 本身是低频数据：

```text
季度财报
公告
盈利预测
行业经营数据
```

因此日度 DAG 不要求：

```text
fundamental_as_of == trade_date
```

要求：

```text
point_in_time valid
known_at <= decision_time
not superseded incorrectly
```

建立：

```text
FRESH
ACCEPTABLE_LAG
STALE_WARNING
MISSING
```

Fundamental 只在：

```text
MISSING critical evidence
```

时阻塞 Decision。

普通滞后只 warning。

这符合你之前已经确定的原则。

---

# 九、Phase 6：Decision Compile 标准化

Decision 不应该直接产出交易。

它产出：

```text
research decision state
```

例如：

```text
ADD_CANDIDATE
HOLD
REDUCE_REVIEW
EXIT_REVIEW
WATCH
REJECT
```

每个 Decision 必须绑定：

```text
Factor
Top100
Theme
Industry
Exposure
Fundamental
Macro
Portfolio state
```

输出：

```text
results/intelligence/decision/
    aggressive_tech_manufacturing/
    YYYYMMDD/
        decision.v2.json
```

每个 symbol 要写：

```json
{
  "symbol": "002463.SZ",
  "decision": "HOLD",
  "evidence_refs": [],
  "blockers": [],
  "confidence": "MEDIUM"
}
```

---

# 十、Phase 7：修复 Store 日度连续性

这是当前第二个真正的大缺口。

你现在：

```text
Factor = 20260904
Store = 20260903
Dashboard = 20260903
```

这是不能长期存在的。

Store 每个交易日必须产生一个正式日终 state：

```text
T-1 holdings
+ T trade/fill changes
+ corporate actions
+ T strict close
= T holdings / cash / NAV
```

即使当天没有交易，也应该产生：

```text
NO_POSITION_CHANGE
```

的新日终状态。

不能因为：

```text
nothing happened
```

就不生成 state。

否则 holdings continuity 永远断。

---

## Store 必须新增 daily close-forward contract

逻辑：

```text
previous_active_closure
        ↓
check fills
check corporate action
check cash movement
        ↓
apply strict close valuation
        ↓
new active closure
```

无交易时：

```text
holdings identical
cash identical
cost basis identical
valuation updated
```

这样 9 月 4 日应该存在正式：

```text
Store 20260904
```

---

# 十一、Phase 8：Corporate Action reconciliation

紫金矿业当前 trailing threshold 不可执行，原因已经暴露：

```text
adj_factor changed
corporate action reconciliation missing
```

需要增加专门节点：

```text
CORPORATE_ACTION_RECON
```

它负责识别：

```text
split
dividend
rights
bonus issue
share conversion
adjustment factor change
```

然后明确：

```text
cost_basis_adjustment
shares_adjustment
threshold_anchor_adjustment
```

在这个节点没闭合之前：

```text
moving threshold = NON_EXECUTABLE
```

这是对的。

不要删除这个保护。

---

# 十二、Phase 9：Dashboard 彻底变成 DAG 最末端

Dashboard 不能拥有独立事实。

Dashboard 只读：

```text
Store
Factor
Top100
Theme
Decision
```

其交易日必须等于：

```text
latest completed daily DAG trade_date
```

例如：

```text
20260904
```

如果 Dashboard 仍是 20260903：

```text
DASHBOARD_STALE
```

系统应该在当天生产链直接失败，而不是第二天 Morning Review 才报告。

---

# 十三、Phase 10：Morning Strategy 变成纯 Consumer

这是非常重要的架构调整。

Morning Strategy 不再承担：

```text
发现昨天有没有跑成功
补 Factor
补 Top100
修 Store
补 Theme
```

Morning Strategy 只负责：

```text
consume last completed EOD DAG
+
same-day quote
+
owner thresholds
=
morning research review
```

因此 T+1 09:45 的 prerequisite 简化为：

```text
previous_trade_date daily DAG == SUCCEEDED
```

如果不是：

```text
MORNING_UPSTREAM_DAG_INCOMPLETE
```

直接显示断点。

---

# 十四、Phase 11：引入 catch-up engine

这是解决你过去 automation 经常断、授权链断裂、某次没运行后一直坏下去的关键。

每次日度 production 启动时，先检查：

```text
last_completed_trade_date
expected_trade_date
```

例如今天发现：

```text
last complete = 20260901
expected = 20260904
```

自动生成：

```text
catch-up:
20260902
20260903
20260904
```

严格按日期顺序重放。

不能直接只跑 9 月 4 日。

否则 prospective history 会断层。

---

# 十五、Phase 12：Automation 简化

最终建议只保留三类 automation。

### A. EOD producer

```text
15:30 or later
daily-production
```

负责：

```text
Market
Factor
Top100
Theme
Fundamental
Macro
Decision
Store
Dashboard
```

---

### B. Night fallback

21:00：

```text
check daily DAG
```

如果：

```text
SUCCEEDED
```

直接：

```text
NO_ACTION
```

如果：

```text
retryable BLOCKED/FAILED
```

从断点继续。

---

### C. Morning review

09:45：

```text
quote-first
+
previous completed DAG
```

绝不运行 maintenance。

---

# 十六、Phase 13：失败分类

Codex 要建立固定 error taxonomy。

至少：

```text
INPUT_MISSING
INPUT_STALE
POINTER_MISMATCH
SHA_MISMATCH
UPSTREAM_INCOMPLETE
PROVIDER_UNAVAILABLE
AUTHORIZATION_BLOCKED
SCHEMA_MISMATCH
IDEMPOTENCY_CONFLICT
DATE_MISMATCH
CALENDAR_MISMATCH
CORPORATE_ACTION_UNRECONCILED
POLICY_BLOCKED
```

每个 blocker 包含：

```text
retryable=true/false
owner_action_required=true/false
recommended_next_node
```

例如现在应该变成：

```json
{
  "node": "top100",
  "status": "BLOCKED",
  "reason": "UPSTREAM_PRODUCT_NOT_PUBLISHED",
  "retryable": true,
  "owner_action_required": false
}
```

这样系统就知道：

> 自动继续执行。

而不是问你授权。

---

# 十七、Phase 14：Prospective Evidence Ledger

这是我认为非常重要的一步。

你真正想知道：

> Factor 到底有效不有效？

那以后必须区分：

```text
historically recomputed evidence
vs
prospectively available evidence
```

每天收盘产生：

```text
results/prospective/YYYYMMDD/evidence-ledger.v1.json
```

记录：

```text
Factor generated_at
Observation registered_at
Top100 generated_at
Decision generated_at
Store closed_at
```

如果像 9 月 4 日 Factor observation 是 9 月 6 日才登记：

```text
prospective=false
backfilled=true
```

以后做因子评价时：

```text
only prospective=true
```

才能算真正 OOS。

---

# 十八、Phase 15：测试

这轮测试重点不是单模块。

必须增加 **full DAG integration tests**。

至少覆盖：

### Case 1

正常交易日：

```text
all nodes success
```

---

### Case 2

Factor 成功、Top100 未运行

期望：

```text
orchestrator automatically runs Top100
```

---

### Case 3

Top100 crash

第二次运行：

```text
resume from Top100
```

不能重跑 Factor。

---

### Case 4

Store 前一天缺失

```text
catch-up store closure
```

---

### Case 5

无交易

仍生成新：

```text
daily Store state
```

---

### Case 6

corporate action unresolved

```text
Store valuation may complete
moving threshold NON_EXECUTABLE
```

---

### Case 7

Fundamental lag

```text
warning
Decision continues
```

---

### Case 8

Market/PIT SHA mismatch

```text
fail closed
```

---

### Case 9

重复运行同一天

```text
idempotent
```

---

### Case 10

周末/节假日

```text
NON_TRADING_DAY_NO_ACTION
```

不产生伪 financial state。

---

# 十九、上线顺序

我建议 Codex 严格按这个顺序：

```text
P0-A DAG inventory
↓
P0-B DAG status schema
↓
P0-C orchestrator skeleton
↓
P0-D Factor → Observation → Top100
↓
P0-E Top100 → Theme → Decision
↓
P0-F Store daily continuity
↓
P0-G Corporate action reconciliation
↓
P0-H Dashboard
↓
P0-I catch-up / resume
↓
P0-J prospective ledger
↓
P0-K integration tests
↓
P0-L automation migration
```

先不要同时大改所有模块。

---

# 二十、明确禁止事项

这轮 Codex 不要：

```text
增加新 Factor
调整 Factor 权重
优化选股逻辑
增加 Theme source
改交易策略
增加真实交易
连接 broker
修改 owner stop
扩大 Paper 权限
修改持仓
为了“跑通”跳过 SHA/PIT/calendar 检查
```

P0 的任务就是：

> **把已经存在的系统组件可靠连接起来。**

---

# 二十一、最终 Done Definition

只有同时满足以下条件，这轮 P0 才算完成：

```text
连续 5 个交易日
```

每天自动产生：

```text
Market/PIT READY
Factor READY
LOW/W80 READY
Top100 READY
Theme READY
Decision READY
Store READY
Dashboard READY
```

并且：

```text
T+1 morning strategy
无需补前一天 production
```

同时满足：

```text
no manual intervention
no stale pointer
no fake backfill
no missing Store date
no missing Top100
no silent failure
```

---

# 二十二、给 Codex 的执行指令

你可以直接把下面这一段作为总指令：

> **Implement P0 Daily Evidence DAG Closure for myQuant. The priority is operational continuity, not model enhancement. Build a deterministic, idempotent, evidence-bound daily production DAG from Market/PIT → Factor → LOW/W80 observations → Top100 → Theme/Industry/Exposure → Fundamental/Macro → Decision → Store → Dashboard. Add explicit node states, dependency resolution, resumable execution, catch-up of missed trade dates, immutable input/output refs and failure taxonomy. Fix the currently observed Factor→Top100 break and Store/Dashboard date continuity first. Morning Strategy must become a pure consumer of the previous completed daily DAG plus same-day quotes; it must not perform maintenance. Preserve all existing safety boundaries: no broker, orders, real trades, actual holdings mutation, unauthorized System/Mainline mutation, or weakened PIT/SHA/calendar validation. Add prospective evidence timestamps so backfilled observations cannot be misclassified as real-time OOS evidence. Validate with full-DAG integration tests and demonstrate at least a synthetic five-trading-day uninterrupted run.**

从你现在系统的状态看，我认为这是**当前优先级最高的一次架构改造**。完成以后，myQuant 才会从“很多功能已经存在的研究系统”真正进入“每天稳定生产投资证据的数据系统”。后面无论做 Factor IC、decay、hit rate、组合 attribution、Theme alpha，才有可信的数据基础。

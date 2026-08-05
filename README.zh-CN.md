<div align="center">

<img src="assets/logo.svg" alt="Quant-Investor" width="520"/>

**一套 fail-closed 的 A 股研究与组合决策系统。**

*确定性控制，AI 只做建议 —— 没有可复现的证据，任何东西都到不了决策。*

[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-17.0.0-FF6B35?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

[English](README.md) · **简体中文**

[要解决的问题](#要解决的问题) · [设计立场](#设计立场) · [架构](#架构) · [因子治理](#因子治理) · [快速开始](#快速开始) · [当前状态](#当前状态)

</div>

---

## 要解决的问题

多数量化系统不会大声失败。它们的失败方式是悄悄给出一个**看起来正是你想要的数字**。Quant-Investor 是围绕其中四种具体情形构建的。

**扛不住自己窗口的证据。** 一个用八个月数据挖出来的因子，可以展示 0.05 的 rank IC 和八门全过的评审，然后在五年尺度上衰减到四分之一。日频 rank IC 对 30 天前瞻收益是高度重叠的，朴素 t 检验会把显著性放大约 √30 倍 —— 足以让一个假象在 p < 1e-4 的水平上看起来像 alpha。

**会自我替换的数据。** 缺失的 Parquet 分区变成 CSV 回退、陈旧快照或推断值。链路照样返回结果，而输出里没有任何一处说明它究竟是由哪些字节算出来的。

**会漂移的定义。** 一个定义为「对当前生产组合取残差」的因子，每当生产集变化，它的含义就变了。它记录在案的证据描述的是一个已经不存在的基线，而且无法仅凭市场数据复现。

**握有决策权的语言模型。** 一个能调整权重的模型，就能幻觉出一个权重。一旦模型输出混进控制路径，「为什么是这个仓位」就不再有确定性的答案。

## 设计立场

针对上述四点的回答，逐一对应。

**缺失权威等于结果不可用，而不是等于可以去创造一个。** 每个公开面只解析唯一一个指针。指针缺失，答案就是 `V17_MAINLINE_UNINITIALIZED`，且不写入任何东西；指针无效，答案就是 `V17_MAINLINE_BLOCKED:<blocker>` —— 没有回退，不扫描目录找最近一次运行，不自举。主线回测不是降级处理，而是直接拒绝：`V17_BACKTEST_UNAVAILABLE`。

**权威是内容寻址的，且只能通过 CAS 推进。** 结果沿固定链路流转，每一环都绑定上一环的字节：

```text
strict CN Parquet + PIT membership
  -> Quant 截面初筛
  -> 同池 Quant + Fundamental 证据
  -> Macro / regime 上下文
  -> 确定性风险与组合闸门
  -> 不可变主线运行 (immutable mainline run)
  -> 受治理的活跃指针 (governed active pointer)
  -> 只读公开运行 (read-only public run)
```

| Artifact | 角色 |
|---|---|
| `myquant.v17.v4.mainline-run.v1` | 不可变的闭合运行 |
| `myquant.v17.v4.mainline-active-pointer.v1` | 某策略唯一的公开权威 |
| `myquant.v17.v4.mainline-public-run.v1` | 只读的公开投影 |

激活是一次针对预期前值的 compare-and-swap，随后做精确回读。**部署代码不会激活任何东西。**

**因子必须能仅凭市场数据加自身 spec 复现。** 依赖可变 registry 状态的定义会在生产闸门被拒 —— 不是因为它分数低，而是因为它的值无法复现。如果确实需要这类定义，它的基线会被 **pin 住**：显式命名并用内容哈希绑定，算术保持一致，但输入不再移动。

**模型负责叙述，从不负责决定。** `NarratorAgent` 是只读角色，不能改动候选标的、风险限额或权重。可选的 LLM review 层只输出建议性提示；没有 key 时降级为 handoff，确定性控制链不受影响。所有权重都只由 `PortfolioConstructor` 生成。

## 架构

一条确定性 DAG，三个证据分支，一条控制链：

```text
snapshot -> DeterministicFunnel -> {quant, fundamental, macro}
  -> Bayesian selection -> RiskGuard -> ICCoordinator
  -> PortfolioConstructor -> NarratorAgent
```

这条链里的**不对称是刻意的**：

- **只有 `quant` 和 `fundamental` 进入 Bayesian likelihood。** `macro` 是 prior 和上下文 —— 它塑造风险预算，但不对个股投票。
- **Fundamental 证据必须自证溯源。** 只有 canonical generation pointer 校验通过、且 lineage 绑定到 `tushare_primary` 时，likelihood 才被采纳。否则分支仍然输出诊断，但其 likelihood 被严格中性化为 `0.50` —— 在场，且可证明不携带信息。
- **Regime 模型只能降低风险。** Markov 的输出以 `min(baseline, suggested)` 作用于目标敞口和单标的最大权重；换手上限只能收紧、不能放松。一个 regime 信号无法把组合说服成更大的仓位。
- **分支不可开关。** 不存在 `enable_quant` 这类开关。带这类字段的请求会失败，而不是被静默忽略。
- **备注分桶且承重。** `investment_risks`、`coverage_notes`、`diagnostic_notes` 按合约彼此分离，且后两者被禁止进入配权。一次数据供应商故障不可能渗透成一个仓位大小。

研究与生产是**靠构造隔离的，不是靠约定**。Shadow 通道（`quant-investor-v17-v4 run-forward`）从内容寻址的密封请求中累积仅面向未来的证据。它的产出永远不授予主线权威，也永远不能作为公开运行返回。

## 因子治理

因子不会因为「有意思」而进入生产。`FactorGovernanceProtocol v4` 的目标集是 **10 个**健康因子；5 到 9 个属于 underfilled 状态，挖掘转入加速的 report-only 模式；低于 5 个则系统处于 `no_new_risk`。

一个因子只有在下列条件**同时**成立时才算健康：

| 要求 | 门槛 |
|---|---|
| Gates 1–8 | 数据安全、覆盖率与稳定性、IC/RankIC、分组收益、成本与换手、中性化与暴露、样本外稳健性、组合增量 |
| 成熟度 | 12 个真实月末 RankIC 交易日，或 8 个非重叠的 30 交易日 cohort，且对齐 strict-Parquet 日历 |
| 多重性校正 | **按 family** 做 Benjamini-Hochberg，`q <= 0.10` |
| 身份 | runtime contract 与 replay hash 均已验证，name 与 slot 唯一 |
| 健康度 | fresh、非 stale、非 data-blocked |
| 配权 | 单因子 ≤ 20%，单 family ≤ 35% |

最后两条上限的相互作用值得单独点出：恰好 5 个因子时，20% 的单因子上限迫使每个因子都正好是 20%，于是同一 family 里放两个就是 40% —— 超过 35% 的 family 上限。**因此五因子集在数学上必须由五个不同 family 构成。** 集中度由算术封顶，而不是由判断封顶。

日历和统计同样重要。交易日来自活跃快照**实际观测到**的结果，稀薄交易日会被排除 —— 在 1,796 个观测到的 CN 交易日中，只有 858 个带有真实截面。在其余交易日上切出的成熟度，会满足算术、但什么也不依托。

## 快速开始

```bash
uv sync
cp .env.example .env
```

本地验证保持离线。只填写获得明确授权的工作流所需要的内容。

以下三条命令解析同一个活跃指针，返回同一条权威链：

```bash
quant-investor research run --workspace-root /absolute/path/to/myQuant --strategy-id <strategy-id>
quant-investor market analyze --workspace-root /absolute/path/to/myQuant --strategy-id <strategy-id>
quant-investor market run --workspace-root /absolute/path/to/myQuant --strategy-id <strategy-id>
```

数据维护是独立且显式的工作流：

```bash
quant-investor market maintain --market CN --staged
quant-investor market storage-validate --market CN
```

Shadow 研究通道，及其其余子命令（`verify`、`status`、`factor-set-status`、`deep-v3-compile`、`forward-shadow-readiness`、`forward-shadow-status`）：

```bash
quant-investor-v17-v4 run-forward --workspace-root /absolute/path/to/myQuant --request-path <request.json> --request-sha256 <sha256>
```

## 当前状态

运行时仅支持 CN，且**不具备任何 broker、order、execution 或 trade 权限**。有两件事仍然敞开，且都是设计的结果，而非未完成的收尾。

**Factor Governance v4 尚未 ready。** 在完整开市日历上实测，5 个候选中有 3 个通过 `q <= 0.10`。未通过的那两个，恰好就是登记证据来自短近窗口的那两个 —— 它们扛不住对 30 天重叠前瞻收益的修正。这道门正在发挥作用，所以 `factor_governance_ready` 保持 `false`，直到因子池能够支撑它。

**主线没有发布器。** `quant_investor/v17_mainline/` 只有读侧，其模块 docstring 自己写明了这一点。在决策输出的生产者出现之前，公开面返回 `V17_MAINLINE_UNINITIALIZED` —— 这是正确答案，不是 bug。

## 项目结构

```text
quant_investor/
  cli/                       公开命令路由
  market/                    CN 维护与市场工作流
  pipeline/                  研究与组合链路
  v17_mainline/              活跃指针权威与公开运行读取
  v17_v4_contract/           V17 v4 schema 与校验器
  v17_v4_runtime/            仅研究用的 Shadow 运行时服务
  factors/                   因子治理
portfolio_dashboard/         只读 dashboard 合约
web/                         本地研究工作台 API
docs/                        架构与运维手册
results/v17_mainline/        受治理的活跃主线结果
results/v17_v4_shadow/       仅研究用的前向证据
```

## 开发

Python 3.13+。优先运行范围最窄的相关测试；需要做大范围 staged-upgrade 时：

```bash
PYTHON=./.venv/bin/python scripts/staged_upgrade_quality_gate.sh
```

除非任务明确授权，本地验证期间不要调用实盘数据、LLM、broker、order、execution 或 trade 接口。

## 文档

- [文档索引](docs/README.md)
- [V17 v4 主线合约](docs/architecture/v17_v4_production_research_contract.md)
- [研究链路与协议](docs/architecture/research_pipeline_and_protocols.md)
- [Factor Governance v4](docs/factor_governance_v4.md)
- [V17 v4 运维](docs/runbooks/v17_v4_operations.md)
- [入口与版本](docs/architecture/entrypoints_and_versioning.md)
- [前向证据运行时](docs/architecture/v17_v4_forward_evidence_runtime.md)
- [模块地图](docs/modules/module_map.md)
- [Agent 指南](AGENTS.md)

## 许可证

[MIT](LICENSE) © 2024 alpha-mw

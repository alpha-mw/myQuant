# v17 Shadow Offline Operations

本 runbook 只覆盖 v17 latest shadow 的本地重放。包版本是 `17.0.0`，但
`market analyze`、`market run` 与 production/default 仍固定走 v15。这里的命令
不会调用 provider、在线 LLM、券商、订单或交易接口，所有输出都必须保持
`authority=false`。

> 当前实现状态：本文后续 lifecycle 命令仍描述尚未获准发布的 v1 wire。
> 在独立 v2 action matrix、ledger、source binding 与 latest pointer 完成并通过
> Architect/Critic 复审前，不得把这些命令接入 main、skill、schedule 或 Dashboard，
> 也不得用它们创建、推进、修复任何正式 v17 shadow run。

## 0. Release 依赖门

安装权威固定为 `pyproject.toml + uv.lock + frozen uv export`。`requirements.txt`
只是兼容外部工具的 runtime＋dev 聚合，不代表默认运行依赖，也不能覆盖 lock。

已批准计划的强制 release gate 只有一条：在目标 CPython 与平台的全新环境中，
`uv sync --locked --all-extras --offline` 必须原样成功。缓存不足时立即停止。

`scripts/v17_offline_dependency_evidence.py` 只生成第三方依赖环境证据，不能证明本地
`quant-investor` wheel/sdist 来源，也不能单独封存 Phase 0。v2 的权威调用方只有下文
单一 session runner；不得把该脚本的独立 CLI、旧 `--native-uv-sync-status` 断言、
pip HTTP cache 或 materialization 路径作为 release gate。session 会把真实 native
command-v2 log、frozen export、目标 marker、精确 installed name/version、pip 缺失、
lock artifact 证据及 source/toolchain/protected-root binding 交叉验证后，再生成
`20_native_dependency.json`。不完整 wheelhouse 可以作为非权威观察保留，但绝不能
替代成功的 native offline sync。

2026-07-23 经 owner 选择的单次受控预取已结束；随后在全新 CPython 3.13 环境中，
原样 `uv sync --locked --all-extras --offline` 已成功安装91个锁定第三方包和本地
17.0.0 项目。该结果只关闭依赖环境门，不授予 runtime、CLI、release 或交易权限。
每个待验收 source state 仍须在从未使用过的环境中重跑同一 native gate，并由外部
receipt 绑定其 porcelain、binary diff、全部 untracked 文件和 package payload。
在该 source-state receipt、Phase 0 复审和后续 Architect→Critic 放行前，仍然
**禁止 v2 runtime/CLI 接线、main/skill/schedule 切换及 v16 results purge**。

### 0.1 当前权威：Phase 0 v2 单会话证据

Architect 与后续 Critic 已批准单一、不可续跑的仓库内 evidence session runner；
外部七日志 harness 与 command receipt v1 已废止。runner 必须由冻结的
CPython 3.13.7 以 `-I -S -B` 启动，只接受固定输入路径，不接受任意 argv、env、
role、resume 或 repair 参数。

每次运行使用两个全新、空、owner-owned `0700` 目录：

- bundle root：只保存 `00_session.json`、十个固定 gate、`60_gate_manifest.json`
  与最终 `70_evidence_index.json`＋sidecar；
- work root：只保存 fresh native venv、package build/install venv、sdist/wheel、
  alternate Git index 与运行临时状态。

两者必须在仓库和四个受保护 v16 路径之外、互不嵌套，并在成功或失败后原样保留。
只有 schema/semantic/readback 全部通过的 `70_evidence_index.json` 与 sidecar 才表示
Phase 0 `SEALED`；中途文件或 `99_unpublished_failure.json` 都不表示成功。

固定执行顺序为：

```text
00 session
10 native uv sync log
20 native dependency v2
30 v2 tests
31 staged + recommended core
32 full suite
33 mypy
34 Black
35 diff check
40 package parity v2
50 hash freeze v2
60 gate manifest v2
70 evidence index v2 + sidecar
```

native gate 仍只有一条原样命令：
`uv sync --python <frozen-base-python> --locked --all-extras --offline`。
native 与 package build 环境必须证明 pip 不存在；pip 25.2 只允许由 package install
环境的 `<install-python> -I -m ensurepip --upgrade` 引入，并绑定 CPython 3.13.7
自带 wheel 的固定 name、size 与 SHA。任何 native `ensurepip`、额外 pip wrapper、
非锁定 distribution 或不同 Python/uv/cache 身份都立即停止。

full-suite skip policy 由独立、固定、离线的 baseline producer 预先冻结。验收口径是
结构化 `(path,line,reason,count)` 行的 `count` 总和严格等于42，不是42行；最终
full suite 必须逐行完全一致，且 fail/error/xfail/xpass 全为0。

候选是 linked worktree。diff checker 必须通过
`git rev-parse --git-path index` 解析真实 index，只在 work root 的
`GIT_INDEX_FILE` alternate index 中执行 `read-tree HEAD`、对全部 Phase0 untracked
路径 `git add -N`，再运行固定的 `git diff --check`；不得读写假定的 `.git/index`。

以下限制是证据本身必须携带的明示边界：

```text
PRECOLLECTION_ALLOWED_TREES_ATTESTED_NOT_COMPLETE_MAIN_RUNTIME_FILESET
NO_DIRECT_LAUNCH_REFERENCE_ONLY_NOT_PROCESS_IO_ATTESTED
OFFLINE_FLAGS_ONLY_NOT_OS_EGRESS_ATTESTED
OWNER_DECLARED_PATHS_ONLY_NOT_CONTENT_PROVENANCE
PIP_25_2_PACKAGE_INSTALL_ENV_ONLY_NATIVE_AND_BUILD_ENV_PIP_ABSENT
```

main-bound full-suite 使用既有、非 hermetic 的环境，不是 fresh release env。
pre-collection policy 只绑定获准的 runtime/plugin trees，不声称覆盖完整 main runtime
fileset。main distribution inventory 中四个已知 invalid dist-info stub 必须继续显式分类
和披露，不能静默归一化为有效 distribution。

### 0.2 经复审的 dual-runtime 门

Phase 0 使用六个互不混用的解释器/环境角色。main-bound lane 是为 Factor
v4.3/v4.4 固定 lexical/runtime 语义批准的正式兼容性门，但它不是 fresh/hermetic
环境，也不能替代任何 fresh release 证据。

| 角色 | 固定身份 | 允许职责 | pip 生命周期 |
|---|---|---|---|
| base session parent | 冻结 CPython 3.13.7，`-I -S -B` | 编排单次不可续跑 session、校验 policy/evidence、启动固定子角色 | pip 不得可导入或已加载，`sys.path` 不得出现 site 路径；Homebrew 安装中超出该运行时作用域的全局 pip 文件允许客观存在；不得 install/sync |
| base skip parent | 同一冻结 CPython 3.13.7，`-I -S -B` | 生成 immutable skip baseline，并启动受 policy 控制的 main-bound collection | 同一 isolated/no-site pip 可见性规则；不得 install/sync |
| fresh native | 从未使用过、由原样 locked offline sync 创建的 venv | native dependency、v2 tests、recommended core、mypy、Black | pip 必须不存在；不得 `ensurepip` |
| main full-suite | 预先存在且逐字节绑定的 main venv；installed `quant-investor` metadata 保持 main-bound，candidate import 只来自 candidate root | 保留 Factor v4.3/v4.4 历史运行时语义的 compatibility/full-suite gate | 不得 install、sync 或 `ensurepip`，不得修改既有环境 |
| package build | 全新 owner-private venv，只含离线安装的精确五包 Hatchling backend | 生成 canonical sdist/wheel，并复核 Hatch selector | `pip` distribution 与 wrapper 必须不存在 |
| package install | 独立全新 owner-private venv，只含 built project 与冻结 install tooling | 以 `--no-deps --no-index --no-compile` 安装同一 wheel，并证明 source/sdist/wheel/installed parity | 唯一允许 `ensurepip` 的角色；pip 必须是冻结 bundled wheel 提供的 25.2 |

main-bound wrapper 必须先重建已哈希绑定的 main venv 正常启动语义，包括
`site.venv` 与 `execsitecustomize`，然后只把 repository import root 替换为 candidate
root。所有 candidate `quant_investor` module 必须来自 candidate worktree；installed
distribution metadata 与其余 runtime 仍须明确标记为 main-bound-not-fresh。

pytest 自动 plugin discovery 必须关闭。精确 pytest argv fragment 是
`-p pytest_cov -p asyncio -p anyio`；pytest 9 使用标准 entry-point resolution，并
保留这三个 registration name，不得用自定义 bootstrap 改名或替换 module。wrapper
还必须校验 plugin-manager 顺序与 hook trace：

1. `pytest_cov` → `pytest_cov.plugin`（`pytest-cov==7.1.0`）
2. `asyncio` → `pytest_asyncio.plugin`（`pytest-asyncio==1.3.0`）
3. `anyio` → `anyio.pytest_plugin`（`anyio==4.13.0`）

Phase 0 只有同时满足以下两项才可 `SEALED`：

1. fresh CPython 3.13 locked offline sync，以及 native dependency、v2、
   recommended-core、mypy、Black、package build/install parity 全部门禁通过；
2. policy-bound main-interpreter compatibility/full-suite gate 通过。

两者互不替代。fresh sync/native gate 未通过时，禁止用 main 环境补位；main
wrapper、冻结 policy、plugin topology 或 full suite 未通过时，禁止用 fresh 结果忽略
Factor historical runtime；package build/install 漂移同样立即停止。

`OFFLINE_FLAGS_ONLY_NOT_OS_EGRESS_ATTESTED` 仍是规范限制。证据可以声明
`offline_policy_enforced=true`，但必须同时声明 `kernel_egress_attested=false` 与
`network_unreachability_proven=false`。旧记录若仍含
`network_actions_performed=false`，它也只代表 runner-declared intent。固定 command
plan 没有显式请求已知 live API，但并未审计所有 test code 的 socket/egress 行为，
因此不能证明没有 network API、网络不可达或没有 socket。如果后续验收要求
OS-level egress attestation，必须停止，不得把现有证据升级解释为已经满足。

每个文件使用 pinned parent dirfd、`O_NOFOLLOW|O_EXCL`、0600 staged inode、
`fsync`、hard-link-to-absent、逐字节 readback 的 exact-once 发布。只允许在成功且
已证明 staged/final 为同一 inode 后 unlink staged link；失败、orphan 或已发布证据
不得覆盖、删除、修复或复用。

在源码与 schemas 完全冻结、ignored bytecode/cache 已从 package source tree 清除后，
先用两个全新的外部目录生成 skip baseline，再用另外两个全新的外部目录运行完整
session。`classification.json`、`skip_baseline.json` 与 frozen export 必须位于仓库外、
owner-owned `0700` 父目录中，并且自身为单链接 `0600` 文件：

```bash
BASE_PYTHON=/opt/homebrew/Cellar/python@3.13/3.13.7/Frameworks/Python.framework/Versions/3.13/bin/python3.13

"$BASE_PYTHON" -I -S -B scripts/v17_phase0_skip_baseline.py \
  --repo-root /private/tmp/myquant-v17-neutral-baseline-20260722 \
  --bundle-root /owner/private/new-skip-bundle \
  --work-root /owner/private/new-skip-work \
  --output-json /owner/private/new-skip-bundle/skip_baseline.json

"$BASE_PYTHON" -I -S -B scripts/v17_phase0_evidence_session.py \
  --repo-root /private/tmp/myquant-v17-neutral-baseline-20260722 \
  --classification-manifest /owner/private/inputs/classification.json \
  --skip-baseline /owner/private/new-skip-bundle/skip_baseline.json \
  --frozen-export /owner/private/inputs/all-extras-nohash.txt \
  --bundle-root /owner/private/new-session-bundle \
  --work-root /owner/private/new-session-work
```

这些命令不支持单 gate、任意 argv/env、resume 或 repair。任一目录曾经存在、任一输入
权限或 inode 不符、或已有同名输出时都必须换用新的路径；不得清理失败 bundle 后重试。

下面直到“1. 先决条件与停止规则”为止的 command receipt v1、external harness 和
package/index v1 描述仅保留作历史记录，不得执行或作为验收依据。

Phase 0 证据索引现在必须使用单一 owner-private `0600` gate manifest 作为必填
输入，而不是一组任意 artifact/log。manifest 必须是 compact sorted JSON 加一个换行，
根 `semantic_sha256` 只排除自身计算；每个 gate 都必须逐项绑定最终 source：
`base_commit`、`source_state_sha256`、`porcelain_sha256`、`binary_diff_sha256`、
`untracked_inventory_sha256`。闭集 role 必须且只能各出现一次：
`native_sync_receipt`、`native_sync_log`、`v2_evidence_tests`、
`recommended_core_tests`、`full_offline_suite`、`mypy`、`black`、`diff_check`、
`package_parity`、`hash_freeze_readback`。缺失、重复、未知、过期 source 或任一 raw
evidence 语义不符时 fail closed；索引只有在全部通过后才可输出 `accepted=true` 和
`status=SEALED`。

`native_sync_receipt` 必须使用 `scripts/v17_offline_dependency_evidence.py` 的真实
canonical receipt，不能手写简化 JSON。索引会 exact-validate 全部 top-level 和 nested
shape，包括 scope、inputs、runtime uv/target venv/Python/platform、expected/installed
reconciliation、artifact records/counts、missing/invalid、全部 acceptance/failure flags
及 `semantic_sha256`；缺字段或多字段都 fail closed。它还要求
`native_dependency_environment_accepted=true`、
`installed_reconciliation.exact_match=true`、`native_uv_sync_status.status=PASSED`、
`operator_asserted_passed=true`、offline/no-network、CPython 3.13，并按 producer 的
原始 HEAD/diff 命令重算 source receipt，再与 index 的 base/head、porcelain 和
untracked bytes 交叉绑定。native log 的 `uv sync`
命令使用独立 base CPython 3.13 解释器；该 `--python` 路径可以不同，但 resolve 后
必须等于 receipt 观察到的 target Python。`UV_PROJECT_ENVIRONMENT` 必须等于 receipt
的 target venv，receipt target Python 必须是该 fresh env 的 `bin/python`，后续所有
Python 命令都必须共用这个 fresh env Python。native log 的 uv argv[0] resolve 后必须
等于 `runtime.uv.path`，tool version 和 cache path 必须分别等于 receipt 的 uv version
和 `inputs.uv_cache.path`；索引会重新读取实际 uv executable，校验 path、byte SHA、
size、mode 和 executable bit。pytest/mypy/Black 的 tool version 还必须等于
`installed_reconciliation.installed` 中同名 distribution 的版本，并绑定同一
CPython patch version。

所有 log role 都必须是单文件 command receipt envelope，第一行严格为
`MYQUANT_PHASE0_COMMAND_RECEIPT=<compact canonical JSON>`，余下 bytes 才是 combined
output。receipt 绑定 role、repo cwd、当前 source binding、命令 argv/env/exit/tool
version、output SHA/size 和自身 semantic SHA；所有 command receipt/claims 的
count、size、exit-code 都必须是 JSON integer，`false` 不得冒充 0；裸的
success-shaped log 无效。最终真实 build report 必须先通过仓库 stdlib-only closed
Draft 2020-12 executor 对 index schema 的 preflight 和 instance validation。

仓库内 index 当前只是独立 validator，不是7个 log role 的 command producer。在另行
复审并纳入仓库的 producer 出现前，这7份 log 只能由同一个固定角色、
owner-private external harness 生成；supplemental audit 必须保存该 harness 的原始
bytes SHA-256 和 invocation record。禁止手写 receipt JSON 或使用可接受任意 argv 的
runner。该 producer provenance 不进入 command-receipt v1、不新增第11个 gate，也尚未被
index schema 绑定，因此是明确的 Phase 0 残余边界；如果验收规则要求 repo-owned
producer，立即停止，不得用现有 harness 冒充。

pytest 三个门禁必须解析最终 summary 并与 manifest claims 的
passed/skipped/failed/errors/xfail/xpass 逐项一致；
除 `full_offline_suite` 外不得有 skips。`full_offline_suite` 的 pytest 命令必须带
`-rs`；claims 必须含 `raw_output_sha256` 和结构化 `skip_allowlist`，每条 allowlist
必须精确匹配 raw pytest `SKIPPED [n] path:line: reason` 行，且 count 总和等于最终
skipped 数。即使 skipped 总数不变，只要 path 或 reason 改变也必须 fail closed；
xfail/xpass 必须为 0。`recommended_core_tests` 同时绑定
`scripts/staged_upgrade_quality_gate.sh` 的闭合证据，不新增 role：claims 必须含
`staged_upgrade_exit_code=0`，raw log 必须唯一包含 `staged_upgrade_exit_code=0`，
并包含 `Running staged upgrade focused tests...`、
`Running staged upgrade focused mypy...` 和 mypy success；schema 将
`staged_upgrade_exit_code` 固定为 integer `0`。`package_parity` 输入必须是
`scripts/v17_phase0_package_evidence.py` 生成并封存的完整
`myquant.v17.v2.phase0-package-parity-evidence.v1`，不能只提供 parity helper 的短
JSON。索引会 exact-validate envelope、当前/before/after source binding、
source-binding artifact、CPython 3.13 与 uv binary 三次 readback、uv 0.10.9、pip
25.2、Hatchling backend 的精确5包 map 与排序 inventory、仓库外 fresh build/install
环境，以及13个命令的固定顺序、完整 argv/env/repo cwd、integer exit 0、tool
identity、sanitized environment、stdout/stderr bytes 和 combined SHA。wheel 必须从
绑定的 sdist 构建，再以 `--no-deps --no-index --no-compile` 安装同一 wheel；
sdist/wheel path、SHA、size、四表面 inventory、非 editable provenance、
`quant-investor` `17.0.0` metadata、RECORD counts 和 dist-info hashes 必须交叉一致。
完整 package report 还必须通过
`scripts/schemas/v17_phase0_package_evidence.v1.schema.json` 的 closed stdlib Draft
2020-12 preflight/instance validation；schema `$id` 为独立的
`myquant.v17.v2.phase0-package-parity-evidence.schema.v1`，report `version` 仍是 artifact
ID。index 必须先执行该 checked-in schema，再执行 cross-field semantic validator；
所有 count/size/exit schema 字段拒绝 boolean。

如果 exact-once 输出过程中出现单边 orphan sidecar 或 report，不要自动修复、删除或
复用。改用新的 owner-private 输出文件名重新生成 output pair；旧 orphan 只有在
显式 owner review 后才可处理。

Phase 0 的权限边界仍只到 contract/evidence closure。它不授权 runtime/CLI/schedule/
Dashboard wiring、main/release cutover、非 Phase0 staging、正式 v17 shadow run 创建
或推进、v16 root 清理、provider/在线 LLM/券商/订单/交易调用。

## 1. 先决条件与停止规则

- 在仓库根目录运行，或对每条命令显式传入同一个绝对 `--repo-root`。
- 输入 JSON 必须已经按 `quant_investor.v17.semantic` 生成并验证
  `semantic_sha256`；不要手工修改已封存文件。
- 对每个输入先执行 `shasum -a 256 <file>`，把读回的 64 位摘要逐字填入
  `--expected-*-sha256`。`EMPTY` 仅表示已确认目标不存在，不能代替未知摘要。
- 新 manifest 的 `--expected-manifest-sha256`、新 run 的
  `--expected-ledger-sha256` 可使用 `EMPTY`。latest 不存在时才可对
  `--expected-latest-sha256` 使用 `EMPTY`。
- 任一命令 exit 2、CAS 不一致、source/resource/schema 漂移、真实 canonical
  authority 缺失或输入语义不明时立即停止。不得猜测摘要、补默认值、切换到
  v15/v3/sample/synthetic 输出，也不得把 no-portfolio 当成完整组合结果。

固定写入根只有：

- 私有来源：`data/private/v17_sources/{objects,manifests}`
- shadow 状态：`results/v17_shadow/{runs,models,outcomes,_latest}`

## 2. Owner-only 风险快照

`v17-risk-policy-seal` 只能由 owner mandate 驱动。mandate 缺失、摘要不符、
PIT/时效/shape 非法时不得生成替代快照；命令会 exit 2 且在验证完成前零写入。

```bash
quant-investor market v17-risk-policy-seal \
  --repo-root /absolute/path/to/myQuant \
  --owner-mandate /owner/private/risk-mandate.json \
  --output data/private/v17_sources/objects/risk-policy.json \
  --expected-owner-mandate-sha256 <owner-mandate-byte-sha256> \
  --validation-cutoff <UTC-cutoff>
```

`AVAILABLE` 风险快照必须是 owner mandate 的真实、未过期、PIT 一致结果；
`UNAVAILABLE` 只能导致正式 no-portfolio 语义，不能被解释成全现金或默认风险上限。

## 3. Source maintain

source plan 必须使用 `myquant.v17.source-maintenance-plan.v1`，包含全部固定 role，
并按 `role, source_id` 排序。`AVAILABLE` role 绑定一个本地普通文件及其字节 SHA；
`UNAVAILABLE` role 只携带明确原因。rank 所需 authority 缺失会阻断 prepare。

canonical 指针 role 必须逐字节绑定现有权威文件：`market_pointer` 使用
`data/parquet/cn/_latest.json`，`market_snapshot_manifest` 使用该指针声明的
`manifest_path`；`fundamental_generation_pointer` 使用
`data/parquet/cn/_fundamental_latest.json`，`fundamental_generation_manifest` 使用该
指针声明的 generation manifest。`data/parquet/cn/latest_manifest.json` 不是
Fundamental generation pointer 或 generation manifest，不得替代。

```bash
quant-investor market v17-source-maintain \
  --repo-root /absolute/path/to/myQuant \
  --plan /owner/private/source-plan.json \
  --expected-plan-sha256 <source-plan-byte-sha256> \
  --expected-manifest-sha256 EMPTY
```

保存命令输出的 `manifest_path` 和 `manifest_sha256`。同一 manifest 的幂等读回必须
传入当前 manifest SHA；不要使用 `EMPTY` 覆盖已存在 manifest。

## 4. Prepare

prepare request 使用 `myquant.v17.shadow-prepare-request.v1`，绑定 source manifest、
包内四个 policy resource SHA、全部 schema SHA、run/strategy/cutoff 和三个单调递增
transition time。resource/schema 摘要必须与安装包冻结清单完全相同。

```bash
quant-investor market v17-shadow-prepare \
  --repo-root /absolute/path/to/myQuant \
  --request /owner/private/prepare-request.json \
  --expected-request-sha256 <prepare-request-byte-sha256> \
  --expected-ledger-sha256 EMPTY
```

prepare 只在 sealed universe 内本地重算 Fundamental 与 deep-request。保存输出的
`ledger_sha256`；随后用 status 复核状态必须为 `DEEP_REQUEST_READY`。

## 5. Receive sealed deep response

Codex 响应必须在仓库外完成审核后以本地文件导入，只能引用 prepare 生成的 sealed
evidence。此步骤不会主动调用模型，也不能扩充股票或数据源。

```bash
quant-investor market v17-shadow-receive \
  --repo-root /absolute/path/to/myQuant \
  --run-id <run-id> \
  --response /owner/private/deep-response.json \
  --expected-response-sha256 <deep-response-byte-sha256> \
  --expected-ledger-sha256 <current-ledger-sha256> \
  --expected-latest-sha256 <current-latest-sha256-or-EMPTY> \
  --failed-at <UTC-timestamp>
```

正常结果必须进入 `DEEP_RESPONSE_RECEIVED`。非法 evidence 会按显式 CAS 写入 hard-stop
终态；出现 hard stop 后不得继续 finalize。

## 6. Finalize

finalization 文件使用 `myquant.v17.shadow-finalization.v1`。它只能提交按
`candidate_id` 排序的 sealed-universe target-weight proposals；Quant、overlay、
权限、PreTrade、成本和优化结果都由本地代码重算，外部文件不能直接提供这些结论。

```bash
quant-investor market v17-shadow-finalize \
  --repo-root /absolute/path/to/myQuant \
  --run-id <run-id> \
  --finalization /owner/private/finalization.json \
  --expected-finalization-sha256 <finalization-byte-sha256> \
  --expected-ledger-sha256 <current-ledger-sha256> \
  --expected-latest-sha256 <current-latest-sha256-or-EMPTY> \
  --failed-at <UTC-timestamp>
```

允许的业务终态只有：

- `SHADOW_COMPLETE_AWAITING_HUMAN_DECISION`
- `SHADOW_RANK_COMPLETE_NO_PORTFOLIO`
- `SHADOW_PORTFOLIO_INFEASIBLE`

这些都是 shadow 结果，不是交易授权。两个 hard-stop 终态表示证据或快照失败，不能
作为业务完成验收。

## 7. Status 与显式 latest repair

status 是只读操作，可在每次状态推进后执行：

```bash
quant-investor market v17-shadow-status \
  --repo-root /absolute/path/to/myQuant \
  --run-id <run-id>
```

只有已逐字节验证为 immutable terminal 的 run，才允许修复 latest。不得直接编辑
`results/v17_shadow/_latest/shadow.json`：

```bash
quant-investor market v17-shadow-latest-repair \
  --repo-root /absolute/path/to/myQuant \
  --run-id <terminal-run-id> \
  --expected-ledger-sha256 <terminal-ledger-sha256> \
  --expected-latest-sha256 <current-latest-sha256-or-EMPTY> \
  --repaired-at <UTC-timestamp>
```

repair 只重发已验证 terminal pointer，不会重算、恢复或创造 run。完成后再次执行
status，并核对 `is_latest=true`、ledger/latest SHA 和 `authority=false`。

# Phase12：三类日常任务的切换准备

状态：本地准备，尚不可应用。正式安装、真实数据配置和最终验收未齐备；本文件不授予新的执行权限，不修改任何任务。精确输入与缺口记录在 `.agent/acceptance/phase12-cutover-bindings.json`，现有四个任务的完整配置和原始 SHA 保存在 `.agent/acceptance/phase12-current-automation-snapshot.json`。

## 当前已确认

2026-09-14 再次读取四份实际配置，全部与上次审计的 SHA 相同；应用的只读查看操作识别四个任务，但没有返回有效的无人值守权限或新的调度运行回执。不能把任务显示 ACTIVE 当作执行成功。

2026-09-17 更新：当前冻结候选为 `5060dee2d4b39b50a735393d056ebe5adb4efe3b`，位置见 `.agent/acceptance/phase15-canonical-scan-location.json`。全部12项 CI 已通过，5185项测试通过、3项跳过；首日完整原生执行与独立回读、Case8 已通过。当前仅1/5正常日期获验证，后续日期、Morning 与其他异常场景尚待当前版本的最终证据。旧候选 c3 的五日与聚合回读、Morning 缺证明失败，以及 d2 的失败与主动停止均保留为历史记录。临时合成验收目录不能作为正式部署绑定。

生产 Calendar 的 v3 正式抓取配置、传输回执、v2 正式证明、零联网恢复和 Morning 正式资格校验已实现，包含在5060完整 CI 中；同版本真实隔离安装的 Calendar 组件检查也已通过，但使用明确披露的离线 HTTPS 与身份检查测试替代，Core 输入仅验证结构。因此真实来源运行、完整正式 Morning 及无人值守证据仍未取得。组件证据见 `.agent/acceptance/phase15-canonical-scan-installed-smoke.json`；不能通过修改 provenance 标签使合成证明获得生产资格。

新安装的原生 `validate_initial_stop_policy` 已接受现有 owner 声明的两行止损，其中一行属于历史持仓重新确认；原始 SHA、文件内容和完整身份均未改变。原来记录的 `MORNING_LEGACY_OWNER_STOP_READER_UNSUPPORTED` 已被该证据替代。此结论仅证明声明格式受支持；完整 Morning 仍必须验证前一日 EOD、行情、持仓和政策来源，缺失的移动止损锚点不能因此获得执行资格。原先准备的 704 个静态来源 SHA 全部一致，但这不证明当前 Top100 覆盖或数据新鲜度。

## 现有任务与目标职责

| 现有任务 ID / 名称 | 当前时间 | 切换后的职责 |
| --- | --- | --- |
| `cn-daily-data-update` / A股收盘生产与数据更新 | 工作日 16:20、17:20、18:20、20:20 | 同一个任务承担 EOD producer。前三个既有维护槽保留；20:20 从同一配置入口产生并运行完整每日 DAG，禁止再追加独立 Factor、Top100 或 Store 生产调用。 |
| `cn-evening-review-fallback` / cn-evening-review-fallback | 工作日 21:00 | 使用与 EOD 相同的配置引用和逻辑 2020 槽，优先复用原请求及其恢复状态。完整时按原生检查返回 NO_ACTION；仅按原生可恢复状态继续。 |
| `automation` / A股量化投资与日度复盘 | 工作日 09:45 | 先封存当日真实行情，再消费前一完成交易日的完整 DAG 与 Morning v3 政策；不运行维护或补前日生产。 |
| `cn-dashboard` / CN aggressive 持仓与业绩 Dashboard | 工作日 21:30 | 在其已有公开展示、脱敏和发布检查职责完成映射前继续保留。只有接替职责已实际验证并完成切换，才暂停该重复定时任务；不能为凑齐三个任务而直接删除这项职责。 |

各任务当前模型、推理设置、项目、运行环境及通知偏好沿用保存配置，不在提示词正文中重复固化这些设置。周度任务、A_quant 和已有观察心跳不在本次切换对象中。

## EOD 与夜间补齐的精确入口

复用正式清洁仓库中的 `scripts/operations/run_cn_daily_slot.sh`。两个任务的完整生产调用必须绑定同一个 `--daily-source-config` 与 `--expected-daily-source-config-sha256`，使用逻辑 `--attempt-slot 2020`，并绑定同一个 `--release-install-input`、`--expected-release-install-input-sha256`、`--release-repository-root`、`--python` 和 `--expected-import-root`。`--workspace-root` 保持 `/Users/maxwell/mySpace/myQuant`，`--run-root` 保持其 `data/private/cn_daily_maintenance`。

配置模式不能同时传入旧的 Factor-context 参数或独立 daily-production-request 参数。实际命令必须由最终已验证的生产绑定值形成；当前绑定文件中的空值是未完成项，不是可执行占位参数，也不得用临时合成引用填补。

原生输入选择负责已有 ACTIVE 请求、已登记请求和旧日恢复的优先级。夜间任务不得自行按时间、最新文件或聊天记忆选择一个新请求。完整回读与有限本地修复的分支不读取凭证；只有既有授权覆盖的 scheduled producer 分支可以使用原来的 PROJECT_ENV 凭证流程和固定官方目的地。保存原始 STARTED/ENDED、真实退出码、请求与安装引用及业务结果，明确区分程序完成和业务闭环完成。

## Morning 与 Dashboard 的承接条件

Morning 保留当前真实时间与 quote-first 顺序，先记录实际时钟、最小安装锚点和持仓代码请求并封存行情，再执行完整原生消费验证。09:45 是调度参考槽，不是伪造的抓取时刻。请求和报告必须绑定同一份已封存行情、前一完成交易日 EOD、trailing 与 initial-stop 政策。任何失败都不得转去运行前日维护、重新抓取覆盖已封存行情或用旧晨报代替本轮结果。

现有 21:30 任务还承担私有导出检查与公开页面脱敏、发布和实际页面回读。每日 DAG 的本地 Dashboard 完成记录不自动证明这些外部职责已经完成。最终三个职责的切换须把这部分原有职责明确放入 EOD 完成后的消费步骤，并保留其官方收盘、基准同日、脱敏和实际字节回读要求；在具体发布动作的既有授权与目标映射未确认前，保持第四个现有任务，不能扩大 EOD 的外部写入范围。

## 完成切换前仍需齐备的证据

1. 当前版本完整五个连续模拟交易日、后续 Morning 及原设计异常案例通过；旧版本或中断后的记录保持各自证明范围。
2. 正式 Store/Event 缺失期间的 owner 账务事实及必要公司行动、锚点复核已确认。当前日期的空事件授权不能用于补造历史事实。
3. 正式 Macro 的原生恢复和写入 veto 已按原路径解决，其他生产来源达到各自门槛；不通过改指针、改 SHA 或放宽新鲜度通过校验。
4. 正式清洁安装、真实 daily-source-config、Factor context、Morning 政策引用及原始 SHA 全部可读且相互绑定；已有请求归属已经核对。预测截止时间未获确认时，不编造当时可用或 OOS 资格。
5. 填入准确最终参数，完成具体提示词、原设置保留、回滚值与外部发布职责的审查。先验证 fallback，再切主 EOD，最后核对 Morning；通过应用工具更新原任务并回读，不创建重复任务。
6. 取得一份独立无人值守原生运行回执，继续完成最终观察要求。手动验证、保存配置和真实无人值守运行分别记录。
7. 完成生产可用的下一交易日 Calendar 证明链及 Morning 非 REPLAY 验证。现有合成发布者、REPLAY 结果与本地测试都不能替代该证据；对应设计已完成 Architect、Critic 审查及本地实现验证；剩余真实来源执行与完整非 REPLAY 消费证据仍须取得。

官方文档要求先测试任务提示词、检查初期运行结果；本地任务还依赖机器、应用、项目和实际生效的权限。其机制说明不能代替本项目的运行证据。[OpenAI Scheduled tasks](https://learn.chatgpt.com/docs/automations?surface=app)

上述任何必要绑定仍为空、原始引用漂移、任务存在冲突请求或验收失败时，保持本准备包不可应用，记录具体缺项。对当前合成验收进程不做重复启动，不修改其已冻结安装和输入。

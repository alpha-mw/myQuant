# myQuant V17 Mainline Dashboard v5

这是一个只读、单策略、fail-closed 的本地静态 Dashboard。唯一可接受的业务输入是：

```text
myquant.v17.v4.mainline-public-run.v1
```

Dashboard 不扫描运行目录，不选择“最近一次”产物，也不从其他协议、研究运行或
历史数据构造替代结果。

## 运行状态

- `ACTIVE`：exact DTO、五个不可变引用、权限标志、目标权重和排序全部通过校验。
- `V17_MAINLINE_UNINITIALIZED`：没有单策略 active pointer；组合详情保持关闭。
- `BLOCKED`：输入存在但不符合 exact DTO；组合详情保持关闭并显示 blocker。

`ACTIVE` 仅表示 public run 已由正式 V17 主线 active pointer 解析。页面始终只读，
不提供 broker、order、execution、trade、provider、LLM control 或 selector write
能力。

## 私有输入

真实输入放在 Git ignored 的：

```text
portfolio_dashboard/private/mainline_public_run.js
```

文件只应赋值一个全局变量：

```js
window.MyQuantV17MainlinePublicRun = { /* exact mainline public run DTO */ };
```

tracked `js/mainline_input.js` 只提供空值占位，并保留先加载的私有 DTO。缺少私有
文件会自然进入 `V17_MAINLINE_UNINITIALIZED`，不会加载 sample 作为业务输入。

## Contract v5

- JSON Schema：`schema/dashboard_contract.v5.schema.json`
- 合成不可用样例：`sample/dashboard_snapshot.v5.json`
- 浏览器校验与状态派生：`js/dashboard_contract_v5.js`

Contract v5 的 `ACTIVE` 分支保留 exact public run 和 active pointer ref；不可用分支
必须把 `strategy_id`、`active_pointer_ref` 和 `public_run` 全部置为 `null`，并至少
提供一个 blocker。

## 本地验证

```bash
node portfolio_dashboard/tests/dashboard_contract_v5.test.js
```

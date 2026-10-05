(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";

  function byId(id) {
    return document.getElementById(id);
  }

  function money(value) {
    var parsed = Number(value);
    return Number.isFinite(parsed)
      ? "¥" + parsed.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
      : "—";
  }

  function signedMoney(value) {
    var parsed = Number(value);
    if (!Number.isFinite(parsed)) return "—";
    return (parsed > 0 ? "+" : "") + money(parsed);
  }

  function pct(value, base) {
    var numerator = Number(value);
    var denominator = Number(base);
    if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) return "—";
    return ((numerator / denominator) * 100).toFixed(2) + "%";
  }

  function tone(value) {
    var parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed === 0) return "";
    return parsed > 0 ? "positive" : "negative";
  }

  function cell(text, className) {
    var td = document.createElement("td");
    td.textContent = text === null || text === undefined || text === "" ? "—" : String(text);
    if (className) td.className = className;
    return td;
  }

  function row(values) {
    var tr = document.createElement("tr");
    values.forEach(function (item) {
      tr.appendChild(cell(item.text, item.className));
    });
    return tr;
  }

  function metricRow(label, value, className) {
    var wrap = document.createElement("div");
    var dt = document.createElement("dt");
    var dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    if (className) dd.className = className;
    wrap.appendChild(dt);
    wrap.appendChild(dd);
    return wrap;
  }

  function renderSummary(bundle) {
    var metrics = byId("paperMetrics");
    var initial = Number(bundle.initial_capital);
    var nav = Number(bundle.nav);
    metrics.textContent = "";
    metrics.appendChild(metricRow("账户净值 NAV", money(nav)));
    metrics.appendChild(metricRow("累计收益", signedMoney(nav - initial) + "（" + pct(nav - initial, initial) + "）", tone(nav - initial)));
    metrics.appendChild(metricRow("现金", money(bundle.cash)));
    metrics.appendChild(metricRow("持仓市值", money(bundle.market_value)));
    metrics.appendChild(metricRow("累计已实现盈亏", signedMoney(bundle.cumulative_realized_pnl), tone(bundle.cumulative_realized_pnl)));
    metrics.appendChild(metricRow("累计费用", money(bundle.cumulative_fees)));
    metrics.appendChild(metricRow("持仓数 / 序号", bundle.positions.length + " / " + bundle.sequence));
    byId("paperValuation").textContent = "估值日 " + bundle.valuation_date + "（严格收盘价）";
    byId("paperStatus").textContent = bundle.boundary.view;
    byId("paperBoundary").textContent = bundle.boundary.note;
    byId("paperEvidence").textContent =
      "pointer " + String(bundle.pointer_sha256).slice(0, 16) + "… · content " +
      String(bundle.content_sha256).slice(0, 16) + "… · 生成于 " + bundle.generated_at;
  }

  function renderCurve(bundle) {
    var host = byId("paperCurve");
    host.textContent = "";
    var points = [{ trade_date: bundle.valuation_date, cash: bundle.cash, realized_pnl: bundle.cumulative_realized_pnl }]
      .concat(bundle.curve.slice().reverse());
    if (points.length < 2) {
      var note = document.createElement("p");
      note.className = "chart-empty";
      note.textContent = "尚无历史点：净值曲线会在第二笔成交后出现。";
      host.appendChild(note);
      return;
    }
    var width = 640;
    var height = 200;
    var padding = 28;
    var values = points.map(function (point) {
      return Number(point.cash) + 0;
    });
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    var span = max - min || 1;
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "现金余额曲线，" + points.length + " 个点");
    var path = values
      .map(function (value, index) {
        var x = padding + (index * (width - padding * 2)) / (values.length - 1);
        var y = height - padding - ((value - min) / span) * (height - padding * 2);
        return (index === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1);
      })
      .join(" ");
    var line = document.createElementNS(SVG_NS, "path");
    line.setAttribute("d", path);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", "var(--accent, #2f6f9f)");
    line.setAttribute("stroke-width", "2");
    svg.appendChild(line);
    var label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", String(padding));
    label.setAttribute("y", String(height - 6));
    label.setAttribute("font-size", "10");
    label.setAttribute("fill", "var(--muted, #6b7280)");
    label.textContent = points[0].trade_date + " → " + points[points.length - 1].trade_date;
    svg.appendChild(label);
    host.appendChild(svg);
  }

  function renderPositions(bundle) {
    var body = byId("paperPositionsBody");
    body.textContent = "";
    if (!bundle.positions.length) {
      body.appendChild(row([{ text: "空仓", className: "identity-cell" }]));
      return;
    }
    bundle.positions.forEach(function (position) {
      body.appendChild(
        row([
          { text: position.symbol + " " + position.name, className: "identity-cell" },
          { text: position.shares, className: "numeric" },
          { text: position.settled_shares, className: "numeric" },
          { text: position.avg_cost, className: "numeric" },
          { text: position.close, className: "numeric" },
          { text: money(position.market_value), className: "numeric" },
          { text: signedMoney(position.unrealized_pnl), className: "numeric " + tone(position.unrealized_pnl) }
        ])
      );
    });
  }

  function renderTrades(bundle) {
    var body = byId("paperTradesBody");
    body.textContent = "";
    if (!bundle.trades.length) {
      body.appendChild(row([{ text: "尚无成交", className: "identity-cell" }]));
      return;
    }
    bundle.trades
      .slice()
      .reverse()
      .forEach(function (trade) {
        body.appendChild(
          row([
            { text: trade.trade_date, className: "period-cell" },
            { text: trade.symbol },
            { text: trade.side },
            { text: trade.owner_declared_price ? trade.action + "（owner 定价）" : trade.action },
            { text: trade.shares, className: "numeric" },
            { text: trade.price, className: "numeric" },
            { text: money(trade.total_fees), className: "numeric" },
            { text: signedMoney(trade.realized_pnl_delta), className: "numeric " + tone(trade.realized_pnl_delta) },
            { text: money(trade.cash_after), className: "numeric" }
          ])
        );
      });
    byId("paperTradesNote").textContent =
      bundle.trades.length + " 笔成交，来自账户不可变记录，按时间正序。";
  }

  function fail(message) {
    var status = byId("paperStatus");
    if (status) {
      status.textContent = "BLOCKED";
      status.className = "status-pill blocked";
    }
    var metrics = byId("paperMetrics");
    if (metrics) {
      metrics.textContent = "";
      metrics.appendChild(metricRow("镜像不可用", message));
    }
  }

  function boot() {
    var bundle = window.CNPaperDashboard;
    if (!bundle || bundle.schema_version !== "cn-paper-dashboard.v1") {
      fail("缺少 cn_paper_dashboard.v1.js（运行 scripts/export_cn_paper_dashboard_data.py --write）");
      return;
    }
    renderSummary(bundle);
    renderCurve(bundle);
    renderPositions(bundle);
    renderTrades(bundle);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();

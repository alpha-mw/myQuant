(function () {
  "use strict";

  var Charts = window.DashboardCharts;

  function escapeHtml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function signedClass(value) {
    if (typeof value !== "number" || !Number.isFinite(value) || value === 0) return "neutral";
    return value > 0 ? "positive" : "negative";
  }

  function renderMessages(errors, warnings, infos) {
    var panel = document.getElementById("messagePanel");
    if (!panel) return;
    function unique(items) {
      var seen = new Set();
      return (items || []).map(function (item) { return String(item || "").trim(); })
        .filter(function (item) {
          if (!item || seen.has(item)) return false;
          seen.add(item);
          return true;
        });
    }
    function messageElement(type, text) {
      var div = document.createElement("div");
      div.className = "message " + type;
      div.textContent = text;
      return div;
    }

    var errorItems = unique(errors);
    var warningItems = unique(warnings);
    var infoItems = unique(infos);
    var primary = [];
    var remaining = [];
    if (errorItems.length === 1) primary.push({ type: "error", text: errorItems[0] });
    else if (errorItems.length > 1) {
      primary.push({
        type: "error",
        text: "数据错误 " + errorItems.length + " 项，Dashboard 已 blocked；请展开查看完整明细。"
      });
      errorItems.forEach(function (text) { remaining.push({ type: "error", text: text }); });
    }
    if (warningItems.length) {
      primary.push({ type: "warning", text: warningItems[0] });
      warningItems.slice(1).forEach(function (text) { remaining.push({ type: "warning", text: text }); });
    }
    if (infoItems.length) {
      primary.push({ type: "info", text: infoItems[0] });
      infoItems.slice(1).forEach(function (text) { remaining.push({ type: "info", text: text }); });
    }
    panel.innerHTML = "";
    panel.classList.toggle("visible", primary.length > 0 || remaining.length > 0);
    primary.forEach(function (message) {
      panel.appendChild(messageElement(message.type, message.text));
    });
    if (!remaining.length) return;

    var details = document.createElement("details");
    details.className = "message-details";
    var summary = document.createElement("summary");
    summary.textContent =
      "展开全部数据状态（错误 " + errorItems.length +
      " / 告警 " + warningItems.length +
      " / 信息 " + infoItems.length + "）";
    details.appendChild(summary);
    var list = document.createElement("div");
    list.className = "message-detail-list";
    remaining.forEach(function (message) {
      list.appendChild(messageElement(message.type, message.text));
    });
    details.appendChild(list);
    panel.appendChild(details);
  }

  function renderKpis(kpis) {
    function fmtPercent(value) {
      return Charts.formatPercent(value);
    }
    function fmtRatio(value) {
      if (value === Infinity) return "∞";
      return Number.isFinite(value) ? value.toFixed(2) : "-";
    }
    function fmtDays(value) {
      return Number.isFinite(value) ? String(value) + " 天" : "-";
    }
    var items = [
      {
        label: "累计 TWR",
        field: "total_return",
        format: fmtPercent,
        subLabel: "年化",
        subValue: function () {
          return kpis.annualization_status === "eligible"
            ? fmtPercent(kpis.annualized_return)
            : (kpis.annualization_status || "insufficient_daily_history");
        }
      },
      { label: "相对财富超额", field: "excess_return", format: fmtPercent, subLabel: "主基准累计", subField: "benchmark_total_return", subFormat: fmtPercent },
      { label: "最大回撤", field: "max_drawdown", format: fmtPercent, subLabel: "最长回撤持续", subField: "drawdown_duration", subFormat: fmtDays },
      { label: "NAV 股票暴露", field: "nav_exposure", format: fmtPercent, subLabel: "现金权重", subField: "cash_weight", subFormat: fmtPercent, neutral: true },
      { label: "股票袖套权重", field: "equity_sleeve_weight", format: fmtPercent, subLabel: "与 NAV 权重分列", subValue: function () { return "equity sleeve"; }, neutral: true },
      { label: "日净值胜率", field: "win_rate", format: fmtPercent, subLabel: "正收益日 / 有效收益日", subValue: function () { return (kpis.win_count || 0) + " / " + (kpis.trading_days || 0); }, neutral: true }
    ];
    var grid = document.getElementById("kpiGrid");
    grid.innerHTML = "";
    items.forEach(function (item) {
      var value = kpis[item.field];
      var subText = typeof item.subValue === "function"
        ? item.subValue()
        : item.subFormat(kpis[item.subField]);
      var card = document.createElement("article");
      card.className = "kpi-card";
      card.innerHTML =
        '<div class="kpi-label">' + escapeHtml(item.label) + '</div>' +
        '<div class="kpi-value ' + (item.neutral ? "neutral" : signedClass(value)) + '">' + escapeHtml(item.format(value)) + '</div>' +
        '<div class="kpi-sub">' + escapeHtml(item.subLabel) + '<strong class="' + (item.neutral ? "neutral" : signedClass(kpis[item.subField])) + '">' + escapeHtml(subText) + '</strong></div>';
      grid.appendChild(card);
    });
  }

  function renderTable(containerId, rows, columns, emptyText) {
    var container = document.getElementById(containerId);
    if (!container) return;
    if (!rows || !rows.length) {
      container.innerHTML = '<div class="empty-state">' + escapeHtml(emptyText || "暂无表格数据") + "</div>";
      return;
    }
    var html = '<table class="data-table"><thead><tr>';
    columns.forEach(function (col) {
      html += '<th class="' + (col.numeric ? "numeric" : "") + '">' + escapeHtml(col.label) + "</th>";
    });
    html += "</tr></thead><tbody>";
    rows.forEach(function (row) {
      html += "<tr>";
      columns.forEach(function (col) {
        var raw = typeof col.value === "function" ? col.value(row) : row[col.value];
        var className = col.numeric ? "numeric" : "";
        if (col.signed) className += " " + signedClass(raw);
        var text = col.format ? col.format(raw, row) : raw;
        html += '<td class="' + className + '" data-label="' + escapeHtml(col.label) + '">' + escapeHtml(text) + "</td>";
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    container.innerHTML = html;
    if (containerId === "overviewHoldingsTable") {
      container.querySelectorAll("tbody tr").forEach(function (row) {
        row.tabIndex = 0;
        row.setAttribute("aria-expanded", "false");
        function toggle() {
          var expanded = row.classList.toggle("expanded");
          row.setAttribute("aria-expanded", expanded ? "true" : "false");
        }
        row.addEventListener("click", toggle);
        row.addEventListener("keydown", function (event) {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            toggle();
          }
        });
      });
    }
  }

  function renderToggleButton(buttonId, expanded, total, limit) {
    var button = document.getElementById(buttonId);
    if (!button) return;
    var hasMore = total > limit;
    button.classList.toggle("hidden", !hasMore);
    button.textContent = expanded ? "收起" : "显示全部 (" + total + ")";
    button.setAttribute("aria-expanded", expanded ? "true" : "false");
  }

  function renderBestWorst(monthly) {
    var container = document.getElementById("bestWorstMonths");
    if (!container) return;
    monthly = (monthly || []).filter(function (row) { return Number.isFinite(row.value); });
    if (!monthly.length) {
      container.innerHTML = '<div class="empty-state">暂无月度收益数据</div>';
      return;
    }
    var best = monthly.slice().sort(function (a, b) { return b.value - a.value; }).slice(0, 5);
    var worst = monthly.slice().sort(function (a, b) { return a.value - b.value; }).slice(0, 5);
    var avg = monthly.reduce(function (total, row) { return total + row.value; }, 0) / monthly.length;
    var up = monthly.filter(function (row) { return row.value > 0; }).length;
    var down = monthly.filter(function (row) { return row.value < 0; }).length;
    var html = '<div class="best-worst-grid">';
    html += '<div class="summary-card">';
    [
      ["平均月收益", Charts.formatPercent(avg), avg],
      ["上涨月份", up + " / " + monthly.length, up],
      ["下跌月份", down + " / " + monthly.length, -down],
      ["最大单月收益", best[0].month + "  " + Charts.formatPercent(best[0].value), best[0].value],
      ["最小单月收益", worst[0].month + "  " + Charts.formatPercent(worst[0].value), worst[0].value]
    ].forEach(function (item) {
      html += '<div class="summary-stat"><span>' + escapeHtml(item[0]) + '</span><span class="' + signedClass(item[2]) + '">' + escapeHtml(item[1]) + "</span></div>";
    });
    html += "</div>";
    [
      ["最好月份 Top 5", best],
      ["最差月份 Bottom 5", worst]
    ].forEach(function (group) {
      html += '<div class="summary-card"><strong>' + group[0] + "</strong>";
      group[1].forEach(function (row) {
        html += '<div class="summary-stat"><span>' + escapeHtml(row.month) + '</span><span class="' + signedClass(row.value) + '">' + Charts.formatPercent(row.value) + "</span></div>";
      });
      html += "</div>";
    });
    html += "</div>";
    container.innerHTML = html;
  }

  function renderTradeSummary(trades) {
    var container = document.getElementById("tradeSummary");
    if (!container) return;
    if (!trades.available) {
      container.innerHTML = '<div class="summary-card"><strong>交易复盘状态</strong><span>未上传交易数据。</span></div>';
      return;
    }
    function fmtPercent(value) {
      return Charts.formatPercent(value);
    }
    function fmtMoney(value) {
      return Charts.formatMoney(value);
    }
    function fmtDays(value) {
      return Number.isFinite(value) ? value.toFixed(1) + " 天" : "-";
    }
    function fmtRatio(value) {
      return Number.isFinite(value) ? value.toFixed(2) : "-";
    }
    function fmtQuantity(value) {
      if (!Number.isFinite(value)) return "-";
      return Math.abs(value - Math.round(value)) < 0.000001 ? String(Math.round(value)) : value.toFixed(3);
    }
	    var cards = [
	      ["买入数量", trades.totals.buy_count],
	      ["卖出数量", trades.totals.sell_count],
	      ["总成交金额", Charts.formatMoney(trades.totals.trade_amount)],
	      ["总费用", Charts.formatMoney(trades.totals.fee)],
	      ["已平仓交易", trades.totals.closed_trade_count || 0],
	      ["毛实现盈亏", fmtMoney(trades.totals.gross_realized_pnl)],
	      ["毛交易胜率", fmtPercent(trades.totals.gross_trade_win_rate)],
	      ["毛盈亏比", fmtRatio(trades.totals.gross_profit_factor)],
	      ["净实现盈亏", fmtMoney(trades.totals.net_realized_pnl)],
	      ["净交易胜率", fmtPercent(trades.totals.trade_win_rate)],
	      ["净盈亏比", fmtRatio(trades.totals.profit_factor)],
	      ["净平均盈利", fmtMoney(trades.totals.avg_win)],
	      ["净平均亏损", fmtMoney(trades.totals.avg_loss)],
	      ["净单笔最大盈利", fmtMoney(trades.totals.max_trade_profit)],
	      ["净单笔最大亏损", fmtMoney(trades.totals.max_trade_loss)],
	      ["平均持仓周期", fmtDays(trades.totals.avg_holding_days)],
	      ["FIFO状态", trades.message]
	    ];
	    if (trades.openingLots && trades.openingLots.available) {
	      cards.push([
	        "期初FIFO lot",
	        trades.openingLots.sourceDate + " 生成 " + trades.openingLots.lotCount + " 个 lot，" +
	        fmtQuantity(trades.openingLots.totalQuantity) + " 股，成本基础 " + fmtMoney(trades.openingLots.costBasis)
	      ]);
	      cards.push([
	        "期初lot匹配",
	        (trades.totals.closed_with_opening_lot_count || 0) + " 笔平仓，匹配 " +
	        fmtQuantity(trades.totals.opening_matched_quantity || 0) + " 股"
	      ]);
	    }
	    if ((trades.unmatchedSells || []).length) {
	      cards.push([
	        "FIFO未匹配",
	        (trades.totals.unmatched_sell_count || trades.unmatchedSells.length) + " 笔卖出，合计 " +
	        fmtQuantity(trades.totals.unmatched_quantity || 0) + " 股；通常是 positions 缺少期初数量/成本或交易 CSV 缺少更早记录。"
	      ]);
	    }
    var otherWarningCount = Math.max(0, (trades.warnings || []).length - (trades.unmatchedSells || []).length);
    if (otherWarningCount) {
      cards.push(["FIFO其他警告", otherWarningCount + " 条，详见原始交易数据。"]);
    }
    container.innerHTML = cards.map(function (item) {
      return '<div class="summary-card"><strong>' + escapeHtml(item[0]) + "</strong><span>" + escapeHtml(item[1]) + "</span></div>";
    }).join("");
  }

  function lineSeriesFromTrend(rows, fields, formatter) {
    return fields.map(function (field, index) {
      return {
        name: field,
        color: Charts.COLORS[index % Charts.COLORS.length],
        points: rows.map(function (row) {
          return { date: row.date, dateObj: row.dateObj, value: row[field] || 0 };
        }),
        formatter: formatter
      };
    });
  }

  function topFieldsFromTrend(rows, limit) {
    var totals = {};
    rows.forEach(function (row) {
      Object.keys(row).forEach(function (key) {
        if (key !== "date" && key !== "dateObj" && Number.isFinite(row[key])) {
          totals[key] = (totals[key] || 0) + row[key];
        }
      });
    });
    return Object.keys(totals).sort(function (a, b) { return totals[b] - totals[a]; }).slice(0, limit || 7);
  }

  function fmtRatio(value) {
    if (value === Infinity) return "∞";
    return Number.isFinite(value) ? value.toFixed(2) : "-";
  }

  function benchmarkColor(index, isMain) {
    if (isMain) return "#101828";
    return Charts.COLORS[(index + 1) % Charts.COLORS.length];
  }

  function selectedBenchmarkSeries(comparison, options) {
    options = options || {};
    var series = [];
    if (options.includePortfolio !== false) {
      series.push({
        name: "Portfolio",
        color: "#2563eb",
        points: options.drawdown ? comparison.portfolioDrawdown : comparison.portfolioSeries,
        width: 3
      });
    }
    (comparison.selectedBenchmarks || []).forEach(function (benchmark, index) {
      series.push({
        name: benchmark.label + (benchmark.isMain ? " · 主基准" : ""),
        color: benchmarkColor(index, benchmark.isMain),
        points: options.drawdown ? benchmark.drawdown : benchmark.series,
        width: benchmark.isMain ? 2.4 : 1.7,
        areaToZero: Boolean(options.drawdown && benchmark.isMain),
        areaOpacity: 0.08
      });
    });
    return series;
  }

  function benchmarkColumns() {
    return [
      { label: "Name", value: function (row) { return row.name; } },
      { label: "Total Return", value: "totalReturn", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "Annualized Return", value: "annualizedReturn", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "Ann. Volatility", value: "annualizedVolatility", format: Charts.formatPercent, numeric: true },
      { label: "Max Drawdown", value: "maxDrawdown", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "Sharpe Ratio", value: "sharpeRatio", format: fmtRatio, numeric: true },
      { label: "Correlation vs Portfolio", value: "correlation", format: fmtRatio, numeric: true },
      { label: "Portfolio Beta to Benchmark", value: "beta", format: fmtRatio, numeric: true },
      { label: "Tracking Error", value: "trackingError", format: Charts.formatPercent, numeric: true },
      { label: "Information Ratio", value: "informationRatio", format: fmtRatio, numeric: true, signed: true },
      { label: "Excess Return vs Portfolio", value: "excessReturn", format: Charts.formatPercent, numeric: true, signed: true }
    ];
  }

  function renderBenchmarkTable(containerId, rows, emptyText) {
    renderTable(containerId, rows, benchmarkColumns(), emptyText || "未识别到 benchmark NAV 字段");
  }

  function sortedBenchmarkRows(rows, view) {
    var field = view && view.benchmarkSortField ? view.benchmarkSortField : "totalReturn";
    var direction = view && view.benchmarkSortDirection === "asc" ? "asc" : "desc";
    return (rows || []).slice().sort(function (a, b) {
      var av = Number.isFinite(a[field]) ? a[field] : (direction === "asc" ? Infinity : -Infinity);
      var bv = Number.isFinite(b[field]) ? b[field] : (direction === "asc" ? Infinity : -Infinity);
      return direction === "asc" ? av - bv : bv - av;
    });
  }

  function renderCorrelationHeatmap(containerId, matrix) {
    var container = document.getElementById(containerId);
    if (!container) return;
    if (!matrix || !matrix.length) {
      container.innerHTML = '<div class="empty-state">暂无相关性数据</div>';
      return;
    }
    var labels = matrix.map(function (row) { return row.label; });
    var columns = labels.length + 1;
    var html = '<div class="corr-grid" style="grid-template-columns: 112px repeat(' + labels.length + ', minmax(72px, 1fr));">';
    html += '<div class="corr-cell header">corr</div>';
    labels.forEach(function (label) {
      html += '<div class="corr-cell header">' + escapeHtml(label) + '</div>';
    });
    matrix.forEach(function (row) {
      html += '<div class="corr-cell header">' + escapeHtml(row.label) + '</div>';
      row.values.forEach(function (cell) {
        var value = cell.value;
        var alpha = Number.isFinite(value) ? Math.min(0.82, Math.abs(value) * 0.72 + 0.1) : 0.08;
        var color = !Number.isFinite(value)
          ? "#eef2f7"
          : value >= 0
            ? "rgba(37,99,235," + alpha + ")"
            : "rgba(21,128,61," + alpha + ")";
        var textColor = Number.isFinite(value) && Math.abs(value) > 0.72 ? "#fff" : "#1f2937";
        html += '<div class="corr-cell" style="background:' + color + ';color:' + textColor + ';">' + escapeHtml(fmtRatio(value)) + '</div>';
      });
    });
    html += "</div>";
    container.innerHTML = html;
  }

  function lastFinite(points) {
    for (var index = (points || []).length - 1; index >= 0; index -= 1) {
      if (Number.isFinite(points[index].value)) return points[index].value;
    }
    return null;
  }

  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function renderPerformanceStory(metrics) {
    var performance = metrics.performance || {};
    var holdings = metrics.holdings || {};
    var trades = metrics.trades || {};
    var tradeTotals = trades.totals || {};
    var kpis = performance.kpis || {};
    var total = kpis.total_return;
    var excess = kpis.excess_return;
    var maxDrawdown = kpis.max_drawdown;
    var headline = "等待数据加载";
    if (Number.isFinite(total)) {
      if (Number.isFinite(excess)) {
        if (total >= 0 && excess >= 0) headline = "组合保持正收益并跑赢基准";
        else if (total >= 0 && excess < 0) headline = "组合为正收益，但相对基准偏弱";
        else if (total < 0 && excess >= 0) headline = "绝对收益承压，但仍好于基准";
        else headline = "组合处于回撤修复状态";
      } else {
        headline = total >= 0 ? "组合保持正收益，未配置 benchmark" : "组合处于回撤状态，未配置 benchmark";
      }
    }
    var narrative = kpis.start_date
      ? "期间 " + kpis.start_date + " 至 " + kpis.end_date +
        "；累计收益 " + Charts.formatPercent(total) +
        "，超额收益 " + Charts.formatPercent(excess) +
        "，最大回撤 " + Charts.formatPercent(maxDrawdown) + "。"
      : "载入 NAV、持仓和交易后，这里会显示组合收益、回撤、集中度和交易数据状态。";
    setText("performanceHeadline", headline);
    setText("performanceNarrative", narrative);
    var quickStats = [
      { label: "Top 5 weight", value: Charts.formatPercent(holdings.top5Weight) },
      { label: "Top 10 weight", value: Charts.formatPercent(holdings.top10Weight) },
      { label: "HHI", value: Number.isFinite(holdings.hhi) ? holdings.hhi.toFixed(3) : "-" },
      { label: "Trades", value: trades.available ? String(tradeTotals.trade_count || 0) + " 笔" : "未上传" }
    ];
    var container = document.getElementById("quickStats");
    if (!container) return;
    container.innerHTML = quickStats.map(function (item) {
      return '<div class="quick-stat"><span>' + escapeHtml(item.label) + "</span><strong>" + escapeHtml(item.value) + "</strong></div>";
    }).join("");
  }

  function renderOverviewFocus(metrics, view) {
    var performance = metrics.performance || {};
    var comparison = metrics.benchmarkComparison || {};
    var kpis = performance.kpis || {};
    var filters = metrics.filters || {};
    var lens = view && view.overviewLens ? view.overviewLens : "nav";
    if (lens === "excess") {
      setText("overviewFocusTitle", "累计超额收益曲线");
      setText("overviewFocusSubtitle", "区间超额收益 " + Charts.formatPercent(kpis.excess_return) + "；主基准 " + (comparison.mainLabel || filters.benchmarkField || "-"));
      if (filters.showExcessCurve === false) {
        Charts.renderEmpty("overviewFocusChart", "已隐藏超额曲线");
      } else {
        Charts.lineChart("overviewFocusChart", [
          { name: "相对主基准超额净值", color: "#2563eb", points: performance.excessSeries, width: 2.7 }
        ], { yFormatter: Charts.formatNumber, empty: "超额收益数据不足", endLabels: true, height: 292 });
      }
      return;
    }
    if (lens === "drawdown") {
      setText("overviewFocusTitle", "多 benchmark 回撤对比");
      setText("overviewFocusSubtitle", "组合最大回撤 " + Charts.formatPercent(kpis.max_drawdown) + "；主基准 " + (comparison.mainLabel || "-"));
      Charts.lineChart("overviewFocusChart", selectedBenchmarkSeries(comparison, {
        includePortfolio: filters.showPortfolioCurve !== false,
        drawdown: true
      }), { yFormatter: Charts.formatPercent, includeZero: true, empty: "回撤数据不足", areaToZero: true, endLabels: true, height: 292 });
      return;
    }
    if (lens === "vol") {
      setText("overviewFocusTitle", "Rolling 20D volatility");
      setText("overviewFocusSubtitle", "当前年化波动 " + Charts.formatPercent(lastFinite(performance.rolling20Vol)) + "；区间年化波动 " + Charts.formatPercent(kpis.annualized_volatility));
      Charts.lineChart("overviewFocusChart", [
        { name: "rolling_20d_volatility", color: "#0f8a8a", points: performance.rolling20Vol, width: 2.5 }
      ], { yFormatter: Charts.formatPercent, includeZero: true, empty: "NAV 日收益不足 20 个交易日", endLabels: true, height: 292 });
      return;
    }
    setText("overviewFocusTitle", "组合净值 vs 多 benchmark 净值");
    setText("overviewFocusSubtitle", "期末组合净值 " + Charts.formatNumber(lastFinite(performance.navSeries)) + "；已选 benchmark " + ((comparison.selectedFields || []).length || 0) + " 个");
    Charts.lineChart("overviewFocusChart", selectedBenchmarkSeries(comparison, {
      includePortfolio: filters.showPortfolioCurve !== false
    }), { yFormatter: Charts.formatNumber, empty: "NAV 数据不足或未选择 benchmark", endLabels: true, height: 292 });
  }

  function dashboardContract() {
    return window.DashboardSnapshotV2 ||
      (window.DashboardGeneratedRecords && window.DashboardGeneratedRecords.contract) || null;
  }

  function renderContractPanels(metrics) {
    var contract = dashboardContract() || {};
    var asOf = contract.as_of_matrix || {};
    var calendar = contract.trading_calendar || {};
    var navProvenance = contract.nav_return_provenance || {};
    var reconciliation = contract.reconciliation || {};
    var kpis = (metrics && metrics.performance && metrics.performance.kpis) || {};
    var asOfRows = [
      { field: "contract status", value: contract.status || "sample", detail: (contract.blockers || []).join(", ") || "no blockers" },
      { field: "strategy record", value: asOf.strategy_record_at || asOf.strategy_record_date, detail: contract.run_id },
      { field: "analysis trading day", value: asOf.analysis_trading_date, detail: "NAV/analysis date" },
      { field: "quote", value: asOf.quote_at, detail: "position quote timestamp" },
      { field: "theme", value: asOf.theme_date, detail: "theme state date" },
      { field: "factor registry", value: asOf.factor_registry_sha, detail: contract.protocol_hash },
      {
        field: "formal trading calendar",
        value: calendar.status || "missing",
        detail: (calendar.source_system || "strict Parquet trade_date unavailable") +
          "；open dates " + (calendar.expected_open_date_count || 0) +
          "；mask " + (calendar.mask_sha256 || "-")
      },
      {
        field: "annualization gate",
        value: kpis.annualization_status || "insufficient_daily_history",
        detail: "coverage " + Charts.formatPercent(kpis.open_day_coverage) +
          "；valid " + (kpis.trading_days || 0) +
          " / strict open dates " + (kpis.expected_open_day_count || 0)
      },
      {
        field: "NAV fee provenance",
        value: navProvenance.gross_or_net || "unknown",
        detail: "fee inclusion " + (navProvenance.trade_fee_inclusion || "unknown") +
          "；secondary deduction " + (navProvenance.secondary_fee_adjustment_allowed === true ? "allowed" : "forbidden")
      }
    ];
    renderTable("asOfMatrix", asOfRows, [
      { label: "as-of layer", value: "field" },
      { label: "value", value: "value" },
      { label: "status / hash", value: "detail" }
    ], "未加载 Dashboard Contract v2");
    var sources = contract.sources || {};
    var sourceRows = Object.keys(sources).map(function (name) {
      var row = sources[name] || {};
      return { source: name, path: row.path_summary, sha256: row.sha256, status: row.status || "" };
    });
    renderTable("sourceAuditTable", sourceRows, [
      { label: "source", value: "source" },
      { label: "path summary", value: "path" },
      { label: "SHA-256", value: "sha256" },
      { label: "status", value: "status" }
    ], "无来源审计数据");
    var reconciliationDaily = (reconciliation.daily || []).slice();
    renderTable("reconciliationTable", reconciliationDaily.slice().reverse(), [
      { label: "date", value: "date" },
      { label: "NAV return", value: "portfolio_return", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "positions contribution", value: "position_contribution", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "cash / fee residual", value: "explicit_cash_fee_residual", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "unexplained", value: "unexplained_residual", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "within 1bp", value: function (row) { return row.within_1bp === null ? "unknown" : row.within_1bp ? "yes" : "no"; } }
    ], "归因记录不足，状态保持 partial");
    renderTable("attributionCoverageSummary", [{
      status: reconciliation.status || "partial",
      valid_nav_return_days: reconciliation.valid_nav_return_days,
      covered_days: reconciliation.covered_days,
      coverage_ratio: reconciliation.coverage_ratio,
      reconciled_days: reconciliation.reconciled_days,
      tolerance: reconciliation.tolerance,
      coverage_basis: reconciliation.coverage_basis,
      blockers: (reconciliation.blockers || []).join(", ") || "none",
      diagnostics: reconciliation.diagnostics || {}
    }], [
      { label: "status", value: "status" },
      { label: "valid NAV days", value: "valid_nav_return_days", numeric: true },
      { label: "covered days", value: "covered_days", numeric: true },
      { label: "coverage", value: "coverage_ratio", format: Charts.formatPercent, numeric: true },
      { label: "within tolerance", value: "reconciled_days", numeric: true },
      { label: "tolerance", value: "tolerance", format: Charts.formatPercent, numeric: true },
      { label: "basis", value: "coverage_basis" },
      { label: "blockers", value: "blockers" },
      { label: "excluded / missing", value: function (row) {
        var diagnostic = row.diagnostics || {};
        return "NAV " + (diagnostic.excluded_nav_return_dates || []).length +
          " / position " + (diagnostic.excluded_position_effective_dates || []).length +
          " / missing effective " + (diagnostic.positions_missing_effective_date_count || 0);
      } }
    ], "无归因覆盖审计");
    var reconciliationExceptions = reconciliationDaily.filter(function (row) {
      var hasPositionContribution = Number.isFinite(row.position_contribution);
      return !hasPositionContribution || row.within_1bp !== true;
    }).reverse();
    renderTable("attributionExceptionsTable", reconciliationExceptions, [
      { label: "date", value: "date" },
      { label: "NAV return", value: "portfolio_return", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "positions", value: "position_contribution", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "cash / fee", value: "explicit_cash_fee_residual", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "unexplained", value: "unexplained_residual", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "reason", value: function (row) {
        if (!Number.isFinite(row.position_contribution)) {
          return Number.isFinite(row.explicit_cash_fee_residual)
            ? "position_uncovered_cash_fee_only"
            : "uncovered";
        }
        return row.within_1bp === true ? "-" : "residual_above_1bp";
      } }
    ], reconciliationDaily.length ? "所有有效 NAV return 日均已覆盖并在 1bp 内勾稽" : "归因记录不足，状态保持 partial");
    var themeProtocol = contract.theme_protocol || {
      status: "blocked",
      blockers: ["theme_protocol_v2_missing"],
      formal_pool: [],
      formal_pool_count: 0,
      readback_verified: false
    };
    renderTable("themeProtocolSummary", [themeProtocol], [
      { label: "status", value: "status" },
      { label: "observer", value: function (row) { return row.observer_enabled === true ? "enabled" : row.observer_enabled === false ? "disabled" : "unknown"; } },
      { label: "formal", value: function (row) { return row.formal_enabled === true ? "enabled" : row.formal_enabled === false ? "observer-only" : "unknown"; } },
      { label: "kill switch", value: function (row) { return row.formal_kill_switch === true ? "active" : row.formal_kill_switch === false ? "inactive" : "unknown"; } },
      { label: "formal pool", value: function (row) { return Number.isFinite(row.formal_pool_count) ? row.formal_pool_count : (row.formal_pool || []).length; }, numeric: true },
      { label: "producer", value: function (row) { return row.formal_producer || "unknown"; } },
      { label: "rollback", value: function (row) { return [row.rollback_status, row.rollback_reason].filter(Boolean).join(": ") || "unknown"; } },
      { label: "readback", value: function (row) { return row.readback_verified === true ? "verified" : "blocked"; } },
      { label: "blockers", value: function (row) { return (row.blockers || []).join(", ") || "none"; } }
    ], "Theme protocol v2 blocked or unavailable");
    var themes = contract.themes || [];
    Charts.scatter("themeAttentionMatrix", themes.map(function (row) {
      return {
        label: row.theme_name || row.theme_id,
        x: row.attention,
        y: row.industrial_validation,
        radius: Number.isFinite(row.nav_weight) ? Math.max(4, Math.min(12, 4 + row.nav_weight * 30)) : 5
      };
    }), {
      xLabel: "attention",
      yLabel: "industrial validation",
      xFormatter: Charts.formatPercent,
      yFormatter: Charts.formatPercent,
      height: 300
    });
    renderTable("themeStateTable", themes, [
      { label: "theme", value: function (row) { return row.theme_name || row.theme_id; } },
      { label: "lane", value: "lane" },
      { label: "lifecycle", value: "lifecycle" },
      { label: "attention", value: "attention", format: Charts.formatPercent, numeric: true },
      { label: "5/20/60/120D trajectory", value: function (row) {
        var trajectory = row.attention_trajectory_120d || {};
        var values = [
          ["5D", trajectory["5d"]],
          ["20D", trajectory["20d"]],
          ["60D", trajectory["60d"]],
          ["120D", trajectory["120d"]]
        ].filter(function (item) { return Number.isFinite(item[1]); });
        return values.length ? values.map(function (item) { return item[0] + " " + Charts.formatPercent(item[1]); }).join(" / ") : "unknown";
      } },
      { label: "industrial", value: "industrial_validation", format: Charts.formatPercent, numeric: true },
      { label: "market", value: "market_confirmation", format: Charts.formatPercent, numeric: true },
      { label: "crowding", value: "crowding", format: Charts.formatPercent, numeric: true },
      { label: "valuation risk", value: "valuation_risk", format: Charts.formatPercent, numeric: true },
      { label: "PE/VC prior", value: "pevc_prior", format: Charts.formatPercent, numeric: true },
      { label: "PE/VC thesis", value: "pevc_thesis_id" },
      { label: "thesis version", value: "pevc_thesis_version" },
      { label: "supply chain", value: function (row) { return Array.isArray(row.supply_chain_roles) && row.supply_chain_roles.length ? row.supply_chain_roles.join(", ") : "unknown"; } },
      { label: "thesis review", value: function (row) {
        var review = row.thesis_review;
        if (!review || typeof review !== "object") return "unknown";
        return [review.status, review.review_by].filter(Boolean).join(" / ") || "unknown";
      } },
      { label: "members", value: "member_count", numeric: true },
      { label: "risk / blockers", value: function (row) {
        return (row.risk_flags || []).concat(row.prequalification_blockers || []).join(", ") || "none";
      } },
      { label: "NAV weight", value: "nav_weight", format: Charts.formatPercent, numeric: true }
    ], "Theme v2 observer 尚无可展示状态");
    var factorProtocol = contract.factor_protocol || {
      status: "blocked",
      blockers: ["factor_protocol_v2_missing"],
      readback_verified: false
    };
    renderTable("factorProtocolSummary", [factorProtocol], [
      { label: "status", value: "status" },
      { label: "protocol hash", value: "protocol_hash" },
      { label: "evidence", value: "evidence_status" },
      { label: "transition", value: function (row) { return row.transition_id || "none"; } },
      { label: "rollback", value: "rollback_status" },
      { label: "canonical producer", value: function (row) { return row.canonical_producer_available === true && row.canonical_production_apply_eligible === true ? "available" : "blocked"; } },
      { label: "readback", value: function (row) { return row.readback_verified === true ? "verified" : "blocked"; } },
      { label: "blockers", value: function (row) { return (row.blockers || []).join(", ") || "none"; } }
    ], "Factor protocol v2 blocked");
    renderTable("factorStateTable", contract.factors || [], [
      { label: "factor", value: "factor_id" },
      { label: "slot", value: "slot" },
      { label: "family", value: "family" },
      { label: "status", value: "status" },
      { label: "weight", value: "weight", format: Charts.formatPercent, numeric: true },
      { label: "health window", value: "health_window" },
      { label: "health", value: "health_status" },
      { label: "challenger", value: "challenger" },
      { label: "last transition", value: "last_transition" }
    ], "Factor protocol readback 不可用或 governance_blocked");
  }

  function renderHoldingsDiscipline(containerId, rows, limit) {
    renderTable(containerId, (rows || []).slice(0, limit || 20), [
      { label: "ticker", value: "ticker" },
      { label: "name", value: "name" },
      { label: "NAV weight", value: "nav_weight", format: Charts.formatPercent, numeric: true },
      { label: "sleeve weight", value: "equity_sleeve_weight", format: Charts.formatPercent, numeric: true },
      { label: "daily return", value: "daily_return", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "NAV contribution", value: "contribution", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "action", value: "recommended_action" },
      { label: "stop", value: "stop_loss", numeric: true },
      { label: "take profit", value: "take_profit", numeric: true },
      { label: "quote age(s)", value: "quote_age_seconds", numeric: true },
      { label: "risk", value: "risk_status" }
    ], "无当前持仓纪律数据");
  }

  function renderDashboard(metrics, view) {
    var performance = metrics.performance;
    var comparison = metrics.benchmarkComparison || {};
    var holdings = metrics.holdings;
    var attribution = metrics.attribution;
    var trades = metrics.trades;
    var filters = metrics.filters || {};
    var tableView = (view && view.tableExpanded) || {};
    var workspace = (view && view.activeWorkspace) || "overview";
    renderContractPanels(metrics);

    if (workspace === "overview") {
      renderPerformanceStory(metrics);
      renderOverviewFocus(metrics, view || { overviewLens: "nav" });
      renderKpis(performance.kpis || {});
      renderHoldingsDiscipline("overviewHoldingsTable", holdings.allCurrent, 8);
      Charts.lineChart("navChart", selectedBenchmarkSeries(comparison, {
        includePortfolio: filters.showPortfolioCurve !== false
      }), { yFormatter: Charts.formatNumber, empty: "NAV 数据不足或未选择 benchmark", endLabels: true });
      if (filters.showExcessCurve === false) Charts.renderEmpty("excessChart", "已隐藏超额曲线");
      else Charts.lineChart("excessChart", [
        { name: "相对主基准超额净值", color: "#2563eb", points: performance.excessSeries, width: 2.2 }
      ], { yFormatter: Charts.formatNumber, empty: "超额收益数据不足", endLabels: true });
      Charts.lineChart("drawdownChart", selectedBenchmarkSeries(comparison, {
        includePortfolio: filters.showPortfolioCurve !== false,
        drawdown: true
      }), { yFormatter: Charts.formatPercent, includeZero: true, empty: "回撤数据不足", areaToZero: true, endLabels: true });
      var overviewRows = (comparison.portfolioRow ? [comparison.portfolioRow] : []).concat((comparison.comparisonRows || []).filter(function (row) { return row.selected; }));
      renderBenchmarkTable("overviewBenchmarkTable", overviewRows, "未识别到 benchmark NAV 字段");
      return;
    }

    if (workspace === "theme" || workspace === "factor") return;

    if (workspace === "audit") {
      Charts.lineChart("benchmarkNavComparisonChart", selectedBenchmarkSeries(comparison, {
        includePortfolio: filters.showPortfolioCurve !== false
      }), { yFormatter: Charts.formatNumber, empty: "未选择 benchmark 或 NAV 数据不足", endLabels: true });
      Charts.lineChart("benchmarkDrawdownComparisonChart", selectedBenchmarkSeries(comparison, {
        includePortfolio: filters.showPortfolioCurve !== false,
        drawdown: true
      }), { yFormatter: Charts.formatPercent, includeZero: true, empty: "未选择 benchmark 或回撤数据不足", areaToZero: true, endLabels: true });
      Charts.scatter("benchmarkRiskReturnScatter", (comparison.scatterRows || []).map(function (row, index) {
        return { label: row.name, x: row.annualizedVolatility, y: row.annualizedReturn, isPortfolio: row.isPortfolio, isMain: row.isMain, color: row.isPortfolio ? "#101828" : benchmarkColor(index, row.isMain) };
      }), { xLabel: "Ann. Volatility", yLabel: "Annualized Return", xFormatter: Charts.formatPercent, yFormatter: Charts.formatPercent, height: 260 });
      renderCorrelationHeatmap("benchmarkCorrelationHeatmap", comparison.correlationMatrix);
      renderBenchmarkTable("benchmarkExcessTable", sortedBenchmarkRows(comparison.comparisonRows || [], view || {}), "未识别到 benchmark NAV 字段");
      Charts.heatmap("monthlyHeatmap", performance.monthly);
      Charts.barChart("monthlyBarChart", performance.monthly.map(function (row) { return { label: row.month, value: row.value }; }), { horizontal: false, valueFormatter: Charts.formatPercent, empty: "暂无月度收益数据", height: 210 });
      renderBestWorst(performance.monthly);
      return;
    }

    if (workspace === "holdings") {
      Charts.lineChart("rollingVolChart", [{ name: "rolling_20d_volatility", color: "#1e5b99", points: performance.rolling20Vol }], { yFormatter: Charts.formatPercent, empty: "NAV 日收益不足 20 个交易日" });
      Charts.lineChart("rollingBetaChart", [{ name: "rolling_60d_beta", color: "#7a5cbd", points: performance.rolling60Beta }], { yFormatter: function (v) { return Number.isFinite(v) ? v.toFixed(2) : "-"; }, empty: "NAV/benchmark 日收益不足 60 个交易日" });
      Charts.barChart("currentThemeWeightChart", holdings.currentThemeWeight, { valueFormatter: Charts.formatPercent, empty: "无当前 theme weight" });
      Charts.barChart("topHoldingsWeightChart", holdings.top10.map(function (row) { return { label: row.ticker + " " + row.name, value: row.nav_weight }; }), { valueFormatter: Charts.formatPercent, empty: "无当前持仓" });
      Charts.lineChart("concentrationChart", [
        { name: "top5_nav_weight", color: "#1e5b99", points: holdings.concentrationTrend.map(function (row) { return { date: row.date, dateObj: row.dateObj, value: row.top5 }; }) },
        { name: "top10_nav_weight", color: "#0f8a8a", points: holdings.concentrationTrend.map(function (row) { return { date: row.date, dateObj: row.dateObj, value: row.top10 }; }) },
        { name: "HHI", color: "#c77819", points: holdings.concentrationTrend.map(function (row) { return { date: row.date, dateObj: row.dateObj, value: row.hhi }; }) }
      ], { yFormatter: Charts.formatPercent, includeZero: true, empty: "无集中度趋势" });
      renderHoldingsDiscipline("holdingsTable", tableView.holdings ? holdings.allCurrent : holdings.top20, tableView.holdings ? holdings.allCurrent.length : 20);
      renderToggleButton("toggleHoldingsRows", Boolean(tableView.holdings), (holdings.allCurrent || []).length, 20);
      return;
    }

    renderTradeSummary(trades);
    var tradeRows = tableView.trades ? trades.all : trades.recent;
    renderTable("tradesTable", tradeRows, [
      { label: "trade_date", value: "trade_date" },
      { label: "ticker", value: "ticker" },
      { label: "name", value: "name" },
      { label: "side", value: "side" },
      { label: "price", value: "price", format: function (v) { return Number.isFinite(v) ? v.toFixed(2) : "-"; }, numeric: true },
      { label: "quantity", value: "quantity", numeric: true },
      { label: "trade_amount", value: "trade_amount", format: Charts.formatMoney, numeric: true },
      { label: "fee", value: "fee", format: Charts.formatMoney, numeric: true },
      { label: "fee_source", value: "fee_source" },
      { label: "decision", value: "decision_id" },
      { label: "order", value: "order_id" },
      { label: "fill", value: "fill_id" },
      { label: "ledger_delta", value: "ledger_delta", format: Charts.formatMoney, numeric: true },
      { label: "reason", value: "reason" },
      { label: "theme", value: "theme" }
    ], "未上传交易数据");
    renderToggleButton("toggleTradesRows", Boolean(tableView.trades), (trades.all || []).length, 20);
    var closedTradeRows = tableView.closedTrades ? trades.closed : trades.closedRecent;
    renderTable("closedTradesTable", closedTradeRows, [
      { label: "sell_date", value: "trade_date" },
      { label: "ticker", value: "ticker" },
      { label: "name", value: "name" },
	      { label: "matched_qty", value: "matched_quantity", numeric: true },
	      { label: "unmatched_qty", value: "unmatched_quantity", numeric: true },
	      { label: "opening_lot_qty", value: "opening_matched_quantity", numeric: true },
	      { label: "avg_buy_price", value: "avg_buy_price", format: function (v) { return Number.isFinite(v) ? v.toFixed(2) : "-"; }, numeric: true },
	      { label: "sell_price", value: "sell_price", format: function (v) { return Number.isFinite(v) ? v.toFixed(2) : "-"; }, numeric: true },
      { label: "gross_pnl", value: "gross_realized_pnl", format: Charts.formatMoney, numeric: true, signed: true },
      { label: "gross_return", value: "gross_realized_return", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "net_pnl", value: "net_realized_pnl", format: Charts.formatMoney, numeric: true, signed: true },
      { label: "net_return", value: "net_realized_return", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "holding_days", value: "holding_days", format: function (v) { return Number.isFinite(v) ? v.toFixed(1) : "-"; }, numeric: true },
      { label: "fee_known", value: function (row) { return row.fee_known === true ? "yes" : "no"; } },
      { label: "total_fee", value: "total_fee", format: Charts.formatMoney, numeric: true },
      { label: "warning", value: "warning" }
    ], "暂无 FIFO 已平仓交易；需要至少一笔买入和后续卖出。");
    renderToggleButton("toggleClosedTradesRows", Boolean(tableView.closedTrades), (trades.closed || []).length, 20);
    Charts.barChart("sideChart", trades.bySide, { valueFormatter: function (v) { return String(v); }, empty: "未上传交易数据" });
    Charts.barChart("tradeThemeChart", trades.byTheme, { valueFormatter: Charts.formatMoney, empty: "未上传交易数据" });
    Charts.barChart("reasonChart", trades.byReason, { valueFormatter: Charts.formatMoney, empty: "未上传交易数据或 reason 字段" });
  }

  function metricRows(metrics) {
    var kpis = metrics.performance.kpis || {};
    var rows = [
      ["metric", "value"],
      ["start_date", kpis.start_date || ""],
      ["end_date", kpis.end_date || ""],
      ["trading_days", kpis.trading_days || 0],
      ["total_return", kpis.total_return],
      ["annualized_return", kpis.annualized_return],
      ["benchmark_total_return", kpis.benchmark_total_return],
      ["excess_return", kpis.excess_return],
      ["max_drawdown", kpis.max_drawdown],
      ["drawdown_duration", kpis.drawdown_duration],
      ["annualized_volatility", kpis.annualized_volatility],
      ["sharpe_ratio_rf_0", kpis.sharpe_ratio],
      ["daily_nav_win_rate", kpis.win_rate],
      ["fee_adjusted_total_return", kpis.fee_adjusted_total_return],
      ["fee_drag_on_nav", kpis.fee_drag_on_nav],
      ["fee_drag_on_profit", kpis.fee_drag_on_profit],
      ["turnover_ratio", kpis.turnover_ratio],
      ["average_portfolio_market_value", kpis.average_portfolio_market_value],
      ["top5_weight", metrics.holdings.top5Weight],
      ["top10_weight", metrics.holdings.top10Weight],
      ["herfindahl_index", metrics.holdings.hhi],
      ["trade_count", metrics.trades.totals.trade_count || 0],
      ["trade_amount", metrics.trades.totals.trade_amount],
      ["trade_fee", metrics.trades.totals.fee],
      ["trade_fee_unknown_count", metrics.trades.totals.fee_unknown_count || 0],
      ["closed_trade_count", metrics.trades.totals.closed_trade_count || 0],
      ["gross_realized_pnl_fifo", metrics.trades.totals.gross_realized_pnl],
      ["gross_trade_win_rate_fifo", metrics.trades.totals.gross_trade_win_rate],
      ["gross_profit_factor_fifo", metrics.trades.totals.gross_profit_factor],
      ["net_realized_pnl_fifo", metrics.trades.totals.net_realized_pnl],
      ["net_trade_win_rate_fifo", metrics.trades.totals.trade_win_rate],
      ["net_profit_factor_fifo", metrics.trades.totals.profit_factor],
      ["avg_win_fifo", metrics.trades.totals.avg_win],
	      ["avg_loss_fifo", metrics.trades.totals.avg_loss],
	      ["max_trade_profit_fifo", metrics.trades.totals.max_trade_profit],
	      ["max_trade_loss_fifo", metrics.trades.totals.max_trade_loss],
	      ["avg_holding_days_fifo", metrics.trades.totals.avg_holding_days],
	      ["unmatched_sell_count", metrics.trades.totals.unmatched_sell_count || 0],
	      ["unmatched_quantity", metrics.trades.totals.unmatched_quantity || 0],
	      ["opening_lot_count", metrics.trades.totals.opening_lot_count || 0],
	      ["opening_lot_quantity", metrics.trades.totals.opening_lot_quantity || 0],
	      ["opening_lot_cost_basis", metrics.trades.totals.opening_lot_cost_basis || 0],
	      ["opening_matched_quantity", metrics.trades.totals.opening_matched_quantity || 0],
	      ["closed_with_opening_lot_count", metrics.trades.totals.closed_with_opening_lot_count || 0]
	    ];
    rows.push([]);
    rows.push(["benchmark", "total_return", "annualized_return", "ann_volatility", "max_drawdown", "correlation", "beta", "tracking_error", "information_ratio", "excess_return"]);
    ((metrics.benchmarkComparison && metrics.benchmarkComparison.comparisonRows) || []).forEach(function (row) {
      rows.push([
        row.name,
        row.totalReturn,
        row.annualizedReturn,
        row.annualizedVolatility,
        row.maxDrawdown,
        row.correlation,
        row.beta,
        row.trackingError,
        row.informationRatio,
        row.excessReturn
      ]);
    });
    return rows;
  }

  window.DashboardUI = {
    renderDashboard: renderDashboard,
    renderMessages: renderMessages,
    renderTable: renderTable,
    metricRows: metricRows,
    escapeHtml: escapeHtml
  };
})();

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
    var messages = [];
    (errors || []).forEach(function (text) { messages.push({ type: "error", text: text }); });
    (warnings || []).slice(0, 8).forEach(function (text) { messages.push({ type: "warning", text: text }); });
    (infos || []).forEach(function (text) { messages.push({ type: "info", text: text }); });
    panel.innerHTML = "";
    panel.classList.toggle("visible", messages.length > 0);
    messages.forEach(function (message) {
      var div = document.createElement("div");
      div.className = "message " + message.type;
      div.textContent = message.text;
      panel.appendChild(div);
    });
  }

  function renderKpis(kpis) {
    function fmtPercent(value) {
      return Charts.formatPercent(value);
    }
    function fmtRatio(value) {
      return Number.isFinite(value) ? value.toFixed(2) : "-";
    }
    function fmtDays(value) {
      return Number.isFinite(value) ? String(value) + " 天" : "-";
    }
    var items = [
      { label: "累计收益", field: "total_return", format: fmtPercent, subLabel: "年化收益 (CAGR)", subField: "annualized_return", subFormat: fmtPercent },
      { label: "基准累计收益", field: "benchmark_total_return", format: fmtPercent, subLabel: "基准年化收益", subField: "benchmark_annualized_return", subFormat: fmtPercent },
      { label: "超额收益", field: "excess_return", format: fmtPercent, subLabel: "年化超额收益", subField: "annualized_excess_return", subFormat: fmtPercent },
      { label: "年化波动率", field: "annualized_volatility", format: fmtPercent, subLabel: "基准波动率", subField: "benchmark_annualized_volatility", subFormat: fmtPercent, neutral: true },
      { label: "Sharpe Ratio", field: "sharpe_ratio", format: fmtRatio, subLabel: "Calmar Ratio", subField: "calmar_ratio", subFormat: fmtRatio, neutral: true },
      { label: "最大回撤", field: "max_drawdown", format: fmtPercent, subLabel: "最长回撤持续", subField: "drawdown_duration", subFormat: fmtDays },
      { label: "日胜率", field: "win_rate", format: fmtPercent, subLabel: "盈利日 / 总交易日", subValue: function () { return (kpis.win_count || 0) + " / " + (kpis.trading_days || 0); }, neutral: true },
      { label: "交易日数量", field: "trading_days", format: function (v) { return Number.isFinite(v) ? String(v) : "-"; }, subLabel: "筛选区间", subValue: function () { return (kpis.start_date || "-") + " 至 " + (kpis.end_date || "-"); }, neutral: true }
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
        html += '<td class="' + className + '">' + escapeHtml(text) + "</td>";
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    container.innerHTML = html;
  }

  function renderBestWorst(monthly) {
    var container = document.getElementById("bestWorstMonths");
    if (!container) return;
    if (!monthly || !monthly.length) {
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
    var cards = [
      ["买入数量", trades.totals.buy_count],
      ["卖出数量", trades.totals.sell_count],
      ["总成交金额", Charts.formatMoney(trades.totals.trade_amount)],
      ["总费用", Charts.formatMoney(trades.totals.fee)],
      ["Post-trade return", trades.message]
    ];
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

  function renderDashboard(metrics, view) {
    var performance = metrics.performance;
    var comparison = metrics.benchmarkComparison || {};
    var holdings = metrics.holdings;
    var attribution = metrics.attribution;
    var trades = metrics.trades;
    var filters = metrics.filters || {};

    renderPerformanceStory(metrics);
    renderOverviewFocus(metrics, view || { overviewLens: "nav" });
    renderKpis(performance.kpis || {});
    Charts.lineChart("navChart", selectedBenchmarkSeries(comparison, {
      includePortfolio: filters.showPortfolioCurve !== false
    }), { yFormatter: Charts.formatNumber, empty: "NAV 数据不足或未选择 benchmark", endLabels: true });
    if (filters.showExcessCurve === false) {
      Charts.renderEmpty("excessChart", "已隐藏超额曲线");
    } else {
      Charts.lineChart("excessChart", [
        { name: "相对主基准超额净值", color: "#2563eb", points: performance.excessSeries, width: 2.2 }
      ], { yFormatter: Charts.formatNumber, empty: "超额收益数据不足", endLabels: true });
    }
    Charts.lineChart("drawdownChart", selectedBenchmarkSeries(comparison, {
      includePortfolio: filters.showPortfolioCurve !== false,
      drawdown: true
    }), { yFormatter: Charts.formatPercent, includeZero: true, empty: "回撤数据不足", areaToZero: true, endLabels: true });
    var overviewBenchmarkRows = (comparison.portfolioRow ? [comparison.portfolioRow] : []).concat((comparison.comparisonRows || []).filter(function (row) {
      return row.selected;
    }));
    renderBenchmarkTable("overviewBenchmarkTable", overviewBenchmarkRows, "未识别到 benchmark NAV 字段");

    Charts.lineChart("benchmarkNavComparisonChart", selectedBenchmarkSeries(comparison, {
      includePortfolio: filters.showPortfolioCurve !== false
    }), { yFormatter: Charts.formatNumber, empty: "未选择 benchmark 或 NAV 数据不足", endLabels: true });
    Charts.lineChart("benchmarkDrawdownComparisonChart", selectedBenchmarkSeries(comparison, {
      includePortfolio: filters.showPortfolioCurve !== false,
      drawdown: true
    }), { yFormatter: Charts.formatPercent, includeZero: true, empty: "未选择 benchmark 或回撤数据不足", areaToZero: true, endLabels: true });
    Charts.scatter("benchmarkRiskReturnScatter", (comparison.scatterRows || []).map(function (row, index) {
      return {
        label: row.name,
        x: row.annualizedVolatility,
        y: row.annualizedReturn,
        isPortfolio: row.isPortfolio,
        isMain: row.isMain,
        color: row.isPortfolio ? "#101828" : benchmarkColor(index, row.isMain)
      };
    }), {
      xLabel: "Ann. Volatility",
      yLabel: "Annualized Return",
      xFormatter: Charts.formatPercent,
      yFormatter: Charts.formatPercent,
      height: 260
    });
    renderCorrelationHeatmap("benchmarkCorrelationHeatmap", comparison.correlationMatrix);
    renderBenchmarkTable("benchmarkExcessTable", sortedBenchmarkRows(comparison.comparisonRows || [], view || {}), "未识别到 benchmark NAV 字段");

    Charts.heatmap("monthlyHeatmap", performance.monthly);
    Charts.barChart("monthlyBarChart", performance.monthly.map(function (row) {
      return { label: row.month, value: row.value };
    }), { horizontal: false, valueFormatter: Charts.formatPercent, empty: "暂无月度收益数据", height: 210 });
    renderBestWorst(performance.monthly);

    Charts.barChart("topContributionChart", attribution.top.map(function (row) {
      return { label: row.label, value: row.value };
    }), { valueFormatter: Charts.formatPercent, empty: "缺少 contribution 或 daily_return" });
    Charts.barChart("bottomContributionChart", attribution.bottom.map(function (row) {
      return { label: row.label, value: row.value };
    }), { valueFormatter: Charts.formatPercent, empty: "缺少 contribution 或 daily_return" });
    Charts.barChart("themeContributionChart", attribution.themeContribution, { valueFormatter: Charts.formatPercent, empty: "缺少 theme contribution" });
    Charts.barChart("sectorContributionChart", attribution.sectorContribution, { valueFormatter: Charts.formatPercent, empty: "positions.csv 未提供 sector 或 contribution" });
    Charts.scatter("scatterChart", attribution.scatter.map(function (row) {
      return { label: row.label, x: row.avg_weight, y: row.value };
    }));

    Charts.lineChart("rollingVolChart", [
      { name: "rolling_20d_volatility", color: "#1e5b99", points: performance.rolling20Vol }
    ], { yFormatter: Charts.formatPercent, empty: "NAV 日收益不足 20 个交易日" });
    Charts.lineChart("rollingBetaChart", [
      { name: "rolling_60d_beta", color: "#7a5cbd", points: performance.rolling60Beta }
    ], { yFormatter: function (v) { return Number.isFinite(v) ? v.toFixed(2) : "-"; }, empty: "NAV/benchmark 日收益不足 60 个交易日" });
    Charts.barChart("currentThemeWeightChart", holdings.currentThemeWeight, { valueFormatter: Charts.formatPercent, empty: "无当前 theme weight" });
    Charts.barChart("topHoldingsWeightChart", holdings.top10.map(function (row) {
      return { label: row.ticker + " " + row.name, value: row.weight };
    }), { valueFormatter: Charts.formatPercent, empty: "无当前持仓" });
    Charts.lineChart("concentrationChart", [
      { name: "top5_weight", color: "#1e5b99", points: holdings.concentrationTrend.map(function (row) { return { date: row.date, dateObj: row.dateObj, value: row.top5 }; }) },
      { name: "top10_weight", color: "#0f8a8a", points: holdings.concentrationTrend.map(function (row) { return { date: row.date, dateObj: row.dateObj, value: row.top10 }; }) },
      { name: "HHI", color: "#c77819", points: holdings.concentrationTrend.map(function (row) { return { date: row.date, dateObj: row.dateObj, value: row.hhi }; }) }
    ], { yFormatter: Charts.formatPercent, includeZero: true, empty: "无集中度趋势" });

    renderTable("holdingsTable", holdings.top20, [
      { label: "date", value: "date" },
      { label: "ticker", value: "ticker" },
      { label: "name", value: "name" },
      { label: "weight", value: "weight", format: Charts.formatPercent, numeric: true },
      { label: "theme", value: "theme" },
      { label: "sector", value: "sector" },
      { label: "daily_return", value: "daily_return", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "contribution", value: "contribution", format: Charts.formatPercent, numeric: true, signed: true },
      { label: "market_value", value: "market_value", format: Charts.formatMoney, numeric: true }
    ], "无当前持仓数据");

    var themeFields = topFieldsFromTrend(holdings.themeWeightTrend, 7);
    var sectorFields = topFieldsFromTrend(holdings.sectorWeightTrend, 7);
    Charts.lineChart("themeWeightTrendChart", lineSeriesFromTrend(holdings.themeWeightTrend, themeFields), { yFormatter: Charts.formatPercent, includeZero: true, empty: "无 theme weight trend" });
    Charts.lineChart("sectorWeightTrendChart", lineSeriesFromTrend(holdings.sectorWeightTrend, sectorFields), { yFormatter: Charts.formatPercent, includeZero: true, empty: "positions.csv 未提供 sector" });
    Charts.barChart("marketValueThemeChart", holdings.marketValueTheme, { valueFormatter: Charts.formatMoney, empty: "positions.csv 未提供 market_value" });
    renderTable("themeWeightTable", holdings.currentThemeWeight, [
      { label: "theme", value: "label" },
      { label: "weight", value: "value", format: Charts.formatPercent, numeric: true }
    ], "无 theme weight");

    renderTradeSummary(trades);
    renderTable("tradesTable", trades.recent, [
      { label: "trade_date", value: "trade_date" },
      { label: "ticker", value: "ticker" },
      { label: "name", value: "name" },
      { label: "side", value: "side" },
      { label: "price", value: "price", format: function (v) { return Number.isFinite(v) ? v.toFixed(2) : "-"; }, numeric: true },
      { label: "quantity", value: "quantity", numeric: true },
      { label: "trade_amount", value: "trade_amount", format: Charts.formatMoney, numeric: true },
      { label: "fee", value: "fee", format: Charts.formatMoney, numeric: true },
      { label: "reason", value: "reason" },
      { label: "theme", value: "theme" }
    ], "未上传交易数据");
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
      ["win_rate", kpis.win_rate],
      ["top5_weight", metrics.holdings.top5Weight],
      ["top10_weight", metrics.holdings.top10Weight],
      ["herfindahl_index", metrics.holdings.hhi],
      ["trade_count", metrics.trades.totals.trade_count || 0],
      ["trade_amount", metrics.trades.totals.trade_amount || 0],
      ["trade_fee", metrics.trades.totals.fee || 0]
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

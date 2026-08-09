(function () {
  "use strict";

  var Contract = window.CNAggressiveDashboardContractV1;
  var Analysis = window.CNAggressiveDashboardAnalysisV1;
  var PublicMode = window.CNPublicDashboard === true;
  var activeBundle = null;
  var activeAnalysis = null;
  var resizeTimer = null;
  var SVG_NS = "http://www.w3.org/2000/svg";

  function byId(id) { return document.getElementById(id); }
  function setText(id, value) {
    var node = byId(id);
    if (node) node.textContent = value === null || value === undefined || value === "" ? "—" : String(value);
  }
  function money(value) {
    var parsed = Number(value);
    return Number.isFinite(parsed) ? "¥" + parsed.toLocaleString("zh-CN", { maximumFractionDigits: 2 }) : "UNKNOWN";
  }
  function compactMoney(value) {
    var parsed = Number(value);
    if (!Number.isFinite(parsed)) return "UNKNOWN";
    if (Math.abs(parsed) >= 100000000) return "¥" + (parsed / 100000000).toFixed(2) + "亿";
    if (Math.abs(parsed) >= 10000) return "¥" + (parsed / 10000).toFixed(1) + "万";
    return money(parsed);
  }
  function number(value) {
    var parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toLocaleString("zh-CN", { maximumFractionDigits: 4 }) : "UNKNOWN";
  }
  function percent(value, signed) {
    var parsed = Number(value);
    if (!Number.isFinite(parsed)) return "UNKNOWN";
    var result = (parsed * 100).toFixed(2) + "%";
    return signed && parsed > 0 ? "+" + result : result;
  }
  function makeCell(row, value, className) {
    var cell = document.createElement("td");
    cell.textContent = value;
    if (className) cell.className = className;
    row.appendChild(cell);
    return cell;
  }
  function metric(list, label, value) {
    var wrapper = document.createElement("div");
    var term = document.createElement("dt");
    var detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = value;
    wrapper.appendChild(term);
    wrapper.appendChild(detail);
    list.appendChild(wrapper);
  }
  function signedClass(value) {
    return Number(value) < 0 ? "negative" : Number(value) > 0 ? "positive" : "";
  }
  function publicRedacted(value) {
    return PublicMode ? "公开版已隐藏" : value;
  }
  function parseDate(value) {
    return Date.parse(value + "T00:00:00Z");
  }
  function shortDate(value) {
    var parts = value.split("-");
    return Number(parts[1]) + "/" + Number(parts[2]);
  }
  function svgNode(name, attributes, textValue) {
    var node = document.createElementNS(SVG_NS, name);
    Object.keys(attributes || {}).forEach(function (key) {
      node.setAttribute(key, attributes[key]);
    });
    if (textValue !== undefined) node.textContent = textValue;
    return node;
  }
  function linear(domainMin, domainMax, rangeMin, rangeMax) {
    var span = domainMax - domainMin || 1;
    return function (value) {
      return rangeMin + (value - domainMin) / span * (rangeMax - rangeMin);
    };
  }
  function pathFor(points, xValue, yValue) {
    return points.map(function (point, index) {
      return (index ? "L" : "M") + xValue(point).toFixed(2) + "," + yValue(point).toFixed(2);
    }).join(" ");
  }
  function areaPath(points, xValue, yValue, baseline) {
    if (!points.length) return "";
    var firstX = xValue(points[0]);
    var lastX = xValue(points[points.length - 1]);
    return "M" + firstX.toFixed(2) + "," + baseline.toFixed(2) + " " +
      pathFor(points, xValue, yValue).replace(/^M/, "L") +
      " L" + lastX.toFixed(2) + "," + baseline.toFixed(2) + " Z";
  }
  function numericTicks(minimum, maximum, count) {
    var ticks = [];
    var step = (maximum - minimum) / Math.max(1, count - 1);
    for (var index = 0; index < count; index += 1) ticks.push(minimum + step * index);
    return ticks;
  }
  function dateTicks(points, count) {
    var chosen = [];
    var seen = {};
    for (var index = 0; index < count; index += 1) {
      var pointIndex = Math.round(index * (points.length - 1) / Math.max(1, count - 1));
      var point = points[pointIndex];
      if (point && !seen[point.date]) {
        chosen.push(point);
        seen[point.date] = true;
      }
    }
    return chosen;
  }
  function spreadLabels(labels, minimum, maximum, gap) {
    var ordered = labels.slice().sort(function (left, right) { return left.y - right.y; });
    ordered.forEach(function (label, index) {
      label.labelY = Math.max(label.y, index === 0 ? minimum : ordered[index - 1].labelY + gap);
    });
    var overflow = ordered.length ? ordered[ordered.length - 1].labelY - maximum : 0;
    if (overflow > 0) ordered.forEach(function (label) { label.labelY -= overflow; });
    for (var index = ordered.length - 2; index >= 0; index -= 1) {
      ordered[index].labelY = Math.min(ordered[index].labelY, ordered[index + 1].labelY - gap);
    }
    return ordered;
  }
  function appendAxes(svg, dimensions, xScale, yScale, points, yMin, yMax, yFormatter) {
    var left = dimensions.left;
    var right = dimensions.width - dimensions.right;
    var top = dimensions.top;
    var bottom = dimensions.height - dimensions.bottom;
    numericTicks(yMin, yMax, 5).forEach(function (tick) {
      var y = yScale(tick);
      svg.appendChild(svgNode("line", { x1: left, x2: right, y1: y, y2: y, class: "chart-grid" }));
      svg.appendChild(svgNode("text", { x: left - 10, y: y + 4, "text-anchor": "end", class: "chart-label" }, yFormatter(tick)));
    });
    dateTicks(points, dimensions.width < 560 ? 4 : 6).forEach(function (point, index, values) {
      var x = xScale(parseDate(point.date));
      svg.appendChild(svgNode("line", { x1: x, x2: x, y1: bottom, y2: bottom + 5, class: "chart-axis" }));
      svg.appendChild(svgNode("text", {
        x: x,
        y: bottom + 22,
        "text-anchor": index === 0 ? "start" : index === values.length - 1 ? "end" : "middle",
        class: "chart-label"
      }, shortDate(point.date)));
    });
    svg.appendChild(svgNode("line", { x1: left, x2: right, y1: bottom, y2: bottom, class: "chart-axis" }));
    svg.appendChild(svgNode("text", { x: left, y: top - 7, class: "chart-axis-title" }, yFormatter === formatIndex ? "起点 = 100" : "回撤 %"));
  }
  function formatIndex(value) { return value.toFixed(0); }
  function formatDrawdown(value) { return value.toFixed(0) + "%"; }

  function renderGrowthChart(bundle, analysis) {
    var host = byId("growthChart");
    host.replaceChildren();
    var points = analysis.points;
    if (points.length < 2) {
      var empty = document.createElement("div");
      empty.className = "chart-empty";
      empty.textContent = "有效历史估值点不足，无法绘制收益曲线";
      host.appendChild(empty);
      return;
    }
    var width = Math.max(320, Math.round(host.getBoundingClientRect().width || 900));
    var height = width < 560 ? 300 : 360;
    var dimensions = { width: width, height: height, left: width < 560 ? 52 : 60, right: width < 560 ? 58 : 86, top: 30, bottom: 40 };
    var start = parseDate(points[0].date);
    var end = parseDate(points[points.length - 1].date);
    var series = [
      { key: "portfolio", label: "组合", values: points.map(function (point) { return { date: point.date, value: point.portfolio_unit_nav * 100 }; }) },
      { key: "benchmark", label: "沪深300", values: points.map(function (point) { return { date: point.date, value: point.csi300_nav * 100 }; }) },
      { key: "excess", label: "累计超额", values: points.map(function (point) { return { date: point.date, value: 100 + point.cumulative_excess_return * 100 }; }) }
    ];
    var allValues = [];
    series.forEach(function (item) { item.values.forEach(function (point) { allValues.push(point.value); }); });
    var minimum = Math.min.apply(null, allValues.concat([100]));
    var maximum = Math.max.apply(null, allValues.concat([100]));
    var padding = Math.max(3, (maximum - minimum) * 0.09);
    var yMin = minimum - padding;
    var yMax = maximum + padding;
    var xScale = linear(start, end, dimensions.left, width - dimensions.right);
    var yScale = linear(yMin, yMax, height - dimensions.bottom, dimensions.top);
    var svg = svgNode("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height, "aria-hidden": "true" });
    svg.appendChild(svgNode("title", {}, "组合、沪深300与累计超额历史曲线"));
    svg.appendChild(svgNode("desc", {}, "从 " + points[0].date + " 到 " + points[points.length - 1].date + "，共同起点归一化为100。"));
    appendAxes(svg, dimensions, xScale, yScale, points, yMin, yMax, formatIndex);

    var portfolioValues = series[0].values;
    svg.appendChild(svgNode("path", {
      d: areaPath(portfolioValues, function (point) { return xScale(parseDate(point.date)); }, function (point) { return yScale(point.value); }, yScale(100)),
      class: "chart-area portfolio"
    }));

    (bundle.history.funding_events || []).forEach(function (event) {
      var eventTime = parseDate(event.date);
      if (eventTime < start || eventTime > end) return;
      var x = xScale(eventTime);
      svg.appendChild(svgNode("line", { x1: x, x2: x, y1: dimensions.top, y2: height - dimensions.bottom, class: "chart-funding-line" }));
      var fundingOnRight = x > width * 0.68;
      svg.appendChild(svgNode("text", {
        x: x + (fundingOnRight ? -5 : 5),
        y: dimensions.top + 12,
        "text-anchor": fundingOnRight ? "end" : "start",
        class: "chart-funding-label"
      }, PublicMode ? event.date.slice(5) + " 外部资金流" : event.date.slice(5) + " 入金 " + compactMoney(event.amount)));
    });

    series.forEach(function (item) {
      svg.appendChild(svgNode("path", {
        d: pathFor(item.values, function (point) { return xScale(parseDate(point.date)); }, function (point) { return yScale(point.value); }),
        class: "chart-line " + item.key
      }));
    });

    var labels = series.map(function (item) {
      var last = item.values[item.values.length - 1];
      return { key: item.key, label: item.label, value: last.value, x: xScale(parseDate(last.date)), y: yScale(last.value) };
    });
    spreadLabels(labels, dimensions.top + 8, height - dimensions.bottom - 8, 17).forEach(function (label) {
      svg.appendChild(svgNode("circle", { cx: label.x, cy: label.y, r: 4, class: "chart-end-dot " + label.key }));
      if (width < 560) return;
      svg.appendChild(svgNode("line", { x1: label.x + 5, x2: width - 96, y1: label.y, y2: label.labelY, class: "chart-axis" }));
      svg.appendChild(svgNode("text", { x: width - 5, y: label.labelY + 4, "text-anchor": "end", class: "chart-direct-label" }, label.label + " " + label.value.toFixed(1)));
    });
    host.appendChild(svg);
  }

  function renderDrawdownChart(analysis) {
    var host = byId("drawdownChart");
    host.replaceChildren();
    var portfolio = analysis.portfolio_drawdown;
    var benchmark = analysis.benchmark_drawdown;
    if (portfolio.length < 2) {
      var empty = document.createElement("div");
      empty.className = "chart-empty";
      empty.textContent = "有效历史估值点不足，无法绘制回撤";
      host.appendChild(empty);
      return;
    }
    var width = Math.max(320, Math.round(host.getBoundingClientRect().width || 900));
    var height = width < 560 ? 180 : 200;
    var dimensions = { width: width, height: height, left: width < 560 ? 52 : 60, right: 18, top: 18, bottom: 36 };
    var start = parseDate(portfolio[0].date);
    var end = parseDate(portfolio[portfolio.length - 1].date);
    var minimum = Math.min.apply(null, portfolio.concat(benchmark).map(function (point) { return point.value * 100; }));
    var yMin = Math.min(-2, minimum * 1.15);
    var xScale = linear(start, end, dimensions.left, width - dimensions.right);
    var yScale = linear(yMin, 0, height - dimensions.bottom, dimensions.top);
    var portfolioValues = portfolio.map(function (point) { return { date: point.date, value: point.value * 100 }; });
    var benchmarkValues = benchmark.map(function (point) { return { date: point.date, value: point.value * 100 }; });
    var svg = svgNode("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height, "aria-hidden": "true" });
    svg.appendChild(svgNode("title", {}, "组合与沪深300历史回撤"));
    svg.appendChild(svgNode("desc", {}, "回撤从各自历史高点计算，数值越低表示距离前高越远。"));
    appendAxes(svg, dimensions, xScale, yScale, activeAnalysis.points, yMin, 0, formatDrawdown);
    svg.appendChild(svgNode("path", {
      d: areaPath(benchmarkValues, function (point) { return xScale(parseDate(point.date)); }, function (point) { return yScale(point.value); }, yScale(0)),
      class: "chart-area drawdown-benchmark"
    }));
    svg.appendChild(svgNode("path", {
      d: areaPath(portfolioValues, function (point) { return xScale(parseDate(point.date)); }, function (point) { return yScale(point.value); }, yScale(0)),
      class: "chart-area drawdown-portfolio"
    }));
    svg.appendChild(svgNode("path", { d: pathFor(benchmarkValues, function (point) { return xScale(parseDate(point.date)); }, function (point) { return yScale(point.value); }), class: "chart-line benchmark" }));
    svg.appendChild(svgNode("path", { d: pathFor(portfolioValues, function (point) { return xScale(parseDate(point.date)); }, function (point) { return yScale(point.value); }), class: "chart-line portfolio" }));
    [
      { item: analysis.deepest_portfolio_drawdown, label: "组合", key: "portfolio" },
      { item: analysis.deepest_benchmark_drawdown, label: "沪深300", key: "benchmark" }
    ].forEach(function (entry, index) {
      if (!entry.item) return;
      var x = xScale(parseDate(entry.item.date));
      var y = yScale(entry.item.value * 100);
      svg.appendChild(svgNode("circle", { cx: x, cy: y, r: 3.5, class: "chart-end-dot " + entry.key }));
      var anchorRight = x > width * 0.72;
      svg.appendChild(svgNode("text", {
        x: x + (anchorRight ? -6 : 6),
        y: y + (index ? 15 : -7),
        "text-anchor": anchorRight ? "end" : "start",
        class: "chart-direct-label"
      }, entry.label + " " + percent(entry.item.value)));
    });
    host.appendChild(svg);
  }

  function renderCharts() {
    if (!activeBundle || !activeAnalysis) return;
    renderGrowthChart(activeBundle, activeAnalysis);
    renderDrawdownChart(activeAnalysis);
  }

  function renderBlockers(blockers) {
    var panel = byId("blockerPanel");
    var list = byId("blockerList");
    list.replaceChildren();
    panel.hidden = blockers.length === 0;
    blockers.forEach(function (blocker) {
      var item = document.createElement("li");
      item.textContent = blocker;
      list.appendChild(item);
    });
  }

  function renderPositions(bundle) {
    var body = byId("positionRows");
    body.replaceChildren();
    bundle.positions.forEach(function (position) {
      var row = document.createElement("tr");
      makeCell(row, position.symbol + " " + position.name, "identity-cell");
      makeCell(row, publicRedacted(number(position.shares)), "numeric");
      makeCell(row, publicRedacted(money(position.avg_cost)), "numeric");
      makeCell(row, publicRedacted(money(position.recorded_price)) + " · " + position.price_date, "numeric");
      makeCell(row, publicRedacted(money(position.market_value)), "numeric");
      makeCell(row, percent(position.nav_weight), "numeric");
      var weightCell = makeCell(row, "", "weight-cell");
      var weightValue = document.createElement("div");
      weightValue.className = "weight-value";
      weightValue.textContent = percent(position.equity_weight);
      var track = document.createElement("div");
      track.className = "weight-track";
      var bar = document.createElement("span");
      bar.style.width = Math.max(0, Math.min(100, position.equity_weight * 100)) + "%";
      track.appendChild(bar);
      weightCell.appendChild(weightValue);
      weightCell.appendChild(track);
      makeCell(row, publicRedacted(money(position.unrealized_pnl)), "numeric " + (PublicMode ? "" : signedClass(position.unrealized_pnl)));
      makeCell(row, position.thesis_status + " · HASH-BOUND", "evidence-value");
      body.appendChild(row);
    });
  }

  function renderAllocation(bundle) {
    var bar = byId("allocationBar");
    var legend = byId("allocationLegend");
    bar.replaceChildren();
    legend.replaceChildren();
    var items = [{ label: "现金", weight: bundle.portfolio.cash_weight, className: "cash" }];
    bundle.positions.forEach(function (position, index) {
      items.push({
        label: position.symbol + " " + position.name,
        weight: position.nav_weight,
        className: index < 4 ? "position-" + index : "position-other"
      });
    });
    bar.setAttribute("aria-label", items.map(function (item) { return item.label + " " + percent(item.weight); }).join("；"));
    items.forEach(function (item) {
      var segment = document.createElement("span");
      segment.className = "allocation-segment " + item.className;
      segment.style.width = Math.max(0, item.weight * 100) + "%";
      segment.setAttribute("aria-hidden", "true");
      bar.appendChild(segment);
      var label = document.createElement("span");
      var swatch = document.createElement("i");
      swatch.className = item.className;
      label.appendChild(swatch);
      label.appendChild(document.createTextNode(item.label + " " + percent(item.weight)));
      legend.appendChild(label);
    });
  }

  function renderChanges(bundle) {
    setText("changeSubtitle", bundle.previous_valid_record + " → " + bundle.latest_valid_record);
    var body = byId("changeRows");
    body.replaceChildren();
    bundle.changes.forEach(function (change) {
      var row = document.createElement("tr");
      makeCell(row, change.symbol + " " + change.name, "identity-cell");
      makeCell(row, change.change_type, "mono");
      makeCell(row, publicRedacted(number(change.previous_shares)), "numeric");
      makeCell(row, publicRedacted(number(change.current_shares)), "numeric");
      makeCell(row, publicRedacted(number(change.share_delta)), "numeric " + (PublicMode ? "" : signedClass(change.share_delta)));
      makeCell(row, percent(change.nav_weight_delta, true), "numeric " + signedClass(change.nav_weight_delta));
      makeCell(row, percent(change.equity_weight_delta, true), "numeric " + signedClass(change.equity_weight_delta));
      body.appendChild(row);
    });
  }

  function renderMetrics(bundle) {
    var portfolio = bundle.portfolio;
    var benchmark = bundle.benchmarks[0];
    var performance = byId("performanceList");
    performance.replaceChildren();
    metric(performance, "统计区间", portfolio.performance_start_date + " → " + portfolio.performance_end_date);
    metric(performance, "收益方法", "funding-aware TWR");
    metric(performance, "最新有效记录区间收益", percent(portfolio.latest_record_interval_return, true));
    metric(performance, "组合 P&L", publicRedacted(money(portfolio.portfolio_pnl)));
    metric(performance, "当前未实现 P&L", publicRedacted(money(portfolio.current_unrealized_pnl)));
    metric(performance, "最新调仓已实现 P&L", publicRedacted(money(portfolio.latest_record_realized_pnl_from_rebalance)));
    metric(performance, "累计已实现 P&L", "UNKNOWN");
    metric(performance, "费用 / 毛净", portfolio.fee_basis + " / " + portfolio.gross_or_net);
    metric(performance, "沪深300覆盖", benchmark.start_date + " → " + benchmark.end_date + " · " + benchmark.missing_dates.length + " gaps");

    var concentration = byId("concentrationList");
    concentration.replaceChildren();
    metric(concentration, "持仓数", String(bundle.concentration.holding_count));
    metric(concentration, "Top 1 权益权重", percent(bundle.concentration.top1_equity_weight));
    metric(concentration, "Top 3 权益权重", percent(bundle.concentration.top3_equity_weight));
    var top5 = bundle.positions.slice().sort(function (left, right) {
      return right.equity_weight - left.equity_weight;
    }).slice(0, 5).reduce(function (total, position) {
      return total + position.equity_weight;
    }, 0);
    metric(concentration, "Top 5 权益权重", percent(top5));
    metric(concentration, "权益 HHI", number(bundle.concentration.equity_hhi));
    metric(concentration, "现金权重", percent(portfolio.cash_weight));
    metric(concentration, "股票仓位", percent(portfolio.gross_exposure));
    metric(concentration, "行业 / 主题", "UNKNOWN · ledger 未 hash-bind 此口径");
    metric(concentration, "数据新鲜度", bundle.latest_data_date + " · " + bundle.data_age_calendar_days + " calendar days");

    var current = byId("currentStateList");
    current.replaceChildren();
    metric(current, "总资产", publicRedacted(money(portfolio.total_value)));
    metric(current, "现金", publicRedacted(money(portfolio.cash)));
    metric(current, "股票市值", publicRedacted(money(portfolio.market_value)));
    metric(current, "股票仓位", percent(portfolio.gross_exposure));
    metric(current, "组合 P&L", publicRedacted(money(portfolio.portfolio_pnl)));
  }

  function historyEvidenceLabel(value) {
    if (value === "ARCHIVE_INCEPTION_EXACT_BYTES_NO_DECLARED_SHA") return "LEGACY BASELINE · exact bytes";
    if (value === "LEGACY_EXACT_BYTES_NO_DECLARED_SHA") return "LEGACY · exact bytes";
    if (value === "HASH_BOUND_CURRENT_CLOSURE") return "CURRENT · hash-bound";
    return value || "UNKNOWN";
  }

  function renderMonthlyPerformance(analysis) {
    var body = byId("monthlyPerformanceRows");
    body.replaceChildren();
    analysis.monthly.forEach(function (period) {
      var row = document.createElement("tr");
      var periodCell = makeCell(row, "", "period-cell");
      var strong = document.createElement("strong");
      var detail = document.createElement("span");
      strong.textContent = period.period;
      detail.textContent = period.point_count + " 个估值点";
      periodCell.appendChild(strong);
      periodCell.appendChild(detail);
      makeCell(row, period.base_date.slice(5) + " → " + period.end_date.slice(5));
      makeCell(row, percent(period.portfolio_return, true), "numeric " + signedClass(period.portfolio_return));
      makeCell(row, percent(period.benchmark_return, true), "numeric " + signedClass(period.benchmark_return));
      makeCell(row, percent(period.excess_return, true), "numeric " + signedClass(period.excess_return));
      makeCell(row, percent(period.portfolio_max_drawdown), "numeric negative");
      makeCell(row, publicRedacted(money(period.ending_total_value)), "numeric");
      makeCell(row, period.evidence_status, "evidence-value");
      body.appendChild(row);
    });
  }

  function insight(list, label, value, detail, className) {
    var row = document.createElement("div");
    var labelNode = document.createElement("span");
    var valueNode = document.createElement("strong");
    var detailNode = document.createElement("p");
    row.className = "insight-row";
    labelNode.textContent = label;
    valueNode.textContent = value;
    if (className) valueNode.className = className;
    detailNode.textContent = detail;
    row.appendChild(labelNode);
    row.appendChild(valueNode);
    row.appendChild(detailNode);
    list.appendChild(row);
  }

  function renderHistoryAnalysis(bundle, analysis) {
    var benchmark = bundle.benchmarks[0];
    var list = byId("historyInsightList");
    list.replaceChildren();
    insight(list, "累计相对表现", percent(benchmark.excess_return, true), "组合 TWR 相对沪深300的累计收益差。", signedClass(benchmark.excess_return));
    if (analysis.best_period) {
      insight(list, "最佳月份", analysis.best_period.period + " · " + percent(analysis.best_period.portfolio_return, true), "当月超额 " + percent(analysis.best_period.excess_return, true) + "。", signedClass(analysis.best_period.portfolio_return));
    }
    if (analysis.weakest_period) {
      insight(list, "最弱月份", analysis.weakest_period.period + " · " + percent(analysis.weakest_period.portfolio_return, true), "月内最大回撤 " + percent(analysis.weakest_period.portfolio_max_drawdown) + "。", signedClass(analysis.weakest_period.portfolio_return));
    }
    if (analysis.deepest_portfolio_drawdown) {
      insight(list, "最深历史回撤", percent(analysis.deepest_portfolio_drawdown.value) + " · " + analysis.deepest_portfolio_drawdown.date, "从此前组合单位净值高点计算。", "negative");
    }

    var funding = byId("fundingEventList");
    funding.replaceChildren();
    if (!bundle.history.funding_events.length) {
      var empty = document.createElement("p");
      empty.className = "section-note";
      empty.textContent = "无已验证外部资金流事件";
      funding.appendChild(empty);
    }
    bundle.history.funding_events.forEach(function (event) {
      var item = document.createElement("div");
      var title = document.createElement("strong");
      var detail = document.createElement("span");
      item.className = "funding-event";
      title.textContent = PublicMode ? event.date + " · 已验证外部资金流" : event.date + " · 入金 " + money(event.amount);
      detail.textContent = PublicMode ? "绝对金额已隐藏 · 不计入投资收益" : money(event.total_value_before) + " → " + money(event.total_value_after) + " · 不计入投资收益";
      item.appendChild(title);
      item.appendChild(detail);
      funding.appendChild(item);
    });

    var summary = byId("historySummary");
    summary.replaceChildren();
    metric(summary, "归档起点", bundle.history.archive_start_date + " · " + bundle.history.archive_start_record);
    metric(summary, "首个 P&L", bundle.history.first_pnl_date + " · " + bundle.history.first_pnl_record);
    metric(summary, "纳入记录 / 估值点", bundle.history.included_record_count + " / " + bundle.history.performance_point_count);
    metric(summary, "旧档 exact-byte", String(bundle.history.legacy_exact_byte_record_count));
    metric(summary, "排除记录", String(bundle.history.rejected_record_count));
    metric(summary, "历史证据", bundle.history.evidence_status);
  }

  function renderHistory(bundle) {
    setText("historyPointCount", bundle.history.performance_point_count + " 个估值点");
    var body = byId("performanceRows");
    body.replaceChildren();
    bundle.portfolio.performance_points.slice().reverse().forEach(function (point) {
      var row = document.createElement("tr");
      makeCell(row, point.date);
      makeCell(row, point.record, "mono");
      makeCell(row, publicRedacted(money(point.total_value)), "numeric");
      makeCell(row, number(point.portfolio_unit_nav), "numeric");
      makeCell(row, percent(point.portfolio_cumulative_return, true), "numeric " + signedClass(point.portfolio_cumulative_return));
      makeCell(row, percent(point.csi300_cumulative_return, true), "numeric " + signedClass(point.csi300_cumulative_return));
      makeCell(row, percent(point.cumulative_excess_return, true), "numeric " + signedClass(point.cumulative_excess_return));
      makeCell(row, historyEvidenceLabel(point.evidence_status), "evidence-value");
      body.appendChild(row);
    });
  }

  function renderRisks(bundle) {
    var grid = byId("riskGrid");
    grid.replaceChildren();
    bundle.risks.forEach(function (risk) {
      var card = document.createElement("article");
      card.className = "risk-card " + risk.severity.toLowerCase();
      var label = document.createElement("span");
      var title = document.createElement("strong");
      var detail = document.createElement("p");
      label.textContent = risk.severity;
      title.textContent = risk.code;
      detail.textContent = risk.detail;
      card.appendChild(label);
      card.appendChild(title);
      card.appendChild(detail);
      grid.appendChild(card);
    });
  }

  function renderEvidence(bundle) {
    var body = byId("evidenceRows");
    body.replaceChildren();
    var entries = [
      ["Manifest", bundle.current_evidence.manifest_path, bundle.current_evidence.manifest_sha256],
      ["Manual execution manifest", bundle.current_evidence.manual_manifest_path, bundle.current_evidence.manual_manifest_sha256],
      ["Effective ledger", bundle.current_evidence.ledger_path, bundle.current_evidence.ledger_sha256],
      ["P&L", bundle.current_evidence.pnl_path, bundle.current_evidence.pnl_sha256],
      ["Archive baseline manifest", bundle.history.baseline_manifest_path, bundle.history.baseline_manifest_sha256],
      ["Archive baseline ledger", bundle.history.baseline_ledger_path, bundle.history.baseline_ledger_sha256],
      ["CSI300", bundle.benchmarks[0].source_path, bundle.benchmarks[0].source_sha256]
    ];
    bundle.history.funding_events.forEach(function (event) {
      entries.push(["Funding " + event.record, event.evidence_path, event.evidence_sha256]);
    });
    entries.forEach(function (entry) {
      var row = document.createElement("tr");
      makeCell(row, entry[0]);
      makeCell(row, entry[1], "mono path-cell");
      makeCell(row, entry[2], "mono hash-cell");
      body.appendChild(row);
    });
  }

  function renderWarnings(bundle) {
    setText("i1Status", bundle.i1_display_status + " · research-only states cannot change holdings");
    var list = byId("warningList");
    list.replaceChildren();
    bundle.warnings.forEach(function (warning) {
      var item = document.createElement("li");
      item.textContent = warning;
      list.appendChild(item);
    });
  }

  function renderBundle(bundle) {
    var benchmark = bundle.benchmarks[0];
    var analysis = Analysis.buildAnalysis(bundle);
    activeBundle = bundle;
    activeAnalysis = analysis;
    setText("latestRecord", bundle.latest_valid_record);
    setText("dataDate", bundle.latest_data_date);
    setText("totalValue", publicRedacted(money(bundle.portfolio.total_value)));
    setText("cashExposure", percent(bundle.portfolio.cash_weight) + " / " + percent(bundle.portfolio.gross_exposure));
    setText("performancePeriod", bundle.portfolio.performance_start_date + " → " + bundle.portfolio.performance_end_date);
    setText("freshnessLabel", bundle.data_age_calendar_days + " calendar days old");
    setText("twrValue", percent(bundle.portfolio.cumulative_twr, true));
    setText("benchmarkReturnValue", percent(benchmark.return, true));
    setText("excessValue", percent(benchmark.excess_return, true));
    setText("drawdownValue", percent(bundle.portfolio.max_drawdown));
    setText("portfolioDrawdownValue", percent(bundle.portfolio.max_drawdown));
    setText("benchmarkDrawdownValue", percent(benchmark.max_drawdown));
    setText("performanceInsight", "组合自 3 月以来跑赢沪深300 " + percent(benchmark.excess_return, true) + "，最大回撤 " + percent(bundle.portfolio.max_drawdown));
    setText("performanceSubtitle", analysis.points.length + " 个有效估值点 · " + bundle.portfolio.performance_start_date + " → " + bundle.portfolio.performance_end_date + " · funding-aware TWR");
    renderPositions(bundle);
    renderAllocation(bundle);
    renderChanges(bundle);
    renderMetrics(bundle);
    renderMonthlyPerformance(analysis);
    renderHistoryAnalysis(bundle, analysis);
    renderHistory(bundle);
    renderRisks(bundle);
    renderEvidence(bundle);
    renderWarnings(bundle);
    if (PublicMode) byId("evidenceDetails").hidden = true;
    byId("dashboardContent").hidden = false;
    window.requestAnimationFrame(renderCharts);
  }

  function render() {
    var snapshot = Contract.deriveSnapshot(window.MyQuantCNAggressiveDashboard);
    var status = byId("runtimeStatus");
    status.textContent = snapshot.status;
    status.className = "status-pill " + snapshot.status.toLowerCase();
    renderBlockers(snapshot.blockers);
    if (snapshot.bundle) renderBundle(snapshot.bundle);
  }

  window.addEventListener("resize", function () {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(renderCharts, 120);
  });
  document.addEventListener("DOMContentLoaded", render);
})();

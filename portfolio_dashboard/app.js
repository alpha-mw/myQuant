(function () {
  "use strict";

  var Contract = window.CNAggressiveDashboardContractV1;
  var Analysis = window.CNAggressiveDashboardAnalysisV1;
  var PublicMode = window.CNPublicDashboard === true;
  var activeBundle = null;
  var activeAnalysis = null;
  var resizeTimer = null;
  var chartInteractions = [];
  var activeChartIndex = null;
  var SVG_NS = "http://www.w3.org/2000/svg";

  document.documentElement.classList.toggle("public-view", PublicMode);

  function byId(id) { return document.getElementById(id); }
  function setText(id, value) {
    var node = byId(id);
    if (node) node.textContent = value === null || value === undefined || value === "" ? "—" : String(value);
  }
  function money(value) {
    var parsed = Number(value);
    return Number.isFinite(parsed) ? "¥" + parsed.toLocaleString("zh-CN", { maximumFractionDigits: 2 }) : "UNKNOWN";
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
  function ratio(value) {
    var parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toFixed(2) : "UNKNOWN";
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
  function benchmarkById(bundle, benchmarkId) {
    return (bundle.benchmarks || []).find(function (benchmark) {
      return benchmark.id === benchmarkId;
    });
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

  function chartTooltip(host) {
    var tooltip = document.createElement("div");
    tooltip.className = "chart-tooltip";
    tooltip.hidden = true;
    host.appendChild(tooltip);
    return tooltip;
  }

  function setTooltipContent(tooltip, dateValue, rows) {
    tooltip.replaceChildren();
    var dateNode = document.createElement("strong");
    dateNode.textContent = dateValue;
    tooltip.appendChild(dateNode);
    rows.forEach(function (row) {
      var line = document.createElement("span");
      var label = document.createElement("i");
      var value = document.createElement("b");
      label.className = "tooltip-key " + row.key;
      label.textContent = row.label;
      value.textContent = row.value;
      line.appendChild(label);
      line.appendChild(value);
      tooltip.appendChild(line);
    });
  }

  function setActiveChartPoint(index, source) {
    activeChartIndex = index;
    chartInteractions.forEach(function (interaction) {
      interaction.update(index, interaction === source);
    });
  }

  function hideChartPoint(source) {
    if (source.hasFocus()) return;
    source.hideTooltip();
  }

  function installChartInteraction(options) {
    var host = options.host;
    var svg = options.svg;
    var points = options.points;
    var line = svgNode("line", {
      y1: options.top,
      y2: options.bottom,
      class: "chart-crosshair",
      hidden: "hidden"
    });
    var dots = options.series.map(function (series) {
      var dot = svgNode("circle", {
        r: 4.5,
        class: "chart-focus-dot " + series.key,
        hidden: "hidden"
      });
      svg.appendChild(dot);
      return dot;
    });
    svg.appendChild(line);
    var overlay = svgNode("rect", {
      x: options.left,
      y: options.top,
      width: options.right - options.left,
      height: options.bottom - options.top,
      class: "chart-hit-area",
      tabindex: "0",
      role: "slider",
      "aria-valuemin": "1",
      "aria-valuemax": String(points.length),
      "aria-label": options.ariaLabel
    });
    svg.appendChild(overlay);
    var tooltip = chartTooltip(host);
    var controller = {
      update: function (index, showTooltip) {
        var bounded = Math.max(0, Math.min(points.length - 1, index));
        var point = points[bounded];
        var x = options.xScale(parseDate(point.date));
        line.setAttribute("x1", x);
        line.setAttribute("x2", x);
        line.removeAttribute("hidden");
        options.series.forEach(function (series, seriesIndex) {
          dots[seriesIndex].setAttribute("cx", x);
          dots[seriesIndex].setAttribute("cy", options.yScale(series.value(point, bounded)));
          dots[seriesIndex].removeAttribute("hidden");
        });
        var rows = options.tooltipRows(point, bounded);
        overlay.setAttribute("aria-valuenow", String(bounded + 1));
        overlay.setAttribute("aria-valuetext", point.date + "，" + rows.map(function (row) {
          return row.label + " " + row.value;
        }).join("，"));
        if (showTooltip) {
          setTooltipContent(tooltip, point.date, rows);
          tooltip.style.left = Math.max(0, Math.min(100, x / options.width * 100)) + "%";
          tooltip.classList.toggle("align-right", x > options.width * 0.72);
          tooltip.hidden = false;
        } else {
          tooltip.hidden = true;
        }
      },
      hideTooltip: function () { tooltip.hidden = true; },
      hasFocus: function () { return document.activeElement === overlay; }
    };
    function indexFromPointer(event) {
      var bounds = svg.getBoundingClientRect();
      var localX = (event.clientX - bounds.left) / Math.max(1, bounds.width) * options.width;
      var ratio = (localX - options.left) / Math.max(1, options.right - options.left);
      return Math.round(Math.max(0, Math.min(1, ratio)) * (points.length - 1));
    }
    overlay.addEventListener("pointermove", function (event) {
      setActiveChartPoint(indexFromPointer(event), controller);
    });
    overlay.addEventListener("pointerdown", function (event) {
      setActiveChartPoint(indexFromPointer(event), controller);
    });
    overlay.addEventListener("pointerleave", function () { hideChartPoint(controller); });
    overlay.addEventListener("focus", function () {
      setActiveChartPoint(activeChartIndex === null ? points.length - 1 : activeChartIndex, controller);
    });
    overlay.addEventListener("blur", function () { controller.hideTooltip(); });
    overlay.addEventListener("keydown", function (event) {
      var next = activeChartIndex === null ? points.length - 1 : activeChartIndex;
      if (event.key === "ArrowLeft") next -= 1;
      else if (event.key === "ArrowRight") next += 1;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = points.length - 1;
      else return;
      event.preventDefault();
      setActiveChartPoint(next, controller);
    });
    chartInteractions.push(controller);
  }

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
      { key: "star50", label: "科创50", values: points.map(function (point) { return { date: point.date, value: point.star50_nav * 100 }; }) },
      { key: "chinext", label: "创业板指", values: points.map(function (point) { return { date: point.date, value: point.chinext_nav * 100 }; }) },
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
    svg.appendChild(svgNode("title", {}, "组合、三项基准与对沪深300累计超额历史曲线"));
    svg.appendChild(svgNode("desc", {}, "从 " + points[0].date + " 到 " + points[points.length - 1].date + "，共同起点归一化为100。"));
    appendAxes(svg, dimensions, xScale, yScale, points, yMin, yMax, formatIndex);

    var portfolioValues = series[0].values;
    svg.appendChild(svgNode("path", {
      d: areaPath(portfolioValues, function (point) { return xScale(parseDate(point.date)); }, function (point) { return yScale(point.value); }, yScale(100)),
      class: "chart-area portfolio"
    }));

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
    installChartInteraction({
      host: host,
      svg: svg,
      points: points,
      width: width,
      left: dimensions.left,
      right: width - dimensions.right,
      top: dimensions.top,
      bottom: height - dimensions.bottom,
      xScale: xScale,
      yScale: yScale,
      ariaLabel: "按日期查看组合、沪深300、科创50、创业板指与累计超额净值",
      series: [
        { key: "portfolio", value: function (point) { return point.portfolio_unit_nav * 100; } },
        { key: "benchmark", value: function (point) { return point.csi300_nav * 100; } },
        { key: "star50", value: function (point) { return point.star50_nav * 100; } },
        { key: "chinext", value: function (point) { return point.chinext_nav * 100; } },
        { key: "excess", value: function (point) { return 100 + point.cumulative_excess_return * 100; } }
      ],
      tooltipRows: function (point) {
        return [
          { key: "portfolio", label: "组合净值", value: point.portfolio_unit_nav.toFixed(4) + "  (" + percent(point.portfolio_cumulative_return, true) + ")" },
          { key: "benchmark", label: "沪深300", value: point.csi300_nav.toFixed(4) + "  (" + percent(point.csi300_cumulative_return, true) + ")" },
          { key: "star50", label: "科创50", value: point.star50_nav.toFixed(4) + "  (" + percent(point.star50_cumulative_return, true) + ")" },
          { key: "chinext", label: "创业板指", value: point.chinext_nav.toFixed(4) + "  (" + percent(point.chinext_cumulative_return, true) + ")" },
          { key: "excess", label: "累计超额", value: percent(point.cumulative_excess_return, true) }
        ];
      }
    });
  }

  function renderDrawdownChart(analysis) {
    var host = byId("drawdownChart");
    host.replaceChildren();
    var portfolio = analysis.portfolio_drawdown;
    var benchmark = analysis.benchmark_drawdown;
    var star50 = analysis.star50_drawdown;
    var chinext = analysis.chinext_drawdown;
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
    var minimum = Math.min.apply(null, portfolio.concat(benchmark, star50, chinext).map(function (point) { return point.value * 100; }));
    var yMin = Math.min(-2, minimum * 1.15);
    var xScale = linear(start, end, dimensions.left, width - dimensions.right);
    var yScale = linear(yMin, 0, height - dimensions.bottom, dimensions.top);
    var portfolioValues = portfolio.map(function (point) { return { date: point.date, value: point.value * 100 }; });
    var benchmarkValues = benchmark.map(function (point) { return { date: point.date, value: point.value * 100 }; });
    var star50Values = star50.map(function (point) { return { date: point.date, value: point.value * 100 }; });
    var chinextValues = chinext.map(function (point) { return { date: point.date, value: point.value * 100 }; });
    var svg = svgNode("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height, "aria-hidden": "true" });
    svg.appendChild(svgNode("title", {}, "组合与三项基准历史回撤"));
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
    svg.appendChild(svgNode("path", { d: pathFor(star50Values, function (point) { return xScale(parseDate(point.date)); }, function (point) { return yScale(point.value); }), class: "chart-line star50" }));
    svg.appendChild(svgNode("path", { d: pathFor(chinextValues, function (point) { return xScale(parseDate(point.date)); }, function (point) { return yScale(point.value); }), class: "chart-line chinext" }));
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
    installChartInteraction({
      host: host,
      svg: svg,
      points: activeAnalysis.points,
      width: width,
      left: dimensions.left,
      right: width - dimensions.right,
      top: dimensions.top,
      bottom: height - dimensions.bottom,
      xScale: xScale,
      yScale: yScale,
      ariaLabel: "按日期查看组合与沪深300、科创50、创业板指回撤",
      series: [
        { key: "portfolio", value: function (point, index) { return portfolioValues[index].value; } },
        { key: "benchmark", value: function (point, index) { return benchmarkValues[index].value; } },
        { key: "star50", value: function (point, index) { return star50Values[index].value; } },
        { key: "chinext", value: function (point, index) { return chinextValues[index].value; } }
      ],
      tooltipRows: function (point, index) {
        return [
          { key: "portfolio", label: "组合回撤", value: percent(portfolio[index].value) },
          { key: "benchmark", label: "沪深300回撤", value: percent(benchmark[index].value) },
          { key: "star50", label: "科创50回撤", value: percent(star50[index].value) },
          { key: "chinext", label: "创业板指回撤", value: percent(chinext[index].value) }
        ];
      }
    });
  }

  function renderCharts() {
    if (!activeBundle || !activeAnalysis) return;
    chartInteractions = [];
    activeChartIndex = null;
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
    if (!body) return;
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
    if (!bar || !legend) return;
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
    if (!byId("changeRows")) return;
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
    var csi300 = benchmarkById(bundle, "CSI300");
    var star50 = benchmarkById(bundle, "STAR50");
    var chinext = benchmarkById(bundle, "CHINEXT");
    var performance = byId("performanceList");
    if (!performance) return;
    performance.replaceChildren();
    metric(performance, "统计区间", portfolio.performance_start_date + " → " + portfolio.performance_end_date);
    metric(performance, "收益方法", "3 月 17 日 100 万唯一资本起点");
    metric(performance, "起始资本", publicRedacted(money(portfolio.performance_initial_capital)));
    metric(performance, "当前组合总资产", publicRedacted(money(portfolio.total_value)));
    metric(performance, "累计利润", publicRedacted(money(portfolio.cumulative_profit_excluding_external_flow)));
    metric(performance, "最新有效记录区间收益", percent(portfolio.latest_record_interval_return, true));
    metric(performance, "最近一次账本数量换手", percent(portfolio.latest_interval_turnover));
    metric(performance, "组合 P&L（100 万基准）", publicRedacted(money(portfolio.portfolio_pnl)));
    metric(performance, "当前未实现 P&L", publicRedacted(money(portfolio.current_unrealized_pnl)));
    metric(performance, "最新调仓已实现 P&L", publicRedacted(money(portfolio.latest_record_realized_pnl_from_rebalance)));
    metric(performance, "累计已实现 P&L", "UNKNOWN");
    metric(performance, "费用 / 毛净", portfolio.fee_basis + " / " + portfolio.gross_or_net);
    [csi300, star50, chinext].forEach(function (benchmark) {
      metric(performance, benchmark.name + "覆盖", benchmark.start_date + " → " + benchmark.end_date + " · " + benchmark.missing_dates.length + " gaps");
    });

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

  function quantMetricCard(grid, label, value, detail, className) {
    var card = document.createElement("div");
    var labelNode = document.createElement("span");
    var valueNode = document.createElement("strong");
    var detailNode = document.createElement("p");
    card.className = "quant-metric";
    labelNode.textContent = label;
    valueNode.textContent = value;
    if (className) valueNode.className = className;
    detailNode.textContent = detail;
    card.appendChild(labelNode);
    card.appendChild(valueNode);
    card.appendChild(detailNode);
    grid.appendChild(card);
  }

  function renderQuantMetrics(analysis) {
    var grid = byId("quantMetricGrid");
    var note = byId("quantMetricNote");
    if (!grid) return;
    grid.replaceChildren();
    var metrics = analysis.quantitative_metrics;
    if (!metrics) {
      quantMetricCard(grid, "量化指标", "UNKNOWN", "至少需要 3 个有效估值点。", "");
      if (note) note.textContent = "有效估值点不足，无法计算风险调整指标。";
      return;
    }
    quantMetricCard(grid, "预计年化收益", percent(metrics.estimated_annualized_return, true), "历史几何折算，非收益承诺", signedClass(metrics.estimated_annualized_return));
    quantMetricCard(grid, "年化波动率", percent(metrics.annualized_volatility), "按实际观测频率年化", "");
    quantMetricCard(grid, "Sharpe", ratio(metrics.sharpe_ratio), "Rf=中国1年期国债收益率", signedClass(metrics.sharpe_ratio));
    quantMetricCard(grid, "Sortino", ratio(metrics.sortino_ratio), "最低可接受收益为 0%", signedClass(metrics.sortino_ratio));
    quantMetricCard(grid, "Calmar", ratio(metrics.calmar_ratio), "预计年化收益 ÷ |最大回撤|", signedClass(metrics.calmar_ratio));
    quantMetricCard(grid, "沪深300 Beta", ratio(metrics.beta_csi300), "相邻验证区间收益回归", "");
    quantMetricCard(grid, "沪深300相关系数", ratio(metrics.correlation_csi300), "区间收益 Pearson 相关", "");
    quantMetricCard(grid, "跟踪误差", percent(metrics.tracking_error), "组合相对沪深300，年化", "");
    quantMetricCard(grid, "信息比率", ratio(metrics.information_ratio), "年化主动收益 ÷ 跟踪误差", signedClass(metrics.information_ratio));
    quantMetricCard(grid, "正收益区间占比", percent(metrics.positive_interval_ratio), "相邻验证区间收益 > 0", "");
    if (note) {
      note.textContent = "历史区间折算，不是未来收益承诺；Sharpe 按中国1年期国债收益率计算（区间折算年化 " +
        percent(metrics.risk_free_rate) + "，最新年收益率 " +
        percent(metrics.risk_free_latest_annual_yield) + "）。" +
        metrics.interval_count + " 个验证区间，年化观测频率 " +
        metrics.annualization_periods_per_year.toFixed(1) + "。";
    }
  }

  function historyEvidenceLabel(value) {
    if (value === "ARCHIVE_INCEPTION_EXACT_BYTES_NO_DECLARED_SHA") return "LEGACY BASELINE · exact bytes";
    if (value === "LEGACY_EXACT_BYTES_NO_DECLARED_SHA") return "LEGACY · exact bytes";
    if (value === "HASH_BOUND_CURRENT_CLOSURE") return "CURRENT · hash-bound";
    if (value === "DASHBOARD_POST_HOC_SHA_REGISTRY_BOUND") return "ARCHIVE · SHA registry";
    return value || "UNKNOWN";
  }

  function renderMonthlyPerformance(analysis) {
    var body = byId("monthlyPerformanceRows");
    if (!body) return;
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
      makeCell(row, percent(period.star50_return, true), "numeric " + signedClass(period.star50_return));
      makeCell(row, percent(period.chinext_return, true), "numeric " + signedClass(period.chinext_return));
      makeCell(row, percent(period.excess_return, true), "numeric " + signedClass(period.excess_return));
      makeCell(row, percent(period.portfolio_max_drawdown), "numeric negative");
      makeCell(row, publicRedacted(money(period.ending_total_value)), "numeric monthly-ending-value");
      makeCell(row, period.evidence_status, "evidence-value monthly-evidence");
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
    var csi300 = benchmarkById(bundle, "CSI300");
    var star50 = benchmarkById(bundle, "STAR50");
    var chinext = benchmarkById(bundle, "CHINEXT");
    var list = byId("historyInsightList");
    list.replaceChildren();
    [csi300, star50, chinext].forEach(function (benchmark) {
      insight(list, "对" + benchmark.name + "超额", percent(benchmark.excess_return, true), "组合自 3 月起的累计收益相对" + benchmark.name + "的收益差。", signedClass(benchmark.excess_return));
    });
    if (analysis.best_period) {
      insight(list, "最佳月份", analysis.best_period.period + " · " + percent(analysis.best_period.portfolio_return, true), "当月超额 " + percent(analysis.best_period.excess_return, true) + "。", signedClass(analysis.best_period.portfolio_return));
    }
    if (analysis.weakest_period) {
      insight(list, "最弱月份", analysis.weakest_period.period + " · " + percent(analysis.weakest_period.portfolio_return, true), "月内最大回撤 " + percent(analysis.weakest_period.portfolio_max_drawdown) + "。", signedClass(analysis.weakest_period.portfolio_return));
    }
    if (analysis.deepest_portfolio_drawdown) {
      insight(list, "最深历史回撤", percent(analysis.deepest_portfolio_drawdown.value) + " · " + analysis.deepest_portfolio_drawdown.date, "从此前组合净值高点计算。", "negative");
    }

    var summary = byId("historySummary");
    if (!summary) return;
    summary.replaceChildren();
    metric(summary, "归档起点", bundle.history.archive_start_date + " · " + bundle.history.archive_start_record);
    metric(summary, "首个 P&L", bundle.history.first_pnl_date + " · " + bundle.history.first_pnl_record);
    metric(summary, "纳入记录 / 估值点", bundle.history.included_record_count + " / " + bundle.history.performance_point_count);
    metric(summary, "旧档 exact-byte", String(bundle.history.legacy_exact_byte_record_count));
    metric(summary, "排除记录", String(bundle.history.rejected_record_count));
    metric(summary, "历史证据", bundle.history.evidence_status);
  }

  function renderHistory(bundle) {
    if (!byId("performanceRows")) return;
    setText("historyPointCount", bundle.history.performance_point_count + " 个估值点");
    var body = byId("performanceRows");
    body.replaceChildren();
    bundle.portfolio.performance_points.slice().reverse().forEach(function (point) {
      var row = document.createElement("tr");
      makeCell(row, point.date);
      makeCell(row, point.record, "mono");
      makeCell(row, publicRedacted(money(point.adjusted_total_value)), "numeric");
      makeCell(row, number(point.portfolio_unit_nav), "numeric");
      makeCell(row, percent(point.portfolio_cumulative_return, true), "numeric " + signedClass(point.portfolio_cumulative_return));
      makeCell(row, percent(point.csi300_cumulative_return, true), "numeric " + signedClass(point.csi300_cumulative_return));
      makeCell(row, percent(point.star50_cumulative_return, true), "numeric " + signedClass(point.star50_cumulative_return));
      makeCell(row, percent(point.chinext_cumulative_return, true), "numeric " + signedClass(point.chinext_cumulative_return));
      makeCell(row, percent(point.cumulative_excess_return, true), "numeric " + signedClass(point.cumulative_excess_return));
      makeCell(row, historyEvidenceLabel(point.evidence_status), "evidence-value");
      body.appendChild(row);
    });
  }

  function renderRisks(bundle) {
    var grid = byId("riskGrid");
    if (!grid) return;
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
    if (!body) return;
    body.replaceChildren();
    var entries = [
      ["Manifest", bundle.current_evidence.manifest_path, bundle.current_evidence.manifest_sha256],
      ["Manual execution manifest", bundle.current_evidence.manual_manifest_path, bundle.current_evidence.manual_manifest_sha256],
      ["Effective ledger", bundle.current_evidence.ledger_path, bundle.current_evidence.ledger_sha256],
      ["P&L", bundle.current_evidence.pnl_path, bundle.current_evidence.pnl_sha256],
      ["Archive baseline manifest", bundle.history.baseline_manifest_path, bundle.history.baseline_manifest_sha256],
      ["Archive baseline ledger", bundle.history.baseline_ledger_path, bundle.history.baseline_ledger_sha256],
      ["沪深300", benchmarkById(bundle, "CSI300").source_path, benchmarkById(bundle, "CSI300").source_sha256],
      ["科创50", benchmarkById(bundle, "STAR50").source_path, benchmarkById(bundle, "STAR50").source_sha256],
      ["创业板指", benchmarkById(bundle, "CHINEXT").source_path, benchmarkById(bundle, "CHINEXT").source_sha256]
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
    if (!byId("warningList")) return;
    setText("i1Status", bundle.i1_display_status + " · research-only states cannot change holdings");
    var list = byId("warningList");
    list.replaceChildren();
    bundle.warnings.forEach(function (warning) {
      var item = document.createElement("li");
      item.textContent = warning;
      list.appendChild(item);
    });
  }

  function renderInternalControl(bundle) {
    if (!byId("internalLatestRecord")) return;
    var risks = bundle.risks || [];
    var highRiskCount = risks.filter(function (risk) { return risk.severity === "HIGH"; }).length;
    var mediumRiskCount = risks.filter(function (risk) { return risk.severity === "MEDIUM"; }).length;
    var execution = bundle.current_evidence || {};
    var performancePointCount = bundle.history && bundle.history.performance_point_count;

    setText("internalLatestRecord", bundle.latest_valid_record);
    setText("internalPreviousRecord", bundle.previous_valid_record);
    setText("internalDataDate", bundle.latest_data_date + " · " + bundle.data_age_calendar_days + " calendar days");
    setText("internalExecutionStatus", (execution.execution_status || "UNKNOWN") + " · " + (execution.execution_kind || "UNKNOWN"));
    setText("internalHistoryEvidence", bundle.history && bundle.history.evidence_status);
    setText("internalTotalValue", money(bundle.portfolio.total_value));
    setText("internalPortfolioPnl", money(bundle.portfolio.portfolio_pnl));
    setText("internalGrossExposure", percent(bundle.portfolio.gross_exposure));
    setText("internalCashWeight", percent(bundle.portfolio.cash_weight));
    setText("internalTurnover", percent(bundle.portfolio.latest_interval_turnover));
    setText("internalHoldingCount", bundle.concentration.holding_count + " 只");
    setText("internalConcentration", percent(bundle.concentration.top1_equity_weight) + " / " + percent(bundle.concentration.top3_equity_weight));
    setText("internalEvidenceCounts", bundle.valid_record_count + " / " + performancePointCount);
    setText("internalRiskCount", risks.length + " 项 · " + highRiskCount + " HIGH · " + mediumRiskCount + " MEDIUM");
    setText("internalWarningCount", (bundle.warnings || []).length + " 项");
    setText("internalI1Status", bundle.i1_display_status + " · research-only · no holding authority");
  }

  function renderBundle(bundle) {
    var csi300 = benchmarkById(bundle, "CSI300");
    var star50 = benchmarkById(bundle, "STAR50");
    var chinext = benchmarkById(bundle, "CHINEXT");
    var analysis = Analysis.buildAnalysis(bundle);
    activeBundle = bundle;
    activeAnalysis = analysis;
    setText("latestRecord", bundle.latest_valid_record);
    setText("dataDate", bundle.latest_data_date);
    setText("totalValue", publicRedacted(money(bundle.portfolio.total_value)));
    setText("cashExposure", percent(bundle.portfolio.cash_weight) + " / " + percent(bundle.portfolio.gross_exposure));
    setText("performancePeriod", bundle.portfolio.performance_start_date + " → " + bundle.portfolio.performance_end_date);
    setText("headerPeriod", bundle.portfolio.performance_start_date + " — " + bundle.portfolio.performance_end_date);
    setText("freshnessLabel", "截至 " + bundle.portfolio.performance_end_date);
    setText("portfolioReturnValue", percent(bundle.portfolio.cumulative_return, true));
    setText("benchmarkReturnValue", percent(csi300.return, true));
    setText("star50ReturnValue", percent(star50.return, true));
    setText("chinextReturnValue", percent(chinext.return, true));
    setText("excessValue", percent(csi300.excess_return, true));
    setText("drawdownValue", percent(bundle.portfolio.max_drawdown));
    setText("portfolioDrawdownValue", percent(bundle.portfolio.max_drawdown));
    setText("benchmarkDrawdownValue", percent(csi300.max_drawdown));
    setText("star50DrawdownValue", percent(star50.max_drawdown));
    setText("chinextDrawdownValue", percent(chinext.max_drawdown));
    setText("performanceInsight", "净值与回撤");
    setText("performanceSubtitle", bundle.portfolio.performance_start_date + " — " + bundle.portfolio.performance_end_date + " · 100 万起点");
    renderQuantMetrics(analysis);
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
    renderInternalControl(bundle);
    byId("dashboardContent").hidden = false;
    window.requestAnimationFrame(renderCharts);
  }

  function render() {
    var snapshot = Contract.deriveSnapshot(window.MyQuantCNAggressiveDashboard);
    var status = byId("runtimeStatus");
    status.textContent = snapshot.status === "FRESH" ? "UPDATED" : snapshot.status;
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

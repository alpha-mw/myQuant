(function () {
  "use strict";

  var COLORS = [
    "#2563eb",
    "#8f98a8",
    "#dc2626",
    "#15835f",
    "#0f8a8a",
    "#c77819",
    "#7a5cbd",
    "#546176",
    "#8a4a5d",
    "#3877a8"
  ];

  function escapeHtml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function el(tag, attrs) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, attrs[key]);
    });
    return node;
  }

  function clear(container) {
    if (typeof container === "string") container = document.getElementById(container);
    if (container) container.innerHTML = "";
    return container;
  }

  function renderEmpty(container, message) {
    container = clear(container);
    if (!container) return;
    var empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = message || "暂无可展示数据";
    container.appendChild(empty);
  }

  function size(container, fallbackHeight) {
    var width = Math.max(320, container.clientWidth || 640);
    var height = fallbackHeight || Math.max(220, container.clientHeight || 260);
    return { width: width, height: height };
  }

  function extent(values) {
    var nums = values.filter(function (value) {
      return typeof value === "number" && Number.isFinite(value);
    });
    if (!nums.length) return [0, 1];
    var min = Math.min.apply(null, nums);
    var max = Math.max.apply(null, nums);
    if (min === max) {
      var pad = Math.abs(min || 1) * 0.1;
      return [min - pad, max + pad];
    }
    var padding = (max - min) * 0.08;
    return [min - padding, max + padding];
  }

  function dateExtent(points) {
    var times = points.map(function (point) { return point.dateObj ? point.dateObj.getTime() : NaN; })
      .filter(Number.isFinite);
    if (!times.length) return [0, 1];
    return [Math.min.apply(null, times), Math.max.apply(null, times)];
  }

  function formatDate(date) {
    if (!date) return "-";
    var d = typeof date === "string" ? window.DashboardData.parseDateOnly(date) : date;
    if (!d) return String(date);
    return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
  }

  function formatPercent(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "-";
    return (value * 100).toFixed(2) + "%";
  }

  function formatNumber(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "-";
    return value.toFixed(3);
  }

  function formatMoney(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "-";
    var abs = Math.abs(value);
    if (abs >= 100000000) return (value / 100000000).toFixed(2) + "亿";
    if (abs >= 10000) return (value / 10000).toFixed(2) + "万";
    return value.toFixed(0);
  }

  function showTooltip(html, event) {
    var tooltip = document.getElementById("chartTooltip");
    if (!tooltip) return;
    tooltip.innerHTML = html;
    tooltip.classList.add("visible");
    var x = Math.min(window.innerWidth - 300, event.clientX + 14);
    var y = Math.min(window.innerHeight - 160, event.clientY + 14);
    tooltip.style.left = Math.max(8, x) + "px";
    tooltip.style.top = Math.max(8, y) + "px";
  }

  function hideTooltip() {
    var tooltip = document.getElementById("chartTooltip");
    if (tooltip) tooltip.classList.remove("visible");
  }

  function addAxes(svg, dims, yMin, yMax, xFormatter, yFormatter) {
    var plot = dims.plot;
    for (var i = 0; i <= 4; i += 1) {
      var y = plot.y + (plot.height * i) / 4;
      svg.appendChild(el("line", { x1: plot.x, x2: plot.x + plot.width, y1: y, y2: y, class: "grid-line" }));
      var value = yMax - ((yMax - yMin) * i) / 4;
      var label = el("text", { x: plot.x - 8, y: y + 4, "text-anchor": "end", class: "axis-label" });
      label.textContent = yFormatter(value);
      svg.appendChild(label);
    }
    svg.appendChild(el("line", { x1: plot.x, x2: plot.x + plot.width, y1: plot.y + plot.height, y2: plot.y + plot.height, class: "axis-line" }));
    for (var j = 0; j <= 3; j += 1) {
      var x = plot.x + (plot.width * j) / 3;
      var labelX = el("text", { x: x, y: plot.y + plot.height + 22, "text-anchor": j === 0 ? "start" : j === 3 ? "end" : "middle", class: "axis-label" });
      labelX.textContent = xFormatter(j / 3);
      svg.appendChild(labelX);
    }
  }

  function hasForwardFilledPoints(series) {
    return (series || []).some(function (item) {
      return (item.points || []).some(function (point) { return point.filled; });
    });
  }

  function makeLegend(container, series) {
    var legend = document.createElement("div");
    legend.className = "legend";
    series.forEach(function (item, index) {
      var entry = document.createElement("span");
      entry.className = "legend-item";
      var dot = document.createElement("span");
      dot.className = "legend-dot";
      dot.style.background = item.color || COLORS[index % COLORS.length];
      entry.appendChild(dot);
      entry.appendChild(document.createTextNode(item.name));
      legend.appendChild(entry);
    });
    if (hasForwardFilledPoints(series)) {
      var fillEntry = document.createElement("span");
      fillEntry.className = "legend-item legend-ffill";
      var line = document.createElement("span");
      line.className = "legend-dash";
      fillEntry.appendChild(line);
      fillEntry.appendChild(document.createTextNode("虚线 = benchmark 前向填充"));
      legend.appendChild(fillEntry);
    }
    container.appendChild(legend);
  }

  function appendLineSegment(svg, segment, dashed, color, width, xScale, yScale) {
    if (!segment || segment.length < 2) return;
    var d = segment.map(function (point, pointIndex) {
      return (pointIndex ? "L" : "M") + xScale(point.dateObj).toFixed(2) + "," + yScale(point.value).toFixed(2);
    }).join(" ");
    var attrs = {
      d: d,
      fill: "none",
      stroke: color,
      "stroke-width": width || 2.2,
      "stroke-linejoin": "round",
      "stroke-linecap": "round"
    };
    if (dashed) attrs["stroke-dasharray"] = "5 5";
    svg.appendChild(el("path", attrs));
  }

  function appendLineSegments(svg, points, color, width, xScale, yScale) {
    if (!points || points.length < 2) return;
    var segment = [points[0]];
    var segmentDashed = null;
    for (var index = 1; index < points.length; index += 1) {
      var prev = points[index - 1];
      var current = points[index];
      var dashed = Boolean(prev.filled || current.filled);
      if (segmentDashed === null) segmentDashed = dashed;
      if (dashed !== segmentDashed) {
        appendLineSegment(svg, segment, segmentDashed, color, width, xScale, yScale);
        segment = [prev, current];
        segmentDashed = dashed;
      } else {
        segment.push(current);
      }
    }
    appendLineSegment(svg, segment, segmentDashed, color, width, xScale, yScale);
  }

  function lineChart(containerId, series, options) {
    var container = clear(containerId);
    options = options || {};
    series = (series || []).filter(function (item) {
      return item.points && item.points.some(function (point) { return Number.isFinite(point.value); });
    });
    if (!container || !series.length) return renderEmpty(container, options.empty || "暂无曲线数据");
    var allPoints = [];
    series.forEach(function (item) { allPoints = allPoints.concat(item.points); });
    var allValues = allPoints.map(function (point) { return point.value; });
    if (options.includeZero) allValues.push(0);
    var yRange = options.yDomain || extent(allValues);
    var xRange = dateExtent(allPoints);
    var dims = size(container, options.height || 260);
    var margin = { top: 14, right: options.endLabels ? 70 : 18, bottom: 38, left: 58 };
    var plot = {
      x: margin.left,
      y: margin.top,
      width: dims.width - margin.left - margin.right,
      height: dims.height - margin.top - margin.bottom
    };
    var svg = el("svg", { viewBox: "0 0 " + dims.width + " " + dims.height, role: "img" });
    var xScale = function (dateObj) {
      var time = dateObj ? dateObj.getTime() : xRange[0];
      return plot.x + ((time - xRange[0]) / Math.max(1, xRange[1] - xRange[0])) * plot.width;
    };
    var yScale = function (value) {
      return plot.y + plot.height - ((value - yRange[0]) / Math.max(0.000001, yRange[1] - yRange[0])) * plot.height;
    };
    addAxes(svg, { plot: plot }, yRange[0], yRange[1], function (ratio) {
      return formatDate(new Date(xRange[0] + (xRange[1] - xRange[0]) * ratio)).slice(0, 7);
    }, options.yFormatter || formatNumber);
    if (yRange[0] < 0 && yRange[1] > 0) {
      var zeroY = yScale(0);
      svg.appendChild(el("line", { x1: plot.x, x2: plot.x + plot.width, y1: zeroY, y2: zeroY, stroke: "#b8c3d0", "stroke-dasharray": "4 4" }));
    }
    series.forEach(function (item, index) {
      var color = item.color || COLORS[index % COLORS.length];
      var points = item.points.filter(function (point) { return Number.isFinite(point.value) && point.dateObj; });
      if ((options.areaToZero || item.areaToZero) && points.length) {
        var zeroYForArea = yScale(0);
        var areaD = "M" + xScale(points[0].dateObj).toFixed(2) + "," + zeroYForArea.toFixed(2) + " " +
          points.map(function (point) {
            return "L" + xScale(point.dateObj).toFixed(2) + "," + yScale(point.value).toFixed(2);
          }).join(" ") +
          " L" + xScale(points[points.length - 1].dateObj).toFixed(2) + "," + zeroYForArea.toFixed(2) + " Z";
        svg.appendChild(el("path", { d: areaD, fill: color, opacity: item.areaOpacity || options.areaOpacity || 0.13 }));
      }
      appendLineSegments(svg, points, color, item.width || 2.2, xScale, yScale);
      if (options.endLabels && points.length) {
        var last = points[points.length - 1];
        var endLabel = el("text", {
          x: Math.min(plot.x + plot.width - 4, xScale(last.dateObj) + 6),
          y: yScale(last.value) + 4,
          fill: color,
          "font-size": "11",
          "font-weight": "700",
          "text-anchor": "start"
        });
        endLabel.textContent = (options.yFormatter || formatNumber)(last.value);
        svg.appendChild(endLabel);
      }
    });
    var overlay = el("rect", { x: plot.x, y: plot.y, width: plot.width, height: plot.height, fill: "transparent" });
    overlay.addEventListener("mousemove", function (event) {
      var rect = svg.getBoundingClientRect();
      var ratio = (event.clientX - rect.left - plot.x) / plot.width;
      var targetTime = xRange[0] + (xRange[1] - xRange[0]) * ratio;
      var html = "";
      series.forEach(function (item) {
        var nearest = item.points.reduce(function (best, point) {
          if (!point.dateObj || !Number.isFinite(point.value)) return best;
          if (!best) return point;
          return Math.abs(point.dateObj.getTime() - targetTime) < Math.abs(best.dateObj.getTime() - targetTime) ? point : best;
        }, null);
        if (nearest && !html) html += '<div class="tooltip-title">' + escapeHtml(formatDate(nearest.dateObj)) + "</div>";
        if (nearest) {
          var formatted = (options.yFormatter || formatNumber)(nearest.value);
          var label = item.name + (nearest.filled ? " · 前向填充" : "");
          html += '<div class="tooltip-row"><span>' + escapeHtml(label) + "</span><strong>" + escapeHtml(formatted) + "</strong></div>";
        }
      });
      showTooltip(html, event);
    });
    overlay.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(overlay);
    container.appendChild(svg);
    makeLegend(container, series);
  }

  function barChart(containerId, rows, options) {
    var container = clear(containerId);
    options = options || {};
    rows = (rows || []).filter(function (row) { return Number.isFinite(row.value); });
    if (!container || !rows.length) return renderEmpty(container, options.empty || "暂无柱状图数据");
    var horizontal = options.horizontal !== false;
    var dims = size(container, options.height || 260);
    var margin = horizontal ? { top: 8, right: 22, bottom: 24, left: 142 } : { top: 14, right: 14, bottom: 58, left: 54 };
    var plot = {
      x: margin.left,
      y: margin.top,
      width: dims.width - margin.left - margin.right,
      height: dims.height - margin.top - margin.bottom
    };
    var svg = el("svg", { viewBox: "0 0 " + dims.width + " " + dims.height, role: "img" });
    var values = rows.map(function (row) { return row.value; });
    values.push(0);
    var range = extent(values);
    var formatter = options.valueFormatter || formatPercent;
    if (horizontal) {
      var band = plot.height / rows.length;
      rows.forEach(function (row, index) {
        var zeroX = plot.x + ((0 - range[0]) / Math.max(0.000001, range[1] - range[0])) * plot.width;
        var x = plot.x + ((Math.min(0, row.value) - range[0]) / Math.max(0.000001, range[1] - range[0])) * plot.width;
        var w = Math.abs(row.value / Math.max(Math.abs(range[0]), Math.abs(range[1]), 0.000001)) * (plot.width / 2);
        if (range[0] >= 0) {
          x = plot.x;
          w = (row.value / Math.max(range[1], 0.000001)) * plot.width;
        } else {
          w = Math.abs((row.value - 0) / Math.max(0.000001, range[1] - range[0])) * plot.width;
        }
        var y = plot.y + index * band + band * 0.18;
        var color = row.color || (row.value < 0 ? "#15835f" : "#dc2626");
        var label = el("text", { x: plot.x - 8, y: y + band * 0.42, "text-anchor": "end", class: "axis-label" });
        label.textContent = String(row.label).slice(0, 22);
        svg.appendChild(label);
        var rect = el("rect", { x: x, y: y, width: Math.max(1, w), height: Math.max(8, band * 0.55), rx: 3, fill: color });
        rect.__data__ = row;
        rect.addEventListener("mousemove", function (event) {
          var data = this.__data__;
          showTooltip('<div class="tooltip-title">' + escapeHtml(data.label) + '</div><div class="tooltip-row"><span>value</span><strong>' + escapeHtml(formatter(data.value)) + "</strong></div>", event);
        });
        rect.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(rect);
        var valueLabel = el("text", { x: row.value < 0 ? x - 5 : x + Math.max(1, w) + 5, y: y + band * 0.42, "text-anchor": row.value < 0 ? "end" : "start", class: "axis-label" });
        valueLabel.textContent = formatter(row.value);
        svg.appendChild(valueLabel);
        svg.appendChild(el("line", { x1: zeroX, x2: zeroX, y1: plot.y, y2: plot.y + plot.height, class: "axis-line" }));
      });
    } else {
      var barWidth = plot.width / rows.length;
      var zeroY = plot.y + plot.height - ((0 - range[0]) / Math.max(0.000001, range[1] - range[0])) * plot.height;
      svg.appendChild(el("line", { x1: plot.x, x2: plot.x + plot.width, y1: zeroY, y2: zeroY, class: "axis-line" }));
      rows.forEach(function (row, index) {
        var x = plot.x + index * barWidth + barWidth * 0.15;
        var y = plot.y + plot.height - ((Math.max(0, row.value) - range[0]) / Math.max(0.000001, range[1] - range[0])) * plot.height;
        var height = Math.abs((row.value - 0) / Math.max(0.000001, range[1] - range[0])) * plot.height;
        if (row.value < 0) y = zeroY;
        var vRect = el("rect", { x: x, y: y, width: Math.max(4, barWidth * 0.7), height: Math.max(1, height), rx: 2, fill: row.value < 0 ? "#15835f" : "#dc2626" });
        vRect.__data__ = row;
        vRect.addEventListener("mousemove", function (event) {
          var data = this.__data__;
          showTooltip('<div class="tooltip-title">' + escapeHtml(data.label) + '</div><div class="tooltip-row"><span>value</span><strong>' + escapeHtml(formatter(data.value)) + "</strong></div>", event);
        });
        vRect.addEventListener("mouseleave", hideTooltip);
        svg.appendChild(vRect);
        if (index % Math.ceil(rows.length / 12) === 0) {
          var label = el("text", { x: x + barWidth * 0.35, y: plot.y + plot.height + 18, "text-anchor": "middle", class: "axis-label", transform: "rotate(-35 " + (x + barWidth * 0.35) + " " + (plot.y + plot.height + 18) + ")" });
          label.textContent = row.label;
          svg.appendChild(label);
        }
      });
    }
    container.appendChild(svg);
  }

  function heatmap(containerId, rows) {
    var container = clear(containerId);
    rows = rows || [];
    if (!container || !rows.length) return renderEmpty(container, "暂无月度收益数据");
    var years = Array.from(new Set(rows.map(function (row) { return row.year; }))).sort();
    var map = {};
    rows.forEach(function (row) { map[row.year + "-" + row.monthNumber] = row; });
    var dims = size(container, 190);
    var margin = { top: 18, right: 10, bottom: 20, left: 54 };
    var plot = {
      x: margin.left,
      y: margin.top,
      width: dims.width - margin.left - margin.right,
      height: dims.height - margin.top - margin.bottom
    };
    var cellW = plot.width / 12;
    var cellH = plot.height / years.length;
    var svg = el("svg", { viewBox: "0 0 " + dims.width + " " + dims.height, role: "img" });
    var months = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"];
    months.forEach(function (month, index) {
      var label = el("text", { x: plot.x + index * cellW + cellW / 2, y: 12, "text-anchor": "middle", class: "axis-label" });
      label.textContent = month;
      svg.appendChild(label);
    });
    years.forEach(function (year, yIndex) {
      var yearLabel = el("text", { x: plot.x - 10, y: plot.y + yIndex * cellH + cellH / 2 + 4, "text-anchor": "end", class: "axis-label" });
      yearLabel.textContent = year;
      svg.appendChild(yearLabel);
      for (var m = 1; m <= 12; m += 1) {
        var row = map[year + "-" + m];
        var value = row ? row.value : null;
        var intensity = value === null ? 0 : Math.min(1, Math.abs(value) / 0.1);
        var color = value === null ? "#eef2f7" : value >= 0 ? "rgba(220,38,38," + (0.14 + intensity * 0.66) + ")" : "rgba(21,131,95," + (0.14 + intensity * 0.66) + ")";
        var rect = el("rect", { x: plot.x + (m - 1) * cellW + 2, y: plot.y + yIndex * cellH + 2, width: Math.max(4, cellW - 4), height: Math.max(18, cellH - 4), rx: 4, fill: color });
        rect.addEventListener("mousemove", function (event) {
          var data = this.__data__;
          showTooltip('<div class="tooltip-title">' + escapeHtml(data.month) + '</div><div class="tooltip-row"><span>monthly_return</span><strong>' + escapeHtml(formatPercent(data.value)) + "</strong></div>", event);
        });
        rect.addEventListener("mouseleave", hideTooltip);
        rect.__data__ = row || { month: year + "-" + String(m).padStart(2, "0"), value: null };
        svg.appendChild(rect);
        if (row) {
          var text = el("text", { x: plot.x + (m - 1) * cellW + cellW / 2, y: plot.y + yIndex * cellH + cellH / 2 + 4, "text-anchor": "middle", fill: Math.abs(value) > 0.055 ? "#fff" : "#24314a", "font-size": "11" });
          text.textContent = (value * 100).toFixed(1) + "%";
          svg.appendChild(text);
        }
      }
    });
    container.appendChild(svg);
  }

  function scatter(containerId, rows, options) {
    var container = clear(containerId);
    options = options || {};
    rows = (rows || []).filter(function (row) { return Number.isFinite(row.x) && Number.isFinite(row.y); });
    if (!container || rows.length < 3) return renderEmpty(container, "数据不足，暂不绘制散点图");
    var dims = size(container, options.height || 260);
    var margin = { top: 14, right: 18, bottom: 42, left: 58 };
    var plot = {
      x: margin.left,
      y: margin.top,
      width: dims.width - margin.left - margin.right,
      height: dims.height - margin.top - margin.bottom
    };
    var xRange = extent(rows.map(function (row) { return row.x; }).concat([0]));
    var yRange = extent(rows.map(function (row) { return row.y; }).concat([0]));
    var xScale = function (value) { return plot.x + ((value - xRange[0]) / Math.max(0.000001, xRange[1] - xRange[0])) * plot.width; };
    var yScale = function (value) { return plot.y + plot.height - ((value - yRange[0]) / Math.max(0.000001, yRange[1] - yRange[0])) * plot.height; };
    var svg = el("svg", { viewBox: "0 0 " + dims.width + " " + dims.height, role: "img" });
    addAxes(svg, { plot: plot }, yRange[0], yRange[1], function (ratio) {
      return formatPercent(xRange[0] + (xRange[1] - xRange[0]) * ratio);
    }, formatPercent);
    rows.forEach(function (row, index) {
      var circle = el("circle", {
        cx: xScale(row.x),
        cy: yScale(row.y),
        r: row.radius || (row.isPortfolio ? 6.5 : 5),
        fill: row.color || (row.isPortfolio ? "#101828" : COLORS[index % COLORS.length]),
        opacity: row.isPortfolio ? 0.95 : 0.82,
        stroke: row.isMain ? "#101828" : "none",
        "stroke-width": row.isMain ? 1.6 : 0
      });
      circle.__data__ = row;
      circle.addEventListener("mousemove", function (event) {
        var data = this.__data__;
        showTooltip(
          '<div class="tooltip-title">' + escapeHtml(data.label) + '</div>' +
          '<div class="tooltip-row"><span>' + escapeHtml(options.xLabel || "avg weight") + '</span><strong>' + escapeHtml((options.xFormatter || formatPercent)(data.x)) + '</strong></div>' +
          '<div class="tooltip-row"><span>' + escapeHtml(options.yLabel || "contribution") + '</span><strong>' + escapeHtml((options.yFormatter || formatPercent)(data.y)) + '</strong></div>',
          event
        );
      });
      circle.addEventListener("mouseleave", hideTooltip);
      svg.appendChild(circle);
      if (row.isPortfolio || row.isMain) {
        var label = el("text", {
          x: xScale(row.x) + 8,
          y: yScale(row.y) - 8,
          fill: "#25324a",
          "font-size": "11",
          "font-weight": "750"
        });
        label.textContent = row.label;
        svg.appendChild(label);
      }
    });
    container.appendChild(svg);
  }

  window.DashboardCharts = {
    lineChart: lineChart,
    barChart: barChart,
    heatmap: heatmap,
    scatter: scatter,
    renderEmpty: renderEmpty,
    formatPercent: formatPercent,
    formatNumber: formatNumber,
    formatMoney: formatMoney,
    formatDate: formatDate,
    COLORS: COLORS
  };
})();

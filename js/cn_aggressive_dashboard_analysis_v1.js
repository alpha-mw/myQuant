(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CNAggressiveDashboardAnalysisV1 = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function orderedPoints(points) {
    if (!Array.isArray(points)) return [];
    return points.slice().filter(function (point) {
      return point && /^\d{4}-\d{2}-\d{2}$/.test(point.date || "") &&
        finite(point.portfolio_unit_nav) && finite(point.csi300_nav) &&
        finite(point.star50_nav) && finite(point.chinext_nav);
    }).sort(function (left, right) {
      return left.date.localeCompare(right.date) || String(left.record).localeCompare(String(right.record));
    });
  }

  function drawdownSeries(points, key) {
    var peak = null;
    return points.map(function (point) {
      var value = point[key];
      peak = peak === null ? value : Math.max(peak, value);
      return {
        date: point.date,
        record: point.record,
        value: peak > 0 ? value / peak - 1 : 0
      };
    });
  }

  function maximumDrawdown(points, key, initialValue) {
    var peak = initialValue;
    var deepest = 0;
    points.forEach(function (point) {
      peak = Math.max(peak, point[key]);
      if (peak > 0) deepest = Math.min(deepest, point[key] / peak - 1);
    });
    return deepest;
  }

  function monthlyPerformance(points) {
    var groups = [];
    points.forEach(function (point) {
      var period = point.date.slice(0, 7);
      var current = groups[groups.length - 1];
      if (!current || current.period !== period) {
        current = { period: period, points: [] };
        groups.push(current);
      }
      current.points.push(point);
    });

    return groups.map(function (group, index) {
      var first = group.points[0];
      var last = group.points[group.points.length - 1];
      var base = index === 0 ? first : groups[index - 1].points[groups[index - 1].points.length - 1];
      var portfolioReturn = last.portfolio_unit_nav / base.portfolio_unit_nav - 1;
      var benchmarkReturn = last.csi300_nav / base.csi300_nav - 1;
      var star50Return = last.star50_nav / base.star50_nav - 1;
      var chinextReturn = last.chinext_nav / base.chinext_nav - 1;
      return {
        period: group.period,
        base_date: base.date,
        start_date: first.date,
        end_date: last.date,
        portfolio_return: portfolioReturn,
        benchmark_return: benchmarkReturn,
        star50_return: star50Return,
        chinext_return: chinextReturn,
        excess_return: portfolioReturn - benchmarkReturn,
        portfolio_max_drawdown: maximumDrawdown(group.points, "portfolio_unit_nav", base.portfolio_unit_nav),
        benchmark_max_drawdown: maximumDrawdown(group.points, "csi300_nav", base.csi300_nav),
        star50_max_drawdown: maximumDrawdown(group.points, "star50_nav", base.star50_nav),
        chinext_max_drawdown: maximumDrawdown(group.points, "chinext_nav", base.chinext_nav),
        ending_total_value: last.total_value,
        point_count: group.points.length,
        evidence_status: group.points.some(function (point) {
          return point.evidence_status !== "HASH_BOUND_CURRENT_CLOSURE" &&
            point.evidence_status !== "DASHBOARD_POST_HOC_SHA_REGISTRY_BOUND";
        }) ? "PARTIAL" : "HASH_BOUND"
      };
    });
  }

  function extreme(rows, key, direction) {
    if (!rows.length) return null;
    return rows.reduce(function (best, row) {
      if (direction === "min") return row[key] < best[key] ? row : best;
      return row[key] > best[key] ? row : best;
    }, rows[0]);
  }

  function buildAnalysis(bundle) {
    var points = orderedPoints(bundle && bundle.portfolio && bundle.portfolio.performance_points);
    var monthly = monthlyPerformance(points);
    var portfolioDrawdown = drawdownSeries(points, "portfolio_unit_nav");
    var benchmarkDrawdown = drawdownSeries(points, "csi300_nav");
    var star50Drawdown = drawdownSeries(points, "star50_nav");
    var chinextDrawdown = drawdownSeries(points, "chinext_nav");
    return {
      points: points,
      portfolio_drawdown: portfolioDrawdown,
      benchmark_drawdown: benchmarkDrawdown,
      star50_drawdown: star50Drawdown,
      chinext_drawdown: chinextDrawdown,
      monthly: monthly,
      best_period: extreme(monthly, "portfolio_return", "max"),
      weakest_period: extreme(monthly, "portfolio_return", "min"),
      best_excess_period: extreme(monthly, "excess_return", "max"),
      deepest_portfolio_drawdown: extreme(portfolioDrawdown, "value", "min"),
      deepest_benchmark_drawdown: extreme(benchmarkDrawdown, "value", "min"),
      deepest_star50_drawdown: extreme(star50Drawdown, "value", "min"),
      deepest_chinext_drawdown: extreme(chinextDrawdown, "value", "min")
    };
  }

  return {
    buildAnalysis: buildAnalysis,
    drawdownSeries: drawdownSeries,
    monthlyPerformance: monthlyPerformance,
    orderedPoints: orderedPoints
  };
});

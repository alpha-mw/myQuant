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

  function average(values) {
    if (!values.length) return null;
    return values.reduce(function (total, value) { return total + value; }, 0) / values.length;
  }

  function sampleDeviation(values) {
    if (values.length < 2) return null;
    var mean = average(values);
    var variance = values.reduce(function (total, value) {
      return total + Math.pow(value - mean, 2);
    }, 0) / (values.length - 1);
    return Math.sqrt(Math.max(0, variance));
  }

  function covariance(left, right) {
    if (left.length !== right.length || left.length < 2) return null;
    var leftMean = average(left);
    var rightMean = average(right);
    return left.reduce(function (total, value, index) {
      return total + (value - leftMean) * (right[index] - rightMean);
    }, 0) / (left.length - 1);
  }

  function utcDay(value) {
    var parts = String(value).split("-").map(Number);
    if (parts.length !== 3 || parts.some(function (part) { return !Number.isFinite(part); })) return null;
    return Date.UTC(parts[0], parts[1] - 1, parts[2]) / 86400000;
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
        ending_total_value: last.adjusted_total_value,
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

  function quantitativeMetrics(points, maximumDrawdown) {
    if (!Array.isArray(points) || points.length < 3) return null;
    var startDay = utcDay(points[0].date);
    var endDay = utcDay(points[points.length - 1].date);
    var elapsedDays = endDay - startDay;
    if (!finite(startDay) || !finite(endDay) || elapsedDays <= 0) return null;

    var portfolioReturns = [];
    var benchmarkReturns = [];
    var riskFreeReturns = [];
    for (var index = 1; index < points.length; index += 1) {
      var previous = points[index - 1];
      var current = points[index];
      var intervalDays = utcDay(current.date) - utcDay(previous.date);
      var portfolioReturn = current.portfolio_unit_nav / previous.portfolio_unit_nav - 1;
      var benchmarkReturn = current.csi300_nav / previous.csi300_nav - 1;
      var annualRiskFreeYield = previous.risk_free_annual_yield;
      var riskFreeReturn = Math.pow(1 + annualRiskFreeYield, intervalDays / 365) - 1;
      if (!finite(portfolioReturn) || !finite(benchmarkReturn) ||
          !finite(annualRiskFreeYield) || annualRiskFreeYield < 0 ||
          !finite(intervalDays) || intervalDays <= 0 || !finite(riskFreeReturn)) return null;
      portfolioReturns.push(portfolioReturn);
      benchmarkReturns.push(benchmarkReturn);
      riskFreeReturns.push(riskFreeReturn);
    }

    var elapsedYears = elapsedDays / 365.2425;
    var periodsPerYear = portfolioReturns.length / elapsedYears;
    var portfolioMean = average(portfolioReturns);
    var excessRiskFreeReturns = portfolioReturns.map(function (value, index) {
      return value - riskFreeReturns[index];
    });
    var excessRiskFreeMean = average(excessRiskFreeReturns);
    var benchmarkMean = average(benchmarkReturns);
    var portfolioDeviation = sampleDeviation(portfolioReturns);
    var activeReturns = portfolioReturns.map(function (value, index) {
      return value - benchmarkReturns[index];
    });
    var activeMean = average(activeReturns);
    var activeDeviation = sampleDeviation(activeReturns);
    var benchmarkDeviation = sampleDeviation(benchmarkReturns);
    var portfolioBenchmarkCovariance = covariance(portfolioReturns, benchmarkReturns);
    var annualizedVolatility = finite(portfolioDeviation) ? portfolioDeviation * Math.sqrt(periodsPerYear) : null;
    var downsideDeviation = Math.sqrt(portfolioReturns.reduce(function (total, value) {
      return total + Math.pow(Math.min(0, value), 2);
    }, 0) / portfolioReturns.length) * Math.sqrt(periodsPerYear);
    var annualizedArithmeticReturn = portfolioMean * periodsPerYear;
    var annualizedExcessRiskFreeReturn = excessRiskFreeMean * periodsPerYear;
    var annualizedActiveReturn = activeMean * periodsPerYear;
    var trackingError = finite(activeDeviation) ? activeDeviation * Math.sqrt(periodsPerYear) : null;
    var estimatedAnnualizedReturn = Math.pow(
      points[points.length - 1].portfolio_unit_nav / points[0].portfolio_unit_nav,
      1 / elapsedYears
    ) - 1;
    var cumulativeRiskFreeGrowth = riskFreeReturns.reduce(function (growth, value) {
      return growth * (1 + value);
    }, 1);
    var annualizedRiskFreeReturn = Math.pow(cumulativeRiskFreeGrowth, 1 / elapsedYears) - 1;
    var beta = finite(portfolioBenchmarkCovariance) && finite(benchmarkDeviation) && benchmarkDeviation > 0
      ? portfolioBenchmarkCovariance / Math.pow(benchmarkDeviation, 2)
      : null;
    var correlation = finite(portfolioBenchmarkCovariance) && finite(portfolioDeviation) && portfolioDeviation > 0 && finite(benchmarkDeviation) && benchmarkDeviation > 0
      ? portfolioBenchmarkCovariance / (portfolioDeviation * benchmarkDeviation)
      : null;

    return {
      estimated_annualized_return: finite(estimatedAnnualizedReturn) ? estimatedAnnualizedReturn : null,
      annualized_volatility: annualizedVolatility,
      sharpe_ratio: finite(annualizedVolatility) && annualizedVolatility > 0 ? annualizedExcessRiskFreeReturn / annualizedVolatility : null,
      sortino_ratio: finite(downsideDeviation) && downsideDeviation > 0 ? annualizedArithmeticReturn / downsideDeviation : null,
      calmar_ratio: finite(maximumDrawdown) && maximumDrawdown < 0 ? estimatedAnnualizedReturn / Math.abs(maximumDrawdown) : null,
      beta_csi300: beta,
      correlation_csi300: correlation,
      tracking_error: trackingError,
      information_ratio: finite(trackingError) && trackingError > 0 ? annualizedActiveReturn / trackingError : null,
      positive_interval_ratio: portfolioReturns.filter(function (value) { return value > 0; }).length / portfolioReturns.length,
      risk_free_rate: finite(annualizedRiskFreeReturn) ? annualizedRiskFreeReturn : null,
      risk_free_latest_annual_yield: points[points.length - 1].risk_free_annual_yield,
      risk_free_tenor: "1Y",
      risk_free_day_count: "ACT/365",
      risk_free_alignment: "interval_start_previous_published_workday",
      minimum_acceptable_return: 0,
      elapsed_days: elapsedDays,
      interval_count: portfolioReturns.length,
      annualization_periods_per_year: periodsPerYear,
      methodology: "calendar_geometric_return_empirical_interval_risk_metrics_cgb_1y_interval_excess"
    };
  }

  function buildAnalysis(bundle) {
    var points = orderedPoints(bundle && bundle.portfolio && bundle.portfolio.performance_points);
    var monthly = monthlyPerformance(points);
    var portfolioDrawdown = drawdownSeries(points, "portfolio_unit_nav");
    var benchmarkDrawdown = drawdownSeries(points, "csi300_nav");
    var star50Drawdown = drawdownSeries(points, "star50_nav");
    var chinextDrawdown = drawdownSeries(points, "chinext_nav");
    var deepestPortfolioDrawdown = extreme(portfolioDrawdown, "value", "min");
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
      deepest_portfolio_drawdown: deepestPortfolioDrawdown,
      deepest_benchmark_drawdown: extreme(benchmarkDrawdown, "value", "min"),
      deepest_star50_drawdown: extreme(star50Drawdown, "value", "min"),
      deepest_chinext_drawdown: extreme(chinextDrawdown, "value", "min"),
      quantitative_metrics: quantitativeMetrics(points, deepestPortfolioDrawdown ? deepestPortfolioDrawdown.value : null)
    };
  }

  return {
    buildAnalysis: buildAnalysis,
    drawdownSeries: drawdownSeries,
    monthlyPerformance: monthlyPerformance,
    orderedPoints: orderedPoints,
    quantitativeMetrics: quantitativeMetrics
  };
});

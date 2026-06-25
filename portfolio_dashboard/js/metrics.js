(function () {
  "use strict";

  var MS_PER_DAY = window.DashboardData.MS_PER_DAY;

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function sum(rows, field) {
    return rows.reduce(function (total, row) {
      return total + (finite(row[field]) ? row[field] : 0);
    }, 0);
  }

  function mean(values) {
    var nums = values.filter(finite);
    if (!nums.length) return null;
    return nums.reduce(function (a, b) { return a + b; }, 0) / nums.length;
  }

  function std(values) {
    var nums = values.filter(finite);
    if (nums.length < 2) return null;
    var avg = mean(nums);
    var variance = nums.reduce(function (acc, value) {
      return acc + Math.pow(value - avg, 2);
    }, 0) / (nums.length - 1);
    return Math.sqrt(variance);
  }

  function covariance(xs, ys) {
    var pairs = [];
    xs.forEach(function (x, index) {
      var y = ys[index];
      if (finite(x) && finite(y)) pairs.push([x, y]);
    });
    if (pairs.length < 2) return null;
    var xAvg = mean(pairs.map(function (pair) { return pair[0]; }));
    var yAvg = mean(pairs.map(function (pair) { return pair[1]; }));
    return pairs.reduce(function (acc, pair) {
      return acc + (pair[0] - xAvg) * (pair[1] - yAvg);
    }, 0) / (pairs.length - 1);
  }

  function variance(values) {
    var nums = values.filter(finite);
    if (nums.length < 2) return null;
    var avg = mean(nums);
    return nums.reduce(function (acc, value) {
      return acc + Math.pow(value - avg, 2);
    }, 0) / (nums.length - 1);
  }

  function byDateRange(rows, startDate, endDate, dateField) {
    return rows.filter(function (row) {
      if (startDate && row[dateField] < startDate) return false;
      if (endDate && row[dateField] > endDate) return false;
      return true;
    });
  }

  function ensureNavReturns(navRows, benchmarkField) {
    var previousBenchmarkNav = null;
    return navRows.map(function (row, index) {
      var prev = navRows[index - 1];
      var rawBenchmarkNav = benchmarkField && finite(row[benchmarkField]) ? row[benchmarkField] : null;
      var benchmarkNav = rawBenchmarkNav;
      if (rawBenchmarkNav === null && previousBenchmarkNav !== null) benchmarkNav = previousBenchmarkNav;
      var portfolioReturn = finite(row.portfolio_return) ? row.portfolio_return : null;
      var benchmarkReturn = benchmarkField === "benchmark_nav" && rawBenchmarkNav !== null && finite(row.benchmark_return) ? row.benchmark_return : null;
      if (portfolioReturn === null && prev && finite(prev.portfolio_nav) && prev.portfolio_nav !== 0) {
        portfolioReturn = row.portfolio_nav / prev.portfolio_nav - 1;
      }
      if (benchmarkReturn === null && finite(previousBenchmarkNav) && previousBenchmarkNav !== 0 && finite(benchmarkNav)) {
        benchmarkReturn = benchmarkNav / previousBenchmarkNav - 1;
      }
      if (rawBenchmarkNav !== null) previousBenchmarkNav = rawBenchmarkNav;
      return Object.assign({}, row, {
        benchmark_nav_selected: benchmarkNav,
        portfolio_return_calc: index === 0 && portfolioReturn === null ? 0 : portfolioReturn,
        benchmark_return_calc: benchmarkReturn,
        daily_excess_return: finite(portfolioReturn) && finite(benchmarkReturn) ? portfolioReturn - benchmarkReturn : null
      });
    });
  }

  function firstLastValid(series) {
    var first = null;
    var last = null;
    (series || []).forEach(function (point) {
      if (!first && finite(point.value)) first = point;
      if (finite(point.value)) last = point;
    });
    return { first: first, last: last };
  }

  function forwardFillSeries(navRows, field, label) {
    var last = null;
    return (navRows || []).map(function (row) {
      var raw = finite(row[field]) ? row[field] : null;
      if (raw !== null) last = raw;
      return {
        date: row.date,
        dateObj: row.dateObj,
        value: last,
        rawValue: raw,
        field: field,
        label: label || field,
        filled: raw === null && last !== null
      };
    });
  }

  function calculateReturnsFromNav(series) {
    var previous = null;
    return (series || []).map(function (point) {
      var dailyReturn = null;
      if (finite(point.value) && finite(previous) && previous !== 0) {
        dailyReturn = point.value / previous - 1;
      }
      if (finite(point.value)) previous = point.value;
      return Object.assign({}, point, { daily_return: dailyReturn });
    });
  }

  function calculateDrawdown(navSeries) {
    var runningMax = -Infinity;
    return (navSeries || []).map(function (point) {
      if (!finite(point.value)) {
        return Object.assign({}, point, { value: null });
      }
      runningMax = Math.max(runningMax, point.value);
      return {
        date: point.date,
        dateObj: point.dateObj,
        value: point.value / runningMax - 1,
        field: point.field,
        label: point.label
      };
    });
  }

  function alignSeriesByDate(seriesList) {
    if (!seriesList || !seriesList.length) return [];
    var master = seriesList[0].points || [];
    return master.map(function (point) {
      var row = { date: point.date, dateObj: point.dateObj };
      seriesList.forEach(function (series) {
        var match = (series.points || []).find(function (item) { return item.date === point.date; });
        row[series.field || series.name] = match && finite(match.value) ? match.value : null;
      });
      return row;
    });
  }

  function correlation(xs, ys) {
    var cov = covariance(xs, ys);
    var xStd = std(xs);
    var yStd = std(ys);
    if (cov === null || !xStd || !yStd) return null;
    return cov / (xStd * yStd);
  }

  function pairedReturns(portfolioReturnRows, benchmarkReturnRows) {
    var rows = [];
    var byDate = {};
    (benchmarkReturnRows || []).forEach(function (row) {
      byDate[row.date] = row.daily_return;
    });
    (portfolioReturnRows || []).forEach(function (row) {
      var benchmarkReturn = byDate[row.date];
      if (finite(row.daily_return) && finite(benchmarkReturn)) {
        rows.push({ date: row.date, portfolio: row.daily_return, benchmark: benchmarkReturn });
      }
    });
    return rows;
  }

  function seriesStats(navSeries) {
    var endpoints = firstLastValid(navSeries);
    if (!endpoints.first || !endpoints.last || endpoints.first.value === 0) {
      return {
        totalReturn: null,
        annualizedReturn: null,
        annualizedVolatility: null,
        maxDrawdown: null,
        sharpeRatio: null,
        returns: calculateReturnsFromNav(navSeries)
      };
    }
    var days = Math.max(1, Math.round((endpoints.last.dateObj - endpoints.first.dateObj) / MS_PER_DAY));
    var returns = calculateReturnsFromNav(navSeries);
    var returnValues = returns.map(function (row) { return row.daily_return; }).filter(finite);
    var totalReturn = endpoints.last.value / endpoints.first.value - 1;
    var annualizedReturn = Math.pow(endpoints.last.value / endpoints.first.value, 365 / days) - 1;
    var volatility = std(returnValues);
    var annualizedVolatility = volatility === null ? null : volatility * Math.sqrt(252);
    var drawdowns = calculateDrawdown(navSeries).map(function (point) { return point.value; }).filter(finite);
    var maxDrawdown = drawdowns.length ? Math.min.apply(null, drawdowns) : null;
    return {
      totalReturn: totalReturn,
      annualizedReturn: annualizedReturn,
      annualizedVolatility: annualizedVolatility,
      maxDrawdown: maxDrawdown,
      sharpeRatio: annualizedVolatility ? annualizedReturn / annualizedVolatility : null,
      returns: returns
    };
  }

  function calculateExcessNav(portfolioReturns, benchmarkReturns) {
    var pairs = pairedReturns(portfolioReturns, benchmarkReturns);
    var excessNav = 1;
    return pairs.map(function (row) {
      excessNav *= 1 + (row.portfolio - row.benchmark);
      return { date: row.date, dateObj: portfolioReturns.find(function (item) { return item.date === row.date; }).dateObj, value: excessNav };
    });
  }

  function calculateBenchmarkMetrics(portfolioSeries, benchmarkSeries) {
    var portfolioStats = seriesStats(portfolioSeries);
    var benchmarkStats = seriesStats(benchmarkSeries);
    var pairs = pairedReturns(portfolioStats.returns, benchmarkStats.returns);
    var portfolioValues = pairs.map(function (row) { return row.portfolio; });
    var benchmarkValues = pairs.map(function (row) { return row.benchmark; });
    var dailyExcess = pairs.map(function (row) { return row.portfolio - row.benchmark; });
    var trackingErrorRaw = std(dailyExcess);
    var trackingError = trackingErrorRaw === null ? null : trackingErrorRaw * Math.sqrt(252);
    var benchmarkVariance = variance(benchmarkValues);
    var portfolioBenchmarkCovariance = covariance(portfolioValues, benchmarkValues);
    var betaValue = benchmarkVariance && portfolioBenchmarkCovariance !== null
      ? portfolioBenchmarkCovariance / benchmarkVariance
      : null;
    var annualizedExcessReturn = finite(portfolioStats.annualizedReturn) && finite(benchmarkStats.annualizedReturn)
      ? portfolioStats.annualizedReturn - benchmarkStats.annualizedReturn
      : null;
    return {
      totalReturn: benchmarkStats.totalReturn,
      annualizedReturn: benchmarkStats.annualizedReturn,
      annualizedVolatility: benchmarkStats.annualizedVolatility,
      maxDrawdown: benchmarkStats.maxDrawdown,
      sharpeRatio: benchmarkStats.sharpeRatio,
      correlation: correlation(portfolioValues, benchmarkValues),
      beta: betaValue,
      trackingError: trackingError,
      informationRatio: trackingError ? annualizedExcessReturn / trackingError : null,
      excessReturn: finite(portfolioStats.totalReturn) && finite(benchmarkStats.totalReturn)
        ? portfolioStats.totalReturn - benchmarkStats.totalReturn
        : null,
      annualizedExcessReturn: annualizedExcessReturn,
      pairedCount: pairs.length
    };
  }

  function computePerformance(navRows, benchmarkField) {
    if (navRows.length < 2) {
      return {
        kpis: {},
        navSeries: [],
        benchmarkSeries: [],
        excessSeries: [],
        drawdownSeries: [],
        rolling20Vol: [],
        rolling60Beta: [],
        monthly: []
      };
    }
    var rows = ensureNavReturns(navRows, benchmarkField);
    var first = rows[0];
    var last = rows[rows.length - 1];
    var days = Math.max(1, Math.round((last.dateObj - first.dateObj) / MS_PER_DAY));
    var returns = rows.slice(1).map(function (row) { return row.portfolio_return_calc; }).filter(finite);
    var benchmarkReturns = rows.slice(1).map(function (row) { return row.benchmark_return_calc; }).filter(finite);
    var totalReturn = last.portfolio_nav / first.portfolio_nav - 1;
    var benchmarkSeries = rows.map(function (row) { return { date: row.date, dateObj: row.dateObj, value: row.benchmark_nav_selected }; });
    var benchmarkEndpoints = firstLastValid(benchmarkSeries);
    var benchmarkTotalReturn = benchmarkEndpoints.first && benchmarkEndpoints.last && benchmarkEndpoints.first.value !== 0
      ? benchmarkEndpoints.last.value / benchmarkEndpoints.first.value - 1
      : null;
    var annualizedReturn = Math.pow(last.portfolio_nav / first.portfolio_nav, 365 / days) - 1;
    var benchmarkDays = benchmarkEndpoints.first && benchmarkEndpoints.last
      ? Math.max(1, Math.round((benchmarkEndpoints.last.dateObj - benchmarkEndpoints.first.dateObj) / MS_PER_DAY))
      : days;
    var benchmarkAnnualizedReturn = benchmarkEndpoints.first && benchmarkEndpoints.last && benchmarkEndpoints.first.value !== 0
      ? Math.pow(benchmarkEndpoints.last.value / benchmarkEndpoints.first.value, 365 / benchmarkDays) - 1
      : null;
    var volatility = std(returns);
    var benchmarkVolatility = std(benchmarkReturns);
    var annualizedVolatility = volatility === null ? null : volatility * Math.sqrt(252);
    var benchmarkAnnualizedVolatility = benchmarkVolatility === null ? null : benchmarkVolatility * Math.sqrt(252);
    var sharpe = annualizedVolatility && annualizedVolatility !== 0 ? annualizedReturn / annualizedVolatility : null;
    var winCount = returns.filter(function (value) { return value > 0; }).length;
    var winRate = returns.length ? winCount / returns.length : null;

    var runningMax = -Infinity;
    var maxDrawdown = 0;
    var duration = 0;
    var maxDuration = 0;
    var drawdownSeries = rows.map(function (row) {
      runningMax = Math.max(runningMax, row.portfolio_nav);
      var drawdown = row.portfolio_nav / runningMax - 1;
      if (drawdown < 0) duration += 1;
      else duration = 0;
      maxDuration = Math.max(maxDuration, duration);
      maxDrawdown = Math.min(maxDrawdown, drawdown);
      return { date: row.date, dateObj: row.dateObj, value: drawdown };
    });

    var cumulativeExcess = 1;
    var excessSeries = rows.map(function (row) {
      if (finite(row.daily_excess_return)) cumulativeExcess *= 1 + row.daily_excess_return;
      return { date: row.date, dateObj: row.dateObj, value: cumulativeExcess };
    });

    var rolling20Vol = [];
    var rolling60Beta = [];
    rows.forEach(function (row, index) {
      if (index >= 20) {
        var windowReturns = rows.slice(index - 19, index + 1).map(function (item) { return item.portfolio_return_calc; });
        var vol = std(windowReturns);
        if (vol !== null) rolling20Vol.push({ date: row.date, dateObj: row.dateObj, value: vol * Math.sqrt(252) });
      }
      if (index >= 60) {
        var p = rows.slice(index - 59, index + 1).map(function (item) { return item.portfolio_return_calc; });
        var b = rows.slice(index - 59, index + 1).map(function (item) { return item.benchmark_return_calc; });
        var cov = covariance(p, b);
        var v = variance(b);
        if (cov !== null && v) rolling60Beta.push({ date: row.date, dateObj: row.dateObj, value: cov / v });
      }
    });

    var monthlyMap = {};
    rows.forEach(function (row) {
      var key = row.date.slice(0, 7);
      if (!monthlyMap[key]) monthlyMap[key] = { month: key, first: row, last: row };
      monthlyMap[key].last = row;
    });
    var monthly = Object.keys(monthlyMap).sort().map(function (key) {
      var bucket = monthlyMap[key];
      return {
        month: key,
        year: key.slice(0, 4),
        monthNumber: Number(key.slice(5, 7)),
        value: bucket.last.portfolio_nav / bucket.first.portfolio_nav - 1
      };
    });

    return {
      kpis: {
        total_return: totalReturn,
        benchmark_total_return: benchmarkTotalReturn,
        excess_return: finite(benchmarkTotalReturn) ? totalReturn - benchmarkTotalReturn : null,
        annualized_return: annualizedReturn,
        benchmark_annualized_return: benchmarkAnnualizedReturn,
        annualized_excess_return: finite(benchmarkAnnualizedReturn) ? annualizedReturn - benchmarkAnnualizedReturn : null,
        annualized_volatility: annualizedVolatility,
        benchmark_annualized_volatility: benchmarkAnnualizedVolatility,
        sharpe_ratio: sharpe,
        calmar_ratio: maxDrawdown < 0 ? annualizedReturn / Math.abs(maxDrawdown) : null,
        win_rate: winRate,
        win_count: winCount,
        max_drawdown: maxDrawdown,
        drawdown_duration: maxDuration,
        start_date: first.date,
        end_date: last.date,
        trading_days: returns.length
      },
      enrichedNav: rows,
      navSeries: rows.map(function (row) { return { date: row.date, dateObj: row.dateObj, value: row.portfolio_nav }; }),
      benchmarkSeries: benchmarkSeries,
      excessSeries: excessSeries,
      drawdownSeries: drawdownSeries,
      rolling20Vol: rolling20Vol,
      rolling60Beta: rolling60Beta,
      monthly: monthly
    };
  }

  function groupSum(rows, keyField, valueField) {
    var map = {};
    rows.forEach(function (row) {
      var key = row[keyField] || "UNCLASSIFIED";
      if (!map[key]) map[key] = { label: key, value: 0, count: 0 };
      map[key].value += finite(row[valueField]) ? row[valueField] : 0;
      map[key].count += 1;
    });
    return Object.keys(map).map(function (key) { return map[key]; });
  }

  function latestDateRows(rows, dateField) {
    if (!rows.length) return [];
    var latest = rows.reduce(function (max, row) {
      return row[dateField] > max ? row[dateField] : max;
    }, rows[0][dateField]);
    return rows.filter(function (row) { return row[dateField] === latest; });
  }

  function computeAttribution(positions) {
    var rowsWithContribution = positions.filter(function (row) { return finite(row.contribution); });
    var tickerMap = {};
    rowsWithContribution.forEach(function (row) {
      var key = row.ticker + " " + row.name;
      if (!tickerMap[key]) {
        tickerMap[key] = {
          label: key,
          ticker: row.ticker,
          name: row.name,
          value: 0,
          weightSum: 0,
          observations: 0
        };
      }
      tickerMap[key].value += row.contribution;
      tickerMap[key].weightSum += finite(row.weight) ? row.weight : 0;
      tickerMap[key].observations += 1;
    });
    var tickerRows = Object.keys(tickerMap).map(function (key) {
      var item = tickerMap[key];
      item.avg_weight = item.observations ? item.weightSum / item.observations : null;
      return item;
    });
    var top = tickerRows.slice().sort(function (a, b) { return b.value - a.value; }).slice(0, 10);
    var bottom = tickerRows.slice().sort(function (a, b) { return a.value - b.value; }).slice(0, 10);
    var themeContribution = groupSum(rowsWithContribution, "theme", "contribution")
      .sort(function (a, b) { return b.value - a.value; });
    var sectorRows = rowsWithContribution.filter(function (row) { return row.sector; });
    var sectorContribution = groupSum(sectorRows, "sector", "contribution")
      .sort(function (a, b) { return b.value - a.value; });
    return {
      top: top,
      bottom: bottom,
      themeContribution: themeContribution,
      sectorContribution: sectorContribution,
      scatter: tickerRows.filter(function (row) { return finite(row.avg_weight) && finite(row.value); })
    };
  }

  function trendByGroup(rows, groupField, valueField) {
    var byDate = {};
    rows.forEach(function (row) {
      if (!row[groupField]) return;
      if (!byDate[row.date]) byDate[row.date] = { date: row.date, dateObj: row.dateObj };
      byDate[row.date][row[groupField]] = (byDate[row.date][row[groupField]] || 0) + (finite(row[valueField]) ? row[valueField] : 0);
    });
    return Object.keys(byDate).sort().map(function (date) { return byDate[date]; });
  }

  function computeHoldings(positions) {
    var latestRows = latestDateRows(positions, "date").sort(function (a, b) {
      return b.weight - a.weight;
    });
    var currentThemeWeight = groupSum(latestRows, "theme", "weight").sort(function (a, b) { return b.value - a.value; });
    var currentSectorWeight = groupSum(latestRows.filter(function (row) { return row.sector; }), "sector", "weight")
      .sort(function (a, b) { return b.value - a.value; });
    var marketValueTheme = groupSum(latestRows.filter(function (row) { return finite(row.market_value); }), "theme", "market_value")
      .sort(function (a, b) { return b.value - a.value; });
    var concentrationTrend = [];
    var byDate = {};
    positions.forEach(function (row) {
      if (!byDate[row.date]) byDate[row.date] = [];
      byDate[row.date].push(row);
    });
    Object.keys(byDate).sort().forEach(function (date) {
      var rows = byDate[date].slice().sort(function (a, b) { return b.weight - a.weight; });
      concentrationTrend.push({
        date: date,
        dateObj: rows[0].dateObj,
        top5: rows.slice(0, 5).reduce(function (total, row) { return total + row.weight; }, 0),
        top10: rows.slice(0, 10).reduce(function (total, row) { return total + row.weight; }, 0),
        hhi: rows.reduce(function (total, row) { return total + row.weight * row.weight; }, 0)
      });
    });
    return {
      latestDate: latestRows.length ? latestRows[0].date : null,
      top20: latestRows.slice(0, 20),
      top10: latestRows.slice(0, 10),
      currentThemeWeight: currentThemeWeight,
      currentSectorWeight: currentSectorWeight,
      marketValueTheme: marketValueTheme,
      themeWeightTrend: trendByGroup(positions, "theme", "weight"),
      sectorWeightTrend: trendByGroup(positions.filter(function (row) { return row.sector; }), "sector", "weight"),
      concentrationTrend: concentrationTrend,
      top5Weight: concentrationTrend.length ? concentrationTrend[concentrationTrend.length - 1].top5 : null,
      top10Weight: concentrationTrend.length ? concentrationTrend[concentrationTrend.length - 1].top10 : null,
      hhi: concentrationTrend.length ? concentrationTrend[concentrationTrend.length - 1].hhi : null
    };
  }

  function computeTrades(trades) {
    if (!trades.length) {
      return {
        available: false,
        message: "未上传交易数据。",
        recent: [],
        bySide: [],
        byTheme: [],
        byReason: [],
        totals: {}
      };
    }
    var buyCount = trades.filter(function (row) { return row.side === "buy"; }).length;
    var sellCount = trades.filter(function (row) { return row.side === "sell"; }).length;
    return {
      available: true,
      message: "缺少价格序列，暂不计算 post-trade return。",
      recent: trades.slice().sort(function (a, b) { return b.dateObj - a.dateObj; }).slice(0, 20),
      bySide: [
        { label: "buy", value: buyCount },
        { label: "sell", value: sellCount }
      ],
      byTheme: groupSum(trades, "theme", "trade_amount").sort(function (a, b) { return b.value - a.value; }),
      byReason: groupSum(trades.filter(function (row) { return row.reason; }), "reason", "trade_amount")
        .sort(function (a, b) { return b.value - a.value; }),
      totals: {
        buy_count: buyCount,
        sell_count: sellCount,
        trade_amount: sum(trades, "trade_amount"),
        fee: sum(trades, "fee"),
        trade_count: trades.length
      }
    };
  }

  function defaultBenchmarkField(benchmarks) {
    var fields = (benchmarks || []).map(function (benchmark) { return benchmark.field; });
    if (fields.indexOf("benchmark_main_nav") >= 0) return "benchmark_main_nav";
    if (fields.indexOf("benchmark_nav") >= 0) return "benchmark_nav";
    return fields[0] || "";
  }

  function benchmarkByField(benchmarks) {
    var map = {};
    (benchmarks || []).forEach(function (benchmark) {
      map[benchmark.field] = benchmark;
    });
    return map;
  }

  function computeCorrelationMatrix(seriesItems) {
    var matrix = [];
    seriesItems.forEach(function (rowItem) {
      var rowStats = seriesStats(rowItem.series);
      var row = {
        field: rowItem.field,
        label: rowItem.label,
        values: []
      };
      seriesItems.forEach(function (colItem) {
        var colStats = seriesStats(colItem.series);
        var pairs = pairedReturns(rowStats.returns, colStats.returns);
        row.values.push({
          field: colItem.field,
          label: colItem.label,
          value: correlation(
            pairs.map(function (pair) { return pair.portfolio; }),
            pairs.map(function (pair) { return pair.benchmark; })
          )
        });
      });
      matrix.push(row);
    });
    return matrix;
  }

  function computeBenchmarkComparison(navRows, benchmarks, mainField, selectedFields) {
    var benchmarkMap = benchmarkByField(benchmarks);
    var availableFields = (benchmarks || []).map(function (benchmark) { return benchmark.field; });
    var benchmarkField = mainField || defaultBenchmarkField(benchmarks);
    var hasExplicitSelection = Array.isArray(selectedFields);
    var selected = (hasExplicitSelection ? selectedFields : []).filter(function (field) {
      return availableFields.indexOf(field) >= 0;
    });
    if (!selected.length && !hasExplicitSelection) selected = availableFields.slice(0, 3);

    var portfolioSeries = (navRows || []).map(function (row) {
      return { date: row.date, dateObj: row.dateObj, value: row.portfolio_nav, field: "portfolio_nav", label: "Portfolio" };
    });
    var portfolioStats = seriesStats(portfolioSeries);
    var benchmarkSeriesMap = {};
    availableFields.forEach(function (field) {
      var label = benchmarkMap[field] ? benchmarkMap[field].label : field;
      benchmarkSeriesMap[field] = forwardFillSeries(navRows, field, label);
    });
    var selectedBenchmarks = selected.map(function (field) {
      var meta = benchmarkMap[field] || { field: field, label: field };
      var series = benchmarkSeriesMap[field] || [];
      var metrics = calculateBenchmarkMetrics(portfolioSeries, series);
      return {
        field: field,
        label: meta.label,
        isMain: field === benchmarkField,
        series: series,
        returns: seriesStats(series).returns,
        drawdown: calculateDrawdown(series),
        excessSeries: calculateExcessNav(portfolioStats.returns, seriesStats(series).returns),
        metrics: metrics
      };
    });
    var comparisonRows = availableFields.map(function (field) {
      var meta = benchmarkMap[field] || { field: field, label: field };
      var metrics = calculateBenchmarkMetrics(portfolioSeries, benchmarkSeriesMap[field]);
      return Object.assign({
        field: field,
        name: meta.label,
        isMain: field === benchmarkField,
        selected: selected.indexOf(field) >= 0
      }, metrics);
    });
    var portfolioRow = {
      field: "portfolio_nav",
      name: "Portfolio",
      isPortfolio: true,
      selected: true,
      totalReturn: portfolioStats.totalReturn,
      annualizedReturn: portfolioStats.annualizedReturn,
      annualizedVolatility: portfolioStats.annualizedVolatility,
      maxDrawdown: portfolioStats.maxDrawdown,
      sharpeRatio: portfolioStats.sharpeRatio,
      correlation: 1,
      beta: null,
      trackingError: null,
      informationRatio: null,
      excessReturn: null,
      annualizedExcessReturn: null
    };
    var scatterRows = [portfolioRow].concat(selectedBenchmarks.map(function (benchmark) {
      return Object.assign({
        field: benchmark.field,
        name: benchmark.label,
        isMain: benchmark.isMain
      }, benchmark.metrics);
    })).filter(function (row) {
      return finite(row.annualizedReturn) && finite(row.annualizedVolatility);
    });
    var matrixItems = [{
      field: "portfolio_nav",
      label: "Portfolio",
      series: portfolioSeries
    }].concat(selectedBenchmarks.map(function (benchmark) {
      return { field: benchmark.field, label: benchmark.label, series: benchmark.series };
    }));
    return {
      availableBenchmarks: benchmarks || [],
      mainField: benchmarkField,
      mainLabel: benchmarkMap[benchmarkField] ? benchmarkMap[benchmarkField].label : benchmarkField,
      selectedFields: selected,
      portfolioSeries: portfolioSeries,
      portfolioReturns: portfolioStats.returns,
      portfolioDrawdown: calculateDrawdown(portfolioSeries),
      selectedBenchmarks: selectedBenchmarks,
      comparisonRows: comparisonRows,
      portfolioRow: portfolioRow,
      scatterRows: scatterRows,
      correlationMatrix: computeCorrelationMatrix(matrixItems),
      alignedSeries: alignSeriesByDate([{ field: "portfolio_nav", points: portfolioSeries }].concat(selectedBenchmarks.map(function (benchmark) {
        return { field: benchmark.field, points: benchmark.series };
      })))
    };
  }

  function computeDashboard(dataset, filters) {
    var startDate = filters.startDate || "";
    var endDate = filters.endDate || "";
    var benchmarkField = filters.benchmarkField || defaultBenchmarkField(dataset.benchmarks || []);
    var navRows = byDateRange(dataset.nav || [], startDate, endDate, "date");
    var positions = byDateRange(dataset.positions || [], startDate, endDate, "date");
    var trades = byDateRange(dataset.trades || [], startDate, endDate, "trade_date");
    var performance = computePerformance(navRows, benchmarkField);
    var attribution = computeAttribution(positions);
    var holdings = computeHoldings(positions);
    var tradeReview = computeTrades(trades);
    var benchmarkComparison = computeBenchmarkComparison(
      navRows,
      dataset.benchmarks || [],
      benchmarkField,
      filters.selectedBenchmarkFields
    );
    return {
      filters: filters,
      performance: performance,
      benchmarkComparison: benchmarkComparison,
      attribution: attribution,
      holdings: holdings,
      trades: tradeReview,
      navRows: navRows,
      positions: positions,
      tradeRows: trades
    };
  }

  window.DashboardMetrics = {
    computeDashboard: computeDashboard,
    computePerformance: computePerformance,
    computeBenchmarkComparison: computeBenchmarkComparison,
    calculateReturnsFromNav: calculateReturnsFromNav,
    calculateBenchmarkMetrics: calculateBenchmarkMetrics,
    calculateExcessNav: calculateExcessNav,
    calculateDrawdown: calculateDrawdown,
    alignSeriesByDate: alignSeriesByDate,
    computeAttribution: computeAttribution,
    computeHoldings: computeHoldings,
    computeTrades: computeTrades,
    mean: mean,
    std: std
  };
})();

(function () {
  "use strict";

  var MS_PER_DAY = window.DashboardData.MS_PER_DAY;
  var TRADING_DAYS_PER_YEAR = 252;

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function formatQuantity(value) {
    if (!finite(value)) return "-";
    return Math.abs(value - Math.round(value)) < 0.000001 ? String(Math.round(value)) : value.toFixed(3);
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

  function annualizedFromTotalReturn(totalReturn, tradingDays) {
    if (!finite(totalReturn) || !finite(tradingDays) || tradingDays <= 0) return null;
    var base = 1 + totalReturn;
    if (base <= 0) return null;
    return Math.pow(base, TRADING_DAYS_PER_YEAR / tradingDays) - 1;
  }

  function formalOpenDates(calendar, startDate, endDate, excludeStart) {
    if (!calendar || calendar.status !== "available" || !Array.isArray(calendar.expected_open_dates)) return [];
    return calendar.expected_open_dates.filter(function (date) {
      if (startDate && (excludeStart ? date <= startDate : date < startDate)) return false;
      if (endDate && date > endDate) return false;
      return true;
    });
  }

  function returnDateSet(returns) {
    var dates = {};
    (returns || []).forEach(function (row) {
      if (finite(row.daily_return)) dates[row.date] = true;
    });
    return dates;
  }

  function annualizationEligibility(firstPoint, lastPoint, returns, calendar) {
    var expectedDates = firstPoint && lastPoint
      ? formalOpenDates(calendar, firstPoint.date, lastPoint.date, true)
      : [];
    var validDates = returnDateSet(returns);
    var validReturnCount = expectedDates.filter(function (date) { return validDates[date]; }).length;
    var coverage = expectedDates.length > 0 ? validReturnCount / expectedDates.length : 0;
    var calendarAvailable = Boolean(calendar && calendar.status === "available");
    var eligible = calendarAvailable && validReturnCount >= 60 && coverage >= 0.95;
    return {
      eligible: eligible,
      status: !calendarAvailable ? "formal_trading_calendar_missing" : eligible ? "eligible" : "insufficient_daily_history",
      validReturnCount: validReturnCount,
      expectedOpenDayCount: expectedDates.length,
      openDayCoverage: coverage,
      coverageBasis: "strict_parquet_trade_date_mask"
    };
  }

  function strictMaskedReturns(returns, calendar, startDate, endDate) {
    var allowed = {};
    formalOpenDates(calendar, startDate, endDate, true).forEach(function (date) {
      allowed[date] = true;
    });
    return (returns || []).filter(function (row) {
      return allowed[row.date] && finite(row.daily_return);
    });
  }

  function strictSeriesPoints(series, calendar) {
    if (!calendar || calendar.status !== "available") return series || [];
    var allowed = {};
    (calendar.expected_open_dates || []).forEach(function (date) {
      allowed[date] = true;
    });
    return (series || []).filter(function (point) { return allowed[point.date]; });
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

  function ensureNavReturns(navRows, benchmarkField, recomputeFromPoints) {
    var previousBenchmarkNav = null;
    return navRows.map(function (row, index) {
      var prev = navRows[index - 1];
      var rawBenchmarkNav = benchmarkField && finite(row[benchmarkField]) ? row[benchmarkField] : null;
      var benchmarkNav = rawBenchmarkNav;
      if (rawBenchmarkNav === null && previousBenchmarkNav !== null) benchmarkNav = previousBenchmarkNav;
      var portfolioReturn = !recomputeFromPoints && finite(row.portfolio_return) ? row.portfolio_return : null;
      var benchmarkReturn = !recomputeFromPoints && benchmarkField === "benchmark_nav" && rawBenchmarkNav !== null && finite(row.benchmark_return) ? row.benchmark_return : null;
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
        label: point.label,
        filled: Boolean(point.filled)
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

  function seriesStats(navSeries, tradingCalendar) {
    navSeries = strictSeriesPoints(navSeries, tradingCalendar);
    var endpoints = firstLastValid(navSeries);
    if (!endpoints.first || !endpoints.last || endpoints.first.value === 0) {
      return {
        totalReturn: null,
        annualizedReturn: null,
        annualizedVolatility: null,
        maxDrawdown: null,
        sharpeRatio: null,
        returns: calculateReturnsFromNav(navSeries),
        tradingDays: 0,
        annualization: {
          eligible: false,
          status: "insufficient_daily_history",
          validReturnCount: 0,
          expectedOpenDayCount: 0,
          openDayCoverage: 0,
          coverageBasis: "strict_parquet_trade_date_mask"
        }
      };
    }
    var returns = calculateReturnsFromNav(navSeries);
    var maskedReturns = strictMaskedReturns(
      returns,
      tradingCalendar,
      endpoints.first.date,
      endpoints.last.date
    );
    var returnValues = maskedReturns.map(function (row) { return row.daily_return; });
    var totalReturn = endpoints.last.value / endpoints.first.value - 1;
    var tradingDays = returnValues.length;
    var eligibility = annualizationEligibility(endpoints.first, endpoints.last, returns, tradingCalendar);
    var annualizedReturn = eligibility.eligible ? annualizedFromTotalReturn(totalReturn, eligibility.validReturnCount) : null;
    var volatility = std(returnValues);
    var annualizedVolatility = eligibility.eligible && volatility !== null ? volatility * Math.sqrt(TRADING_DAYS_PER_YEAR) : null;
    var drawdowns = calculateDrawdown(navSeries).map(function (point) { return point.value; }).filter(finite);
    var maxDrawdown = drawdowns.length ? Math.min.apply(null, drawdowns) : null;
    return {
      totalReturn: totalReturn,
      annualizedReturn: annualizedReturn,
      annualizedVolatility: annualizedVolatility,
      maxDrawdown: maxDrawdown,
      sharpeRatio: annualizedVolatility && annualizedReturn !== null ? annualizedReturn / annualizedVolatility : null,
      returns: returns,
      tradingDays: tradingDays,
      annualization: eligibility
    };
  }

  function calculateExcessNav(portfolioReturns, benchmarkReturns) {
    var pairs = pairedReturns(portfolioReturns, benchmarkReturns);
    var excessNav = 1;
    return pairs.map(function (row) {
      if (1 + row.benchmark <= 0) return { date: row.date, dateObj: null, value: null };
      excessNav *= (1 + row.portfolio) / (1 + row.benchmark);
      return { date: row.date, dateObj: portfolioReturns.find(function (item) { return item.date === row.date; }).dateObj, value: excessNav };
    });
  }

  function calculateBenchmarkMetrics(portfolioSeries, benchmarkSeries, tradingCalendar) {
    var portfolioStats = seriesStats(portfolioSeries, tradingCalendar);
    var benchmarkStats = seriesStats(benchmarkSeries, tradingCalendar);
    var pairs = pairedReturns(portfolioStats.returns, benchmarkStats.returns);
    var portfolioEndpoints = firstLastValid(portfolioSeries);
    var strictDates = {};
    if (portfolioEndpoints.first && portfolioEndpoints.last) {
      formalOpenDates(
        tradingCalendar,
        portfolioEndpoints.first.date,
        portfolioEndpoints.last.date,
        true
      ).forEach(function (date) { strictDates[date] = true; });
    }
    pairs = pairs.filter(function (row) { return strictDates[row.date]; });
    var portfolioValues = pairs.map(function (row) { return row.portfolio; });
    var benchmarkValues = pairs.map(function (row) { return row.benchmark; });
    var dailyExcess = pairs.map(function (row) { return row.portfolio - row.benchmark; });
    var trackingErrorRaw = std(dailyExcess);
    var pairedReturnRows = pairs.map(function (row) {
      return { date: row.date, daily_return: row.portfolio - row.benchmark };
    });
    var endpoints = portfolioEndpoints;
    var pairedEligibility = annualizationEligibility(
      endpoints.first,
      endpoints.last,
      pairedReturnRows,
      tradingCalendar
    );
    var trackingError = pairedEligibility.eligible && trackingErrorRaw !== null
      ? trackingErrorRaw * Math.sqrt(TRADING_DAYS_PER_YEAR)
      : null;
    var benchmarkVariance = variance(benchmarkValues);
    var portfolioBenchmarkCovariance = covariance(portfolioValues, benchmarkValues);
    var betaValue = benchmarkVariance && portfolioBenchmarkCovariance !== null
      ? portfolioBenchmarkCovariance / benchmarkVariance
      : null;
    var relativeTotal = finite(portfolioStats.totalReturn) && finite(benchmarkStats.totalReturn) && 1 + benchmarkStats.totalReturn > 0
      ? (1 + portfolioStats.totalReturn) / (1 + benchmarkStats.totalReturn) - 1
      : null;
    var annualizedExcessReturn = pairedEligibility.eligible
      ? annualizedFromTotalReturn(relativeTotal, pairedEligibility.validReturnCount)
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
      excessReturn: relativeTotal,
      annualizedExcessReturn: annualizedExcessReturn,
      pairedCount: pairs.length
    };
  }

  function computePerformance(navRows, benchmarkField, monthlyAnchor, tradingCalendar) {
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
    // Compress raw NAV and benchmark points to the canonical open-day mask
    // before deriving returns.  Returns computed across weekend/non-open
    // snapshots can otherwise move a jump to the wrong trading day and
    // contaminate every downstream metric.
    var rows = strictSeriesPoints(navRows, tradingCalendar);
    rows = ensureNavReturns(rows, benchmarkField, Boolean(tradingCalendar && tradingCalendar.status === "available"));
    if (rows.length < 2) {
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
    var first = rows[0];
    var last = rows[rows.length - 1];
    var totalReturn = last.portfolio_nav / first.portfolio_nav - 1;
    var benchmarkSeries = rows.map(function (row) { return { date: row.date, dateObj: row.dateObj, value: row.benchmark_nav_selected }; });
    var benchmarkEndpoints = firstLastValid(benchmarkSeries);
    var benchmarkTotalReturn = benchmarkEndpoints.first && benchmarkEndpoints.last && benchmarkEndpoints.first.value !== 0
      ? benchmarkEndpoints.last.value / benchmarkEndpoints.first.value - 1
      : null;
    var portfolioReturnRows = rows.slice(1).map(function (row) {
      return { date: row.date, daily_return: row.portfolio_return_calc };
    });
    var benchmarkReturnRows = rows.slice(1).map(function (row) {
      return { date: row.date, daily_return: row.benchmark_return_calc };
    });
    var returns = strictMaskedReturns(
      portfolioReturnRows,
      tradingCalendar,
      first.date,
      last.date
    ).map(function (row) { return row.daily_return; });
    var benchmarkReturns = benchmarkEndpoints.first && benchmarkEndpoints.last
      ? strictMaskedReturns(
        benchmarkReturnRows,
        tradingCalendar,
        benchmarkEndpoints.first.date,
        benchmarkEndpoints.last.date
      ).map(function (row) { return row.daily_return; })
      : [];
    var annualization = annualizationEligibility(first, last, portfolioReturnRows, tradingCalendar);
    var benchmarkAnnualization = benchmarkEndpoints.first && benchmarkEndpoints.last
      ? annualizationEligibility(benchmarkEndpoints.first, benchmarkEndpoints.last, benchmarkReturnRows, tradingCalendar)
      : annualizationEligibility(null, null, [], tradingCalendar);
    var annualizedReturn = annualization.eligible ? annualizedFromTotalReturn(totalReturn, annualization.validReturnCount) : null;
    var benchmarkAnnualizedReturn = benchmarkEndpoints.first && benchmarkEndpoints.last && benchmarkEndpoints.first.value !== 0 && benchmarkAnnualization.eligible
      ? annualizedFromTotalReturn(benchmarkTotalReturn, benchmarkAnnualization.validReturnCount)
      : null;
    var volatility = std(returns);
    var benchmarkVolatility = std(benchmarkReturns);
    var annualizedVolatility = annualization.eligible && volatility !== null ? volatility * Math.sqrt(TRADING_DAYS_PER_YEAR) : null;
    var benchmarkAnnualizedVolatility = benchmarkAnnualization.eligible && benchmarkVolatility !== null ? benchmarkVolatility * Math.sqrt(TRADING_DAYS_PER_YEAR) : null;
    var sharpe = annualizedVolatility && annualizedVolatility !== 0 && annualizedReturn !== null ? annualizedReturn / annualizedVolatility : null;
    var formalReturnDates = {};
    formalOpenDates(tradingCalendar, first.date, last.date, true).forEach(function (date) {
      formalReturnDates[date] = true;
    });
    var formalReturnValues = portfolioReturnRows.filter(function (row) {
      return formalReturnDates[row.date] && finite(row.daily_return);
    }).map(function (row) { return row.daily_return; });
    var winCount = formalReturnValues.filter(function (value) { return value > 0; }).length;
    var winRate = formalReturnValues.length ? winCount / formalReturnValues.length : null;

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
      if (finite(row.portfolio_return_calc) && finite(row.benchmark_return_calc) && 1 + row.benchmark_return_calc > 0) {
        cumulativeExcess *= (1 + row.portfolio_return_calc) / (1 + row.benchmark_return_calc);
      }
      return { date: row.date, dateObj: row.dateObj, value: cumulativeExcess };
    });

    var rolling20Vol = [];
    var rolling60Beta = [];
    var portfolioReturnByDate = {};
    var benchmarkReturnByDate = {};
    rows.forEach(function (row) {
      if (finite(row.portfolio_return_calc)) portfolioReturnByDate[row.date] = row.portfolio_return_calc;
      if (finite(row.benchmark_return_calc)) benchmarkReturnByDate[row.date] = row.benchmark_return_calc;
    });
    var allExpectedDates = formalOpenDates(
      tradingCalendar,
      rows.length ? rows[0].date : "",
      rows.length ? rows[rows.length - 1].date : "",
      true
    );
    rows.forEach(function (row) {
      var expectedThroughDate = allExpectedDates.filter(function (date) { return date <= row.date; });
      var expected20 = expectedThroughDate.slice(-20);
      var windowReturns = expected20.map(function (date) { return portfolioReturnByDate[date]; }).filter(finite);
      if (expected20.length === 20 && windowReturns.length / 20 >= 0.95) {
        var vol = std(windowReturns);
        if (vol !== null) rolling20Vol.push({ date: row.date, dateObj: row.dateObj, value: vol * Math.sqrt(TRADING_DAYS_PER_YEAR) });
      }
      var expected60 = expectedThroughDate.slice(-60);
      var pairedDates = expected60.filter(function (date) {
        return finite(portfolioReturnByDate[date]) && finite(benchmarkReturnByDate[date]);
      });
      if (expected60.length === 60 && pairedDates.length / 60 >= 0.95) {
        var p = pairedDates.map(function (date) { return portfolioReturnByDate[date]; });
        var b = pairedDates.map(function (date) { return benchmarkReturnByDate[date]; });
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
    var previousMonthEnd = monthlyAnchor && finite(monthlyAnchor.portfolio_nav) ? monthlyAnchor.portfolio_nav : null;
    var monthly = Object.keys(monthlyMap).sort().map(function (key) {
      var bucket = monthlyMap[key];
      var value = finite(previousMonthEnd) && previousMonthEnd !== 0
        ? bucket.last.portfolio_nav / previousMonthEnd - 1
        : null;
      previousMonthEnd = bucket.last.portfolio_nav;
      return {
        month: key,
        year: key.slice(0, 4),
        monthNumber: Number(key.slice(5, 7)),
        value: value,
        anchor: "previous_month_end"
      };
    });

    var relativeTotalReturn = finite(benchmarkTotalReturn) && 1 + benchmarkTotalReturn > 0
      ? (1 + totalReturn) / (1 + benchmarkTotalReturn) - 1
      : null;
    var relativeReturnRows = rows.slice(1).map(function (row) {
      return {
        date: row.date,
        daily_return: finite(row.portfolio_return_calc) && finite(row.benchmark_return_calc) && 1 + row.benchmark_return_calc > 0
          ? (1 + row.portfolio_return_calc) / (1 + row.benchmark_return_calc) - 1
          : null
      };
    });
    var relativeAnnualization = annualizationEligibility(first, last, relativeReturnRows, tradingCalendar);
    var annualizedRelativeReturn = relativeAnnualization.eligible
      ? annualizedFromTotalReturn(relativeTotalReturn, relativeAnnualization.validReturnCount)
      : null;

    return {
      kpis: {
        total_return: totalReturn,
        benchmark_total_return: benchmarkTotalReturn,
        excess_return: relativeTotalReturn,
        annualized_return: annualizedReturn,
        benchmark_annualized_return: benchmarkAnnualizedReturn,
        annualized_excess_return: annualizedRelativeReturn,
        annualized_volatility: annualizedVolatility,
        benchmark_annualized_volatility: benchmarkAnnualizedVolatility,
        sharpe_ratio: sharpe,
        calmar_ratio: maxDrawdown < 0 && annualizedReturn !== null ? annualizedReturn / Math.abs(maxDrawdown) : null,
        win_rate: winRate,
        win_count: winCount,
        max_drawdown: maxDrawdown,
        drawdown_duration: maxDuration,
        start_date: first.date,
        end_date: last.date,
        trading_days: formalReturnValues.length,
        annualization_status: annualization.status,
        open_day_coverage: annualization.openDayCoverage,
        expected_open_day_count: annualization.expectedOpenDayCount,
        annualization_coverage_basis: annualization.coverageBasis
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
    var equitySleeveTotal = latestRows.reduce(function (total, row) {
      return total + (finite(row.equity_sleeve_weight) ? row.equity_sleeve_weight : 0);
    }, 0);
    var navExposure = latestRows.reduce(function (total, row) {
      return total + (finite(row.nav_weight) ? row.nav_weight : 0);
    }, 0);
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
      allCurrent: latestRows,
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
      hhi: concentrationTrend.length ? concentrationTrend[concentrationTrend.length - 1].hhi : null,
      navExposure: navExposure,
      equitySleeveTotal: equitySleeveTotal
    };
  }

  function safeQuantity(row) {
    return finite(row.quantity) && row.quantity > 0 ? row.quantity : null;
  }

  function positionQuantity(row) {
    return finite(row.quantity) && row.quantity > 0 ? row.quantity : null;
  }

  function positionCostBasis(row) {
    var quantity = positionQuantity(row);
    if (finite(row.cost_basis) && row.cost_basis > 0) return row.cost_basis;
    if (quantity && finite(row.avg_cost) && row.avg_cost > 0) return quantity * row.avg_cost;
    return null;
  }

  function tradeAmount(row) {
    if (finite(row.trade_amount)) return row.trade_amount;
    if (finite(row.price) && finite(row.quantity)) return row.price * row.quantity;
    return null;
  }

  function sortTradesDesc(trades) {
    return trades.slice().sort(function (a, b) {
      return b.dateObj - a.dateObj || String(a.ticker || "").localeCompare(String(b.ticker || ""));
    });
  }

  function buildClosedTradeRecord(sell, aggregate, unmatchedQuantity) {
    var matchedQty = aggregate.matchedQuantity;
    var netCostBasis = aggregate.feesKnown ? aggregate.buyCost + aggregate.buyFee : null;
    var grossRealizedPnl = aggregate.sellProceeds - aggregate.buyCost;
    var netRealizedPnl = aggregate.feesKnown
      ? grossRealizedPnl - aggregate.buyFee - aggregate.sellFee
      : null;
    return {
      trade_date: sell.trade_date,
      dateObj: sell.dateObj,
      ticker: sell.ticker,
      name: sell.name,
      side: sell.side,
      matched_quantity: matchedQty,
      sell_quantity: sell.quantity,
      unmatched_quantity: unmatchedQuantity || 0,
      avg_buy_price: matchedQty ? aggregate.buyCost / matchedQty : null,
      sell_price: finite(sell.price) ? sell.price : matchedQty ? aggregate.sellProceeds / matchedQty : null,
      buy_cost: aggregate.buyCost,
      sell_proceeds: aggregate.sellProceeds,
      fee_known: aggregate.feesKnown,
      buy_fee: aggregate.feesKnown ? aggregate.buyFee : null,
      sell_fee: aggregate.feesKnown ? aggregate.sellFee : null,
      total_fee: aggregate.feesKnown ? aggregate.buyFee + aggregate.sellFee : null,
      gross_realized_pnl: grossRealizedPnl,
      gross_realized_return: aggregate.buyCost ? grossRealizedPnl / aggregate.buyCost : null,
      net_realized_pnl: netRealizedPnl,
      net_realized_return: netCostBasis ? netRealizedPnl / netCostBasis : null,
      realized_pnl: netRealizedPnl,
      realized_return: netCostBasis ? netRealizedPnl / netCostBasis : null,
      holding_days: matchedQty ? aggregate.holdingDayQuantity / matchedQty : null,
      match_count: aggregate.matchCount,
      opening_matched_quantity: aggregate.openingMatchedQuantity,
      uses_opening_lot: aggregate.openingMatchedQuantity > 0,
      reason: sell.reason,
      theme: sell.theme,
      warning: unmatchedQuantity > 0 ? "卖出数量超过 FIFO 可用持仓，未匹配 " + unmatchedQuantity + " 股。" : ""
    };
  }

  function buildOpeningLots(positions, orderedTrades, warnings) {
    var empty = {
      lotsByTicker: {},
      sourceDate: null,
      lotCount: 0,
      totalQuantity: 0,
      costBasis: 0,
      adjustedForSameDayTrades: false,
      available: false,
      message: ""
    };
    if (!orderedTrades.length || !(positions || []).length) return empty;
    var firstTrade = orderedTrades[0];
    var candidateDates = (positions || []).filter(function (row) {
      return row.dateObj && row.dateObj <= firstTrade.dateObj;
    }).map(function (row) { return row.date; }).sort();
    if (!candidateDates.length) return empty;
    var sourceDate = candidateDates[candidateDates.length - 1];
    var snapshotRows = positions.filter(function (row) { return row.date === sourceDate; });
    if (!snapshotRows.length) return empty;
    var hasQuantityField = snapshotRows.some(function (row) { return positionQuantity(row); });
    var hasCostField = snapshotRows.some(function (row) { return positionCostBasis(row); });
    if (!hasQuantityField || !hasCostField) {
      warnings.push(
        "positions.csv 缺少 quantity/shares 或 avg_cost/cost_basis，无法为 FIFO 生成期初持仓 lot；未匹配卖出仍会保留警告。"
      );
      return empty;
    }
    var sameDayTrades = {};
    orderedTrades.forEach(function (trade) {
      if (trade.trade_date !== sourceDate) return;
      var qty = safeQuantity(trade);
      if (!qty) return;
      var ticker = trade.ticker || "UNKNOWN_TICKER";
      if (!sameDayTrades[ticker]) sameDayTrades[ticker] = { buy: 0, sell: 0 };
      if (trade.side === "buy") sameDayTrades[ticker].buy += qty;
      if (trade.side === "sell") sameDayTrades[ticker].sell += qty;
    });
    var lotsByTicker = {};
    var summary = {
      lotsByTicker: lotsByTicker,
      sourceDate: sourceDate,
      lotCount: 0,
      totalQuantity: 0,
      costBasis: 0,
      adjustedForSameDayTrades: sourceDate === firstTrade.trade_date,
      available: false,
      message: ""
    };
    snapshotRows.forEach(function (row) {
      var quantity = positionQuantity(row);
      var costBasis = positionCostBasis(row);
      if (!quantity || !finite(costBasis) || costBasis <= 0) return;
      var sameDay = sameDayTrades[row.ticker] || { buy: 0, sell: 0 };
      var openingQuantity = quantity;
      if (summary.adjustedForSameDayTrades) {
        openingQuantity = quantity - sameDay.buy + sameDay.sell;
      }
      if (!finite(openingQuantity) || openingQuantity <= 0) return;
      var avgCost = costBasis / quantity;
      var ticker = row.ticker || "UNKNOWN_TICKER";
      if (!lotsByTicker[ticker]) lotsByTicker[ticker] = [];
      lotsByTicker[ticker].push({
        ticker: ticker,
        date: sourceDate,
        dateObj: row.dateObj,
        originalQuantity: openingQuantity,
        remainingQuantity: openingQuantity,
        remainingCost: avgCost * openingQuantity,
        remainingFee: 0,
        feeKnown: false,
        source: row,
        sourceType: "opening_position"
      });
      summary.lotCount += 1;
      summary.totalQuantity += openingQuantity;
      summary.costBasis += avgCost * openingQuantity;
    });
    summary.available = summary.lotCount > 0;
    if (summary.available) {
      summary.message = "FIFO 已使用 " + sourceDate + " 持仓成本基础生成期初 lot";
      if (summary.adjustedForSameDayTrades) summary.message += "，并按当日买卖回推期初数量";
      summary.message += "。";
    }
    return summary;
  }

  function computeTrades(trades, options) {
    options = options || {};
    if (!trades.length) {
      return {
        available: false,
        message: "未上传交易数据。",
        all: [],
        recent: [],
        closed: [],
        closedRecent: [],
        unmatchedSells: [],
        warnings: [],
        openingLots: {
          available: false,
          sourceDate: null,
          lotCount: 0,
          totalQuantity: 0,
          costBasis: 0,
          adjustedForSameDayTrades: false,
          message: ""
        },
        bySide: [],
        byTheme: [],
        byReason: [],
        totals: {}
      };
    }
    var warnings = [];
    var unknownFeeCount = trades.filter(function (row) { return !finite(row.fee); }).length;
    if (unknownFeeCount) {
      warnings.push(unknownFeeCount + " 笔交易费用未知；费用后收益和费用拖累指标已禁用。");
    }
    var unmatchedSells = [];
    var closedTrades = [];
    var orderedTrades = trades.slice().sort(function (a, b) {
      return a.dateObj - b.dateObj || String(a.ticker || "").localeCompare(String(b.ticker || ""));
    });
    var openingLots = buildOpeningLots(options.positions || [], orderedTrades, warnings);
    var lotsByTicker = openingLots.lotsByTicker;
    orderedTrades.forEach(function (row) {
      var qty = safeQuantity(row);
      var amount = tradeAmount(row);
      if (!qty || !finite(amount)) {
        warnings.push(row.trade_date + " " + row.ticker + " 缺少有效 quantity 或 trade_amount，已跳过 FIFO 配对。");
        return;
      }
      var ticker = row.ticker || "UNKNOWN_TICKER";
      if (!lotsByTicker[ticker]) lotsByTicker[ticker] = [];
      var fee = finite(row.fee) ? Math.max(0, row.fee) : 0;
      var feeKnown = finite(row.fee);
      if (row.side === "buy") {
        lotsByTicker[ticker].push({
          ticker: ticker,
          date: row.trade_date,
          dateObj: row.dateObj,
          originalQuantity: qty,
          remainingQuantity: qty,
          remainingCost: amount,
          remainingFee: fee,
          feeKnown: feeKnown,
          source: row
        });
        return;
      }
      if (row.side !== "sell") {
        warnings.push(row.trade_date + " " + row.ticker + " side=" + row.side + " 非 buy/sell，未参与 FIFO 配对。");
        return;
      }
      var remainingSellQuantity = qty;
      var aggregate = {
        matchedQuantity: 0,
        buyCost: 0,
        sellProceeds: 0,
        buyFee: 0,
        sellFee: 0,
        feesKnown: feeKnown,
        holdingDayQuantity: 0,
        matchCount: 0,
        openingMatchedQuantity: 0
      };
      var lots = lotsByTicker[ticker];
      while (remainingSellQuantity > 0 && lots.length) {
        var lot = lots[0];
        var beforeLotQty = lot.remainingQuantity;
        var matched = Math.min(remainingSellQuantity, beforeLotQty);
        var lotRatio = matched / beforeLotQty;
        var sellRatio = matched / qty;
        var buyCost = lot.remainingCost * lotRatio;
        var buyFee = lot.remainingFee * lotRatio;
        var sellProceeds = amount * sellRatio;
        var sellFee = fee * sellRatio;
        var holdingDays = Math.max(0, Math.round((row.dateObj - lot.dateObj) / MS_PER_DAY));
        aggregate.matchedQuantity += matched;
        aggregate.buyCost += buyCost;
        aggregate.sellProceeds += sellProceeds;
        aggregate.buyFee += buyFee;
        aggregate.sellFee += sellFee;
        aggregate.feesKnown = aggregate.feesKnown && lot.feeKnown === true;
        aggregate.holdingDayQuantity += holdingDays * matched;
        aggregate.matchCount += 1;
        if (lot.sourceType === "opening_position") aggregate.openingMatchedQuantity += matched;
        lot.remainingQuantity -= matched;
        lot.remainingCost -= buyCost;
        lot.remainingFee -= buyFee;
        remainingSellQuantity -= matched;
        if (lot.remainingQuantity <= 0.000001) lots.shift();
      }
      if (remainingSellQuantity > 0.000001) {
        var unmatched = {
          trade_date: row.trade_date,
          ticker: row.ticker,
          name: row.name,
          sell_quantity: qty,
          unmatched_quantity: remainingSellQuantity,
          message: row.trade_date + " " + row.ticker + " 卖出 " + qty + "，FIFO 可用持仓不足，未匹配 " + remainingSellQuantity + "。"
        };
        unmatchedSells.push(unmatched);
        warnings.push(unmatched.message);
      }
      if (aggregate.matchedQuantity > 0) {
        closedTrades.push(buildClosedTradeRecord(row, aggregate, remainingSellQuantity));
      }
    });
    var buyCount = trades.filter(function (row) { return row.side === "buy"; }).length;
    var sellCount = trades.filter(function (row) { return row.side === "sell"; }).length;
    var closedUnknownFeeCount = closedTrades.filter(function (row) { return row.fee_known !== true; }).length;
    var netMetricsComplete = closedTrades.length > 0 && closedUnknownFeeCount === 0;
    if (closedUnknownFeeCount) {
      warnings.push(
        closedUnknownFeeCount + " 笔平仓缺少买入或卖出费用；净实现盈亏、净胜率和净盈亏比保持 unavailable。"
      );
    }
    var wins = netMetricsComplete ? closedTrades.filter(function (row) { return row.net_realized_pnl > 0; }) : [];
    var losses = netMetricsComplete ? closedTrades.filter(function (row) { return row.net_realized_pnl < 0; }) : [];
    var netProfit = wins.reduce(function (total, row) { return total + row.net_realized_pnl; }, 0);
    var netLoss = losses.reduce(function (total, row) { return total + row.net_realized_pnl; }, 0);
    var grossWins = closedTrades.filter(function (row) { return row.gross_realized_pnl > 0; });
    var grossLosses = closedTrades.filter(function (row) { return row.gross_realized_pnl < 0; });
    var grossProfit = grossWins.reduce(function (total, row) { return total + row.gross_realized_pnl; }, 0);
    var grossLoss = grossLosses.reduce(function (total, row) { return total + row.gross_realized_pnl; }, 0);
    var matchedQuantity = closedTrades.reduce(function (total, row) {
      return total + (finite(row.matched_quantity) ? row.matched_quantity : 0);
    }, 0);
    var weightedHoldingDays = closedTrades.reduce(function (total, row) {
      return total + (finite(row.holding_days) && finite(row.matched_quantity) ? row.holding_days * row.matched_quantity : 0);
    }, 0);
    var allTrades = sortTradesDesc(trades);
    var closedSorted = closedTrades.slice().sort(function (a, b) {
      return b.dateObj - a.dateObj || String(a.ticker || "").localeCompare(String(b.ticker || ""));
    });
	    var fifoMessage = "交易胜率基于 FIFO 平仓交易；日净值胜率基于 portfolio_nav 日收益。";
	    if (openingLots.message) fifoMessage += " " + openingLots.message;
	    return {
	      available: true,
	      message: fifoMessage,
      all: allTrades,
      recent: allTrades.slice(0, 20),
      closed: closedSorted,
      closedRecent: closedSorted.slice(0, 20),
      unmatchedSells: unmatchedSells,
      warnings: warnings,
      openingLots: openingLots,
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
        fee: unknownFeeCount ? null : sum(trades, "fee"),
        fee_unknown_count: unknownFeeCount,
        closed_fee_unknown_count: closedUnknownFeeCount,
        trade_count: trades.length,
        closed_trade_count: closedTrades.length,
        gross_realized_pnl: closedTrades.reduce(function (total, row) { return total + row.gross_realized_pnl; }, 0),
        gross_trade_win_rate: closedTrades.length ? grossWins.length / closedTrades.length : null,
        gross_profit_factor: grossLoss < 0 ? grossProfit / Math.abs(grossLoss) : null,
        net_realized_pnl: netMetricsComplete ? closedTrades.reduce(function (total, row) { return total + row.net_realized_pnl; }, 0) : null,
        realized_pnl: netMetricsComplete ? closedTrades.reduce(function (total, row) { return total + row.net_realized_pnl; }, 0) : null,
        trade_win_rate: netMetricsComplete ? wins.length / closedTrades.length : null,
        profit_factor: netMetricsComplete && netLoss < 0 ? netProfit / Math.abs(netLoss) : null,
        avg_win: netMetricsComplete && wins.length ? netProfit / wins.length : null,
        avg_loss: netMetricsComplete && losses.length ? netLoss / losses.length : null,
        max_trade_profit: netMetricsComplete && wins.length ? Math.max.apply(null, wins.map(function (row) { return row.net_realized_pnl; })) : null,
        max_trade_loss: netMetricsComplete && losses.length ? Math.min.apply(null, losses.map(function (row) { return row.net_realized_pnl; })) : null,
        avg_holding_days: matchedQuantity ? weightedHoldingDays / matchedQuantity : null,
        unmatched_sell_count: unmatchedSells.length,
        unmatched_quantity: unmatchedSells.reduce(function (total, row) {
          return total + (finite(row.unmatched_quantity) ? row.unmatched_quantity : 0);
        }, 0),
        opening_lot_count: openingLots.lotCount,
        opening_lot_quantity: openingLots.totalQuantity,
        opening_lot_cost_basis: openingLots.costBasis,
        opening_matched_quantity: closedTrades.reduce(function (total, row) {
          return total + (finite(row.opening_matched_quantity) ? row.opening_matched_quantity : 0);
        }, 0),
        closed_with_opening_lot_count: closedTrades.filter(function (row) { return row.uses_opening_lot; }).length
      }
    };
  }

  function summarizeTradeWarnings(tradeReview) {
    if (!tradeReview || !tradeReview.available) return [];
    var warnings = [];
    var unmatchedSells = tradeReview.unmatchedSells || [];
    if (unmatchedSells.length) {
      var unmatchedQuantity = unmatchedSells.reduce(function (total, row) {
        return total + (finite(row.unmatched_quantity) ? row.unmatched_quantity : 0);
      }, 0);
      warnings.push(
        "FIFO 有 " + unmatchedSells.length + " 笔卖出未能完全匹配历史买入，合计未匹配 " +
        formatQuantity(unmatchedQuantity) + " 股；通常表示交易 CSV 缺少区间起点前的建仓/成本记录，详见交易复盘区。"
      );
    }
	    var unmatchedMessages = {};
	    unmatchedSells.forEach(function (row) { unmatchedMessages[row.message] = true; });
	    var otherWarnings = (tradeReview.warnings || []).filter(function (warning) {
	      return !unmatchedMessages[warning];
	    });
	    if (otherWarnings.length <= 2) {
	      warnings = warnings.concat(otherWarnings);
	    } else if (otherWarnings.length) {
	      warnings.push("交易 FIFO 配对还有 " + otherWarnings.length + " 条非未匹配警告，详见交易复盘区。");
	    }
	    return warnings;
	  }

  function computePortfolioMarketValueStats(positions) {
    var byDate = {};
    (positions || []).forEach(function (row) {
      if (!finite(row.market_value)) return;
      if (!byDate[row.date]) byDate[row.date] = { date: row.date, total: 0, count: 0 };
      byDate[row.date].total += row.market_value;
      byDate[row.date].count += 1;
    });
    var rows = Object.keys(byDate).sort().map(function (date) { return byDate[date]; })
      .filter(function (row) { return row.total > 0; });
    var average = rows.length ? rows.reduce(function (total, row) { return total + row.total; }, 0) / rows.length : null;
    return {
      average: average,
      observationCount: rows.length,
      latest: rows.length ? rows[rows.length - 1].total : null
    };
  }

  function enrichPerformanceWithTradingCosts(performance, tradeReview, marketValueStats, navProvenance) {
    var kpis = performance.kpis || {};
    var totals = tradeReview.totals || {};
    var warnings = [];
    var avgMarketValue = marketValueStats.average;
    navProvenance = navProvenance || {};
    kpis.nav_return_basis = navProvenance.return_method || "unknown";
    kpis.nav_gross_or_net = navProvenance.gross_or_net || "unknown";
    kpis.nav_trade_fee_inclusion = navProvenance.trade_fee_inclusion || "unknown";
    kpis.average_portfolio_market_value = avgMarketValue;
    kpis.turnover_ratio = finite(avgMarketValue) && avgMarketValue > 0 ? (totals.trade_amount || 0) / avgMarketValue : null;
    var feesComplete = !totals.fee_unknown_count && finite(totals.fee);
    kpis.fee_drag_on_nav = feesComplete && finite(avgMarketValue) && avgMarketValue > 0 ? totals.fee / avgMarketValue : null;
    var profitBase = finite(kpis.total_return) && finite(avgMarketValue) ? Math.abs(kpis.total_return * avgMarketValue) : null;
    kpis.fee_drag_on_profit = feesComplete && profitBase && profitBase > 0 ? totals.fee / profitBase : null;
    var secondaryAdjustmentAllowed = navProvenance.secondary_fee_adjustment_allowed === true && navProvenance.gross_or_net === "gross";
    kpis.fee_adjusted_total_return = secondaryAdjustmentAllowed && finite(kpis.total_return) && finite(kpis.fee_drag_on_nav)
      ? kpis.total_return - kpis.fee_drag_on_nav
      : null;
    if (!finite(avgMarketValue) || avgMarketValue <= 0) {
      warnings.push("positions.csv 缺少可用 market_value，费用拖累和换手率无法按组合市值计算。");
    }
    if (tradeReview.available && !feesComplete) {
      warnings.push("交易费用存在 unknown，费用后业绩指标保持 unavailable，不按 0 估算。");
    }
    if (!secondaryAdjustmentAllowed) {
      warnings.push("NAV gross/net 或费用嵌入口径未明确，禁止再次从 TWR 扣减交易费用。");
    }
    return warnings;
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

  function computeCorrelationMatrix(seriesItems, tradingCalendar) {
    var matrix = [];
    seriesItems.forEach(function (rowItem) {
      var rowStats = seriesStats(rowItem.series, tradingCalendar);
      var row = {
        field: rowItem.field,
        label: rowItem.label,
        values: []
      };
      seriesItems.forEach(function (colItem) {
        var colStats = seriesStats(colItem.series, tradingCalendar);
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

  function computeBenchmarkComparison(navRows, benchmarks, mainField, selectedFields, tradingCalendar) {
    var benchmarkMap = benchmarkByField(benchmarks);
    var availableFields = (benchmarks || []).map(function (benchmark) { return benchmark.field; });
    var benchmarkField = mainField || defaultBenchmarkField(benchmarks);
    var hasExplicitSelection = Array.isArray(selectedFields);
    var selected = (hasExplicitSelection ? selectedFields : []).filter(function (field) {
      return availableFields.indexOf(field) >= 0;
    });
    if (!selected.length && !hasExplicitSelection) selected = availableFields.slice(0, 3);

    var strictRows = strictSeriesPoints(navRows || [], tradingCalendar);
    var portfolioSeries = strictRows.map(function (row) {
      return { date: row.date, dateObj: row.dateObj, value: row.portfolio_nav, field: "portfolio_nav", label: "Portfolio" };
    });
    var portfolioStats = seriesStats(portfolioSeries, tradingCalendar);
    var benchmarkSeriesMap = {};
    availableFields.forEach(function (field) {
      var label = benchmarkMap[field] ? benchmarkMap[field].label : field;
      benchmarkSeriesMap[field] = forwardFillSeries(strictRows, field, label);
    });
    var selectedBenchmarks = selected.map(function (field) {
      var meta = benchmarkMap[field] || { field: field, label: field };
      var series = benchmarkSeriesMap[field] || [];
      var metrics = calculateBenchmarkMetrics(portfolioSeries, series, tradingCalendar);
      return {
        field: field,
        label: meta.label,
        isMain: field === benchmarkField,
        series: series,
        returns: seriesStats(series, tradingCalendar).returns,
        drawdown: calculateDrawdown(series),
        excessSeries: calculateExcessNav(portfolioStats.returns, seriesStats(series, tradingCalendar).returns),
        metrics: metrics
      };
    });
    var comparisonRows = availableFields.map(function (field) {
      var meta = benchmarkMap[field] || { field: field, label: field };
      var metrics = calculateBenchmarkMetrics(portfolioSeries, benchmarkSeriesMap[field], tradingCalendar);
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
      correlationMatrix: computeCorrelationMatrix(matrixItems, tradingCalendar),
      alignedSeries: alignSeriesByDate([{ field: "portfolio_nav", points: portfolioSeries }].concat(selectedBenchmarks.map(function (benchmark) {
        return { field: benchmark.field, points: benchmark.series };
      })))
    };
  }

  function formalChartAnalysisCap(dataset) {
    var contract = dataset && dataset.contract ? dataset.contract : {};
    var blockers = Array.isArray(contract.blockers) ? contract.blockers : [];
    var asOf = contract.as_of_matrix || {};
    var hasFutureAsOfBlocker = blockers.some(function (blocker) {
      return /_after_analysis_date$/.test(String(blocker || ""));
    });
    return hasFutureAsOfBlocker && /^\d{4}-\d{2}-\d{2}$/.test(String(asOf.analysis_trading_date || ""))
      ? asOf.analysis_trading_date
      : "";
  }

  function computeDashboard(dataset, filters) {
    var startDate = filters.startDate || "";
    var endDate = filters.endDate || "";
    var analysisCap = formalChartAnalysisCap(dataset);
    var formalEndDate = analysisCap && (!endDate || analysisCap < endDate)
      ? analysisCap
      : endDate;
    var benchmarkField = filters.benchmarkField || defaultBenchmarkField(dataset.benchmarks || []);
    var strictDatasetNav = strictSeriesPoints(
      dataset.nav || [],
      dataset.tradingCalendar
    );
    var navRows = byDateRange(strictDatasetNav, startDate, formalEndDate, "date");
    var monthlyAnchor = null;
    if (navRows.length) {
      strictDatasetNav.forEach(function (row) {
        if (row.date < navRows[0].date && (!monthlyAnchor || row.date > monthlyAnchor.date)) {
          monthlyAnchor = row;
        }
      });
    }
    var positions = byDateRange(dataset.positions || [], startDate, endDate, "date");
    var trades = byDateRange(dataset.trades || [], startDate, endDate, "trade_date");
    var performance = computePerformance(
      navRows,
      benchmarkField,
      monthlyAnchor,
      dataset.tradingCalendar
    );
    var attribution = computeAttribution(positions);
    var holdings = computeHoldings(positions);
	if (performance.kpis) {
	  performance.kpis.nav_exposure = holdings.navExposure;
	  performance.kpis.equity_sleeve_weight = holdings.equitySleeveTotal;
	  var latestNav = navRows.length ? navRows[navRows.length - 1] : null;
	  performance.kpis.cash_weight = latestNav && finite(latestNav.cash_weight)
	    ? latestNav.cash_weight
	    : finite(holdings.navExposure) ? 1 - holdings.navExposure : null;
	}
	    var tradeReview = computeTrades(trades, { positions: dataset.positions || [] });
    var marketValueStats = computePortfolioMarketValueStats(positions);
    var metricWarnings = enrichPerformanceWithTradingCosts(
      performance,
      tradeReview,
      marketValueStats,
      dataset.navReturnProvenance
    );
    if (analysisCap && formalEndDate === analysisCap) {
      metricWarnings.push(
        "formal NAV/benchmark charts capped at analysis_trading_date=" + analysisCap +
        " because a future as-of blocker is active."
      );
    }
    var benchmarkComparison = computeBenchmarkComparison(
      navRows,
      dataset.benchmarks || [],
      benchmarkField,
      filters.selectedBenchmarkFields,
      dataset.tradingCalendar
    );
    return {
      filters: filters,
      performance: performance,
      benchmarkComparison: benchmarkComparison,
      attribution: attribution,
      holdings: holdings,
      trades: tradeReview,
	      warnings: metricWarnings.concat(summarizeTradeWarnings(tradeReview)),
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
    computePortfolioMarketValueStats: computePortfolioMarketValueStats,
    mean: mean,
    std: std
  };
})();

(function () {
  "use strict";

  var MS_PER_DAY = 24 * 60 * 60 * 1000;
  var STOCKS = [
    ["688001.SH", "华芯精密", "Semiconductor Equipment", "Technology", "Etch Equipment"],
    ["688002.SH", "中微装备", "Semiconductor Equipment", "Technology", "Deposition Tools"],
    ["688003.SH", "北方晶工", "Semiconductor Equipment", "Technology", "Metrology"],
    ["300101.SZ", "星河机器人", "Robotics", "Industrials", "Industrial Robots"],
    ["300102.SZ", "灵动智造", "Robotics", "Industrials", "Motion Control"],
    ["300103.SZ", "锐控自动化", "Industrial Automation", "Industrials", "Factory Automation"],
    ["300104.SZ", "华工数控", "Industrial Automation", "Industrials", "CNC Systems"],
    ["600201.SH", "天穹航制", "Aerospace Manufacturing", "Defense", "Aircraft Parts"],
    ["600202.SH", "云航科技", "Aerospace Manufacturing", "Defense", "Avionics"],
    ["600203.SH", "星航动力", "Aerospace Manufacturing", "Defense", "Propulsion"],
    ["002301.SZ", "新材复合", "Advanced Materials", "Materials", "Composite Materials"],
    ["002302.SZ", "钛晶材料", "Advanced Materials", "Materials", "Specialty Metals"],
    ["002303.SZ", "高分子新材", "Advanced Materials", "Materials", "Polymer Materials"],
    ["300401.SZ", "绿能装备", "New Energy Equipment", "Industrials", "Battery Equipment"],
    ["300402.SZ", "光伏智造", "New Energy Equipment", "Industrials", "PV Equipment"],
    ["300403.SZ", "储能精机", "New Energy Equipment", "Industrials", "Energy Storage"],
    ["600501.SH", "雷达电子", "Defense Electronics", "Defense", "Radar Systems"],
    ["600502.SH", "电科传感", "Defense Electronics", "Defense", "Sensors"],
    ["600503.SH", "北斗模块", "Defense Electronics", "Defense", "Navigation Modules"],
    ["600504.SH", "航电集成", "Defense Electronics", "Defense", "Avionics Integration"],
    ["688005.SH", "硅谷测控", "Semiconductor Equipment", "Technology", "Testing Tools"],
    ["300105.SZ", "协作智臂", "Robotics", "Industrials", "Collaborative Robots"],
    ["002304.SZ", "碳纤维科技", "Advanced Materials", "Materials", "Carbon Fiber"],
    ["300404.SZ", "锂电装配", "New Energy Equipment", "Industrials", "Lithium Equipment"]
  ];

  var TRADE_REASONS = [
    "Earnings upgrade",
    "Policy catalyst",
    "Valuation reset",
    "Stop loss",
    "Position sizing",
    "Industry rotation",
    "Risk control"
  ];

  var BENCHMARK_LABELS = {
    benchmark_main_nav: "主基准",
    benchmark_nav: "旧版基准",
    csi300_nav: "沪深300",
    csi500_nav: "中证500",
    csi1000_nav: "中证1000",
    star50_nav: "科创50",
    chinext_nav: "创业板指",
    semiconductor_nav: "半导体",
    robotics_nav: "机器人",
    high_end_manufacturing_nav: "高端制造"
  };

  function seededRandom(seed) {
    var x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  }

  function formatDate(date) {
    var y = date.getFullYear();
    var m = String(date.getMonth() + 1).padStart(2, "0");
    var d = String(date.getDate()).padStart(2, "0");
    return y + "-" + m + "-" + d;
  }

  function parseDateOnly(value) {
    if (!value) return null;
    var text = String(value).trim();
    var match = text.match(/^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$/);
    if (!match) return null;
    var date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
    if (Number.isNaN(date.getTime())) return null;
    return date;
  }

  function isWeekday(date) {
    var day = date.getDay();
    return day !== 0 && day !== 6;
  }

  function monthEndWorkdays(start, end) {
    var rows = [];
    var cursor = new Date(start.getFullYear(), start.getMonth(), 1);
    while (cursor <= end) {
      var last = new Date(cursor.getFullYear(), cursor.getMonth() + 1, 0);
      while (!isWeekday(last)) last.setDate(last.getDate() - 1);
      if (last >= start && last <= end) rows.push(new Date(last));
      cursor = new Date(cursor.getFullYear(), cursor.getMonth() + 1, 1);
    }
    return rows;
  }

  function csvEscape(value) {
    if (value === null || value === undefined) return "";
    var text = String(value);
    if (/[",\n\r]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
    return text;
  }

  function buildNavSampleCSV() {
    var lines = [
      "date,portfolio_nav,benchmark_main_nav,csi300_nav,csi500_nav,csi1000_nav,star50_nav,chinext_nav,semiconductor_nav,robotics_nav,high_end_manufacturing_nav,cash_weight,gross_exposure,net_exposure"
    ];
    var navs = {
      portfolio_nav: 1,
      csi300_nav: 1,
      csi500_nav: 1,
      csi1000_nav: 1,
      star50_nav: 1,
      chinext_nav: 1,
      semiconductor_nav: 1,
      robotics_nav: 1,
      high_end_manufacturing_nav: 1
    };
    var index = 0;
    for (var date = new Date(2024, 0, 1); date <= new Date(2025, 11, 31); date.setDate(date.getDate() + 1)) {
      if (!isWeekday(date)) continue;
      var macro = seededRandom(index + 101) - 0.5;
      var growth = seededRandom(index + 211) - 0.5;
      var tech = seededRandom(index + 307) - 0.5;
      var industrial = seededRandom(index + 419) - 0.5;
      var cycle = Math.sin(index / 19) * 0.0028 + Math.cos(index / 43) * 0.0015;
      var returns = {
        csi300_nav: 0.00024 + Math.sin(index / 31) * 0.0011 + macro * 0.0085,
        csi500_nav: 0.00036 + cycle * 0.45 + macro * 0.007 + industrial * 0.008,
        csi1000_nav: 0.00048 + cycle * 0.7 + macro * 0.006 + growth * 0.013,
        star50_nav: 0.00062 + cycle * 0.85 + macro * 0.006 + tech * 0.016,
        chinext_nav: 0.00055 + cycle * 0.78 + macro * 0.006 + growth * 0.015,
        semiconductor_nav: 0.00074 + cycle * 0.95 + macro * 0.005 + tech * 0.019,
        robotics_nav: 0.00068 + cycle * 0.82 + macro * 0.005 + industrial * 0.017,
        high_end_manufacturing_nav: 0.00052 + cycle * 0.62 + macro * 0.006 + industrial * 0.013
      };
      returns.portfolio_nav = 0.00096 + cycle * 1.05 + macro * 0.006 + tech * 0.017 + industrial * 0.010 + growth * 0.006;
      if (index === 125 || index === 306 || index === 441) returns.portfolio_nav -= 0.038;
      if (index === 208 || index === 388) returns.portfolio_nav += 0.031;
      if (index === 126 || index === 307 || index === 442) {
        returns.semiconductor_nav -= 0.018;
        returns.star50_nav -= 0.014;
        returns.robotics_nav -= 0.012;
      }
      if (index === 0) {
        Object.keys(returns).forEach(function (field) {
          returns[field] = 0;
        });
      } else {
        Object.keys(returns).forEach(function (field) {
          var limit = field === "portfolio_nav" ? 0.08 : field === "csi300_nav" ? 0.035 : 0.06;
          returns[field] = Math.max(-limit, Math.min(limit, returns[field]));
          navs[field] *= 1 + returns[field];
        });
      }
      var benchmarkMainNav = 0.5 * navs.star50_nav + 0.3 * navs.semiconductor_nav + 0.2 * navs.high_end_manufacturing_nav;
      var grossExposure = 0.0426;
      var netExposure = grossExposure;
      var cashWeight = 1 - netExposure;
      lines.push([
        formatDate(date),
        navs.portfolio_nav.toFixed(6),
        benchmarkMainNav.toFixed(6),
        navs.csi300_nav.toFixed(6),
        navs.csi500_nav.toFixed(6),
        navs.csi1000_nav.toFixed(6),
        navs.star50_nav.toFixed(6),
        navs.chinext_nav.toFixed(6),
        navs.semiconductor_nav.toFixed(6),
        navs.robotics_nav.toFixed(6),
        navs.high_end_manufacturing_nav.toFixed(6),
        cashWeight.toFixed(4),
        grossExposure.toFixed(4),
        netExposure.toFixed(4)
      ].join(","));
      index += 1;
    }
    return lines.join("\n");
  }

	  function buildPositionsSampleCSV() {
	    var lines = [
	      "date,ticker,name,weight,nav_weight,equity_sleeve_weight,industry,sector,sub_sector,daily_return,contribution,market_value,quantity,avg_cost,cost_basis,current_price"
	    ];
    var dates = monthEndWorkdays(new Date(2024, 0, 1), new Date(2025, 11, 31));
    dates.forEach(function (date, dateIndex) {
      var scores = STOCKS.map(function (stock, stockIndex) {
        return {
          stock: stock,
          score: 0.55 + seededRandom(dateIndex * 37 + stockIndex * 19) + (stockIndex % 7) * 0.04
        };
      });
      scores.sort(function (a, b) {
        return b.score - a.score;
      });
      var selected = scores.slice(0, 18);
      var totalScore = selected.reduce(function (sum, item) {
        return sum + item.score;
      }, 0);
      selected.forEach(function (item, rank) {
        var stock = item.stock;
	        var sleeveWeight = item.score / totalScore;
	        var navWeight = sleeveWeight * 0.0426;
	        var dailyReturn = Math.sin((dateIndex + rank) / 4.2) * 0.012 + (seededRandom(dateIndex * 91 + rank * 17) - 0.5) * 0.055;
	        var contribution = navWeight * dailyReturn;
	        var marketValue = 10000 * navWeight;
	        var currentPrice = 18 + seededRandom(dateIndex * 41 + rank * 23) * 132;
	        var quantity = Math.max(100, Math.round((marketValue / currentPrice) / 100) * 100);
	        var avgCost = currentPrice * (0.72 + seededRandom(dateIndex * 53 + rank * 31) * 0.45);
	        var costBasis = avgCost * quantity;
	        lines.push([
	          formatDate(date),
	          stock[0],
	          stock[1],
	          navWeight.toFixed(8),
	          navWeight.toFixed(8),
	          sleeveWeight.toFixed(8),
          csvEscape(stock[2]),
          csvEscape(stock[3]),
	          csvEscape(stock[4]),
	          dailyReturn.toFixed(6),
	          contribution.toFixed(6),
	          marketValue.toFixed(2),
	          quantity,
	          avgCost.toFixed(4),
	          costBasis.toFixed(2),
	          currentPrice.toFixed(4)
	        ].join(","));
      });
    });
    return lines.join("\n");
  }

  function buildTradesSampleCSV() {
    var lines = [
      "trade_date,ticker,name,side,price,quantity,trade_amount,fee,reason,industry"
    ];
    var tradeDays = [];
    for (var date = new Date(2024, 0, 9); date <= new Date(2025, 11, 23); date.setDate(date.getDate() + 7)) {
      while (!isWeekday(date)) date.setDate(date.getDate() + 1);
      tradeDays.push(new Date(date));
    }
    for (var i = 0; i < 84; i += 1) {
      var stock = STOCKS[(i * 5 + Math.floor(seededRandom(i + 8) * STOCKS.length)) % STOCKS.length];
      var side = i % 5 === 3 || i % 7 === 4 ? "sell" : "buy";
      var price = 18 + seededRandom(i * 31 + 7) * 132;
      var quantity = (Math.floor(2 + seededRandom(i * 43) * 18) * 100);
      var amount = price * quantity;
      var fee = amount * (0.00028 + seededRandom(i * 13) * 0.00012);
      var reason = TRADE_REASONS[(i * 3 + Math.floor(seededRandom(i + 3) * 3)) % TRADE_REASONS.length];
      lines.push([
        formatDate(tradeDays[i % tradeDays.length]),
        stock[0],
        stock[1],
        side,
        price.toFixed(2),
        quantity,
        amount.toFixed(2),
        fee.toFixed(2),
        reason,
        csvEscape(stock[2])
      ].join(","));
    }
    return lines.join("\n");
  }

  var SAMPLE_CSV = {
    nav: buildNavSampleCSV(),
    positions: buildPositionsSampleCSV(),
    trades: buildTradesSampleCSV()
  };

  function parseCSV(text) {
    var rows = [];
    var row = [];
    var field = "";
    var inQuotes = false;
    var input = String(text || "");
    for (var i = 0; i < input.length; i += 1) {
      var char = input[i];
      if (inQuotes) {
        if (char === '"') {
          if (input[i + 1] === '"') {
            field += '"';
            i += 1;
          } else {
            inQuotes = false;
          }
        } else {
          field += char;
        }
      } else if (char === '"') {
        inQuotes = true;
      } else if (char === ",") {
        row.push(field);
        field = "";
      } else if (char === "\n") {
        row.push(field);
        rows.push(row);
        row = [];
        field = "";
      } else if (char !== "\r") {
        field += char;
      }
    }
    row.push(field);
    rows.push(row);

    rows = rows.filter(function (items) {
      return items.some(function (item) {
        return String(item || "").trim() !== "";
      });
    });
    if (!rows.length) return [];
    var headers = rows.shift().map(function (header) {
      return String(header || "").trim();
    });
    return rows.map(function (items) {
      var obj = {};
      headers.forEach(function (header, index) {
        obj[header] = items[index] === undefined ? "" : items[index].trim();
      });
      return obj;
    });
  }

  function parseNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    if (typeof value === "number") return Number.isFinite(value) ? value : null;
    var text = String(value).trim().replace(/,/g, "");
    if (!text) return null;
    var isPercent = text.endsWith("%");
    if (isPercent) text = text.slice(0, -1).trim();
    var num = Number(text);
    if (!Number.isFinite(num)) return null;
    return isPercent ? num / 100 : num;
  }

  function normalizeWeightUnit(unit) {
    var text = String(unit || "").trim().toLowerCase();
    if (!text) return "";
    if (["percent", "percentage", "pct", "%"].indexOf(text) >= 0) return "percent";
    if (["decimal", "ratio", "fraction"].indexOf(text) >= 0) return "decimal";
    return "";
  }

  function addWeightWarning(options, message) {
    if (!options || !options.warnings) return;
    var key = options.warningKey || message;
    if (options.warningLedger) {
      if (options.warningLedger[key]) return;
      options.warningLedger[key] = true;
    }
    options.warnings.push(message);
  }

  function parseWeight(value, options) {
    options = options || {};
    var num = parseNumber(value);
    if (num === null) return null;
    var rawText = String(value || "").trim();
    if (typeof value === "string" && rawText.endsWith("%")) return num;
    var unit = normalizeWeightUnit(options.unit);
    if (unit === "percent") return num / 100;
    if (unit === "decimal") return num;
    if (num > 1) {
      addWeightWarning(
        options,
        (options.fieldName || "weight") + " 第 " + (options.rowNumber || "-") +
          " 行为裸数字 " + rawText + "，已按小数权重解析；如需百分比请写 " + rawText + "% 或提供 weight_unit=percent。"
      );
    }
    return num;
  }

  function hasField(row, field) {
    return Object.prototype.hasOwnProperty.call(row, field) && String(row[field] || "").trim() !== "";
  }

  function validateRequired(rows, fields, label) {
    var errors = [];
    if (!rows.length) {
      errors.push(label + " 为空或无法解析。");
      return errors;
    }
    fields.forEach(function (field) {
      if (!Object.prototype.hasOwnProperty.call(rows[0], field)) {
        errors.push(label + " 缺少必填字段：" + field);
      }
    });
    return errors;
  }

  function benchmarkLabel(field) {
    if (BENCHMARK_LABELS[field]) return BENCHMARK_LABELS[field];
    return String(field || "")
      .replace(/_nav$/i, "")
      .split("_")
      .filter(Boolean)
      .map(function (part) {
        var upper = part.toUpperCase();
        if (/^(CSI|ETF|STAR|CN|US)$/.test(upper)) return upper;
        if (/^\d+$/.test(part)) return part;
        return part.charAt(0).toUpperCase() + part.slice(1);
      })
      .join(" ");
  }

  function detectBenchmarks(navData) {
    var fields = {};
    (navData || []).forEach(function (row) {
      Object.keys(row.raw || row).forEach(function (field) {
        if (field.endsWith("_nav") && field !== "portfolio_nav") fields[field] = true;
      });
    });
    var preferred = [
      "benchmark_main_nav",
      "benchmark_nav",
      "csi300_nav",
      "csi500_nav",
      "csi1000_nav",
      "star50_nav",
      "chinext_nav",
      "semiconductor_nav",
      "robotics_nav",
      "high_end_manufacturing_nav"
    ];
    return Object.keys(fields).sort(function (a, b) {
      var ai = preferred.indexOf(a);
      var bi = preferred.indexOf(b);
      if (ai >= 0 || bi >= 0) return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi);
      return a.localeCompare(b);
    }).map(function (field) {
      return {
        field: field,
        label: benchmarkLabel(field),
        isDefaultMain: field === "benchmark_main_nav" || field === "benchmark_nav"
      };
    });
  }

  function normalizeNavRows(rows) {
    var errors = validateRequired(rows, ["date", "portfolio_nav"], "nav.csv");
    var warnings = [];
    var weightWarningLedger = {};
    var normalized = [];
    rows.forEach(function (row, index) {
      var date = parseDateOnly(row.date);
      var portfolioNav = parseNumber(row.portfolio_nav);
      if (!date || portfolioNav === null) {
        warnings.push("nav.csv 第 " + (index + 2) + " 行 date 或 portfolio_nav 无效，已跳过。");
        return;
      }
      var item = {
        date: formatDate(date),
        dateObj: date,
        portfolio_nav: portfolioNav,
        portfolio_return: hasField(row, "portfolio_return") ? parseNumber(row.portfolio_return) : null,
        benchmark_return: hasField(row, "benchmark_return") ? parseNumber(row.benchmark_return) : null,
        cash_weight: hasField(row, "cash_weight") ? parseWeight(row.cash_weight, {
          unit: row.cash_weight_unit || row.weight_unit || row.unit,
          fieldName: "cash_weight",
          rowNumber: index + 2,
          warnings: warnings,
          warningLedger: weightWarningLedger,
          warningKey: "nav.cash_weight.bare_gt_one"
        }) : null,
        gross_exposure: hasField(row, "gross_exposure") ? parseWeight(row.gross_exposure, {
          unit: row.gross_exposure_unit || row.weight_unit || row.unit,
          fieldName: "gross_exposure",
          rowNumber: index + 2,
          warnings: warnings,
          warningLedger: weightWarningLedger,
          warningKey: "nav.gross_exposure.bare_gt_one"
        }) : null,
        net_exposure: hasField(row, "net_exposure") ? parseWeight(row.net_exposure, {
          unit: row.net_exposure_unit || row.weight_unit || row.unit,
          fieldName: "net_exposure",
          rowNumber: index + 2,
          warnings: warnings,
          warningLedger: weightWarningLedger,
          warningKey: "nav.net_exposure.bare_gt_one"
        }) : null,
        portfolio_nav_raw: hasField(row, "portfolio_nav_raw") ? parseNumber(row.portfolio_nav_raw) : null,
        portfolio_nav_rebased: hasField(row, "portfolio_nav_rebased") ? parseNumber(row.portfolio_nav_rebased) : null,
        portfolio_units: hasField(row, "portfolio_units") ? parseNumber(row.portfolio_units) : null,
        initial_capital: hasField(row, "initial_capital") ? parseNumber(row.initial_capital) : null,
        total_value_after: hasField(row, "total_value_after") ? parseNumber(row.total_value_after) : null,
        external_funding_cash_flow: hasField(row, "external_funding_cash_flow") ? parseNumber(row.external_funding_cash_flow) : null,
        raw: row
      };
      Object.keys(row).forEach(function (field) {
        if (field.endsWith("_nav")) {
          item[field] = parseNumber(row[field]);
        }
      });
      normalized.push(item);
    });
    normalized.sort(function (a, b) {
      return a.dateObj - b.dateObj;
    });
    return { rows: normalized, errors: errors, warnings: warnings };
  }

  function normalizePositionsRows(rows) {
    var errors = validateRequired(rows, ["date", "ticker", "name"], "positions.csv");
    var warnings = [];
    var weightWarningLedger = {};
    var normalized = [];
    rows.forEach(function (row, index) {
      var date = parseDateOnly(row.date);
      var navWeight = parseWeight(hasField(row, "nav_weight") ? row.nav_weight : row.weight, {
        unit: row.weight_unit || row.unit,
        fieldName: "positions.csv nav_weight",
        rowNumber: index + 2,
        warnings: warnings,
        warningLedger: weightWarningLedger,
        warningKey: "positions.weight.bare_gt_one"
      });
      var sleeveWeight = parseWeight(
        hasField(row, "equity_sleeve_weight") ? row.equity_sleeve_weight : row.weight,
        {
          unit: row.weight_unit || row.unit,
          fieldName: "positions.csv equity_sleeve_weight",
          rowNumber: index + 2,
          warnings: warnings,
          warningLedger: weightWarningLedger,
          warningKey: "positions.sleeve_weight.bare_gt_one"
        }
      );
      if (!date || !row.ticker || navWeight === null) {
        warnings.push("positions.csv 第 " + (index + 2) + " 行 date/ticker/nav_weight 无效，已跳过。");
        return;
      }
      var dailyReturn = hasField(row, "daily_return") ? parseNumber(row.daily_return) : null;
      var contribution = hasField(row, "contribution") ? parseNumber(row.contribution) : null;
      if (contribution === null && dailyReturn !== null) contribution = navWeight * dailyReturn;
	      normalized.push({
	        date: formatDate(date),
	        dateObj: date,
	        ticker: String(row.ticker || "").trim(),
	        name: String(row.name || "").trim() || "UNKNOWN_NAME",
	        weight: navWeight,
	        nav_weight: navWeight,
	        equity_sleeve_weight: sleeveWeight,
	        industry: String(row.industry || "").trim() || null,
	        industry_source: String(row.industry_source || "").trim() || null,
	        sector: String(row.sector || "").trim(),
	        sub_sector: String(row.sub_sector || "").trim(),
	        daily_return: dailyReturn,
	        contribution: contribution,
	        contribution_effective_date: String(row.contribution_effective_date || "").trim(),
	        contribution_date_source: String(row.contribution_date_source || "").trim(),
	        market_value: hasField(row, "market_value") ? parseNumber(row.market_value) : null,
	        quantity: hasField(row, "quantity") ? parseNumber(row.quantity) : hasField(row, "shares") ? parseNumber(row.shares) : null,
	        avg_cost: hasField(row, "avg_cost") ? parseNumber(row.avg_cost) : hasField(row, "cost_price") ? parseNumber(row.cost_price) : null,
	        cost_basis: hasField(row, "cost_basis") ? parseNumber(row.cost_basis) : null,
	        current_price: hasField(row, "current_price") ? parseNumber(row.current_price) : null,
	        unrealized_pnl: hasField(row, "unrealized_pnl") ? parseNumber(row.unrealized_pnl) : null,
	        recommended_action: String(row.recommended_action || "").trim(),
	        stop_loss: hasField(row, "stop_loss") ? parseNumber(row.stop_loss) : null,
	        take_profit: hasField(row, "take_profit") ? parseNumber(row.take_profit) : null,
	        quote_at: String(row.quote_at || "").trim(),
	        quote_age_seconds: hasField(row, "quote_age_seconds") ? parseNumber(row.quote_age_seconds) : null,
	        thesis: String(row.thesis || "").trim(),
	        risk_status: String(row.risk_status || "").trim(),
	        raw: row
	      });
    });
    normalized.sort(function (a, b) {
      return a.dateObj - b.dateObj || b.weight - a.weight;
    });
    return { rows: normalized, errors: errors, warnings: warnings };
  }

  function normalizeTradesRows(rows) {
    if (!rows.length) return { rows: [], errors: [], warnings: ["未上传交易数据。"] };
    var errors = validateRequired(
      rows,
      ["trade_date", "ticker", "name", "side", "price", "quantity", "trade_amount", "fee", "reason"],
      "trades.csv"
    );
    var warnings = [];
    var normalized = [];
    rows.forEach(function (row, index) {
      var date = parseDateOnly(row.trade_date);
      var price = parseNumber(row.price);
      var quantity = parseNumber(row.quantity);
      var amount = hasField(row, "trade_amount") ? parseNumber(row.trade_amount) : null;
      if (amount === null && price !== null && quantity !== null) amount = price * quantity;
      if (!date || !row.ticker || !row.side || amount === null) {
        warnings.push("trades.csv 第 " + (index + 2) + " 行 trade_date/ticker/side/trade_amount 无效，已跳过。");
        return;
      }
      normalized.push({
        trade_date: formatDate(date),
        dateObj: date,
        ticker: String(row.ticker || "").trim(),
        name: String(row.name || "").trim() || "UNKNOWN_NAME",
        side: String(row.side || "").trim().toLowerCase(),
        price: price,
        quantity: quantity,
        trade_amount: amount,
        fee: hasField(row, "fee") ? parseNumber(row.fee) : null,
        fee_source: String(row.fee_source || (hasField(row, "fee") ? "provided" : "unknown")).trim(),
        slippage: hasField(row, "slippage") ? parseNumber(row.slippage) : null,
        ledger_delta: hasField(row, "ledger_delta") ? parseNumber(row.ledger_delta) : null,
        recommendation_id: String(row.recommendation_id || "").trim(),
        decision_id: String(row.decision_id || "").trim(),
        order_id: String(row.order_id || "").trim(),
        fill_id: String(row.fill_id || "").trim(),
        reason: String(row.reason || "").trim(),
        industry: String(row.industry || "").trim() || null,
        raw: row
      });
    });
    normalized.sort(function (a, b) {
      return a.dateObj - b.dateObj;
    });
    return { rows: normalized, errors: errors, warnings: warnings };
  }

  function readFileAsText(file) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () {
        resolve(String(reader.result || ""));
      };
      reader.onerror = function () {
        reject(new Error("无法读取文件：" + file.name));
      };
      reader.readAsText(file);
    });
  }

  function extractBenchmarkFields(navRows) {
    return detectBenchmarks(navRows).map(function (benchmark) {
      return benchmark.field;
    });
  }

  function parseDataset(csvBundle, contract) {
    var nav = normalizeNavRows(parseCSV(csvBundle.nav || ""));
    var positions = normalizePositionsRows(parseCSV(csvBundle.positions || ""));
    var trades = normalizeTradesRows(csvBundle.trades ? parseCSV(csvBundle.trades) : []);
    return {
      nav: nav.rows,
      positions: positions.rows,
      trades: trades.rows,
      errors: nav.errors.concat(positions.errors, trades.errors),
      warnings: nav.warnings.concat(positions.warnings, trades.warnings),
      benchmarks: detectBenchmarks(nav.rows),
      benchmarkFields: extractBenchmarkFields(nav.rows),
      tradingCalendar: contract && contract.trading_calendar ? contract.trading_calendar : null,
      navReturnProvenance: contract && contract.nav_return_provenance ? contract.nav_return_provenance : null,
      contract: contract || null
    };
  }

  window.DashboardData = {
    SAMPLE_CSV: SAMPLE_CSV,
    BENCHMARK_LABELS: BENCHMARK_LABELS,
    STOCKS: STOCKS,
    buildNavSampleCSV: buildNavSampleCSV,
    buildPositionsSampleCSV: buildPositionsSampleCSV,
    buildTradesSampleCSV: buildTradesSampleCSV,
    parseCSV: parseCSV,
    parseNumber: parseNumber,
    parseWeight: parseWeight,
    formatDate: formatDate,
    parseDateOnly: parseDateOnly,
    readFileAsText: readFileAsText,
    normalizeNavRows: normalizeNavRows,
    normalizePositionsRows: normalizePositionsRows,
    normalizeTradesRows: normalizeTradesRows,
    parseDataset: parseDataset,
    detectBenchmarks: detectBenchmarks,
    benchmarkLabel: benchmarkLabel,
    extractBenchmarkFields: extractBenchmarkFields,
    MS_PER_DAY: MS_PER_DAY
  };
})();

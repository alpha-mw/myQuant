(function () {
  "use strict";

  var Data = window.DashboardData;
  var Metrics = window.DashboardMetrics;
  var UI = window.DashboardUI;
  var Generated = window.DashboardGeneratedRecords || {};
  var USER_UPLOAD_STORAGE_KEY = "portfolioDashboardUserUploads.v1";
  var STORAGE_VERSION = 1;
  var storageWarning = "";

  function hasGeneratedRecords(records) {
    return Boolean(
      records &&
      records.csv &&
      records.csv.nav &&
      records.csv.positions
    );
  }

  function readPersistedUploads() {
    try {
      if (!window.localStorage) return null;
      var raw = window.localStorage.getItem(USER_UPLOAD_STORAGE_KEY);
      if (!raw) return null;
      var parsed = JSON.parse(raw);
      if (!parsed || parsed.version !== STORAGE_VERSION || !parsed.csvBundle) return null;
      return parsed;
    } catch (error) {
      storageWarning = "无法读取浏览器本地保存的用户 CSV，已回退到当前数据源：" + error.message;
      return null;
    }
  }

  var persistedUploads = readPersistedUploads();

  function initialCsvBundle() {
    var bundle;
    if (hasGeneratedRecords(Generated)) {
      bundle = {
        nav: Generated.csv.nav,
        positions: Generated.csv.positions,
        trades: Generated.csv.trades || ""
      };
    } else {
      bundle = {
        nav: Data.SAMPLE_CSV.nav,
        positions: Data.SAMPLE_CSV.positions,
        trades: Data.SAMPLE_CSV.trades
      };
    }
    if (persistedUploads && persistedUploads.csvBundle) {
      ["nav", "positions", "trades"].forEach(function (kind) {
        if (typeof persistedUploads.csvBundle[kind] === "string") bundle[kind] = persistedUploads.csvBundle[kind];
      });
    }
    return bundle;
  }

  function initialSource() {
    var source = hasGeneratedRecords(Generated)
      ? { nav: "records", positions: "records", trades: Generated.csv.trades ? "records" : "sample" }
      : { nav: "sample", positions: "sample", trades: "sample" };
    if (persistedUploads && persistedUploads.csvBundle) {
      ["nav", "positions", "trades"].forEach(function (kind) {
        if (typeof persistedUploads.csvBundle[kind] === "string") source[kind] = "user";
      });
    }
    return source;
  }

  function initialFileNames() {
    var names = hasGeneratedRecords(Generated)
      ? { nav: "generated_records.js", positions: "generated_records.js", trades: Generated.csv.trades ? "generated_records.js" : "trades_sample.csv" }
      : { nav: "nav_sample.csv", positions: "positions_sample.csv", trades: "trades_sample.csv" };
    if (persistedUploads && persistedUploads.fileNames) {
      ["nav", "positions", "trades"].forEach(function (kind) {
        if (typeof persistedUploads.csvBundle[kind] === "string") {
          names[kind] = persistedUploads.fileNames[kind] || (kind + ".csv");
        }
      });
    }
    return names;
  }

  var state = {
    csvBundle: initialCsvBundle(),
    source: initialSource(),
    fileNames: initialFileNames(),
    userUploadSavedAt: persistedUploads ? persistedUploads.savedAt : "",
    filters: {
      startDate: "",
      endDate: "",
      benchmarkField: "",
      selectedBenchmarkFields: [],
      showPortfolioCurve: true,
      showExcessCurve: true
    },
    view: {
      overviewLens: "nav",
      benchmarkSortField: "totalReturn",
      benchmarkSortDirection: "desc",
      benchmarkSelectionTouched: false,
      tableExpanded: {
        holdings: false,
        trades: false,
        closedTrades: false
      }
    },
    dataset: null,
    metrics: null
  };

  function $(id) {
    return document.getElementById(id);
  }

  function csvEscape(value) {
    if (value === null || value === undefined) return "";
    var text = String(value);
    if (/[",\n\r]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
    return text;
  }

  function formatStoredTimestamp(value) {
    if (!value) return "";
    var date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.getFullYear() + "-" + String(date.getMonth() + 1).padStart(2, "0") + "-" +
      String(date.getDate()).padStart(2, "0") + " " +
      String(date.getHours()).padStart(2, "0") + ":" + String(date.getMinutes()).padStart(2, "0");
  }

  function persistUserUploads() {
    try {
      if (!window.localStorage) return;
      var csvBundle = {};
      var fileNames = {};
      ["nav", "positions", "trades"].forEach(function (kind) {
        if (state.source[kind] === "user") {
          csvBundle[kind] = state.csvBundle[kind] || "";
          fileNames[kind] = state.fileNames[kind] || (kind + ".csv");
        }
      });
      if (!Object.keys(csvBundle).length) {
        window.localStorage.removeItem(USER_UPLOAD_STORAGE_KEY);
        state.userUploadSavedAt = "";
        return;
      }
      var payload = {
        version: STORAGE_VERSION,
        savedAt: new Date().toISOString(),
        csvBundle: csvBundle,
        fileNames: fileNames
      };
      window.localStorage.setItem(USER_UPLOAD_STORAGE_KEY, JSON.stringify(payload));
      state.userUploadSavedAt = payload.savedAt;
      storageWarning = "";
    } catch (error) {
      storageWarning = "无法写入浏览器本地保存的用户 CSV；刷新后可能回退到默认数据：" + error.message;
    }
  }

  function clearPersistedUploads() {
    try {
      if (window.localStorage) window.localStorage.removeItem(USER_UPLOAD_STORAGE_KEY);
      storageWarning = "";
    } catch (error) {
      storageWarning = "无法清除浏览器本地保存的用户 CSV：" + error.message;
    }
    state.userUploadSavedAt = "";
  }

  function setStatus() {
    var status = $("dataStatus");
    var dateStatus = $("dateStatus");
    var modeLabel = $("dataModeLabel");
    var sampleRiskTag = $("sampleRiskTag");
    var overviewPeriod = $("overviewPeriod");
    var monthlyPeriod = $("monthlyPeriod");
    var usingUserData = Object.keys(state.source).some(function (key) {
      return state.source[key] === "user";
    });
    var usingRecordData = !usingUserData && Object.keys(state.source).some(function (key) {
      return state.source[key] === "records";
    });
    if (usingUserData) {
      status.textContent = "当前使用用户上传数据";
      if (modeLabel) modeLabel.textContent = "（用户上传数据）";
    } else if (usingRecordData) {
      status.textContent = "当前使用本地记录数据";
      if (modeLabel) modeLabel.textContent = "（本地记录数据）";
    } else {
      status.textContent = "当前使用示例数据";
      if (modeLabel) modeLabel.textContent = "（示例数据）";
    }
    status.className = "status-pill " + (usingUserData ? "user" : usingRecordData ? "records" : "sample");
    if (sampleRiskTag) sampleRiskTag.classList.toggle("hidden", usingRecordData || usingUserData);
    var kpis = state.metrics && state.metrics.performance.kpis;
    var periodText = kpis && kpis.start_date ? kpis.start_date + " 至 " + kpis.end_date : "-";
    dateStatus.textContent = "日期区间：" + periodText;
    if (overviewPeriod) overviewPeriod.textContent = "期间：" + periodText;
    if (monthlyPeriod) monthlyPeriod.textContent = "期间：" + periodText;
  }

  function setUploadNames() {
    [
      ["nav", "navUploadName"],
      ["positions", "positionsUploadName"],
      ["trades", "tradesUploadName"]
    ].forEach(function (item) {
      var el = $(item[1]);
      if (el) el.textContent = state.fileNames[item[0]] || "-";
    });
  }

  function syncDateInputsToDataset() {
    if (!state.dataset || !state.dataset.nav.length) return;
    var first = state.dataset.nav[0].date;
    var last = state.dataset.nav[state.dataset.nav.length - 1].date;
    ["startDate", "endDate"].forEach(function (id) {
      var input = $(id);
      input.min = first;
      input.max = last;
    });
    if (!state.filters.startDate) {
      state.filters.startDate = first;
      $("startDate").value = first;
    }
    if (!state.filters.endDate) {
      state.filters.endDate = last;
      $("endDate").value = last;
    }
  }

  function preferredMainBenchmark(fields) {
    if (fields.indexOf("benchmark_main_nav") >= 0) return "benchmark_main_nav";
    if (fields.indexOf("benchmark_nav") >= 0) return "benchmark_nav";
    return fields[0] || "";
  }

  function defaultSelectedBenchmarks(fields) {
    var preferred = ["benchmark_main_nav", "csi300_nav", "star50_nav", "semiconductor_nav"];
    var selected = preferred.filter(function (field) {
      return fields.indexOf(field) >= 0;
    });
    return selected.length ? selected : fields.slice(0, 3);
  }

  function benchmarkLabel(benchmark) {
    return (benchmark && benchmark.label) || (benchmark && benchmark.field) || "";
  }

  function updateBenchmarkControls(benchmarks) {
    benchmarks = benchmarks || [];
    var fields = benchmarks.map(function (benchmark) { return benchmark.field; });
    var select = $("benchmarkSelect");
    var multi = $("benchmarkMultiSelect");
    var previous = state.filters.benchmarkField || "";
    select.innerHTML = "";
    if (!fields.length) {
      var emptyOption = document.createElement("option");
      emptyOption.value = "";
      emptyOption.textContent = "无 benchmark";
      select.appendChild(emptyOption);
      select.disabled = true;
      if (multi) {
        multi.innerHTML = "";
        multi.disabled = true;
      }
      state.filters.benchmarkField = "";
      state.filters.selectedBenchmarkFields = [];
      updateBenchmarkWarnings();
      return;
    }
    select.disabled = false;
    benchmarks.forEach(function (benchmark) {
      var option = document.createElement("option");
      option.value = benchmark.field;
      option.textContent = benchmarkLabel(benchmark) + " (" + benchmark.field + ")";
      select.appendChild(option);
    });
    state.filters.benchmarkField = fields.indexOf(previous) >= 0 ? previous : preferredMainBenchmark(fields);
    select.value = state.filters.benchmarkField;
    if (multi) {
      var selected = (state.filters.selectedBenchmarkFields || []).filter(function (field) {
        return fields.indexOf(field) >= 0;
      });
      if (!selected.length && !state.view.benchmarkSelectionTouched) selected = defaultSelectedBenchmarks(fields);
      state.filters.selectedBenchmarkFields = selected;
      multi.disabled = false;
      multi.innerHTML = "";
      benchmarks.forEach(function (benchmark) {
        var multiOption = document.createElement("option");
        multiOption.value = benchmark.field;
        multiOption.textContent = benchmarkLabel(benchmark) + " (" + benchmark.field + ")";
        multiOption.selected = selected.indexOf(benchmark.field) >= 0;
        multi.appendChild(multiOption);
      });
    }
    updateBenchmarkWarnings();
  }

  function selectedBenchmarkValues() {
    var multi = $("benchmarkMultiSelect");
    if (!multi) return [];
    return Array.from(multi.selectedOptions).map(function (option) {
      return option.value;
    });
  }

  function updateBenchmarkWarnings() {
    var warning = $("benchmarkCrowdingWarning");
    if (warning) warning.classList.toggle("hidden", (state.filters.selectedBenchmarkFields || []).length <= 5);
    var showPortfolio = $("showPortfolioCurve");
    var showExcess = $("showExcessCurve");
    if (showPortfolio) showPortfolio.checked = state.filters.showPortfolioCurve !== false;
    if (showExcess) showExcess.checked = state.filters.showExcessCurve !== false;
    document.querySelectorAll("[data-benchmark-sort]").forEach(function (button) {
      button.classList.toggle("active", button.dataset.benchmarkSort === state.view.benchmarkSortField);
    });
  }

  function refresh(options) {
    options = options || {};
    state.dataset = Data.parseDataset(state.csvBundle);
    if (options.resetDateRange) {
      state.filters.startDate = "";
      state.filters.endDate = "";
      $("startDate").value = "";
      $("endDate").value = "";
    }
    updateBenchmarkControls(state.dataset.benchmarks);
    syncDateInputsToDataset();
    state.metrics = Metrics.computeDashboard(state.dataset, state.filters);
    renderCurrentDashboard();
    var infos = [];
    var usingUser = Object.keys(state.source).some(function (key) { return state.source[key] === "user"; });
    var usingRecords = !usingUser && Object.keys(state.source).some(function (key) { return state.source[key] === "records"; });
    var warnings = state.dataset.warnings.slice().concat((state.metrics && state.metrics.warnings) || []);
    if (storageWarning) warnings.push(storageWarning);
    if (usingRecords) {
      infos.push("已自动加载本地记录数据：" + (Generated.latestRecord || "latest") + "；生成时间：" + (Generated.generatedAt || "-") + "。");
      infos.push("所有计算仍在浏览器本地完成；records 数据来自本机 strategy_records 导出。");
      if ((state.dataset.benchmarks || []).length > 1) {
        infos.push("已识别 " + state.dataset.benchmarks.length + " 个 benchmark NAV 字段，可在顶部多选对比。");
      }
      warnings = warnings.concat(Generated.warnings || []);
      infos = infos.concat(Generated.infos || []);
    } else if (usingUser) {
      infos.push("已加载用户上传 CSV；未上传的模块保留当前数据源，所有计算在浏览器本地完成。");
      if (state.userUploadSavedAt) infos.push("用户上传数据已保存于本机浏览器：" + formatStoredTimestamp(state.userUploadSavedAt) + "。");
    } else {
      infos.push("示例数据和 sample benchmark 均为模拟数据，仅用于演示，不代表真实业绩或真实指数；所有计算在浏览器本地完成。");
    }
    UI.renderMessages(state.dataset.errors, warnings, infos);
    setUploadNames();
    setStatus();
  }

  function handleFileUpload(kind, file) {
    if (!file) return;
    Data.readFileAsText(file)
      .then(function (text) {
        state.csvBundle[kind] = text;
        state.source[kind] = "user";
        state.fileNames[kind] = file.name || (kind + ".csv");
        persistUserUploads();
        if (kind === "nav") {
          state.view.benchmarkSelectionTouched = false;
          refresh({ resetDateRange: true });
        }
        else refresh();
      })
      .catch(function (error) {
        UI.renderMessages([error.message], [], []);
      });
  }

  function resetToSample() {
    state.csvBundle = {
      nav: Data.SAMPLE_CSV.nav,
      positions: Data.SAMPLE_CSV.positions,
      trades: Data.SAMPLE_CSV.trades
    };
    state.source = { nav: "sample", positions: "sample", trades: "sample" };
    state.fileNames = { nav: "nav_sample.csv", positions: "positions_sample.csv", trades: "trades_sample.csv" };
    clearPersistedUploads();
    ["navUpload", "positionsUpload", "tradesUpload"].forEach(function (id) {
      $(id).value = "";
    });
    state.filters = {
      startDate: "",
      endDate: "",
      benchmarkField: "",
      selectedBenchmarkFields: [],
      showPortfolioCurve: true,
      showExcessCurve: true
    };
    state.view.overviewLens = "nav";
    state.view.benchmarkSortField = "totalReturn";
    state.view.benchmarkSortDirection = "desc";
    state.view.benchmarkSelectionTouched = false;
    state.view.tableExpanded = { holdings: false, trades: false, closedTrades: false };
    syncOverviewLensTabs();
    refresh({ resetDateRange: true });
  }

  function exportMetricsCSV() {
    if (!state.metrics) return;
    var rows = UI.metricRows(state.metrics).map(function (row) {
      return row.map(csvEscape).join(",");
    }).join("\n");
    var blob = new Blob([rows], { type: "text/csv;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "dashboard_metrics_" + (state.metrics.performance.kpis.end_date || "current") + ".csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function bindEvents() {
    $("navUpload").addEventListener("change", function (event) {
      handleFileUpload("nav", event.target.files[0]);
    });
    $("positionsUpload").addEventListener("change", function (event) {
      handleFileUpload("positions", event.target.files[0]);
    });
    $("tradesUpload").addEventListener("change", function (event) {
      handleFileUpload("trades", event.target.files[0]);
    });
    $("resetButton").addEventListener("click", resetToSample);
    $("exportButton").addEventListener("click", exportMetricsCSV);
    $("startDate").addEventListener("change", function (event) {
      state.filters.startDate = event.target.value;
      refresh();
    });
    $("endDate").addEventListener("change", function (event) {
      state.filters.endDate = event.target.value;
      refresh();
    });
    $("benchmarkSelect").addEventListener("change", function (event) {
      state.filters.benchmarkField = event.target.value;
      refresh();
    });
    $("benchmarkMultiSelect").addEventListener("change", function () {
      state.view.benchmarkSelectionTouched = true;
      state.filters.selectedBenchmarkFields = selectedBenchmarkValues();
      refresh();
    });
    $("selectAllBenchmarks").addEventListener("click", function () {
      state.view.benchmarkSelectionTouched = true;
      state.filters.selectedBenchmarkFields = (state.dataset.benchmarks || []).map(function (benchmark) {
        return benchmark.field;
      });
      refresh();
    });
    $("clearBenchmarks").addEventListener("click", function () {
      state.view.benchmarkSelectionTouched = true;
      state.filters.selectedBenchmarkFields = [];
      refresh();
    });
    $("showPortfolioCurve").addEventListener("change", function (event) {
      state.filters.showPortfolioCurve = event.target.checked;
      refresh();
    });
    $("showExcessCurve").addEventListener("change", function (event) {
      state.filters.showExcessCurve = event.target.checked;
      refresh();
    });
    document.querySelectorAll("[data-benchmark-sort]").forEach(function (button) {
      button.addEventListener("click", function () {
        var field = button.dataset.benchmarkSort;
        if (state.view.benchmarkSortField === field) {
          state.view.benchmarkSortDirection = state.view.benchmarkSortDirection === "desc" ? "asc" : "desc";
        } else {
          state.view.benchmarkSortField = field;
          state.view.benchmarkSortDirection = "desc";
        }
        updateBenchmarkWarnings();
        renderCurrentDashboard();
      });
    });
    bindOverviewLens();
    bindSectionTabs();
    bindTableFilters();
    bindTableToggles();
    window.addEventListener("resize", debounce(function () {
      if (state.metrics) renderCurrentDashboard();
    }, 160));
  }

  function renderCurrentDashboard() {
    if (!state.metrics) return;
    UI.renderDashboard(state.metrics, state.view);
    applyTableFilters();
  }

  function syncOverviewLensTabs() {
    document.querySelectorAll("#overviewLensTabs button").forEach(function (button) {
      button.classList.toggle("active", button.dataset.lens === state.view.overviewLens);
    });
  }

  function bindOverviewLens() {
    var tabs = $("overviewLensTabs");
    if (!tabs) return;
    tabs.addEventListener("click", function (event) {
      var button = event.target.closest("button[data-lens]");
      if (!button) return;
      state.view.overviewLens = button.dataset.lens || "nav";
      syncOverviewLensTabs();
      renderCurrentDashboard();
    });
  }

  function bindSectionTabs() {
    var tabs = Array.from(document.querySelectorAll(".section-tab"));
    var sections = tabs.map(function (tab) {
      var href = tab.getAttribute("href") || "";
      return { tab: tab, section: href.charAt(0) === "#" ? document.querySelector(href) : null };
    }).filter(function (item) { return item.section; });
    function setActive(hash) {
      tabs.forEach(function (tab) {
        tab.classList.toggle("active", tab.getAttribute("href") === hash);
      });
    }
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        setActive(tab.getAttribute("href"));
      });
    });
    if ("IntersectionObserver" in window && sections.length) {
      var observer = new IntersectionObserver(function (entries) {
        var visible = entries.filter(function (entry) { return entry.isIntersecting; })
          .sort(function (a, b) { return b.intersectionRatio - a.intersectionRatio; })[0];
        if (visible) setActive("#" + visible.target.id);
      }, { rootMargin: "-20% 0px -68% 0px", threshold: [0.12, 0.3, 0.6] });
      sections.forEach(function (item) { observer.observe(item.section); });
    }
  }

  function bindTableFilters() {
    document.querySelectorAll("input[data-filter-table]").forEach(function (input) {
      input.addEventListener("input", function () {
        filterTable(input.dataset.filterTable, input.value);
      });
    });
  }

  function bindTableToggles() {
    [
      ["toggleHoldingsRows", "holdings"],
      ["toggleTradesRows", "trades"],
      ["toggleClosedTradesRows", "closedTrades"]
    ].forEach(function (item) {
      var button = $(item[0]);
      if (!button) return;
      button.addEventListener("click", function () {
        state.view.tableExpanded[item[1]] = !state.view.tableExpanded[item[1]];
        renderCurrentDashboard();
      });
    });
  }

  function applyTableFilters() {
    document.querySelectorAll("input[data-filter-table]").forEach(function (input) {
      filterTable(input.dataset.filterTable, input.value);
    });
  }

  function filterTable(containerId, query) {
    var container = $(containerId);
    if (!container) return;
    var rows = container.querySelectorAll("tbody tr");
    var needle = String(query || "").trim().toLowerCase();
    rows.forEach(function (row) {
      row.classList.toggle("filtered-out", Boolean(needle) && row.textContent.toLowerCase().indexOf(needle) < 0);
    });
  }

  function debounce(fn, wait) {
    var timeout = null;
    return function () {
      var args = arguments;
      clearTimeout(timeout);
      timeout = setTimeout(function () {
        fn.apply(null, args);
      }, wait);
    };
  }

  document.addEventListener("DOMContentLoaded", function () {
    bindEvents();
    refresh({ resetDateRange: true });
  });
})();

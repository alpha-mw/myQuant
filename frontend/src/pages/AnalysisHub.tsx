import { Component, Suspense, lazy, useDeferredValue, useEffect, useState, type ReactNode } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useSearchParams } from 'react-router-dom';
import { ChevronRight, X } from 'lucide-react';
import {
  fetchAnalysisJob,
  fetchAnalysisOptions,
  fetchAnalysisResult,
  fetchRecentJobs,
  runAnalysis,
} from '../api/analysis';
import { fetchStocks } from '../api/data';
import { fetchPortfolioState } from '../api/portfolio';
import type {
  AnalysisBranchConfig,
  AnalysisLlmDebateConfig,
  AnalysisPortfolioConfig,
  AnalysisRiskConfig,
  AnalysisRunRequest,
  AnalysisOptionsResponse,
  StockInfo,
} from '../types/api';

const BranchDetailModal = lazy(() => import('../components/analysis/BranchDetailModal'));
const ReportMarkdown = lazy(() => import('../components/analysis/ReportMarkdown'));

const defaultBranches: Record<string, AnalysisBranchConfig> = {
  kline: { enabled: true, settings: { prediction_horizon: '20d', trend_window: '60d', regime_filter: 'auto', backend: 'heuristic' } },
  quant: { enabled: true, settings: { factor_pack: 'core', rebalance: 'monthly', neutralize: true } },
  llm_debate: { enabled: true, settings: { rounds: 2 } },
  intelligence: { enabled: true, settings: { event_risk: true, capital_flow: true, breadth: true } },
  macro: { enabled: true, settings: { overlay_strength: 'medium', scope: 'market' } },
};

const defaultRisk: AnalysisRiskConfig = {
  capital: 1_000_000,
  risk_level: '中等',
  max_single_position: 0.2,
  max_drawdown_limit: 0.15,
  default_stop_loss: 0.08,
  keep_cash_buffer: true,
};

const defaultPortfolio: AnalysisPortfolioConfig = {
  candidate_limit: 8,
  allocation_mode: 'target_weight',
  allow_cash_buffer: true,
};

const defaultLlm: AnalysisLlmDebateConfig = {
  enabled: true,
  models: [],
  rounds: 2,
  assignment_mode: 'random_balanced',
  judge_mode: 'auto',
  judge_model: null,
  assignments: [],
};

type SelectedTarget = {
  symbol: string;
  name: string | null;
  market: string;
};

const defaultTargets: SelectedTarget[] = [
  { symbol: '000001.SZ', name: '平安银行', market: 'CN' },
];

function dedupeTargets(targets: SelectedTarget[]) {
  const next = new Map<string, SelectedTarget>();
  targets.forEach((item) => {
    next.set(item.symbol, item);
  });
  return Array.from(next.values());
}

function stockToTarget(stock: StockInfo): SelectedTarget {
  return {
    symbol: stock.ts_code,
    name: stock.name,
    market: stock.market ?? 'CN',
  };
}

function normalizeManualTargetInput(value: string, market: string): string | null {
  const normalized = value.trim().toUpperCase().replace(/-/g, '.');
  if (!normalized) return null;

  if (market === 'CN') {
    const exact = normalized.match(/^(\d{6})\.(SH|SZ|BJ)$/);
    if (exact) return `${exact[1]}.${exact[2]}`;

    const prefixed = normalized.match(/^(SH|SZ|BJ)(\d{6})$/);
    if (prefixed) return `${prefixed[2]}.${prefixed[1]}`;

    const digits = normalized.replace(/\D/g, '');
    if (digits.length === 6) {
      if (/^[695]/.test(digits)) return `${digits}.SH`;
      if (/^[48]/.test(digits)) return `${digits}.BJ`;
      return `${digits}.SZ`;
    }
    return null;
  }

  return /^[A-Z][A-Z0-9.]{0,9}$/.test(normalized) ? normalized : null;
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatPrice(value: number) {
  return value.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatDateTime(value: string) {
  return value.replace('T', ' ').slice(0, 16);
}

function cloneState<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function asRecord(value: unknown) {
  return typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : {};
}

function mergeBranches(source: Record<string, unknown> | undefined) {
  const next = cloneState(defaultBranches);
  const incoming = source ?? {};
  Object.entries(incoming).forEach(([key, raw]) => {
    const current = next[key] ?? { enabled: true, settings: {} };
    const config = asRecord(raw);
    next[key] = {
      enabled: typeof config.enabled === 'boolean' ? config.enabled : current.enabled,
      settings: { ...current.settings, ...asRecord(config.settings) },
    };
  });
  return next;
}

function buildPresetState(options: AnalysisOptionsResponse | undefined, presetId: string) {
  const preset = options?.presets.find((item) => item.id === presetId);
  const defaults = asRecord(preset?.defaults);
  return {
    mode: preset?.mode ?? 'single',
    branches: mergeBranches(defaults.branches as Record<string, unknown> | undefined),
    risk: { ...cloneState(defaultRisk), ...asRecord(defaults.risk) } as AnalysisRiskConfig,
    portfolio: { ...cloneState(defaultPortfolio), ...asRecord(defaults.portfolio) } as AnalysisPortfolioConfig,
    llmDebate: { ...cloneState(defaultLlm), ...asRecord(defaults.llm_debate) } as AnalysisLlmDebateConfig,
  };
}

export default function AnalysisHub() {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedAnalysisId = searchParams.get('id');

  const [mode, setMode] = useState('single');
  const [preset, setPreset] = useState('quick_scan');
  const [market, setMarket] = useState('CN');
  const [manualTargets, setManualTargets] = useState<SelectedTarget[]>(defaultTargets);
  const [targetSearch, setTargetSearch] = useState('');
  const [selectedHoldingAccount, setSelectedHoldingAccount] = useState('ALL');
  const [excludedHoldingSymbols, setExcludedHoldingSymbols] = useState<string[]>([]);
  const [excludedWatchlistSymbols, setExcludedWatchlistSymbols] = useState<string[]>([]);
  const [branches, setBranches] = useState<Record<string, AnalysisBranchConfig>>(cloneState(defaultBranches));
  const [risk, setRisk] = useState<AnalysisRiskConfig>(cloneState(defaultRisk));
  const [portfolio, setPortfolio] = useState<AnalysisPortfolioConfig>(cloneState(defaultPortfolio));
  const [llmDebate, setLlmDebate] = useState<AnalysisLlmDebateConfig>(cloneState(defaultLlm));
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [currentJobStatus, setCurrentJobStatus] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [expandedBranch, setExpandedBranch] = useState<string | null>(null);
  const deferredTargetSearch = useDeferredValue(targetSearch.trim());

  const optionsQuery = useQuery({
    queryKey: ['analysis-options'],
    queryFn: fetchAnalysisOptions,
  });

  const portfolioQuery = useQuery({
    queryKey: ['portfolio-state'],
    queryFn: fetchPortfolioState,
  });

  const stockSearchQuery = useQuery({
    queryKey: ['analysis-target-search', market, deferredTargetSearch],
    queryFn: () => fetchStocks({ market, search: deferredTargetSearch, limit: 12 }),
    enabled: deferredTargetSearch.length >= 1 && mode === 'single',
  });
  const marketUniverseQuery = useQuery({
    queryKey: ['analysis-market-universe', market],
    queryFn: () => fetchStocks({ market, limit: 1 }),
    enabled: mode === 'market',
  });

  const jobsQuery = useQuery({
    queryKey: ['analysis-jobs', 8],
    queryFn: () => fetchRecentJobs(8),
    refetchInterval: currentJobId ? 2_000 : 15_000,
  });

  const holdingAccounts = portfolioQuery.data?.summary.accounts ?? [];
  const effectiveHoldingAccount = selectedHoldingAccount === 'ALL' || holdingAccounts.includes(selectedHoldingAccount)
    ? selectedHoldingAccount
    : 'ALL';
  const holdingTargets = dedupeTargets(
    (portfolioQuery.data?.holdings ?? [])
      .filter((item) => item.market === market)
      .filter((item) => effectiveHoldingAccount === 'ALL' || item.account_name === effectiveHoldingAccount)
      .map((item) => ({ symbol: item.symbol, name: item.name, market: item.market })),
  );
  const watchlistTargets = dedupeTargets(
    (portfolioQuery.data?.watchlist ?? [])
      .filter((item) => item.market === market)
      .map((item) => ({ symbol: item.symbol, name: item.name, market: item.market })),
  );
  const activeTargets = mode === 'holdings'
    ? holdingTargets.filter((item) => !excludedHoldingSymbols.includes(item.symbol))
    : mode === 'watchlist'
      ? watchlistTargets.filter((item) => !excludedWatchlistSymbols.includes(item.symbol))
      : mode === 'single'
        ? manualTargets
        : [];
  const targets = activeTargets.map((item) => item.symbol);
  const manualCandidateSymbol = mode === 'single' ? normalizeManualTargetInput(targetSearch, market) : null;
  const manualCandidateAlreadySelected = manualCandidateSymbol
    ? activeTargets.some((item) => item.symbol === manualCandidateSymbol)
    : false;
  const manualCandidateAlreadyMatched = manualCandidateSymbol
    ? (stockSearchQuery.data?.items ?? []).some((item) => item.ts_code === manualCandidateSymbol)
    : false;

  useEffect(() => {
    if (!currentJobId) return;

    let cancelled = false;
    let nextPollTimer: number | null = null;

    const pollJob = async () => {
      try {
        const response = await fetchAnalysisJob(currentJobId);
        if (cancelled) return;

        setCurrentJobStatus(response.status);

        if (response.status === 'completed' && response.result) {
          queryClient.setQueryData(['analysis-result', response.result.analysis_id], response.result);
          setCurrentJobId(null);
          setCurrentJobStatus(null);
          setSearchParams((current) => {
            const next = new URLSearchParams(current);
            next.set('id', response.result!.analysis_id);
            return next;
          }, { replace: true });
          await Promise.all([
            queryClient.invalidateQueries({ queryKey: ['analysis-history'] }),
            queryClient.invalidateQueries({ queryKey: ['analysis-history-paged'] }),
            queryClient.invalidateQueries({ queryKey: ['analysis-jobs'] }),
          ]);
          return;
        }

        if (response.status === 'failed') {
          setRunError(response.error ?? '分析失败');
          setCurrentJobId(null);
          setCurrentJobStatus(null);
          void queryClient.invalidateQueries({ queryKey: ['analysis-jobs'] });
          return;
        }

        nextPollTimer = window.setTimeout(pollJob, 1500);
      } catch (error) {
        if (cancelled) return;
        setRunError(error instanceof Error ? error.message : '任务状态获取失败');
        setCurrentJobId(null);
        setCurrentJobStatus(null);
      }
    };

    void pollJob();

    return () => {
      cancelled = true;
      if (nextPollTimer != null) {
        window.clearTimeout(nextPollTimer);
      }
    };
  }, [currentJobId, queryClient, setSearchParams]);

  const detailQuery = useQuery({
    queryKey: ['analysis-result', selectedAnalysisId],
    queryFn: () => fetchAnalysisResult(selectedAnalysisId!),
    enabled: !!selectedAnalysisId,
  });

  function applyPreset(nextPreset: string) {
    setPreset(nextPreset);
    const next = buildPresetState(optionsQuery.data, nextPreset);
    changeMode(next.mode);
    setBranches(next.branches);
    setRisk(next.risk);
    setPortfolio(next.portfolio);
    setLlmDebate(next.llmDebate);
  }

  function changeMode(nextMode: string) {
    setMode(nextMode);
    setTargetSearch('');
    setExcludedHoldingSymbols([]);
    setExcludedWatchlistSymbols([]);
    if (nextMode === 'single') {
      setManualTargets((current) => (current.length > 1 ? current.slice(0, 1) : current));
    }
  }

  function addTarget(stock: StockInfo) {
    const target = stockToTarget(stock);
    setManualTargets((current) => {
      return mode === 'single' ? [target] : dedupeTargets([...current, target]);
    });
    setTargetSearch('');
  }

  function addManualTarget(symbol: string) {
    const target: SelectedTarget = {
      symbol,
      name: null,
      market,
    };
    setManualTargets((current) => (mode === 'single' ? [target] : dedupeTargets([...current, target])));
    setTargetSearch('');
  }

  function removeTarget(symbol: string) {
    if (mode === 'holdings') {
      setExcludedHoldingSymbols((current) => [...current, symbol]);
      return;
    }
    if (mode === 'watchlist') {
      setExcludedWatchlistSymbols((current) => [...current, symbol]);
      return;
    }
    setManualTargets((current) => current.filter((item) => item.symbol !== symbol));
  }

  function updateBranch(branchName: string, updater: (branch: AnalysisBranchConfig) => AnalysisBranchConfig) {
    setBranches((current) => ({
      ...current,
      [branchName]: updater(current[branchName] ?? { enabled: true, settings: {} }),
    }));
  }

  function toggleModel(modelId: string) {
    setLlmDebate((current) => ({
      ...current,
      models: current.models.includes(modelId)
        ? current.models.filter((item) => item !== modelId)
        : [...current.models, modelId],
    }));
  }

  const runMutation = useMutation({
    mutationFn: () => {
      const payload: AnalysisRunRequest = {
        mode,
        targets,
        stocks: targets,
        preset,
        market,
        branches,
        risk,
        portfolio,
        llm_debate: { ...llmDebate, enabled: branches.llm_debate?.enabled ?? llmDebate.enabled },
      };
      return runAnalysis(payload);
    },
    onMutate: () => setRunError(null),
    onSuccess: (response) => {
      if (!response.ok || !response.job_id) {
        setRunError(response.error ?? '分析失败');
        return;
      }
      setCurrentJobId(response.job_id);
      setCurrentJobStatus(response.status);
    },
    onError: (error: Error) => setRunError(error.message),
  });

  const activeResult = detailQuery.data ?? null;
  const runningJobs = jobsQuery.data?.filter((item) => item.status === 'queued' || item.status === 'running') ?? [];
  const expandedBranchData = expandedBranch
    ? activeResult?.branches.find((b) => b.branch_name === expandedBranch) ?? null
    : null;

  return (
    <div className="space-y-6">
      {/* Config Section */}
      <section className="paper-card">
        <div className="section-header">
          <div>
            <p className="panel-kicker">配置</p>
            <h3>研究参数</h3>
          </div>
        </div>

        <div className="space-y-5">
          {/* Always visible: mode, preset, market, targets */}
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="space-y-2">
              <label className="filter-label">研究模式</label>
              <div className="mode-grid">
                {[
                  { id: 'single', label: '单只股票' },
                  { id: 'holdings', label: '我的持仓' },
                  { id: 'watchlist', label: '自选池' },
                  { id: 'market', label: '全市场' },
                ].map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => changeMode(item.id)}
                    className={`mode-chip ${mode === item.id ? 'is-active' : ''}`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            <SelectField
              label="预设模板"
              value={preset}
              onChange={applyPreset}
              options={(optionsQuery.data?.presets ?? []).map((item) => ({
                value: item.id,
                label: item.label,
              }))}
            />

            <SelectField
              label="市场"
              value={market}
              onChange={(value) => {
                setMarket(value);
                setExcludedHoldingSymbols([]);
                setExcludedWatchlistSymbols([]);
              }}
              options={[
                { value: 'CN', label: 'A股' },
                { value: 'US', label: '美股' },
              ]}
            />

            <div className="space-y-2 xl:col-span-2">
              <label className="filter-label" htmlFor="analysis-targets">
                研究标的 ({mode === 'market' ? `${marketUniverseQuery.data?.total ?? 0} 只` : `${targets.length} 只`})
              </label>
              {mode === 'holdings' ? (
                <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(12,33,60,0.04)] p-4">
                  <div className="flex flex-col gap-3">
                    <div>
                      <div className="text-sm font-semibold text-[var(--ink)]">我的持仓自动带入</div>
                      <div className="mt-1 text-sm text-[var(--muted)]">
                        已带入 {targets.length} 只 {market === 'CN' ? 'A股' : '美股'} 持仓股票，可以临时删减但不会修改真实仓位。
                      </div>
                    </div>
                    <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
                      <SelectField
                        label="账户范围"
                        value={effectiveHoldingAccount}
                        onChange={(value) => {
                          setSelectedHoldingAccount(value);
                          setExcludedHoldingSymbols([]);
                        }}
                        options={[
                          { value: 'ALL', label: '全部账户' },
                          ...holdingAccounts.map((account) => ({ value: account, label: account })),
                        ]}
                      />
                      <div className="flex items-end">
                        <Link to="/watchlists" className="secondary-button justify-center">
                          管理持仓
                        </Link>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {targets.length ? (
                      activeTargets.map((item) => (
                        <button
                          key={item.symbol}
                          type="button"
                          className="candidate-chip inline-flex items-center gap-2"
                          onClick={() => removeTarget(item.symbol)}
                        >
                          <span>{item.symbol}{item.name ? ` · ${item.name}` : ''}</span>
                          <X size={14} />
                        </button>
                      ))
                    ) : (
                      <div className="empty-inline">当前市场下没有持仓，先去“持仓与观察”里维护仓位。</div>
                    )}
                  </div>
                </div>
              ) : mode === 'watchlist' ? (
                <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(12,33,60,0.04)] p-4">
                  <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div>
                      <div className="text-sm font-semibold text-[var(--ink)]">自选池自动带入</div>
                      <div className="mt-1 text-sm text-[var(--muted)]">
                        已带入 {targets.length} 只待观察股票，可以临时删减本次分析范围。
                      </div>
                    </div>
                    <Link to="/watchlists" className="secondary-button justify-center">
                      管理自选池
                    </Link>
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {targets.length ? (
                      activeTargets.map((item) => (
                        <button
                          key={item.symbol}
                          type="button"
                          className="candidate-chip inline-flex items-center gap-2"
                          onClick={() => removeTarget(item.symbol)}
                        >
                          <span>{item.symbol}{item.name ? ` · ${item.name}` : ''}</span>
                          <X size={14} />
                        </button>
                      ))
                    ) : (
                      <div className="empty-inline">当前市场下没有自选池股票，先去“持仓与自选池”里维护。</div>
                    )}
                  </div>
                </div>
              ) : mode === 'market' ? (
                <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(12,33,60,0.04)] p-4">
                  <div className="text-sm font-semibold text-[var(--ink)]">全市场分析</div>
                  <div className="mt-1 text-sm text-[var(--muted)]">
                    将直接分析当前市场下的全部股票：
                    {marketUniverseQuery.isLoading ? ' 正在统计...' : ` ${marketUniverseQuery.data?.total ?? 0} 只`}
                  </div>
                  <div className="mt-3 rounded-[18px] bg-white/80 px-3 py-2 text-sm text-[var(--muted)]">
                    当前选择：{market === 'CN' ? 'A股全市场' : '美股全市场'}。建议优先使用偏全市场扫描的模板。
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <input
                    id="analysis-targets"
                    value={targetSearch}
                    onChange={(event) => setTargetSearch(event.target.value)}
                    placeholder="搜索一只股票代码或名称"
                    className="app-input"
                  />

                  <div className="flex flex-wrap gap-2">
                    {activeTargets.length ? (
                      activeTargets.map((item) => (
                        <button
                          key={item.symbol}
                          type="button"
                          className="candidate-chip inline-flex items-center gap-2"
                          onClick={() => removeTarget(item.symbol)}
                        >
                          <span>{item.symbol}{item.name ? ` · ${item.name}` : ''}</span>
                          <X size={14} />
                        </button>
                      ))
                    ) : (
                      <div className="empty-inline">先从下方搜索结果里选择一只股票。</div>
                    )}
                  </div>

                  {targetSearch && (
                    <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.82)] p-2">
                      {stockSearchQuery.isLoading ? (
                        <div className="empty-inline">搜索中...</div>
                      ) : stockSearchQuery.data?.items.length ? (
                        <div className="space-y-2">
                          {stockSearchQuery.data.items.map((item) => (
                            <button
                              key={item.ts_code}
                              type="button"
                              className="flex w-full items-center justify-between rounded-[16px] px-3 py-2 text-left text-sm transition-colors hover:bg-[rgba(12,33,60,0.05)]"
                              onClick={() => addTarget(item)}
                            >
                              <span className="text-[var(--ink)]">
                                {item.ts_code}
                                {item.name ? ` · ${item.name}` : ''}
                              </span>
                              <span className="text-[var(--muted)]">{item.market ?? market}</span>
                            </button>
                          ))}
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className="empty-inline">没有匹配结果，换个代码或名称试试。</div>
                          {manualCandidateSymbol && !manualCandidateAlreadySelected && !manualCandidateAlreadyMatched ? (
                            <button
                              type="button"
                              className="secondary-button w-full justify-center"
                              onClick={() => addManualTarget(manualCandidateSymbol)}
                            >
                              使用 {manualCandidateSymbol} 作为新股票代码，分析前自动下载入库
                            </button>
                          ) : null}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Collapsible: branches */}
          <details className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
            <summary className="cursor-pointer text-sm font-semibold text-[var(--ink)]">五维分支配置</summary>
            <div className="mt-4 space-y-4">
              <BranchPanel
                title="K线分析"
                branch={branches.kline}
                onToggle={(enabled) => updateBranch('kline', (b) => ({ ...b, enabled }))}
              >
                <SelectField
                  label="分析后端"
                  value={String(branches.kline?.settings.backend ?? 'heuristic')}
                  onChange={(v) => updateBranch('kline', (b) => ({ ...b, settings: { ...b.settings, backend: v } }))}
                  options={[
                    { value: 'heuristic', label: '启发式模型' },
                    { value: 'kronos', label: 'Kronos 预训练模型' },
                    { value: 'chronos', label: 'Chronos-2 时序模型' },
                  ]}
                />
                <SelectField
                  label="预测周期"
                  value={String(branches.kline?.settings.prediction_horizon ?? '')}
                  onChange={(v) => updateBranch('kline', (b) => ({ ...b, settings: { ...b.settings, prediction_horizon: v } }))}
                  options={[
                    { value: '5d', label: '5 个交易日' },
                    { value: '20d', label: '20 个交易日' },
                    { value: '60d', label: '60 个交易日' },
                    { value: '120d', label: '120 个交易日' },
                  ]}
                />
                <SelectField
                  label="趋势窗口"
                  value={String(branches.kline?.settings.trend_window ?? '')}
                  onChange={(v) => updateBranch('kline', (b) => ({ ...b, settings: { ...b.settings, trend_window: v } }))}
                  options={[
                    { value: '20d', label: '20 日' },
                    { value: '60d', label: '60 日' },
                    { value: '120d', label: '120 日' },
                    { value: '250d', label: '250 日' },
                  ]}
                />
                <SelectField
                  label="Regime 过滤"
                  value={String(branches.kline?.settings.regime_filter ?? 'auto')}
                  onChange={(v) => updateBranch('kline', (b) => ({ ...b, settings: { ...b.settings, regime_filter: v } }))}
                  options={[
                    { value: 'auto', label: '自动' },
                    { value: 'trend_follow', label: '顺势优先' },
                    { value: 'mean_reversion', label: '均值回归' },
                    { value: 'defensive', label: '防守模式' },
                  ]}
                />
              </BranchPanel>

              <BranchPanel
                title="传统量化"
                branch={branches.quant}
                onToggle={(enabled) => updateBranch('quant', (b) => ({ ...b, enabled }))}
              >
                <SelectField
                  label="因子包"
                  value={String(branches.quant?.settings.factor_pack ?? '')}
                  onChange={(v) => updateBranch('quant', (b) => ({ ...b, settings: { ...b.settings, factor_pack: v } }))}
                  options={[
                    { value: 'core', label: '核心因子' },
                    { value: 'expanded', label: '扩展因子' },
                    { value: 'portfolio', label: '组合构建' },
                    { value: 'quality', label: '质量优先' },
                    { value: 'momentum', label: '动量优先' },
                  ]}
                />
                <SelectField
                  label="调仓频率"
                  value={String(branches.quant?.settings.rebalance ?? '')}
                  onChange={(v) => updateBranch('quant', (b) => ({ ...b, settings: { ...b.settings, rebalance: v } }))}
                  options={[
                    { value: 'weekly', label: '每周' },
                    { value: 'biweekly', label: '双周' },
                    { value: 'monthly', label: '每月' },
                    { value: 'quarterly', label: '每季' },
                  ]}
                />
                <SelectField
                  label="中性化"
                  value={String(Boolean(branches.quant?.settings.neutralize))}
                  onChange={(v) => updateBranch('quant', (b) => ({ ...b, settings: { ...b.settings, neutralize: v === 'true' } }))}
                  options={[
                    { value: 'true', label: '开启' },
                    { value: 'false', label: '关闭' },
                  ]}
                />
              </BranchPanel>

              <BranchPanel
                title="LLM 多空辩论"
                branch={branches.llm_debate}
                onToggle={(enabled) => {
                  updateBranch('llm_debate', (b) => ({ ...b, enabled }));
                  setLlmDebate((c) => ({ ...c, enabled }));
                }}
              >
                <SelectField
                  label="辩论轮次"
                  value={String(llmDebate.rounds)}
                  onChange={(v) => setLlmDebate((c) => ({ ...c, rounds: Number(v || 0) }))}
                  options={[
                    { value: '1', label: '1 轮' },
                    { value: '2', label: '2 轮' },
                    { value: '3', label: '3 轮' },
                    { value: '4', label: '4 轮' },
                  ]}
                />
                <SelectField
                  label="角色分配"
                  value={llmDebate.assignment_mode}
                  onChange={(v) => setLlmDebate((c) => ({ ...c, assignment_mode: v }))}
                  options={[
                    { value: 'random_balanced', label: '随机均衡' },
                    { value: 'bull_bias', label: '多头优先' },
                    { value: 'bear_bias', label: '空头优先' },
                  ]}
                />
                <SelectField
                  label="裁判模式"
                  value={llmDebate.judge_mode}
                  onChange={(v) => setLlmDebate((c) => ({ ...c, judge_mode: v }))}
                  options={[
                    { value: 'auto', label: '自动' },
                    { value: 'manual', label: '手动' },
                  ]}
                />
                <div className="space-y-2">
                  <label className="filter-label">模型池</label>
                  <div className="space-y-2">
                    {(optionsQuery.data?.llm_models ?? []).map((model) => (
                      <label
                        key={model.id}
                        className={`flex items-center justify-between rounded-[18px] border px-3 py-2 text-sm ${
                          model.enabled
                            ? 'border-[var(--line)] bg-[rgba(12,33,60,0.04)] text-[var(--ink)]'
                            : 'border-[rgba(190,92,44,0.18)] bg-[rgba(190,92,44,0.07)] text-[var(--muted)]'
                        }`}
                      >
                        <span>{model.label} · {model.provider}</span>
                        <input
                          type="checkbox"
                          checked={llmDebate.models.includes(model.id)}
                          onChange={() => toggleModel(model.id)}
                          disabled={!model.enabled}
                        />
                      </label>
                    ))}
                  </div>
                </div>
              </BranchPanel>

              <BranchPanel
                title="多维智能融合"
                branch={branches.intelligence}
                onToggle={(enabled) => updateBranch('intelligence', (b) => ({ ...b, enabled }))}
              >
                <ToggleRow label="事件风险" checked={Boolean(branches.intelligence?.settings.event_risk)} onChange={(c) => updateBranch('intelligence', (b) => ({ ...b, settings: { ...b.settings, event_risk: c } }))} />
                <ToggleRow label="资金流" checked={Boolean(branches.intelligence?.settings.capital_flow)} onChange={(c) => updateBranch('intelligence', (b) => ({ ...b, settings: { ...b.settings, capital_flow: c } }))} />
                <ToggleRow label="市场广度" checked={Boolean(branches.intelligence?.settings.breadth)} onChange={(c) => updateBranch('intelligence', (b) => ({ ...b, settings: { ...b.settings, breadth: c } }))} />
                <SelectField
                  label="融合重心"
                  value={String(branches.intelligence?.settings.focus ?? 'balanced')}
                  onChange={(v) => updateBranch('intelligence', (b) => ({ ...b, settings: { ...b.settings, focus: v } }))}
                  options={[
                    { value: 'balanced', label: '均衡融合' },
                    { value: 'event_first', label: '事件优先' },
                    { value: 'flow_first', label: '资金流优先' },
                    { value: 'breadth_first', label: '市场广度优先' },
                  ]}
                />
              </BranchPanel>

              <BranchPanel
                title="宏观分支"
                branch={branches.macro}
                onToggle={(enabled) => updateBranch('macro', (b) => ({ ...b, enabled }))}
              >
                <SelectField
                  label="宏观覆盖"
                  value={String(branches.macro?.settings.scope ?? 'market')}
                  onChange={(v) => updateBranch('macro', (b) => ({ ...b, settings: { ...b.settings, scope: v } }))}
                  options={[
                    { value: 'market', label: '市场级' },
                    { value: 'sector', label: '行业级' },
                    { value: 'global', label: '全球宏观' },
                  ]}
                />
                <SelectField
                  label="Overlay 强度"
                  value={String(branches.macro?.settings.overlay_strength ?? 'medium')}
                  onChange={(v) => updateBranch('macro', (b) => ({ ...b, settings: { ...b.settings, overlay_strength: v } }))}
                  options={[
                    { value: 'low', label: '低' },
                    { value: 'medium', label: '中' },
                    { value: 'high', label: '高' },
                  ]}
                />
                <SelectField
                  label="周期聚焦"
                  value={String(branches.macro?.settings.cycle_focus ?? 'liquidity')}
                  onChange={(v) => updateBranch('macro', (b) => ({ ...b, settings: { ...b.settings, cycle_focus: v } }))}
                  options={[
                    { value: 'liquidity', label: '流动性' },
                    { value: 'policy', label: '政策' },
                    { value: 'inflation', label: '通胀' },
                    { value: 'growth', label: '增长周期' },
                  ]}
                />
              </BranchPanel>
            </div>
          </details>

          {/* Collapsible: risk */}
          <details className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
            <summary className="cursor-pointer text-sm font-semibold text-[var(--ink)]">风控设置</summary>
            <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              <SelectField
                label="总资金"
                value={String(risk.capital)}
                onChange={(v) => setRisk((c) => ({ ...c, capital: Number(v || 0) }))}
                options={[
                  { value: '500000', label: '50 万' },
                  { value: '1000000', label: '100 万' },
                  { value: '3000000', label: '300 万' },
                  { value: '10000000', label: '1000 万' },
                ]}
              />
              <SelectField
                label="风险偏好"
                value={risk.risk_level}
                onChange={(v) => setRisk((c) => ({ ...c, risk_level: v }))}
                options={[
                  { value: '保守', label: '保守' },
                  { value: '中等', label: '中等' },
                  { value: '积极', label: '积极' },
                ]}
              />
              <SelectField
                label="单票上限"
                value={String(risk.max_single_position)}
                onChange={(v) => setRisk((c) => ({ ...c, max_single_position: Number(v || 0) }))}
                options={[
                  { value: '0.1', label: '10%' },
                  { value: '0.15', label: '15%' },
                  { value: '0.2', label: '20%' },
                  { value: '0.25', label: '25%' },
                  { value: '0.3', label: '30%' },
                ]}
              />
              <SelectField
                label="回撤红线"
                value={String(risk.max_drawdown_limit)}
                onChange={(v) => setRisk((c) => ({ ...c, max_drawdown_limit: Number(v || 0) }))}
                options={[
                  { value: '0.08', label: '8%' },
                  { value: '0.1', label: '10%' },
                  { value: '0.12', label: '12%' },
                  { value: '0.15', label: '15%' },
                  { value: '0.18', label: '18%' },
                  { value: '0.2', label: '20%' },
                ]}
              />
              <SelectField
                label="默认止损"
                value={String(risk.default_stop_loss)}
                onChange={(v) => setRisk((c) => ({ ...c, default_stop_loss: Number(v || 0) }))}
                options={[
                  { value: '0.05', label: '5%' },
                  { value: '0.06', label: '6%' },
                  { value: '0.08', label: '8%' },
                  { value: '0.1', label: '10%' },
                  { value: '0.12', label: '12%' },
                ]}
              />
              <SelectField
                label="候选数量"
                value={String(portfolio.candidate_limit)}
                onChange={(v) => setPortfolio((c) => ({ ...c, candidate_limit: Number(v || 0) }))}
                options={[
                  { value: '5', label: '5 只' },
                  { value: '8', label: '8 只' },
                  { value: '10', label: '10 只' },
                  { value: '15', label: '15 只' },
                  { value: '20', label: '20 只' },
                ]}
              />
              <SelectField
                label="分配模式"
                value={portfolio.allocation_mode}
                onChange={(v) => setPortfolio((c) => ({ ...c, allocation_mode: v }))}
                options={[
                  { value: 'target_weight', label: '目标权重' },
                  { value: 'conviction_weight', label: '信念权重' },
                  { value: 'risk_budget', label: '风险预算' },
                ]}
              />
              <ToggleRow
                label="保留现金缓冲"
                checked={risk.keep_cash_buffer}
                onChange={(c) => {
                  setRisk((cur) => ({ ...cur, keep_cash_buffer: c }));
                  setPortfolio((cur) => ({ ...cur, allow_cash_buffer: c }));
                }}
              />
            </div>
          </details>

          {/* Status banners + launch */}
          {runError && <div className="error-banner">{runError}</div>}
          {currentJobId && (
            <div className="info-banner">
              任务已提交，后台状态：{currentJobStatus === 'running' ? '运行中' : '排队中'}。
            </div>
          )}
          {runningJobs.length > 0 && !currentJobId && (
            <div className="info-banner">
              {runningJobs.length} 个任务运行中。
            </div>
          )}

          <button
            type="button"
            className="primary-button w-full justify-center"
            disabled={((mode === 'market' ? (marketUniverseQuery.data?.total ?? 0) === 0 : targets.length === 0)) || runMutation.isPending || !!currentJobId}
            onClick={() => runMutation.mutate()}
          >
            {runMutation.isPending || currentJobId ? '运行中...' : '启动分析'}
          </button>
        </div>
      </section>

      {/* Results Section */}
      {!activeResult ? (
        <div className="paper-card">
          <div className="empty-card">配置参数后启动任务，或从过往分析页面选择结果。</div>
        </div>
      ) : (
        <>
          {/* Final Decision */}
          <section className="paper-card">
            <div className="section-header">
              <div>
                <p className="panel-kicker">总决策</p>
                <h3>{activeResult.final_decision}</h3>
              </div>
              <div className="text-right text-sm text-[var(--muted)]">
                <div>{formatDateTime(activeResult.created_at)}</div>
                <div className="mt-1">总耗时 {activeResult.total_time.toFixed(1)} 秒</div>
              </div>
            </div>

            <div className="metric-grid">
              <ResultMetric label="目标仓位" value={formatPercent(activeResult.target_exposure)} />
              <ResultMetric label="风格偏好" value={activeResult.style_bias} />
              <ResultMetric label="风险等级" value={activeResult.risk.risk_level} />
              <ResultMetric label="研究模式" value={modeLabelMap[activeResult.request.mode] ?? activeResult.request.mode} />
            </div>

            <div className="mt-4 rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
              <div className="text-sm font-semibold text-[var(--ink)]">候选标的</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {activeResult.candidate_symbols.length ? (
                  activeResult.candidate_symbols.map((symbol) => (
                    <span key={symbol} className="candidate-chip">{symbol}</span>
                  ))
                ) : (
                  <div className="empty-inline">暂无候选。</div>
                )}
              </div>
            </div>
          </section>

          {/* Five Branches - Clickable */}
          <section className="paper-card">
            <div className="section-header">
              <div>
                <p className="panel-kicker">五维结论</p>
                <h3>点击查看分支详细分析</h3>
              </div>
            </div>

            <div className="grid gap-4 xl:grid-cols-2">
              {activeResult.branches.map((branch) => (
                <button
                  key={branch.branch_name}
                  type="button"
                  onClick={() => setExpandedBranch(branch.branch_name)}
                  className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4 text-left transition-colors hover:border-[var(--accent)] cursor-pointer"
                >
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <div className="text-base font-semibold text-[var(--ink)]">{branchTitleMap[branch.branch_name] ?? branch.branch_name}</div>
                      <div className="mt-1 text-xs text-[var(--muted)]">
                        {branch.enabled ? '已启用' : '未启用'} · 置信度 {formatPercent(branch.confidence)}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className={`text-xl font-semibold ${branch.score >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                        {branch.score >= 0 ? '+' : ''}{branch.score.toFixed(2)}
                      </div>
                      <ChevronRight size={16} className="text-[var(--muted)]" />
                    </div>
                  </div>
                  <p className="mt-3 text-sm leading-7 text-[var(--muted)] line-clamp-2">{branch.explanation || '暂无分支解释。'}</p>
                  {branch.top_symbols.length > 0 && (
                    <div className="mt-3 text-sm text-[var(--ink-soft)]">支持标的：{branch.top_symbols.join('、')}</div>
                  )}
                  {branch.risks.length > 0 && (
                    <div className="mt-3 rounded-[18px] bg-[rgba(190,92,44,0.08)] px-3 py-2 text-sm text-[var(--danger)]">
                      风险：{branch.risks[0]}
                    </div>
                  )}
                </button>
              ))}
            </div>
          </section>

          {/* Execution Plan */}
          <section className="paper-card">
            <div className="section-header">
              <div>
                <p className="panel-kicker">执行建议</p>
                <h3>买卖区间、仓位与理由</h3>
              </div>
            </div>

            <div className="grid gap-4">
              {activeResult.execution_plan.symbol_decisions.length ? (
                activeResult.execution_plan.symbol_decisions.map((item) => (
                  <div key={item.symbol} className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
                    <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                      <div>
                        <div className="text-lg font-semibold text-[var(--ink)]">{item.symbol}</div>
                        <div className="mt-1 text-sm text-[var(--muted)]">
                          {item.action} · 趋势状态 {item.trend_regime || '未标注'}
                        </div>
                        <p className="mt-3 text-sm leading-7 text-[var(--muted)]">{item.rationale}</p>
                      </div>
                      <div className="grid gap-3 md:grid-cols-3">
                        <ResultMetric label="建议仓位" value={formatPercent(item.suggested_weight)} />
                        <ResultMetric label="买入价" value={formatPrice(item.recommended_entry_price)} />
                        <ResultMetric label="目标 / 止损" value={`${formatPrice(item.target_price)} / ${formatPrice(item.stop_loss_price)}`} />
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty-card">暂无执行建议。</div>
              )}
            </div>
          </section>

          {/* Risk + Evidence */}
          <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
            <section className="paper-card">
              <div className="section-header">
                <div>
                  <p className="panel-kicker">风控</p>
                  <h3>风险指标与约束</h3>
                </div>
              </div>

              <div className="metric-grid">
                <ResultMetric label="波动率" value={formatPercent(activeResult.risk.volatility)} />
                <ResultMetric label="最大回撤" value={formatPercent(activeResult.risk.max_drawdown)} />
                <ResultMetric label="夏普比率" value={activeResult.risk.sharpe_ratio.toFixed(2)} />
                <ResultMetric label="单票上限" value={formatPercent(activeResult.risk.max_single_position)} />
              </div>

              <div className="mt-4 space-y-3">
                <div className="rounded-[18px] bg-[rgba(12,33,60,0.04)] px-4 py-3 text-sm text-[var(--muted)]">
                  回撤红线 {formatPercent(activeResult.risk.max_drawdown_limit)} · 默认止损 {formatPercent(activeResult.risk.default_stop_loss)}
                </div>
                {activeResult.risk.warnings.length > 0 && (
                  <div className="rounded-[18px] bg-[rgba(190,92,44,0.08)] px-4 py-3 text-sm text-[var(--danger)]">
                    {activeResult.risk.warnings.join('；')}
                  </div>
                )}
              </div>
            </section>

            <section className="paper-card">
              <div className="section-header">
                <div>
                  <p className="panel-kicker">证据与日志</p>
                  <h3>报告、日志与模型分配</h3>
                </div>
              </div>

              <div className="space-y-4">
                <details className="rounded-[18px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
                  <summary className="cursor-pointer text-sm font-semibold text-[var(--ink)]">查看 Markdown 报告</summary>
                  <div className="mt-4 max-h-96 overflow-auto rounded-[18px] border border-[var(--line)] bg-white p-5 prose prose-sm prose-slate max-w-none">
                    <Suspense fallback={<div className="empty-inline">报告加载中...</div>}>
                      <ReportMarkdown content={activeResult.report_markdown} />
                    </Suspense>
                  </div>
                </details>

                <details className="rounded-[18px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
                  <summary className="cursor-pointer text-sm font-semibold text-[var(--ink)]">查看执行日志</summary>
                  <div className="mt-4 space-y-2">
                    {activeResult.execution_log.length ? (
                      activeResult.execution_log.map((item) => (
                        <div key={item} className="rounded-[14px] bg-[rgba(12,33,60,0.04)] px-3 py-2 text-xs text-[var(--muted)]">
                          {item}
                        </div>
                      ))
                    ) : (
                      <div className="empty-inline">暂无日志。</div>
                    )}
                  </div>
                </details>

                {activeResult.llm_assignments.length > 0 && (
                  <div className="rounded-[18px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
                    <div className="text-sm font-semibold text-[var(--ink)]">LLM 模型分配</div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {activeResult.llm_assignments.map((item, index) => (
                        <span key={`assignment-${index}`} className="candidate-chip">
                          {String(item.model)} · {String(item.role)}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </section>
          </div>
        </>
      )}

      {/* Branch Detail Modal */}
      {expandedBranch && expandedBranchData && (
        <BranchDetailBoundary onClose={() => setExpandedBranch(null)}>
          <Suspense fallback={<BranchDetailLoadingOverlay />}>
            <BranchDetailModal
              branch={expandedBranchData}
              onClose={() => setExpandedBranch(null)}
            />
          </Suspense>
        </BranchDetailBoundary>
      )}
    </div>
  );
}

class BranchDetailBoundary extends Component<
  { children: ReactNode; onClose: () => void },
  { hasError: boolean }
> {
  constructor(props: { children: ReactNode; onClose: () => void }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: unknown) {
    console.error('Branch detail modal failed to render', error);
  }

  componentDidUpdate(prevProps: { children: ReactNode }) {
    if (prevProps.children !== this.props.children && this.state.hasError) {
      this.setState({ hasError: false });
    }
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-6 backdrop-blur-sm">
        <div className="w-full max-w-lg rounded-[28px] border border-[var(--line)] bg-[var(--surface)] p-6 shadow-xl">
          <div className="text-lg font-semibold text-[var(--ink)]">分支详情暂时无法展示</div>
          <p className="mt-3 text-sm leading-7 text-[var(--muted)]">
            当前分析结果里的分支数据结构不完整，页面已阻止弹窗崩溃。可以关闭后继续查看其它部分。
          </p>
          <button type="button" className="primary-button mt-6 w-full justify-center" onClick={this.props.onClose}>
            关闭
          </button>
        </div>
      </div>
    );
  }
}

function BranchDetailLoadingOverlay() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-6 backdrop-blur-sm">
      <div className="rounded-[24px] border border-[var(--line)] bg-[var(--surface)] px-6 py-4 text-sm text-[var(--muted)] shadow-xl">
        正在加载分支详情...
      </div>
    </div>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <div className="space-y-2">
      <label className="filter-label">{label}</label>
      <select value={value} onChange={(event) => onChange(event.target.value)} className="app-input">
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function ToggleRow({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="flex items-center justify-between rounded-[18px] border border-[var(--line)] bg-[rgba(12,33,60,0.04)] px-3 py-2 text-sm text-[var(--ink)]">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function BranchPanel({ title, branch, onToggle, children }: { title: string; branch: AnalysisBranchConfig | undefined; onToggle: (enabled: boolean) => void; children: ReactNode }) {
  return (
    <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(12,33,60,0.04)] p-3">
      <div className="mb-3 flex items-center justify-between">
        <div className="text-sm font-semibold text-[var(--ink)]">{title}</div>
        <input type="checkbox" checked={branch?.enabled ?? false} onChange={(event) => onToggle(event.target.checked)} />
      </div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function ResultMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-block">
      <div className="metric-label">{label}</div>
      <div className="metric-value text-[1.15rem]">{value}</div>
      <div className="metric-note">&nbsp;</div>
    </div>
  );
}

const branchTitleMap: Record<string, string> = {
  kline: 'K线分析',
  quant: '传统量化分支',
  llm_debate: 'LLM 多空辩论',
  intelligence: '多维智能融合',
  macro: '宏观分支',
};

const modeLabelMap: Record<string, string> = {
  single: '单只股票',
  holdings: '我的持仓',
  watchlist: '自选池',
  market: '全市场',
};

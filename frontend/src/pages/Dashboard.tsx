import { useDeferredValue, useState, type FormEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router-dom';
import {
  ArrowRight,
  CircleAlert,
  Database,
  FlaskConical,
  TrendingUp,
} from 'lucide-react';
import { fetchRecentJobs } from '../api/analysis';
import { fetchMarketOverview, fetchStocks } from '../api/data';
import { fetchPortfolioState } from '../api/portfolio';
import type { StockInfo } from '../types/api';

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatSignedPercent(value: number) {
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
}

function formatDateTime(value: string) {
  return value.replace('T', ' ').slice(0, 16);
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [quickSymbol, setQuickSymbol] = useState('000001.SZ');
  const deferredQuickSearch = useDeferredValue(quickSymbol.trim());

  const { data: overview, isLoading } = useQuery({
    queryKey: ['market-overview'],
    queryFn: fetchMarketOverview,
  });

  const { data: jobs } = useQuery({
    queryKey: ['analysis-jobs', 8],
    queryFn: () => fetchRecentJobs(8),
  });
  const { data: portfolio } = useQuery({
    queryKey: ['portfolio-state'],
    queryFn: fetchPortfolioState,
  });
  const quickSearchQuery = useQuery({
    queryKey: ['dashboard-quick-search', deferredQuickSearch],
    queryFn: () => fetchStocks({ search: deferredQuickSearch, limit: 8 }),
    enabled: deferredQuickSearch.length > 0,
  });

  const runningJobs = jobs?.filter((job) => job.status === 'queued' || job.status === 'running') ?? [];
  const failedJobs = jobs?.filter((job) => job.status === 'failed') ?? [];

  async function handleQuickOpen(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const matched = await findMatchedStock(quickSymbol, quickSearchQuery.data?.items ?? []);
    const symbol = matched?.ts_code ?? quickSymbol.trim().toUpperCase();
    if (!symbol) return;
    navigate(`/stocks/${symbol}`);
  }

  return (
    <div className="space-y-6">
      <section className="hero-panel">
        <div>
          <p className="panel-kicker">快速入口</p>
          <h1 className="hero-title">数据覆盖 → 五维分析 → 交易决策</h1>
          <p className="hero-copy">
            当前数据量、市场状态、可执行机会与风险一览。
          </p>
        </div>
        <div className="hero-actions">
          <Link to="/stocks" className="primary-button">
            <Database size={16} />
            数据浏览
          </Link>
          <Link to="/research" className="secondary-button">
            <FlaskConical size={16} />
            新建研究任务
          </Link>
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="paper-card">
          <div className="section-header">
            <div>
              <p className="panel-kicker">市场快照</p>
              <h3>数据覆盖与状态</h3>
            </div>
            <Link to="/market" className="inline-flex items-center gap-1 text-sm text-[var(--accent-strong)]">
              详细市场状态
              <ArrowRight size={14} />
            </Link>
          </div>

          {isLoading || !overview ? (
            <div className="empty-card">加载中...</div>
          ) : (
            <div className="space-y-4">
              <div className="metric-grid">
                <MetricBlock label="股票覆盖" value={String(overview.summary.total_stocks)} note="总标的数" />
                <MetricBlock label="A股 / 美股" value={`${overview.summary.cn_count} / ${overview.summary.us_count}`} note="市场覆盖" />
                <MetricBlock
                  label="20日市场宽度"
                  value={formatPercent(overview.market_pulse.positive_ratio_20d)}
                  note={overview.market_pulse.breadth_label}
                />
                <MetricBlock
                  label="20日平均收益"
                  value={formatSignedPercent(overview.market_pulse.avg_return_20d)}
                  note={`样本 ${overview.market_pulse.sampled_stocks} 只`}
                />
              </div>

              <div className="tone-band">
                <div>
                  <div className="tone-band-label">市场状态</div>
                  <div className="tone-band-value">
                    {overview.market_pulse.risk_state === 'constructive'
                      ? '偏积极'
                      : overview.market_pulse.risk_state === 'risk_off'
                        ? '偏防守'
                        : '中性分化'}
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="paper-card">
          <div className="section-header">
            <div>
              <p className="panel-kicker">任务状态</p>
              <h3>运行中与最近失败</h3>
            </div>
            <TrendingUp size={18} className="text-[var(--accent-strong)]" />
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(12,33,60,0.04)] p-4">
              <div className="text-sm font-semibold text-[var(--ink)]">运行中任务</div>
              <div className="mt-3 space-y-2 text-sm text-[var(--ink-soft)]">
                {runningJobs.length ? (
                  runningJobs.map((job) => (
                    <div key={job.job_id} className="flex items-center justify-between rounded-2xl bg-white/80 px-3 py-2">
                      <span>{job.status === 'running' ? '运行中' : '排队中'}</span>
                      <span className="font-mono text-xs">{job.job_id.slice(-8)}</span>
                    </div>
                  ))
                ) : (
                  <div className="empty-inline">无运行中任务。</div>
                )}
              </div>
            </div>

            <div className="rounded-[24px] border border-[rgba(192,94,32,0.16)] bg-[rgba(192,94,32,0.06)] p-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-[var(--ink)]">
                <CircleAlert size={16} />
                失败任务提示
              </div>
              <div className="mt-3 space-y-2 text-sm text-[var(--ink-soft)]">
                {failedJobs.length ? (
                  failedJobs.slice(0, 3).map((job) => (
                    <div key={job.job_id} className="rounded-2xl bg-white/85 px-3 py-2">
                      <div>{formatDateTime(job.updated_at)}</div>
                      <div className="mt-1 text-xs text-[var(--danger)]">{job.error ?? '任务失败'}</div>
                    </div>
                  ))
                ) : (
                  <div className="empty-inline">最近没有失败任务。</div>
                )}
              </div>
            </div>
          </div>
        </section>

        <section className="paper-card xl:col-span-2">
          <div className="section-header">
            <div>
              <p className="panel-kicker">快捷入口</p>
              <h3>个股、持仓、自选池与档案研究</h3>
            </div>
          </div>

          <div className="space-y-4">
            <form onSubmit={handleQuickOpen} className="space-y-3 rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
              <label className="text-sm font-semibold text-[var(--ink)]" htmlFor="quick-symbol">
                搜索股票代码或公司名称
              </label>
              <div className="flex flex-col gap-3">
                <div className="flex flex-col gap-3 md:flex-row">
                  <input
                    id="quick-symbol"
                    value={quickSymbol}
                    onChange={(event) => setQuickSymbol(event.target.value)}
                    placeholder="例如 平安银行 / Apple / 000001.SZ / AAPL"
                    className="app-input flex-1"
                  />
                  <button type="submit" className="primary-button justify-center md:min-w-40">
                    打开
                  </button>
                </div>

                {quickSymbol.trim() && (
                  <div className="rounded-[20px] border border-[var(--line)] bg-white/85 p-2">
                    {quickSearchQuery.isLoading ? (
                      <div className="empty-inline">搜索中...</div>
                    ) : quickSearchQuery.data?.items.length ? (
                      <div className="space-y-2">
                        {quickSearchQuery.data.items.map((item) => (
                          <button
                            key={item.ts_code}
                            type="button"
                            className="flex w-full items-center justify-between rounded-[16px] px-3 py-2 text-left text-sm transition-colors hover:bg-[rgba(12,33,60,0.05)]"
                            onClick={() => navigate(`/stocks/${item.ts_code}`)}
                          >
                            <span className="text-[var(--ink)]">
                              {item.ts_code}
                              {item.name ? ` · ${item.name}` : ''}
                            </span>
                            <span className="text-[var(--muted)]">{item.market ?? '-'}</span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="empty-inline">没有匹配结果，支持输入股票代码或公司名称。</div>
                    )}
                  </div>
                )}
              </div>
            </form>

            <div className="grid gap-3 md:grid-cols-4">
              <QuickAction title="单只股票研究" description="选择一只股票深度分析" to="/research" />
              <QuickAction title="我的持仓研究" description="自动带入真实仓位" to="/research" />
              <QuickAction title="过往分析" description="查看历史记录" to="/history" />
              <QuickAction title="持仓与观察" description="维护自选池" to="/watchlists" />
            </div>
          </div>
        </section>

        <section className="paper-card xl:col-span-2">
          <div className="section-header">
            <div>
              <p className="panel-kicker">用户资产池</p>
              <h3>持仓与观察列表</h3>
            </div>
            <Link to="/watchlists" className="inline-flex items-center gap-1 text-sm text-[var(--accent-strong)]">
              打开模块
              <ArrowRight size={14} />
            </Link>
          </div>

          <div className="metric-grid">
            <MetricBlock label="当前持仓" value={String(portfolio?.summary.holdings_count ?? 0)} note="自选池自动引用" />
            <MetricBlock label="观察列表" value={String(portfolio?.summary.watchlist_count ?? 0)} note="备用候选" />
            <MetricBlock
              label="持仓样本"
              value={portfolio?.summary.holding_symbols.slice(0, 3).join('、') || '暂无'}
              note="分析中心一键带出"
            />
            <MetricBlock
              label="观察样本"
              value={portfolio?.summary.watchlist_symbols.slice(0, 3).join('、') || '暂无'}
              note="持续追踪"
            />
          </div>
        </section>
      </div>
    </div>
  );
}

function MetricBlock({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="metric-block">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </div>
  );
}

function resolveMatchedStock(input: string, items: StockInfo[]) {
  const normalized = input.trim().toUpperCase();
  if (!normalized) {
    return null;
  }
  return items.find((item) => item.ts_code.toUpperCase() === normalized) ?? items[0] ?? null;
}

async function findMatchedStock(input: string, items: StockInfo[]) {
  const localMatched = resolveMatchedStock(input, items);
  if (localMatched) {
    return localMatched;
  }
  const keyword = input.trim();
  if (!keyword) {
    return null;
  }
  try {
    const response = await fetchStocks({ search: keyword, limit: 1 });
    return response.items[0] ?? null;
  } catch {
    return null;
  }
}

function QuickAction({ title, description, to }: { title: string; description: string; to: string }) {
  return (
    <Link to={to} className="action-card">
      <div className="text-base font-semibold text-[var(--ink)]">{title}</div>
      <div className="mt-2 text-sm leading-6 text-[var(--muted)]">{description}</div>
      <div className="mt-4 inline-flex items-center gap-1 text-sm font-semibold text-[var(--accent-strong)]">
        进入
        <ArrowRight size={14} />
      </div>
    </Link>
  );
}

import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { fetchMarketOverview } from '../api/data';

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatSignedPercent(value: number) {
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
}

function marketLabel(value: string) {
  return value === 'CN' ? 'A股' : value === 'US' ? '美股' : value;
}

export default function MarketStatus() {
  const { data: overview, isLoading } = useQuery({
    queryKey: ['market-overview'],
    queryFn: fetchMarketOverview,
  });

  if (isLoading || !overview) {
    return (
      <div className="space-y-6">
        <section className="hero-panel">
          <div>
            <p className="panel-kicker">市场状态</p>
            <h2 className="hero-title text-[2rem]">数据覆盖与市场脉搏</h2>
          </div>
        </section>
        <div className="paper-card">
          <div className="empty-card">加载中...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <section className="hero-panel">
        <div>
          <p className="panel-kicker">市场状态</p>
          <h2 className="hero-title text-[2rem]">数据覆盖与市场脉搏</h2>
          <p className="hero-copy">
            当前数据量、市场宽度、风险状态与行业分布一览。
          </p>
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="paper-card">
          <div className="section-header">
            <div>
              <p className="panel-kicker">数据覆盖</p>
              <h3>股票与记录统计</h3>
            </div>
          </div>
          <div className="metric-grid">
            <MetricBlock label="股票覆盖" value={String(overview.summary.total_stocks)} note="总标的数" />
            <MetricBlock label="总记录数" value={overview.summary.total_records.toLocaleString()} note="日线数据" />
            <MetricBlock label="A股" value={String(overview.summary.cn_count)} note="CN" />
            <MetricBlock label="美股" value={String(overview.summary.us_count)} note="US" />
          </div>
          <div className="mt-4 rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
            <div className="text-sm font-semibold text-[var(--ink)]">最近数据更新</div>
            <div className="mt-2 text-sm text-[var(--muted)]">{overview.summary.last_data_update ?? '未知'}</div>
          </div>
        </section>

        <section className="paper-card">
          <div className="section-header">
            <div>
              <p className="panel-kicker">市场脉搏</p>
              <h3>宽度与风险状态</h3>
            </div>
          </div>
          <div className="metric-grid">
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
            <MetricBlock
              label="20日波动率"
              value={formatPercent(overview.market_pulse.avg_volatility_20d)}
              note="平均波动"
            />
            <MetricBlock
              label="上涨占比"
              value={`${overview.market_pulse.rising_count_20d}`}
              note={`共 ${overview.market_pulse.sampled_stocks} 只`}
            />
          </div>
          <div className="mt-4 tone-band">
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
            <div className="tone-band-detail">
              宽度标签：{overview.market_pulse.breadth_label}
              {overview.market_pulse.last_trade_date && ` · 最近交易日 ${overview.market_pulse.last_trade_date}`}
            </div>
          </div>
        </section>

        <section className="paper-card">
          <div className="section-header">
            <div>
              <p className="panel-kicker">数据完整度</p>
              <h3>各维度就绪统计</h3>
            </div>
          </div>
          <div className="metric-grid">
            <MetricBlock label="技术面已齐" value={String(overview.completeness.technical_ready)} note="有足够日线" />
            <MetricBlock label="基本面已齐" value={String(overview.completeness.fundamentals_ready)} note="有标准化快照" />
            <MetricBlock label="公司档案已齐" value={String(overview.completeness.profile_ready)} note="有业务/画像" />
            <MetricBlock label="竞对关系已齐" value={String(overview.completeness.competitors_ready)} note="有可比公司池" />
            <MetricBlock label="行业标注已齐" value={String(overview.completeness.industry_ready)} note="有行业分类" />
            <MetricBlock label="业务标注已齐" value={String(overview.completeness.business_ready)} note="有产品/业务" />
          </div>
        </section>

        <section className="paper-card">
          <div className="section-header">
            <div>
              <p className="panel-kicker">候选池</p>
              <h3>可跟进标的</h3>
            </div>
            <Link to="/research" className="inline-flex items-center gap-1 text-sm text-[var(--accent-strong)]">
              继续研究
              <ArrowRight size={14} />
            </Link>
          </div>
          <div className="flex flex-wrap gap-2">
            {overview.candidate_symbols.length ? (
              overview.candidate_symbols.map((symbol) => (
                <Link key={symbol} to={`/stocks/${symbol}`} className="candidate-chip">
                  {symbol}
                </Link>
              ))
            ) : (
              <div className="empty-inline">暂无候选。</div>
            )}
          </div>
          {overview.watch_candidates.length > 0 && (
            <div className="mt-4 space-y-3">
              {overview.watch_candidates.map((item) => (
                <Link key={`${item.symbol}-${item.created_at}`} to={`/stocks/${item.symbol}`} className="list-card">
                  <div className="list-card-main">
                    <div className="list-card-title">{item.symbol}</div>
                    <div className="list-card-subtitle">{item.title}</div>
                  </div>
                  <div className="list-card-meta">
                    <div>{item.created_at.replace('T', ' ').slice(0, 16)}</div>
                    <div>{item.summary}</div>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </section>
      </div>

      <section className="paper-card">
        <div className="section-header">
          <div>
            <p className="panel-kicker">行业分布</p>
            <h3>全量行业覆盖</h3>
          </div>
          <div className="text-sm text-[var(--muted)]">{overview.sector_distribution.length} 个行业</div>
        </div>
        <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
          {overview.sector_distribution.map((item) => (
            <div key={`${item.market}-${item.industry}`} className="flex items-center justify-between gap-3 rounded-[18px] bg-[rgba(12,33,60,0.04)] px-4 py-3 text-sm">
              <div className="flex items-center gap-2">
                <span className="data-chip">{marketLabel(item.market)}</span>
                <span className="text-[var(--ink)]">{item.industry}</span>
              </div>
              <span className="font-mono text-[var(--ink-soft)]">{item.count}</span>
            </div>
          ))}
        </div>
      </section>
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

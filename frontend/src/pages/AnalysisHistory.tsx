import { useDeferredValue, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { ChevronLeft, ChevronRight, Search } from 'lucide-react';
import { clearAnalysisHistory, deleteAnalysis, fetchAnalysisHistoryPaged } from '../api/analysis';

const PAGE_SIZE = 20;

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDateTime(value: string) {
  return value.replace('T', ' ').slice(0, 16);
}

export default function AnalysisHistory() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState('');
  const [market, setMarket] = useState('');
  const [page, setPage] = useState(0);
  const deferredSearch = useDeferredValue(search);

  const { data, isLoading } = useQuery({
    queryKey: ['analysis-history-paged', deferredSearch, market, page],
    queryFn: () =>
      fetchAnalysisHistoryPaged({
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
        search: deferredSearch || undefined,
        market: market || undefined,
      }),
  });

  const deleteMutation = useMutation({
    mutationFn: deleteAnalysis,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['analysis-history-paged'] });
      void queryClient.invalidateQueries({ queryKey: ['analysis-jobs'] });
    },
  });

  const clearMutation = useMutation({
    mutationFn: clearAnalysisHistory,
    onSuccess: () => {
      setPage(0);
      void queryClient.invalidateQueries({ queryKey: ['analysis-history-paged'] });
      void queryClient.invalidateQueries({ queryKey: ['analysis-jobs'] });
      void queryClient.invalidateQueries({ queryKey: ['portfolio-state'] });
    },
  });

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0;

  return (
    <div className="space-y-6">
      <section className="hero-panel">
        <div>
          <p className="panel-kicker">过往分析</p>
          <h2 className="hero-title text-[2rem]">研究记录与回溯</h2>
          <p className="hero-copy">
            查看所有历史分析结果，点击进入详情。
          </p>
        </div>
      </section>

      <section className="paper-card">
        <div className="section-header">
          <div>
            <p className="panel-kicker">筛选</p>
            <h3>搜索与过滤</h3>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-sm text-[var(--muted)]">共 {data?.total ?? 0} 条</div>
            <button
              type="button"
              className="secondary-button"
              onClick={() => {
                if (window.confirm('将删除当前所有 Web 分析历史和任务缓存，是否继续？')) {
                  clearMutation.mutate();
                }
              }}
              disabled={clearMutation.isPending || (data?.total ?? 0) === 0}
            >
              {clearMutation.isPending ? '清空中...' : '清空历史'}
            </button>
          </div>
        </div>

        <div className="flex flex-col gap-4 md:flex-row">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" size={16} />
            <input
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(0); }}
              placeholder="搜索标的、标题..."
              className="app-input pl-10"
            />
          </div>
          <select
            value={market}
            onChange={(e) => { setMarket(e.target.value); setPage(0); }}
            className="app-input md:w-40"
          >
            <option value="">全部市场</option>
            <option value="CN">A股</option>
            <option value="US">美股</option>
          </select>
        </div>
      </section>

      <section className="paper-card">
        <div className="space-y-3">
          {isLoading ? (
            <div className="empty-card">加载中...</div>
          ) : data?.items.length ? (
            data.items.map((item, index) => (
              <div
                key={`${item.analysis_id}-${index}`}
                className="list-card"
              >
                <div className="list-card-main">
                  <Link to={`/research?id=${item.analysis_id}`} className="list-card-title hover:underline">
                    {item.title}
                  </Link>
                  <div className="list-card-subtitle">
                    {item.candidate_symbols.length
                      ? `候选：${item.candidate_symbols.slice(0, 5).join('、')}`
                      : '暂无明确候选标的'}
                  </div>
                </div>
                <div className="list-card-meta">
                  <div>{formatDateTime(item.created_at)}</div>
                  <div>
                    仓位 {formatPercent(item.target_exposure)} · 风险 {item.risk_level} · {item.stock_count} 只
                  </div>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => {
                      if (window.confirm(`删除分析记录 ${item.analysis_id}？`)) {
                        deleteMutation.mutate(item.analysis_id);
                      }
                    }}
                    disabled={deleteMutation.isPending}
                  >
                    删除
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="empty-card">无匹配结果。</div>
          )}
        </div>

        {totalPages > 1 && (
          <div className="table-footer">
            <span>
              第 {page + 1} / {totalPages} 页
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="icon-button"
                disabled={page === 0}
                onClick={() => setPage((v) => Math.max(0, v - 1))}
              >
                <ChevronLeft size={16} />
              </button>
              <button
                type="button"
                className="icon-button"
                disabled={page >= totalPages - 1}
                onClick={() => setPage((v) => Math.min(totalPages - 1, v + 1))}
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

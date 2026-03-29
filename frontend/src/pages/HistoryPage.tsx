import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { History, Trash2 } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'
import { getResearchHistory, deleteResearchRun } from '../api/research'
import { Badge } from '../components/Badge'
import { EmptyState } from '../components/EmptyState'
import { formatDateTime } from '../lib/format'

const PAGE_SIZE = 20

function parsePage(value: string | null) {
  const page = Number(value)
  if (!Number.isInteger(page) || page < 1) {
    return 1
  }
  return page
}

function summarizeStockPool(symbols: string[]) {
  const preview = symbols.slice(0, 6).join(', ')
  if (symbols.length <= 6) {
    return preview
  }
  return `${preview} +${symbols.length - 6} more`
}

export function HistoryPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const page = parsePage(searchParams.get('page'))
  const marketFilter = ['CN', 'US'].includes(searchParams.get('market') ?? '')
    ? searchParams.get('market') ?? undefined
    : undefined
  const queryClient = useQueryClient()

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['history', page, marketFilter],
    queryFn: () => getResearchHistory(page, PAGE_SIZE, marketFilter),
    staleTime: 30_000,
  })

  const deleteMutation = useMutation({
    mutationFn: (jobId: string) => deleteResearchRun(jobId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['history'] }),
  })

  const items = data?.items ?? []
  const total = data?.total ?? 0
  const totalPages = Math.ceil(total / PAGE_SIZE)

  function setQueryState(nextPage: number, nextMarket?: string) {
    const nextParams = new URLSearchParams(searchParams)
    if (nextPage > 1) {
      nextParams.set('page', String(nextPage))
    } else {
      nextParams.delete('page')
    }

    if (nextMarket) {
      nextParams.set('market', nextMarket)
    } else {
      nextParams.delete('market')
    }

    setSearchParams(nextParams)
  }

  function handleDelete(jobId: string) {
    if (!window.confirm(`Delete run ${jobId}? This also removes its saved trade suggestions.`)) {
      return
    }
    deleteMutation.mutate(jobId)
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="border-b border-white/10 px-4 py-4 lg:px-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.24em] text-teal-300/70">
              <History size={14} aria-hidden="true" />
              Run History
            </div>
            <h1 className="mt-2 text-2xl font-semibold text-white">Review Prior Research Runs</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Filter completed and failed runs, deep-link the current view, and reopen any stored report from the mainline workspace.
            </p>
          </div>
          <div className="rounded-3xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-slate-300">
            <p>
              <span className="font-semibold text-white">{total}</span>
              {' '}
              recorded runs
            </p>
            <p className="mt-1 text-xs uppercase tracking-[0.18em] text-slate-500">
              {marketFilter ? `${marketFilter} market only` : 'All markets'}
            </p>
          </div>
        </div>
      </header>

      <div className="border-b border-white/8 px-4 py-3 lg:px-6">
        <div className="flex flex-wrap gap-2">
          {[undefined, 'CN', 'US'].map((m) => (
            <button
              key={m ?? 'all'}
              type="button"
              onClick={() => setQueryState(1, m)}
              className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
                marketFilter === m
                  ? 'border-teal-400/30 bg-teal-400/10 text-teal-100'
                  : 'border-white/10 bg-white/[0.02] text-slate-400 hover:border-white/16 hover:text-slate-200'
              }`}
            >
              {m ?? 'All'}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 lg:px-6">
        {isLoading ? (
          <div className="rounded-3xl border border-white/8 bg-white/[0.03] p-6 text-sm text-slate-400">
            Loading…
          </div>
        ) : isError ? (
          <div className="rounded-3xl border border-red-500/20 bg-red-500/10 p-6 text-sm text-red-200">
            {error instanceof Error ? error.message : 'Failed to load run history.'}
          </div>
        ) : items.length === 0 ? (
          <EmptyState title="No runs yet" description="Run your first research from the workspace." />
        ) : (
          <div className="overflow-hidden rounded-[1.75rem] border border-white/8 bg-slate-950/50 shadow-2xl shadow-black/20">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[54rem] text-sm">
                <thead>
                  <tr className="border-b border-white/8 text-slate-400">
                    <th className="px-4 py-3 text-left font-medium">Job ID</th>
                    <th className="px-4 py-3 text-left font-medium">Status</th>
                    <th className="px-4 py-3 text-left font-medium">Market</th>
                    <th className="px-4 py-3 text-left font-medium">Stocks</th>
                    <th className="px-4 py-3 text-left font-medium">Runtime</th>
                    <th className="px-4 py-3 text-left font-medium">Created</th>
                    <th className="px-4 py-3 text-right font-medium">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => (
                    <tr key={item.job_id} className="border-b border-white/6 align-top last:border-b-0 hover:bg-white/[0.03]">
                      <td className="px-4 py-3">
                        <Link
                          to={`/history/${item.job_id}`}
                          className="font-medium text-sky-300 hover:text-sky-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
                        >
                          {item.job_id}
                        </Link>
                      </td>
                      <td className="px-4 py-3"><Badge status={item.status} /></td>
                      <td className="px-4 py-3 text-slate-300">{item.market}</td>
                      <td className="max-w-[20rem] px-4 py-3 text-slate-400">
                        <p className="truncate">{summarizeStockPool(item.stock_pool)}</p>
                      </td>
                      <td className="px-4 py-3 text-slate-400">
                        {item.total_time != null ? `${item.total_time.toFixed(1)}s` : '—'}
                      </td>
                      <td className="px-4 py-3 text-slate-500">
                        {formatDateTime(item.created_at)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          type="button"
                          onClick={() => handleDelete(item.job_id)}
                          aria-label={`Delete run ${item.job_id}`}
                          className="inline-flex items-center gap-2 rounded-full border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-red-400/30 hover:bg-red-500/10 hover:text-red-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
                        >
                          <Trash2 size={14} aria-hidden="true" />
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex shrink-0 items-center justify-center gap-2 border-t border-white/8 px-4 py-3">
          <button
            type="button"
            onClick={() => setQueryState(Math.max(1, page - 1), marketFilter)}
            disabled={page === 1}
            className="rounded-full border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-white/16 hover:text-white disabled:opacity-30"
          >
            Prev
          </button>
          <span className="text-xs text-slate-500">{page} / {totalPages}</span>
          <button
            type="button"
            onClick={() => setQueryState(Math.min(totalPages, page + 1), marketFilter)}
            disabled={page === totalPages}
            className="rounded-full border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-white/16 hover:text-white disabled:opacity-30"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}

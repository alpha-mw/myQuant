import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getResearchHistory, deleteResearchRun } from '../api/research'
import { Badge } from '../components/Badge'
import { EmptyState } from '../components/EmptyState'

export function HistoryPage() {
  const [page, setPage] = useState(1)
  const [marketFilter, setMarketFilter] = useState<string | undefined>()
  const queryClient = useQueryClient()

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['history', page, marketFilter],
    queryFn: () => getResearchHistory(page, 20, marketFilter),
    staleTime: 30_000,
  })

  const deleteMutation = useMutation({
    mutationFn: (jobId: string) => deleteResearchRun(jobId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['history'] }),
  })

  const items = data?.items ?? []
  const total = data?.total ?? 0
  const totalPages = Math.ceil(total / 20)

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 shrink-0">
        <h1 className="text-sm font-medium text-gray-200">Run History</h1>
        <div className="flex gap-1">
          {[undefined, 'CN', 'US'].map((m) => (
            <button
              key={m ?? 'all'}
              onClick={() => { setMarketFilter(m); setPage(1) }}
              className={`px-2 py-1 text-xs rounded border transition-colors ${
                marketFilter === m
                  ? 'border-emerald-700 text-emerald-400 bg-emerald-900/30'
                  : 'border-gray-700 text-gray-500 hover:text-gray-300'
              }`}
            >
              {m ?? 'All'}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-4 text-xs text-gray-500">Loading...</div>
        ) : isError ? (
          <div className="p-4 text-xs text-red-400">
            {error instanceof Error ? error.message : 'Failed to load run history.'}
          </div>
        ) : items.length === 0 ? (
          <EmptyState title="No runs yet" description="Run your first research from the workspace." />
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 border-b border-gray-800">
                <th className="text-left px-4 py-2 font-medium">Job ID</th>
                <th className="text-left px-4 py-2 font-medium">Status</th>
                <th className="text-left px-4 py-2 font-medium">Market</th>
                <th className="text-left px-4 py-2 font-medium">Stocks</th>
                <th className="text-left px-4 py-2 font-medium">Time</th>
                <th className="text-left px-4 py-2 font-medium">Date</th>
                <th className="text-right px-4 py-2 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.job_id} className="border-b border-gray-900 hover:bg-gray-900/50">
                  <td className="px-4 py-2">
                    <Link to={`/history/${item.job_id}`} className="text-blue-400 hover:text-blue-300">
                      {item.job_id}
                    </Link>
                  </td>
                  <td className="px-4 py-2"><Badge status={item.status} /></td>
                  <td className="px-4 py-2 text-gray-400">{item.market}</td>
                  <td className="px-4 py-2 text-gray-400 max-w-[200px] truncate">
                    {item.stock_pool.join(', ')}
                  </td>
                  <td className="px-4 py-2 text-gray-500">
                    {item.total_time != null ? `${item.total_time.toFixed(1)}s` : '-'}
                  </td>
                  <td className="px-4 py-2 text-gray-600">
                    {new Date(item.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <button
                      onClick={() => deleteMutation.mutate(item.job_id)}
                      className="text-gray-600 hover:text-red-400"
                    >
                      &times;
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 px-4 py-2 border-t border-gray-800 shrink-0">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-2 py-1 text-xs text-gray-500 hover:text-gray-300 disabled:opacity-30"
          >
            Prev
          </button>
          <span className="text-xs text-gray-600">{page} / {totalPages}</span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="px-2 py-1 text-xs text-gray-500 hover:text-gray-300 disabled:opacity-30"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}

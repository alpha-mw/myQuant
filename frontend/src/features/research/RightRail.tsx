import { useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getStartupContext } from '../../api/research'
import type { PhaseInfo } from '../../hooks/useSSE'

interface Props {
  logs: string[]
  progress: number
  phase: PhaseInfo
  isRunning: boolean
  isCompleted: boolean
  isFailed: boolean
  selectionSummary?: { count: number; keys: string[]; market: string } | null
}

export function RightRail({
  logs,
  progress,
  phase,
  isRunning,
  isCompleted,
  isFailed,
  selectionSummary,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isRunning) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs.length, isRunning]) // eslint-disable-line react-hooks/exhaustive-deps

  const { data: startupCtx } = useQuery({
    queryKey: ['startup-context'],
    queryFn: getStartupContext,
    staleTime: 60_000,
    // Only load on mount when idle, not during a run
    enabled: !isRunning,
  })

  const pct = Math.round(progress * 100)
  const showProgress = isRunning || isCompleted || isFailed

  return (
    <div className="h-full flex flex-col overflow-hidden text-xs">
      {/* ── Progress + phase ──────────────────────────── */}
      {showProgress && (
        <div className="shrink-0 border-b border-gray-800 p-3 space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-gray-400 font-medium">
              {isFailed ? '运行失败' : isCompleted ? '已完成' : phase.phase_label || '运行中...'}
            </span>
            <span className={`font-mono ${isFailed ? 'text-red-400' : 'text-emerald-400'}`}>
              {pct}%
            </span>
          </div>
          <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 rounded-full ${
                isFailed ? 'bg-red-600' : 'bg-emerald-600'
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
          {phase.phase_key && !isCompleted && !isFailed && (
            <p className="text-[10px] text-gray-600">{phase.phase_key}</p>
          )}
        </div>
      )}

      {/* ── Live log stream ───────────────────────────── */}
      {(isRunning || isCompleted || isFailed) && logs.length > 0 && (
        <div className="shrink-0 border-b border-gray-800">
          <div className="px-3 py-1.5 flex items-center justify-between">
            <span className="text-gray-500 uppercase tracking-wider text-[10px]">日志</span>
            <span className="text-gray-600 text-[10px]">{logs.length} 行</span>
          </div>
          <div className="overflow-y-auto max-h-48 px-3 pb-2 font-mono leading-relaxed">
            {logs.slice(-80).map((line, i) => (
              <div key={i} className="text-gray-500 hover:text-gray-300 truncate">
                {line}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>
      )}

      {/* ── Selection summary ─────────────────────────── */}
      {selectionSummary && (
        <div className="shrink-0 border-b border-gray-800 p-3 space-y-1">
          <p className="text-gray-500 uppercase tracking-wider text-[10px]">当前股票池</p>
          <p className="text-gray-200">
            <span className="text-emerald-400 font-medium">{selectionSummary.count}</span> 只
            {selectionSummary.keys.length > 0 && (
              <span className="text-gray-600 ml-1">· {selectionSummary.keys.join(' + ')}</span>
            )}
          </p>
          <p className="text-gray-600">{selectionSummary.market}</p>
        </div>
      )}

      {/* ── Startup recall summary ────────────────────── */}
      {!isRunning && startupCtx && startupCtx.recent_runs.length > 0 && (
        <div className="flex-1 overflow-y-auto p-3 space-y-3">
          <p className="text-gray-500 uppercase tracking-wider text-[10px]">历史运行</p>

          {/* recall summary stats */}
          {startupCtx.recall_summary && (
            <div className="grid grid-cols-2 gap-1.5">
              <div className="bg-gray-900 rounded p-2">
                <p className="text-gray-600 text-[10px]">运行次数</p>
                <p className="text-gray-200 font-medium">
                  {startupCtx.recall_summary.run_count as number ?? 0}
                </p>
              </div>
              <div className="bg-gray-900 rounded p-2">
                <p className="text-gray-600 text-[10px]">待确认交易</p>
                <p className="text-gray-200 font-medium">
                  {startupCtx.recall_summary.pending_trades as number ?? 0}
                </p>
              </div>
            </div>
          )}

          {/* recent run list */}
          <div className="space-y-1.5">
            {startupCtx.recent_runs.map((run) => (
              <div
                key={run.job_id}
                className="bg-gray-900 rounded p-2 space-y-0.5"
              >
                <div className="flex items-center justify-between">
                  <span className={`text-[10px] font-medium ${
                    run.status === 'completed' ? 'text-emerald-400' : 'text-red-400'
                  }`}>
                    {run.status === 'completed' ? '✓' : '✗'} {run.market}
                  </span>
                  <span className="text-gray-600 text-[10px]">
                    {new Date(run.created_at).toLocaleDateString('zh-CN')}
                  </span>
                </div>
                <p className="text-gray-400 text-[10px]">
                  {run.stock_pool.slice(0, 5).join(', ')}
                  {run.stock_pool.length > 5 && ` +${run.stock_pool.length - 5}`}
                </p>
                {run.recall_context?.conviction && (
                  <p className="text-gray-600 text-[10px]">
                    观点: {String(run.recall_context.conviction)}
                  </p>
                )}
              </div>
            ))}
          </div>

          {/* pending suggested trades */}
          {startupCtx.suggested_trades.length > 0 && (
            <>
              <p className="text-gray-500 uppercase tracking-wider text-[10px] pt-1">建议交易</p>
              <div className="space-y-1">
                {startupCtx.suggested_trades.slice(0, 8).map((t, i) => (
                  <div key={i} className="flex items-center gap-2 text-[10px]">
                    <span className={`px-1 py-0.5 rounded text-[9px] font-medium ${
                      t.direction === 'buy'
                        ? 'bg-emerald-900/40 text-emerald-400'
                        : 'bg-red-900/40 text-red-400'
                    }`}>
                      {t.direction === 'buy' ? '买' : '卖'}
                    </span>
                    <span className="text-gray-300 font-mono">{String(t.symbol)}</span>
                    <span className="text-gray-600 truncate flex-1">{String(t.rationale || '').slice(0, 40)}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* Empty state when idle with no history */}
      {!isRunning && !isCompleted && !isFailed && (!startupCtx || startupCtx.recent_runs.length === 0) && (
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-gray-700 text-[10px] text-center leading-relaxed">
            运行后此处显示<br />进度、日志和历史记录
          </p>
        </div>
      )}
    </div>
  )
}

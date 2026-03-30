import { useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Bot, Clock3, History, ShieldCheck } from 'lucide-react'
import { getStartupContext } from '../../api/research'
import type { PhaseInfo } from '../../hooks/useSSE'
import { formatDate, formatInteger } from '../../lib/format'

interface Props {
  logs: string[]
  progress: number
  phase: PhaseInfo
  isRunning: boolean
  isCompleted: boolean
  isFailed: boolean
  selectionSummary?: { count: number; keys: string[]; market: string } | null
  activeBranchCount: number
  capitalLabel: string
  riskLevel: string
  agentLayerEnabled: boolean
}

function getRecallConvictionText(value: Record<string, unknown> | null | undefined) {
  const conviction = value?.conviction
  if (typeof conviction === 'string' || typeof conviction === 'number') {
    return String(conviction)
  }
  return null
}

export function RightRail({
  logs,
  progress,
  phase,
  isRunning,
  isCompleted,
  isFailed,
  selectionSummary,
  activeBranchCount,
  capitalLabel,
  riskLevel,
  agentLayerEnabled,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isRunning) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs.length, isRunning])

  const { data: startupCtx } = useQuery({
    queryKey: ['startup-context'],
    queryFn: getStartupContext,
    staleTime: 60_000,
    enabled: !isRunning,
  })

  const pct = Math.round(progress * 100)
  const toneClassName = isFailed
    ? 'border-red-300/18 bg-red-300/10 text-red-100'
    : isCompleted
      ? 'border-emerald-300/18 bg-emerald-300/10 text-emerald-100'
      : isRunning
        ? 'border-cyan-300/18 bg-cyan-300/10 text-cyan-100'
        : 'border-white/10 bg-white/[0.04] text-slate-200'
  const statusLabel = isFailed ? 'Run failed' : isCompleted ? 'Run complete' : isRunning ? 'Run active' : 'Idle'
  const recentRuns = startupCtx?.recent_runs ?? []
  const suggestedTrades = startupCtx?.suggested_trades ?? []

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-white/8 px-4 py-3">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.22em] text-cyan-200/65">
          <History size={14} aria-hidden="true" />
          Workspace context
        </div>
        <p className="mt-2 text-sm leading-6 text-slate-400">
          Track the current run, recent history, and the structural constraints that stay in force.
        </p>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <div className="space-y-4">
          <section className={`rounded-[1.5rem] border px-4 py-4 ${toneClassName}`}>
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className={`text-[11px] uppercase tracking-[0.18em] ${isFailed ? 'text-red-50/75' : isCompleted ? 'text-emerald-50/75' : isRunning ? 'text-cyan-50/75' : 'text-slate-400'}`}>
                  Run pulse
                </p>
                <p className="mt-2 text-base font-semibold text-white">{statusLabel}</p>
              </div>
              <div className="text-right">
                <p className={`text-[11px] uppercase tracking-[0.18em] ${isFailed ? 'text-red-50/75' : isCompleted ? 'text-emerald-50/75' : isRunning ? 'text-cyan-50/75' : 'text-slate-400'}`}>
                  Progress
                </p>
                <p className="mt-2 text-lg font-semibold tabular-nums text-white">{pct}%</p>
              </div>
            </div>

            <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-slate-950/85">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isFailed
                    ? 'bg-red-400'
                    : isCompleted
                      ? 'bg-emerald-400'
                      : 'bg-[linear-gradient(90deg,rgba(34,211,238,1),rgba(16,185,129,0.92))]'
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>

            <p className={`mt-3 text-sm leading-6 ${isFailed ? 'text-red-50/80' : isCompleted ? 'text-emerald-50/80' : isRunning ? 'text-cyan-50/80' : 'text-slate-300'}`}>
              {phase.phase_label || (isRunning ? 'Waiting for the next phase update.' : 'No active phase.')}
            </p>
            {phase.phase_key && (
              <p className={`mt-1 text-xs ${isFailed ? 'text-red-50/65' : isCompleted ? 'text-emerald-50/65' : isRunning ? 'text-cyan-50/65' : 'text-slate-500'}`}>
                {phase.phase_key}
              </p>
            )}
          </section>

          <section className="rounded-[1.5rem] border border-white/10 bg-white/[0.03] px-4 py-4">
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
              <Clock3 size={14} aria-hidden="true" />
              Current setup
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <MiniMetric label="Selection" value={selectionSummary ? `${formatInteger(selectionSummary.count)} names` : 'Not set'} />
              <MiniMetric label="Branches" value={`${formatInteger(activeBranchCount)} active`} />
              <MiniMetric label="Capital" value={capitalLabel} />
              <MiniMetric label="Risk" value={riskLevel} />
            </div>

            {selectionSummary?.keys.length ? (
              <p className="mt-4 text-sm leading-6 text-slate-400">Universe keys: {selectionSummary.keys.join(' + ')}</p>
            ) : (
              <p className="mt-4 text-sm leading-6 text-slate-400">Manual stock selection is currently active.</p>
            )}
          </section>

          <section className="rounded-[1.5rem] border border-white/10 bg-white/[0.03] px-4 py-4">
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
              <ShieldCheck size={14} aria-hidden="true" />
              Guardrails
            </div>
            <ul className="mt-3 space-y-2 text-sm leading-6 text-slate-300">
              <li>RiskGuard hard veto stays authoritative.</li>
              <li>PortfolioConstructor stays deterministic.</li>
              <li>{agentLayerEnabled ? 'Agent review is live for this run.' : 'Agent review is bypassed for this run.'}</li>
            </ul>
          </section>

          {!isRunning && recentRuns.length > 0 && (
            <section className="rounded-[1.5rem] border border-white/10 bg-white/[0.03] px-4 py-4">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
                <History size={14} aria-hidden="true" />
                Recent runs
              </div>

              {startupCtx?.recall_summary && (
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <MiniMetric
                    label="Run count"
                    value={formatInteger(Number(startupCtx.recall_summary.run_count ?? 0))}
                  />
                  <MiniMetric
                    label="Pending trades"
                    value={formatInteger(Number(startupCtx.recall_summary.pending_trades ?? 0))}
                  />
                </div>
              )}

              <div className="mt-4 space-y-3">
                {recentRuns.map((run) => (
                  <div key={run.job_id} className="rounded-2xl border border-white/8 bg-slate-950/72 px-3 py-3">
                    <div className="flex items-center justify-between gap-3">
                      <span
                        className={`rounded-full px-2.5 py-1 text-[11px] font-medium ${
                          run.status === 'completed'
                            ? 'border border-emerald-300/20 bg-emerald-300/12 text-emerald-100'
                            : 'border border-red-300/20 bg-red-300/12 text-red-100'
                        }`}
                      >
                        {run.status}
                      </span>
                      <span className="text-xs text-slate-500">{formatDate(run.created_at)}</span>
                    </div>
                    <p className="mt-2 text-sm font-medium tabular-nums text-white">{run.market}</p>
                    <p className="mt-1 text-sm leading-6 text-slate-400">
                      {run.stock_pool.slice(0, 4).join(', ')}
                      {run.stock_pool.length > 4 && ` +${run.stock_pool.length - 4}`}
                    </p>
                    {getRecallConvictionText(run.recall_context) && (
                      <p className="mt-2 text-xs text-slate-500">
                        Recall: {getRecallConvictionText(run.recall_context)}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {!isRunning && suggestedTrades.length > 0 && (
            <section className="rounded-[1.5rem] border border-white/10 bg-white/[0.03] px-4 py-4">
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
                <Bot size={14} aria-hidden="true" />
                Suggested trades
              </div>
              <div className="mt-4 space-y-2">
                {suggestedTrades.slice(0, 8).map((trade, index) => (
                  <div key={`${String(trade.symbol)}-${index}`} className="flex items-start gap-3 rounded-2xl border border-white/8 bg-slate-950/72 px-3 py-3">
                    {(() => {
                      const direction = String(trade.direction)

                      return (
                        <>
                          <span
                            className={`mt-0.5 rounded-full px-2.5 py-1 text-[11px] font-medium ${
                              direction === 'buy'
                                ? 'border border-emerald-300/20 bg-emerald-300/12 text-emerald-100'
                                : 'border border-red-300/20 bg-red-300/12 text-red-100'
                            }`}
                          >
                            {direction}
                          </span>
                          <div className="min-w-0 flex-1">
                            <p className="text-sm font-medium text-white">{String(trade.symbol)}</p>
                            <p className="mt-1 text-sm leading-6 text-slate-400">{String(trade.rationale || '').slice(0, 72)}</p>
                          </div>
                        </>
                      )
                    })()}
                  </div>
                ))}
              </div>
            </section>
          )}

          {!isRunning && !isCompleted && !isFailed && recentRuns.length === 0 && (
            <section className="rounded-[1.5rem] border border-dashed border-white/10 bg-white/[0.02] px-4 py-6 text-center">
              <p className="text-sm leading-6 text-slate-400">
                Run history, recall context, and suggested trades will appear here after the first completed session.
              </p>
            </section>
          )}

          {isRunning && logs.length > 0 && (
            <section className="rounded-[1.5rem] border border-white/10 bg-white/[0.03] px-4 py-4">
              <div className="flex items-center justify-between gap-3 text-[11px] uppercase tracking-[0.18em] text-slate-500">
                <span>Recent log tail</span>
                <span>{logs.length} lines</span>
              </div>
              <div className="mt-3 space-y-2 font-mono text-xs leading-5 text-slate-300">
                {logs.slice(-8).map((line, index) => (
                  <div key={`${index}-${line}`} className="truncate rounded-xl bg-slate-950/70 px-3 py-2">
                    {line}
                  </div>
                ))}
                <div ref={bottomRef} />
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-slate-950/76 px-3 py-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-sm font-medium tabular-nums text-white">{value}</p>
    </div>
  )
}

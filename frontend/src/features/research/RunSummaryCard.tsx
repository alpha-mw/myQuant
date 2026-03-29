import { Badge } from '../../components/Badge'

interface Props {
  status: string
  totalTime?: number
  llmCost?: number
  stockCount?: number
  market?: string
}

export function RunSummaryCard({ status, totalTime, llmCost, stockCount, market }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 text-xs lg:px-6">
      <Badge status={status} />

      {totalTime != null && (
        <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 tabular-nums text-slate-300">
          {totalTime.toFixed(1)}s total
        </span>
      )}

      {llmCost != null && llmCost > 0 && (
        <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 tabular-nums text-slate-300">
          ${llmCost.toFixed(4)} LLM cost
        </span>
      )}

      {stockCount != null && (
        <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 tabular-nums text-slate-300">
          {stockCount} stocks
        </span>
      )}

      {market && (
        <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-slate-300">
          {market}
        </span>
      )}
    </div>
  )
}

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
    <div className="flex items-center gap-4 px-3 py-2 bg-gray-900/50 border-t border-gray-800 text-xs shrink-0">
      <Badge status={status} />
      {totalTime != null && (
        <span className="text-gray-500">
          {totalTime.toFixed(1)}s
        </span>
      )}
      {llmCost != null && llmCost > 0 && (
        <span className="text-gray-500">
          ${llmCost.toFixed(4)}
        </span>
      )}
      {stockCount != null && (
        <span className="text-gray-500">
          {stockCount} stocks
        </span>
      )}
      {market && (
        <span className="text-gray-500">{market}</span>
      )}
    </div>
  )
}

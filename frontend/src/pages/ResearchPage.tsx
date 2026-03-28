import { useUIStore } from '../stores/uiStore'
import { useResearchStore } from '../stores/researchStore'
import { useResearchRun } from '../hooks/useResearchRun'
import { ResearchConfigPanel } from '../features/research/ResearchConfigPanel'
import { ReportViewer } from '../features/research/ReportViewer'
import { RunSummaryCard } from '../features/research/RunSummaryCard'
import { RightRail } from '../features/research/RightRail'
import { EmptyState } from '../components/EmptyState'

export function ResearchPage() {
  const collapsed = useUIStore((s) => s.configPanelCollapsed)
  const setCollapsed = useUIStore((s) => s.setConfigPanelCollapsed)
  const {
    submit,
    isSubmitting,
    status,
    logs,
    progress,
    phase,
    report,
    isRunning,
    isCompleted,
    isFailed,
    error,
    resultSummary,
  } = useResearchRun()

  const stockPool = useResearchStore((s) => s.stock_pool)
  const universeKeys = useResearchStore((s) => s.universe_keys) ?? []
  const market = useResearchStore((s) => s.market)

  const summary = resultSummary as Record<string, unknown> | null
  const llmUsage = summary?.llm_usage_summary as Record<string, number> | undefined

  const selectionSummary =
    stockPool.length > 0
      ? { count: stockPool.length, keys: universeKeys, market }
      : null

  return (
    <div className="flex flex-col h-full">
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Config panel */}
        <div
          className={`border-r border-gray-800 bg-gray-950 transition-all duration-200 shrink-0 overflow-hidden ${
            collapsed ? 'w-0' : 'w-80'
          }`}
        >
          <ResearchConfigPanel submit={submit} isSubmitting={isSubmitting} isRunning={isRunning} />
        </div>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-5 shrink-0 flex items-center justify-center border-r border-gray-800 bg-gray-900 text-gray-600 hover:text-gray-300 hover:bg-gray-800 transition-colors"
          title={collapsed ? 'Show config' : 'Hide config'}
        >
          {collapsed ? '\u25B6' : '\u25C0'}
        </button>

        {/* Center: main content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {isCompleted && report && <ReportViewer markdown={report} />}
          {isFailed && (
            <div className="flex-1 flex items-center justify-center p-8">
              <div className="text-center">
                <p className="text-red-400 text-sm font-medium">Run Failed</p>
                <p className="text-gray-500 text-xs mt-1">{error}</p>
              </div>
            </div>
          )}
          {!status && error && (
            <EmptyState title="Unable to start research" description={error} />
          )}
          {!isCompleted && !isFailed && !error && (
            <EmptyState
              title={isRunning ? '分析运行中...' : 'Ready to run research'}
              description={
                isRunning
                  ? '查看右侧进度栏了解实时日志'
                  : 'Configure parameters and click Run Research to start.'
              }
            />
          )}
        </div>

        {/* Right: persistent rail */}
        <div className="w-64 shrink-0 border-l border-gray-800 bg-gray-950 overflow-hidden">
          <div className="px-3 py-2 border-b border-gray-800">
            <h2 className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">
              进度 / 日志 / 记录
            </h2>
          </div>
          <div className="h-[calc(100%-33px)] overflow-y-auto">
            <RightRail
              logs={logs}
              progress={progress}
              phase={phase}
              isRunning={isRunning}
              isCompleted={isCompleted}
              isFailed={isFailed}
              selectionSummary={selectionSummary}
            />
          </div>
        </div>
      </div>

      {/* Status bar */}
      {status && (
        <RunSummaryCard
          status={status}
          totalTime={summary?.total_time as number | undefined}
          llmCost={llmUsage?.estimated_cost_usd}
          stockCount={(summary?.stock_pool as string[] | undefined)?.length}
          market={summary?.market as string | undefined}
        />
      )}
    </div>
  )
}

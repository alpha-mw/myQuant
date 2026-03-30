import {
  AlertTriangle,
  Bot,
  ChartNoAxesCombined,
  ChevronLeft,
  ChevronRight,
  Clock3,
  ShieldCheck,
  Workflow,
} from 'lucide-react'
import { useUIStore } from '../stores/uiStore'
import { useResearchStore } from '../stores/researchStore'
import { useResearchRun } from '../hooks/useResearchRun'
import { formatCurrency, formatInteger } from '../lib/format'
import { ResearchConfigPanel } from '../features/research/ResearchConfigPanel'
import { ReportViewer } from '../features/research/ReportViewer'
import { RunSummaryCard } from '../features/research/RunSummaryCard'
import { RightRail } from '../features/research/RightRail'
import { LiveLogPanel } from '../features/research/LiveLogPanel'
import { RunButton } from '../features/research/RunButton'

const BRANCHES = [
  { name: 'K-Line', key: 'enable_kline', description: 'Trend structure, regime shifts, and timing pressure.' },
  { name: 'Quant', key: 'enable_quant', description: 'Factor signals, alpha estimates, and crowding pressure.' },
  { name: 'Fundamental', key: 'enable_fundamental', description: 'Quality, valuation, earnings, and governance coverage.' },
  { name: 'Intelligence', key: 'enable_intelligence', description: 'Events, sentiment, flows, and information asymmetry.' },
  { name: 'Macro', key: 'enable_macro', description: 'Shared macro regime context for the full run.' },
] as const

const CONTROL_CHAIN = [
  {
    name: 'Research Agents',
    description: 'Parallel branches build structured evidence for the current stock pool.',
  },
  {
    name: 'RiskGuard',
    description: 'Hard vetoes and exposure caps remain authoritative.',
  },
  {
    name: 'ICCoordinator',
    description: 'Consensus and candidate ranking stay inside risk boundaries.',
  },
  {
    name: 'PortfolioConstructor',
    description: 'Final weights stay deterministic for identical inputs.',
  },
  {
    name: 'NarratorAgent',
    description: 'The report explains the outcome without changing the decision.',
  },
] as const

type BranchName = (typeof BRANCHES)[number]['key']

type WorkspaceState = 'idle' | 'running' | 'completed' | 'failed'

interface BranchState {
  name: string
  enabled: boolean
  description: string
}

function getWorkspaceState(isRunning: boolean, isCompleted: boolean, isFailed: boolean): WorkspaceState {
  if (isFailed) {
    return 'failed'
  }
  if (isCompleted) {
    return 'completed'
  }
  if (isRunning) {
    return 'running'
  }
  return 'idle'
}

function getWorkspaceTone(state: WorkspaceState) {
  if (state === 'running') {
    return {
      pillClassName: 'border-cyan-300/25 bg-cyan-300/12 text-cyan-100',
      label: 'Run active',
    }
  }
  if (state === 'completed') {
    return {
      pillClassName: 'border-emerald-300/25 bg-emerald-300/12 text-emerald-100',
      label: 'Report ready',
    }
  }
  if (state === 'failed') {
    return {
      pillClassName: 'border-red-300/25 bg-red-300/12 text-red-100',
      label: 'Run halted',
    }
  }
  return {
    pillClassName: 'border-white/12 bg-white/[0.05] text-slate-200',
    label: 'Idle',
  }
}

function formatMarketLabel(market: string) {
  return market === 'CN' ? 'A股 CN' : 'US'
}

function formatCapitalLabel(value: number, market: string) {
  return formatCurrency(value, market === 'CN' ? 'CNY' : 'USD')
}

export function ResearchPage() {
  const collapsed = useUIStore((state) => state.configPanelCollapsed)
  const setCollapsed = useUIStore((state) => state.setConfigPanelCollapsed)
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

  const stockPool = useResearchStore((state) => state.stock_pool)
  const universeKeys = useResearchStore((state) => state.universe_keys) ?? []
  const market = useResearchStore((state) => state.market)
  const capital = useResearchStore((state) => state.capital)
  const riskLevel = useResearchStore((state) => state.risk_level)
  const enableAgentLayer = useResearchStore((state) => state.enable_agent_layer)
  const enableKline = useResearchStore((state) => state.enable_kline)
  const enableQuant = useResearchStore((state) => state.enable_quant)
  const enableFundamental = useResearchStore((state) => state.enable_fundamental)
  const enableIntelligence = useResearchStore((state) => state.enable_intelligence)
  const enableMacro = useResearchStore((state) => state.enable_macro)

  const branches: BranchState[] = BRANCHES.map((branch) => ({
    name: branch.name,
    enabled: Boolean(
      {
        enable_kline: enableKline,
        enable_quant: enableQuant,
        enable_fundamental: enableFundamental,
        enable_intelligence: enableIntelligence,
        enable_macro: enableMacro,
      }[branch.key as BranchName],
    ),
    description: branch.description,
  }))
  const activeBranches = branches.filter((branch) => branch.enabled)
  const selectionSummary =
    stockPool.length > 0
      ? { count: stockPool.length, keys: universeKeys, market }
      : null

  const summary = resultSummary as Record<string, unknown> | null
  const llmUsage = summary?.llm_usage_summary as Record<string, number> | undefined
  const workspaceState = getWorkspaceState(isRunning, isCompleted, isFailed)
  const tone = getWorkspaceTone(workspaceState)
  const leadCopy =
    workspaceState === 'running'
      ? phase.phase_label || 'Streaming logs and phase updates from the active run.'
      : workspaceState === 'completed'
        ? 'The report bundle is ready. Review the narrative and copy it when needed.'
        : workspaceState === 'failed'
          ? 'Inspect the latest error and logs before retrying with a new configuration.'
          : 'Set the stock pool, confirm branch coverage, and launch the deterministic mainline pipeline.'

  return (
    <div className="relative flex h-full min-h-0 flex-col overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.12),_transparent_24%),radial-gradient(circle_at_82%_18%,_rgba(16,185,129,0.10),_transparent_18%)]" />

      <header className="relative shrink-0 border-b border-white/8 bg-slate-950/45 backdrop-blur-xl">
        <div className="px-4 py-4 lg:px-6 lg:py-5">
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1.25fr)_minmax(20rem,0.85fr)] xl:items-end">
            <div>
              <div className="flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.24em] text-cyan-200/70">
                <span>Research Workspace</span>
                <span className="rounded-full border border-white/12 bg-white/[0.04] px-2.5 py-1 text-[10px] text-slate-300">
                  Mainline only
                </span>
              </div>

              <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
                <div className="max-w-3xl">
                  <h1 className="text-2xl font-semibold tracking-tight text-white text-balance sm:text-[2rem]">
                    Mainline Research Console
                  </h1>
                  <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
                    Configure the stock pool, run the deterministic control chain, and inspect live logs or the final
                    report without leaving the workspace.
                  </p>
                </div>

                <div
                  className={`inline-flex items-center gap-2 rounded-full border px-3.5 py-2 text-sm font-medium ${tone.pillClassName}`}
                >
                  <span className="inline-flex h-2.5 w-2.5 rounded-full bg-current opacity-80" aria-hidden="true" />
                  {tone.label}
                </div>
              </div>

              <div className="mt-4 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setCollapsed(!collapsed)}
                  aria-label={collapsed ? 'Show configuration dock' : 'Hide configuration dock'}
                  aria-pressed={!collapsed}
                  className="inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-white/18 hover:bg-white/[0.08] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
                >
                  {collapsed ? <ChevronRight size={14} aria-hidden="true" /> : <ChevronLeft size={14} aria-hidden="true" />}
                  {collapsed ? 'Open config dock' : 'Collapse config dock'}
                </button>

                {selectionSummary ? (
                    <span className="inline-flex items-center gap-2 rounded-full border border-cyan-300/18 bg-cyan-300/10 px-3 py-1.5 text-xs text-cyan-50">
                    <span className="font-semibold tabular-nums">{formatInteger(selectionSummary.count)}</span>
                    <span>symbols</span>
                    {selectionSummary.keys.length > 0 && (
                      <span className="text-cyan-100/70">{selectionSummary.keys.join(' + ')}</span>
                    )}
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-slate-400">
                    Select a stock pool to unlock the run button
                  </span>
                )}

                <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-slate-300">
                  <Bot size={14} aria-hidden="true" />
                  {enableAgentLayer ? 'Agent review enabled' : 'Agent review disabled'}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 border-t border-white/8 pt-4 sm:grid-cols-4 xl:border-t-0 xl:border-l xl:pl-6 xl:pt-0">
              <Metric label="Selection" value={selectionSummary ? `${formatInteger(selectionSummary.count)} names` : 'Not set'} />
              <Metric label="Market" value={formatMarketLabel(market)} />
              <Metric label="Risk" value={riskLevel} />
              <Metric label="Branches" value={`${formatInteger(activeBranches.length)} active`} />
            </div>
          </div>
        </div>
      </header>

      {status && (
        <div className="relative shrink-0 border-b border-white/8 bg-slate-950/40 backdrop-blur-xl">
          <RunSummaryCard
            status={status}
            totalTime={summary?.total_time as number | undefined}
            llmCost={llmUsage?.estimated_cost_usd}
            stockCount={(summary?.stock_pool as string[] | undefined)?.length}
            market={summary?.market as string | undefined}
          />
        </div>
      )}

      <div className="relative flex min-h-0 flex-1 flex-col lg:flex-row">
        {!collapsed && (
          <aside className="w-full shrink-0 border-b border-white/8 bg-slate-950/52 backdrop-blur-xl lg:w-[23rem] lg:border-b-0 lg:border-r">
            <ResearchConfigPanel submit={submit} isSubmitting={isSubmitting} isRunning={isRunning} />
          </aside>
        )}

        <div className="min-w-0 flex-1">
          <div className="grid h-full min-h-0 grid-rows-[minmax(0,1fr)_18rem] xl:grid-cols-[minmax(0,1fr)_21rem] xl:grid-rows-1">
            <section className="min-h-0 border-b border-white/8 bg-black/10 xl:border-b-0 xl:border-r">
              <div className="flex h-full min-h-0 flex-col">
                <div className="shrink-0 border-b border-white/8 px-4 py-3 lg:px-6">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div className="max-w-3xl">
                      <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Workspace surface</p>
                      <h2 className="mt-2 text-xl font-semibold text-white">
                        {workspaceState === 'completed'
                          ? 'Research report'
                          : workspaceState === 'running'
                            ? 'Live execution'
                            : workspaceState === 'failed'
                              ? 'Run recovery'
                              : 'Preflight briefing'}
                      </h2>
                      <p className="mt-2 text-sm leading-6 text-slate-400">{leadCopy}</p>
                    </div>

                    <div className="flex flex-wrap gap-2 text-xs text-slate-400">
                      <WorkspaceStat icon={ChartNoAxesCombined} label="Capital" value={formatCapitalLabel(capital, market)} />
                      <WorkspaceStat
                        icon={Clock3}
                        label="Phase"
                        value={workspaceState === 'running' ? phase.phase_label || 'Preparing run' : tone.label}
                      />
                    </div>
                  </div>
                </div>

                <div className="min-h-0 flex-1 overflow-y-auto">
                  {isCompleted && report && <ReportViewer markdown={report} />}

                  {isFailed && (
                    <FailureState
                      error={error}
                      phaseLabel={phase.phase_label}
                      selectionLabel={selectionSummary ? `${selectionSummary.count} symbols` : 'No stock pool'}
                    />
                  )}

                  {isRunning && (
                    <LiveLogPanel
                      logs={logs}
                      progress={progress}
                      phaseLabel={phase.phase_label}
                      phaseKey={phase.phase_key}
                    />
                  )}

                  {!isRunning && !isCompleted && !isFailed && !error && (
                    <IdleSummary
                      market={market}
                      capital={capital}
                      riskLevel={riskLevel}
                      stockPool={stockPool}
                      universeKeys={universeKeys}
                      activeBranches={activeBranches}
                      branches={branches}
                      enableAgentLayer={enableAgentLayer}
                    />
                  )}

                  {!status && error && !isFailed && (
                    <FailureState error={error} phaseLabel={phase.phase_label} selectionLabel="Request could not start" />
                  )}
                </div>

                {collapsed && !isRunning && !isCompleted && (
                  <div className="shrink-0 border-t border-white/8 bg-slate-950/80 px-4 py-3 lg:px-6">
                    <RunButton submit={submit} isSubmitting={isSubmitting} isRunning={isRunning} />
                  </div>
                )}
              </div>
            </section>

            <aside className="min-h-0 bg-slate-950/46 backdrop-blur-xl">
              <RightRail
                logs={logs}
                progress={progress}
                phase={phase}
                isRunning={isRunning}
                isCompleted={isCompleted}
                isFailed={isFailed}
                selectionSummary={selectionSummary}
                activeBranchCount={activeBranches.length}
                capitalLabel={formatCapitalLabel(capital, market)}
                riskLevel={riskLevel}
                agentLayerEnabled={enableAgentLayer}
              />
            </aside>
          </div>
        </div>
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-[0.22em] text-slate-500">{label}</dt>
      <dd className="mt-2 text-sm font-medium tabular-nums text-white">{value}</dd>
    </div>
  )
}

function WorkspaceStat({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Clock3
  label: string
  value: string
}) {
  return (
    <div className="min-w-[12rem] rounded-2xl border border-white/10 bg-white/[0.04] px-3 py-2.5">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
        <Icon size={14} aria-hidden="true" />
        {label}
      </div>
      <p className="mt-2 text-sm font-medium tabular-nums text-white">{value}</p>
    </div>
  )
}

interface IdleSummaryProps {
  market: string
  capital: number
  riskLevel: string
  stockPool: string[]
  universeKeys: string[]
  activeBranches: BranchState[]
  branches: BranchState[]
  enableAgentLayer: boolean
}

function IdleSummary({
  market,
  capital,
  riskLevel,
  stockPool,
  universeKeys,
  activeBranches,
  branches,
  enableAgentLayer,
}: IdleSummaryProps) {
  const preview = stockPool.slice(0, 12)
  const remaining = Math.max(stockPool.length - preview.length, 0)
  const hasSelection = stockPool.length > 0

  return (
    <div className="px-4 py-5 lg:px-6 lg:py-6">
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.15fr)_minmax(19rem,0.85fr)]">
        <section className="rounded-[1.8rem] border border-white/10 bg-[linear-gradient(180deg,rgba(13,20,35,0.96),rgba(7,11,22,0.88))] p-5 shadow-[0_24px_80px_rgba(1,8,20,0.28)]">
          <div className="flex flex-wrap items-start justify-between gap-4 border-b border-white/8 pb-4">
            <div>
              <p className="text-[11px] uppercase tracking-[0.22em] text-cyan-200/65">Current brief</p>
              <h3 className="mt-2 text-xl font-semibold text-balance text-white">
                {hasSelection ? `${formatInteger(stockPool.length)} symbols staged for review` : 'Start with a stock pool'}
              </h3>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.04] px-3 py-2 text-right">
              <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Market</p>
              <p className="mt-1 text-sm font-medium tabular-nums text-white">{formatMarketLabel(market)}</p>
            </div>
          </div>

          <div className="mt-4 grid gap-4 sm:grid-cols-3">
            <BriefingMetric label="Capital" value={formatCapitalLabel(capital, market)} />
            <BriefingMetric label="Risk posture" value={riskLevel} />
            <BriefingMetric
              label="Agent review"
              value={enableAgentLayer ? 'Enabled' : 'Disabled'}
            />
          </div>

          <div className="mt-5 border-t border-white/8 pt-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm font-medium text-white">Selection preview</p>
              {universeKeys.length > 0 && (
                <p className="text-xs text-slate-400">Universe: {universeKeys.join(' + ')}</p>
              )}
            </div>

            {hasSelection ? (
              <div className="mt-3 flex flex-wrap gap-2">
                {preview.map((symbol) => (
                  <span
                    key={symbol}
                    className="rounded-full border border-cyan-300/18 bg-cyan-300/10 px-3 py-1.5 text-xs font-medium text-cyan-50"
                  >
                    {symbol}
                  </span>
                ))}
                {remaining > 0 && (
                  <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs text-slate-400">
                    +{formatInteger(remaining)} more
                  </span>
                )}
              </div>
            ) : (
              <p className="mt-3 max-w-xl text-sm leading-6 text-slate-400">
                Load a preset universe, browse the symbol list, or paste tickers manually in the config dock. The run
                button stays disabled until at least one symbol is staged.
              </p>
            )}
          </div>

          <div className="mt-5 border-t border-white/8 pt-4">
            <p className="text-sm font-medium text-white">Branch coverage</p>
            <div className="mt-3 space-y-2">
              {branches.map((branch) => (
                <div
                  key={branch.name}
                  className={`flex items-start justify-between gap-4 rounded-2xl border px-3 py-3 ${
                    branch.enabled
                      ? 'border-emerald-300/18 bg-emerald-300/10'
                      : 'border-white/8 bg-white/[0.03]'
                  }`}
                >
                  <div>
                    <p className={`text-sm font-medium ${branch.enabled ? 'text-white' : 'text-slate-500'}`}>
                      {branch.name}
                    </p>
                    <p className="mt-1 text-xs leading-5 text-slate-400">{branch.description}</p>
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-2.5 py-1 text-[11px] font-medium ${
                      branch.enabled
                        ? 'border border-emerald-300/20 bg-emerald-300/12 text-emerald-100'
                        : 'border border-white/10 bg-white/[0.04] text-slate-500'
                    }`}
                  >
                    {branch.enabled ? 'On' : 'Off'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="rounded-[1.8rem] border border-white/10 bg-slate-950/78 p-5">
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.22em] text-cyan-200/65">
            <Workflow size={14} aria-hidden="true" />
            Control chain
          </div>

          <div className="mt-4 space-y-4">
            {CONTROL_CHAIN.map((step, index) => (
              <div key={step.name} className="relative pl-6">
                {index < CONTROL_CHAIN.length - 1 && (
                  <span className="absolute left-[0.58rem] top-6 h-[calc(100%+0.5rem)] w-px bg-white/10" aria-hidden="true" />
                )}
                <span className="absolute left-0 top-1.5 inline-flex h-4 w-4 items-center justify-center rounded-full border border-cyan-300/25 bg-cyan-300/12 text-[10px] font-semibold text-cyan-100">
                  {index + 1}
                </span>
                <p className="text-sm font-medium text-white">{step.name}</p>
                <p className="mt-1 text-sm leading-6 text-slate-400">{step.description}</p>
              </div>
            ))}
          </div>

          <div className="mt-5 border-t border-white/8 pt-4">
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
              <ShieldCheck size={14} aria-hidden="true" />
              Governance checks
            </div>
            <ul className="mt-3 space-y-2 text-sm leading-6 text-slate-300">
              <li>RiskGuard retains hard veto authority across every run.</li>
              <li>PortfolioConstructor remains deterministic for identical inputs.</li>
              <li>NarratorAgent explains the output after weights are finalized.</li>
              <li>MacroAgent runs once per top-level research session.</li>
            </ul>
          </div>

          <div className="mt-5 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
            <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Ready state</p>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              {stockPool.length === 0
                ? 'Add symbols first. The workspace will expose the run action as soon as the pool is non-empty.'
                : activeBranches.length === 0
                  ? 'Enable at least one branch before you start the mainline pipeline.'
                  : `The workspace is ready to evaluate ${formatInteger(stockPool.length)} symbols across ${formatInteger(activeBranches.length)} active branches.`}
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}

function BriefingMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.04] px-3 py-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-2 text-sm font-medium tabular-nums text-white">{value}</p>
    </div>
  )
}

function FailureState({
  error,
  phaseLabel,
  selectionLabel,
}: {
  error: string | null
  phaseLabel: string
  selectionLabel: string
}) {
  return (
    <div className="px-4 py-6 lg:px-6">
      <div className="rounded-[1.8rem] border border-red-400/20 bg-[linear-gradient(180deg,rgba(63,16,22,0.45),rgba(22,8,14,0.82))] p-5 shadow-[0_18px_70px_rgba(40,8,14,0.28)]">
        <div className="flex flex-wrap items-start gap-4">
          <div className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-red-300/20 bg-red-300/12 text-red-100">
            <AlertTriangle size={20} aria-hidden="true" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-[11px] uppercase tracking-[0.22em] text-red-100/70">Run recovery</p>
            <h3 className="mt-2 text-xl font-semibold text-balance text-white">The last research run did not complete.</h3>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-red-50/75">
              {error ?? 'The workspace reported a failure without an explicit error message.'}
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              <span className="rounded-full border border-red-200/16 bg-red-200/10 px-3 py-1.5 text-xs text-red-50/85">
                {phaseLabel || 'Phase unavailable'}
              </span>
              <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs text-slate-300">
                {selectionLabel}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

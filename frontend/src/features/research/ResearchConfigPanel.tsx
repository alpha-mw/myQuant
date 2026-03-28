import { useResearchStore } from '../../stores/researchStore'
import { StockPoolInput } from './StockPoolInput'
import { MarketSwitcher } from './MarketSwitcher'
import { RiskLevelSelector } from './RiskLevelSelector'
import { BranchToggleGroup } from './BranchToggleGroup'
import { KlineBackendSelector } from './KlineBackendSelector'
import { LLMModelSelector } from './LLMModelSelector'
import { PresetSelector } from './PresetSelector'
import { RunButton } from './RunButton'
import type { ResearchRunRequest } from '../../types/research'

interface Props {
  submit: (req: ResearchRunRequest) => void
  isSubmitting: boolean
  isRunning: boolean
}

export function ResearchConfigPanel({ submit, isSubmitting, isRunning }: Props) {
  const { capital, lookback_years, agent_timeout, master_timeout, setField } = useResearchStore()

  return (
    <div className="h-full flex flex-col overflow-y-auto">
      <div className="px-3 py-2 border-b border-gray-800 shrink-0">
        <h2 className="text-xs font-medium text-gray-300 uppercase tracking-wider">Research Config</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        <PresetSelector />

        <div className="border-t border-gray-800 pt-3">
          <StockPoolInput />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <MarketSwitcher />
          <RiskLevelSelector />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Capital</label>
            <input
              type="number"
              value={capital}
              onChange={(e) => setField('capital', Number(e.target.value))}
              className="w-full px-2 py-1.5 text-xs bg-gray-900 border border-gray-700 rounded text-gray-200 outline-none focus:border-emerald-700"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Lookback (yr)</label>
            <input
              type="number"
              step="0.5"
              value={lookback_years}
              onChange={(e) => setField('lookback_years', Number(e.target.value))}
              className="w-full px-2 py-1.5 text-xs bg-gray-900 border border-gray-700 rounded text-gray-200 outline-none focus:border-emerald-700"
            />
          </div>
        </div>

        <BranchToggleGroup />
        <KlineBackendSelector />
        <LLMModelSelector />

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Agent Timeout (s)</label>
            <input
              type="number"
              value={agent_timeout}
              onChange={(e) => setField('agent_timeout', Number(e.target.value))}
              className="w-full px-2 py-1.5 text-xs bg-gray-900 border border-gray-700 rounded text-gray-200 outline-none focus:border-emerald-700"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Master Timeout (s)</label>
            <input
              type="number"
              value={master_timeout}
              onChange={(e) => setField('master_timeout', Number(e.target.value))}
              className="w-full px-2 py-1.5 text-xs bg-gray-900 border border-gray-700 rounded text-gray-200 outline-none focus:border-emerald-700"
            />
          </div>
        </div>
      </div>

      <div className="p-3 border-t border-gray-800 shrink-0">
        <RunButton submit={submit} isSubmitting={isSubmitting} isRunning={isRunning} />
      </div>
    </div>
  )
}

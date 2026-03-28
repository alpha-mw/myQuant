import { useResearchStore } from '../../stores/researchStore'
import type { ResearchRunRequest } from '../../types/research'

const BRANCHES: { key: keyof Pick<ResearchRunRequest, 'enable_macro' | 'enable_quant' | 'enable_kline' | 'enable_fundamental' | 'enable_intelligence'>; label: string }[] = [
  { key: 'enable_kline', label: 'K-Line' },
  { key: 'enable_quant', label: 'Quant' },
  { key: 'enable_fundamental', label: 'Fundamental' },
  { key: 'enable_intelligence', label: 'Intelligence' },
  { key: 'enable_macro', label: 'Macro' },
]

export function BranchToggleGroup() {
  const store = useResearchStore()

  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1">Research Branches</label>
      <div className="grid grid-cols-2 gap-1">
        {BRANCHES.map(({ key, label }) => {
          const enabled = store[key] as boolean
          return (
            <button
              key={key}
              onClick={() => store.setField(key, !enabled)}
              className={`px-2 py-1 text-xs rounded border transition-colors text-left ${
                enabled
                  ? 'bg-gray-800 border-gray-600 text-gray-200'
                  : 'bg-gray-900 border-gray-800 text-gray-600 line-through'
              }`}
            >
              <span className={`inline-block w-2 h-2 rounded-full mr-1.5 ${enabled ? 'bg-emerald-500' : 'bg-gray-700'}`} />
              {label}
            </button>
          )
        })}
        <button
          onClick={() => store.setField('enable_agent_layer', !store.enable_agent_layer)}
          className={`px-2 py-1 text-xs rounded border transition-colors text-left ${
            store.enable_agent_layer
              ? 'bg-gray-800 border-gray-600 text-gray-200'
              : 'bg-gray-900 border-gray-800 text-gray-600 line-through'
          }`}
        >
          <span className={`inline-block w-2 h-2 rounded-full mr-1.5 ${store.enable_agent_layer ? 'bg-blue-500' : 'bg-gray-700'}`} />
          Agent Review
        </button>
      </div>
    </div>
  )
}

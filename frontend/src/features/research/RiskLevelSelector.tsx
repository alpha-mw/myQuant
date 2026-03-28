import { useResearchStore } from '../../stores/researchStore'

const LEVELS = [
  { value: '保守', label: '保守 Conservative' },
  { value: '中等', label: '中等 Moderate' },
  { value: '积极', label: '积极 Aggressive' },
]

export function RiskLevelSelector() {
  const riskLevel = useResearchStore((s) => s.risk_level)
  const setField = useResearchStore((s) => s.setField)

  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1">Risk Level</label>
      <div className="flex gap-1">
        {LEVELS.map((l) => (
          <button
            key={l.value}
            onClick={() => setField('risk_level', l.value)}
            className={`flex-1 px-2 py-1.5 text-xs rounded border transition-colors ${
              riskLevel === l.value
                ? 'bg-emerald-900/40 border-emerald-700 text-emerald-400'
                : 'bg-gray-900 border-gray-700 text-gray-500 hover:text-gray-300'
            }`}
          >
            {l.label}
          </button>
        ))}
      </div>
    </div>
  )
}

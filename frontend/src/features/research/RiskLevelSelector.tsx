import { useResearchStore } from '../../stores/researchStore'

const LEVELS = [
  { value: '保守', label: '保守', description: 'Lean defensive with tighter exposure expectations.' },
  { value: '中等', label: '中等', description: 'Balanced posture for standard research runs.' },
  { value: '积极', label: '积极', description: 'Allow more offensive positioning and swing.' },
]

export function RiskLevelSelector() {
  const riskLevel = useResearchStore((state) => state.risk_level)
  const setField = useResearchStore((state) => state.setField)

  return (
    <div>
      <label className="mb-2 block text-[11px] font-medium uppercase tracking-[0.18em] text-slate-500">
        Risk level
      </label>
      <div className="space-y-2">
        {LEVELS.map((level) => (
          <button
            key={level.value}
            type="button"
            onClick={() => setField('risk_level', level.value)}
            className={`w-full rounded-2xl border px-3 py-3 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300 ${
              riskLevel === level.value
                ? 'border-cyan-300/22 bg-cyan-300/10 text-white'
                : 'border-white/10 bg-white/[0.03] text-slate-400 hover:border-white/16 hover:bg-white/[0.05] hover:text-slate-200'
            }`}
          >
            <span className="block font-medium">
              {level.label}
            </span>
            <span className={`mt-1 block text-xs leading-5 ${riskLevel === level.value ? 'text-slate-100/80' : 'text-slate-500'}`}>
              {level.description}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}

import { useResearchStore } from '../../stores/researchStore'

export function MarketSwitcher() {
  const market = useResearchStore((s) => s.market)
  const setField = useResearchStore((s) => s.setField)

  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1">Market</label>
      <div className="flex gap-1">
        {(['CN', 'US'] as const).map((m) => (
          <button
            key={m}
            onClick={() => setField('market', m)}
            className={`flex-1 px-3 py-1.5 text-xs rounded border transition-colors ${
              market === m
                ? 'bg-emerald-900/40 border-emerald-700 text-emerald-400'
                : 'bg-gray-900 border-gray-700 text-gray-500 hover:text-gray-300'
            }`}
          >
            {m === 'CN' ? 'A\u80A1 CN' : 'US'}
          </button>
        ))}
      </div>
    </div>
  )
}

import { useQuery } from '@tanstack/react-query'
import { getLLMModels } from '../../api/settings'
import { useResearchStore } from '../../stores/researchStore'

export function LLMModelSelector() {
  const agentModel = useResearchStore((s) => s.agent_model)
  const masterModel = useResearchStore((s) => s.master_model)
  const setField = useResearchStore((s) => s.setField)

  const { data } = useQuery({
    queryKey: ['settings', 'models'],
    queryFn: getLLMModels,
    staleTime: 60_000,
  })

  const available = data?.models?.filter((m) => m.available) ?? []

  return (
    <div className="space-y-2">
      <div>
        <label className="block text-xs text-gray-400 mb-1">Branch Agent Model</label>
        <select
          value={agentModel}
          onChange={(e) => setField('agent_model', e.target.value)}
          className="w-full px-2 py-1.5 text-xs bg-gray-900 border border-gray-700 rounded text-gray-200 outline-none focus:border-emerald-700"
        >
          <option value="">default (claude-sonnet-4-6)</option>
          {available.map((m) => (
            <option key={m.id} value={m.id}>{m.label} — ${m.prompt_price}/1M</option>
          ))}
        </select>
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Master IC Model</label>
        <select
          value={masterModel}
          onChange={(e) => setField('master_model', e.target.value)}
          className="w-full px-2 py-1.5 text-xs bg-gray-900 border border-gray-700 rounded text-gray-200 outline-none focus:border-emerald-700"
        >
          <option value="">default (gpt-5.4-mini)</option>
          {available.map((m) => (
            <option key={m.id} value={m.id}>{m.label} — ${m.prompt_price}/1M</option>
          ))}
        </select>
      </div>
    </div>
  )
}

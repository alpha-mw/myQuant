import { useQuery } from '@tanstack/react-query'
import { getLLMModels } from '../../api/settings'
import { useResearchStore } from '../../stores/researchStore'

const FIELD_CLASS_NAME =
  'w-full rounded-2xl border border-white/10 bg-slate-950/72 px-3 py-2.5 text-sm text-slate-100 outline-none transition-colors focus:border-cyan-300/35 focus:bg-slate-950/90'
const LABEL_CLASS_NAME = 'mb-2 block text-[11px] font-medium uppercase tracking-[0.18em] text-slate-500'

export function LLMModelSelector() {
  const agentModel = useResearchStore((state) => state.agent_model)
  const agentFallbackModel = useResearchStore((state) => state.agent_fallback_model)
  const masterModel = useResearchStore((state) => state.master_model)
  const masterFallbackModel = useResearchStore((state) => state.master_fallback_model)
  const setField = useResearchStore((state) => state.setField)

  const { data } = useQuery({
    queryKey: ['settings', 'models'],
    queryFn: getLLMModels,
    staleTime: 60_000,
  })

  const available = data?.models?.filter((model) => model.available) ?? []
  const slots = [
    { id: 'agent_model', label: 'Subagent primary', value: agentModel },
    { id: 'agent_fallback_model', label: 'Subagent fallback', value: agentFallbackModel },
    { id: 'master_model', label: 'Master primary', value: masterModel },
    { id: 'master_fallback_model', label: 'Master fallback', value: masterFallbackModel },
  ] as const

  return (
    <div className="grid gap-3">
      {slots.map((slot) => (
        <div key={slot.id}>
          <label className={LABEL_CLASS_NAME} htmlFor={slot.id}>
            {slot.label}
          </label>
          <select
            id={slot.id}
            name={slot.id}
            value={slot.value}
            onChange={(event) => setField(slot.id, event.target.value)}
            className={FIELD_CLASS_NAME}
          >
            <option value="">use default</option>
            {available.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label} - ${model.prompt_price}/1M
              </option>
            ))}
          </select>
        </div>
      ))}
    </div>
  )
}

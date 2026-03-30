import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listPresets, createPreset, deletePreset } from '../../api/presets'
import { useResearchStore } from '../../stores/researchStore'
import type { ResearchRunRequest } from '../../types/research'

export function PresetSelector() {
  const queryClient = useQueryClient()
  const loadPreset = useResearchStore((state) => state.loadPreset)
  const toRequest = useResearchStore((state) => state.toRequest)
  const [saving, setSaving] = useState(false)
  const [name, setName] = useState('')

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['presets'],
    queryFn: listPresets,
    staleTime: 60_000,
  })

  const saveMutation = useMutation({
    mutationFn: () => {
      const { preset_id, ...config } = toRequest()
      void preset_id
      return createPreset(name, '', config as ResearchRunRequest)
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['presets'] })
      setSaving(false)
      setName('')
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deletePreset(id),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['presets'] })
    },
  })

  const presets = data?.presets ?? []

  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <label className="text-[11px] font-medium uppercase tracking-[0.18em] text-slate-500">Saved presets</label>
        <button
          type="button"
          onClick={() => setSaving((value) => !value)}
          className="rounded-full border border-white/12 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-white/18 hover:bg-white/[0.08] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
        >
          {saving ? 'Cancel' : 'Save current'}
        </button>
      </div>

      {saving && (
        <div className="mt-3 flex gap-2">
          <input
            aria-label="Preset name"
            autoComplete="off"
            name="preset_name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Preset name…"
            className="flex-1 rounded-2xl border border-white/10 bg-slate-950/72 px-3 py-2.5 text-sm text-slate-100 outline-none transition-colors placeholder:text-slate-500 focus:border-cyan-300/35 focus:bg-slate-950/90"
          />
          <button
            type="button"
            onClick={() => saveMutation.mutate()}
            disabled={!name.trim()}
            className="rounded-2xl border border-cyan-300/22 bg-cyan-300/10 px-4 py-2.5 text-sm font-medium text-cyan-50 transition-colors hover:border-cyan-300/32 hover:bg-cyan-300/16 disabled:cursor-not-allowed disabled:border-white/8 disabled:bg-white/[0.03] disabled:text-slate-500"
          >
            Save
          </button>
        </div>
      )}

      {presets.length > 0 && (
        <div className="mt-3 max-h-40 space-y-2 overflow-y-auto">
          {presets.map((preset) => (
            <div
              key={preset.preset_id}
              className="group flex items-center gap-2 rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-3"
            >
              <button
                type="button"
                onClick={() => loadPreset(preset.preset_id, preset.config as unknown as Partial<ResearchRunRequest>)}
                className="min-w-0 flex-1 text-left focus-visible:outline-none"
              >
                <span className="block truncate text-sm font-medium text-white">{preset.name}</span>
                <span className="mt-1 block text-xs text-slate-500">Load this configuration into the dock</span>
              </button>
              <button
                type="button"
                onClick={() => deleteMutation.mutate(preset.preset_id)}
                className="rounded-full border border-transparent px-2 py-1 text-xs text-slate-500 transition-colors hover:border-red-300/20 hover:bg-red-300/10 hover:text-red-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      )}

      {isLoading && presets.length === 0 && <p className="mt-3 text-sm text-slate-500">Loading presets…</p>}

      {isError && (
        <p className="mt-3 text-sm text-red-300">
          {error instanceof Error ? error.message : 'Failed to load presets.'}
        </p>
      )}

      {saveMutation.isError && (
        <p className="mt-3 text-sm text-red-300">
          {saveMutation.error instanceof Error ? saveMutation.error.message : 'Failed to save preset.'}
        </p>
      )}

      {deleteMutation.isError && (
        <p className="mt-3 text-sm text-red-300">
          {deleteMutation.error instanceof Error ? deleteMutation.error.message : 'Failed to delete preset.'}
        </p>
      )}
    </div>
  )
}

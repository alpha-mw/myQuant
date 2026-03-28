import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { listPresets, createPreset, deletePreset } from '../../api/presets'
import { useResearchStore } from '../../stores/researchStore'
import type { ResearchRunRequest } from '../../types/research'

export function PresetSelector() {
  const queryClient = useQueryClient()
  const loadPreset = useResearchStore((s) => s.loadPreset)
  const toRequest = useResearchStore((s) => s.toRequest)
  const [saving, setSaving] = useState(false)
  const [name, setName] = useState('')

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['presets'],
    queryFn: listPresets,
    staleTime: 60_000,
  })

  const saveMutation = useMutation({
    mutationFn: () => {
      const { preset_id: _presetId, ...config } = toRequest()
      return createPreset(name, '', config as ResearchRunRequest)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['presets'] })
      setSaving(false)
      setName('')
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deletePreset(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['presets'] }),
  })

  const presets = data?.presets ?? []

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-xs text-gray-400">Presets</label>
        <button
          onClick={() => setSaving(!saving)}
          className="text-xs text-emerald-500 hover:text-emerald-400"
        >
          {saving ? 'Cancel' : '+ Save'}
        </button>
      </div>

      {saving && (
        <div className="flex gap-1 mb-2">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Preset name"
            className="flex-1 px-2 py-1 text-xs bg-gray-900 border border-gray-700 rounded text-gray-200 outline-none"
          />
          <button
            onClick={() => saveMutation.mutate()}
            disabled={!name.trim()}
            className="px-2 py-1 text-xs bg-emerald-800 text-emerald-200 rounded hover:bg-emerald-700 disabled:opacity-30"
          >
            Save
          </button>
        </div>
      )}

      {presets.length > 0 && (
        <div className="space-y-0.5 max-h-24 overflow-y-auto">
          {presets.map((p) => (
            <div key={p.preset_id} className="flex items-center group">
              <button
                onClick={() => loadPreset(p.preset_id, p.config as unknown as Partial<ResearchRunRequest>)}
                className="flex-1 text-left px-2 py-1 text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded truncate"
              >
                {p.name}
              </button>
              <button
                onClick={() => deleteMutation.mutate(p.preset_id)}
                className="text-xs text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 px-1"
              >
                &times;
              </button>
            </div>
          ))}
        </div>
      )}

      {isLoading && presets.length === 0 && (
        <p className="mt-2 text-xs text-gray-600">Loading presets...</p>
      )}

      {isError && (
        <p className="mt-2 text-xs text-red-400">
          {error instanceof Error ? error.message : 'Failed to load presets.'}
        </p>
      )}

      {saveMutation.isError && (
        <p className="mt-2 text-xs text-red-400">
          {saveMutation.error instanceof Error ? saveMutation.error.message : 'Failed to save preset.'}
        </p>
      )}

      {deleteMutation.isError && (
        <p className="mt-2 text-xs text-red-400">
          {deleteMutation.error instanceof Error ? deleteMutation.error.message : 'Failed to delete preset.'}
        </p>
      )}
    </div>
  )
}

import { useEffect, useRef, useState, useCallback } from 'react'

export interface PhaseInfo {
  phase_key: string
  phase_label: string
}

interface SSEState {
  logs: string[]
  progress: number
  phase: PhaseInfo
  isComplete: boolean
  error: string | null
}

export function useSSE(jobId: string | null): SSEState {
  const [state, setState] = useState<SSEState>({
    logs: [],
    progress: 0,
    phase: { phase_key: '', phase_label: '' },
    isComplete: false,
    error: null,
  })
  const esRef = useRef<EventSource | null>(null)

  const cleanup = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!jobId) {
      cleanup()
      return
    }

    // Reset state for new job
    setState({ logs: [], progress: 0, phase: { phase_key: '', phase_label: '' }, isComplete: false, error: null })

    const es = new EventSource(`/api/research/${jobId}/stream`)
    esRef.current = es

    es.addEventListener('log', (e) => {
      setState((prev) => ({
        ...prev,
        logs: [...prev.logs, e.data],
      }))
    })

    es.addEventListener('progress', (e) => {
      try {
        const data = JSON.parse(e.data) as {
          progress_pct: number
          phase_key?: string
          phase_label?: string
        }
        setState((prev) => ({
          ...prev,
          progress: data.progress_pct,
          phase: {
            phase_key: data.phase_key ?? prev.phase.phase_key,
            phase_label: data.phase_label ?? prev.phase.phase_label,
          },
        }))
      } catch {
        // ignore parse errors
      }
    })

    es.addEventListener('completed', () => {
      setState((prev) => ({
        ...prev,
        isComplete: true,
        progress: 1,
        phase: { phase_key: 'done', phase_label: '完成' },
      }))
      cleanup()
    })

    es.addEventListener('failed', (e) => {
      let errorMsg = 'Run failed'
      try {
        const data = JSON.parse(e.data) as { error?: string }
        if (data.error) errorMsg = data.error
      } catch {
        // ignore
      }
      setState((prev) => ({ ...prev, isComplete: true, error: errorMsg }))
      cleanup()
    })

    es.onerror = () => {
      setState((prev) => {
        if (prev.isComplete) return prev
        return prev
      })
    }

    return cleanup
  }, [jobId, cleanup])

  return state
}

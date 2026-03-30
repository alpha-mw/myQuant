import { startTransition, useCallback, useEffect, useRef, useState } from 'react'

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

interface SSEInternalState extends SSEState {
  jobId: string | null
}

const MAX_LOG_LINES = 200

function createInitialState(jobId: string | null = null): SSEInternalState {
  return {
    jobId,
    logs: [],
    progress: 0,
    phase: { phase_key: '', phase_label: '' },
    isComplete: false,
    error: null,
  }
}

export function useSSE(jobId: string | null): SSEState {
  const [state, setState] = useState<SSEInternalState>(() => createInitialState())
  const esRef = useRef<EventSource | null>(null)

  const cleanup = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
  }, [])

  const updateState = useCallback((updater: (prev: SSEInternalState) => SSEInternalState) => {
    startTransition(() => {
      setState(updater)
    })
  }, [])

  useEffect(() => {
    cleanup()
    if (!jobId) {
      return
    }

    const es = new EventSource(`/api/research/${jobId}/stream`)
    esRef.current = es
    const baseState = createInitialState(jobId)

    es.addEventListener('log', (e) => {
      updateState((prev) => {
        const current = prev.jobId === jobId ? prev : baseState
        const nextLogs = [...current.logs, e.data]
        return {
          ...current,
          logs: nextLogs.length > MAX_LOG_LINES ? nextLogs.slice(-MAX_LOG_LINES) : nextLogs,
        }
      })
    })

    es.addEventListener('progress', (e) => {
      try {
        const data = JSON.parse(e.data) as {
          progress_pct: number
          phase_key?: string
          phase_label?: string
        }
        updateState((prev) => {
          const current = prev.jobId === jobId ? prev : baseState
          return {
            ...current,
            progress: data.progress_pct,
            phase: {
              phase_key: data.phase_key ?? current.phase.phase_key,
              phase_label: data.phase_label ?? current.phase.phase_label,
            },
          }
        })
      } catch {
        // ignore parse errors
      }
    })

    es.addEventListener('completed', () => {
      updateState((prev) => ({
        ...(prev.jobId === jobId ? prev : baseState),
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
      updateState((prev) => ({
        ...(prev.jobId === jobId ? prev : baseState),
        isComplete: true,
        error: errorMsg,
      }))
      cleanup()
    })

    es.onerror = () => {
      updateState((prev) => {
        if (prev.isComplete) return prev
        return { ...prev, isComplete: true, error: 'Connection lost' }
      })
      cleanup()
    }

    return cleanup
  }, [jobId, cleanup, updateState])

  if (state.jobId !== jobId) {
    return createInitialState()
  }

  return state
}

import { useMutation, useQuery } from '@tanstack/react-query'
import { submitResearchRun, getResearchJob, getResearchReport } from '../api/research'
import { useSSE } from './useSSE'
import { useUIStore } from '../stores/uiStore'
import type { ResearchRunRequest } from '../types/research'

export function useResearchRun() {
  const activeJobId = useUIStore((s) => s.activeJobId)
  const setActiveJobId = useUIStore((s) => s.setActiveJobId)
  const setConfigPanelCollapsed = useUIStore((s) => s.setConfigPanelCollapsed)

  const submitMutation = useMutation({
    mutationFn: (req: ResearchRunRequest) => submitResearchRun(req),
    onSuccess: (data) => {
      setActiveJobId(data.job_id)
      setConfigPanelCollapsed(true)
    },
  })

  const jobQuery = useQuery({
    queryKey: ['job', activeJobId],
    queryFn: () => getResearchJob(activeJobId!),
    enabled: !!activeJobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (status === 'completed' || status === 'failed') return false
      return 2000
    },
  })

  const sse = useSSE(
    activeJobId && jobQuery.data?.status !== 'completed' && jobQuery.data?.status !== 'failed'
      ? activeJobId
      : null,
  )

  const isCompleted = jobQuery.data?.status === 'completed'
  const isFailed = jobQuery.data?.status === 'failed'

  const reportQuery = useQuery({
    queryKey: ['job', activeJobId, 'report'],
    queryFn: () => getResearchReport(activeJobId!),
    enabled: !!activeJobId && isCompleted,
    staleTime: Infinity,
  })

  return {
    submit: submitMutation.mutate,
    isSubmitting: submitMutation.isPending,
    jobId: activeJobId,
    status: jobQuery.data?.status ?? null,
    logs: sse.logs,
    progress: sse.progress,
    phase: sse.phase,
    report: reportQuery.data?.markdown ?? null,
    isRunning: !!activeJobId && !isCompleted && !isFailed,
    isCompleted,
    isFailed,
    error:
      jobQuery.data?.error ??
      sse.error ??
      (submitMutation.error instanceof Error ? submitMutation.error.message : null),
    resultSummary: jobQuery.data?.result_summary ?? null,
  }
}

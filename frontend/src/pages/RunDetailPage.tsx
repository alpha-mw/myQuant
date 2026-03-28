import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getResearchJob, getResearchReport } from '../api/research'
import { ReportViewer } from '../features/research/ReportViewer'
import { Badge } from '../components/Badge'
import { LoadingSpinner } from '../components/LoadingSpinner'

export function RunDetailPage() {
  const { jobId } = useParams<{ jobId: string }>()

  const jobQuery = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getResearchJob(jobId!),
    enabled: !!jobId,
  })

  const reportQuery = useQuery({
    queryKey: ['job', jobId, 'report'],
    queryFn: () => getResearchReport(jobId!),
    enabled: !!jobId && jobQuery.data?.status === 'completed',
    staleTime: Infinity,
  })

  const job = jobQuery.data
  const summary = job?.result_summary as Record<string, unknown> | undefined
  const llmUsage = summary?.llm_usage_summary as Record<string, number> | undefined

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-800 shrink-0">
        <Link to="/history" className="text-xs text-gray-500 hover:text-gray-300">&larr; History</Link>
        <span className="text-xs text-gray-600">/</span>
        <span className="text-xs text-gray-300">{jobId}</span>
        {job && <Badge status={job.status} />}
      </div>

      {/* Metadata bar */}
      {job && (
        <div className="flex items-center gap-4 px-4 py-1.5 border-b border-gray-800 text-xs text-gray-500 shrink-0">
          {summary?.total_time != null && <span>{(summary.total_time as number).toFixed(1)}s</span>}
          {llmUsage?.estimated_cost_usd != null && llmUsage.estimated_cost_usd > 0 && (
            <span>${llmUsage.estimated_cost_usd.toFixed(4)}</span>
          )}
          {llmUsage?.total_calls != null && <span>{llmUsage.total_calls} LLM calls</span>}
          <span>{new Date(job.created_at).toLocaleString()}</span>
        </div>
      )}

      {/* Report */}
      <div className="flex-1 overflow-hidden">
        {jobQuery.isError ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-red-400 text-xs">
              {jobQuery.error instanceof Error ? jobQuery.error.message : 'Failed to load run details.'}
            </p>
          </div>
        ) : reportQuery.isLoading ? (
          <div className="flex items-center justify-center h-full">
            <LoadingSpinner />
          </div>
        ) : reportQuery.isError ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-red-400 text-xs">
              {reportQuery.error instanceof Error ? reportQuery.error.message : 'Failed to load report.'}
            </p>
          </div>
        ) : reportQuery.data?.markdown ? (
          <ReportViewer markdown={reportQuery.data.markdown} />
        ) : job?.status === 'failed' ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-red-400 text-sm">Run failed</p>
              <p className="text-gray-500 text-xs mt-1">{job.error}</p>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-600 text-xs">No report available</p>
          </div>
        )}
      </div>
    </div>
  )
}

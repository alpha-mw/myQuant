import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft } from 'lucide-react'
import { getResearchJob, getResearchReport } from '../api/research'
import { ReportViewer } from '../features/research/ReportViewer'
import { Badge } from '../components/Badge'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { formatCurrency, formatDateTime } from '../lib/format'

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
    <div className="flex h-full min-h-0 flex-col">
      <header className="border-b border-white/10 px-4 py-4 lg:px-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <Link
              to="/history"
              className="inline-flex items-center gap-2 text-sm text-slate-400 transition-colors hover:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
            >
              <ArrowLeft size={14} aria-hidden="true" />
              Back to History
            </Link>
            <h1 className="mt-3 text-2xl font-semibold text-white">Run Details</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              Inspect execution metadata, report output, and final run state for job
              {' '}
              <span className="font-medium text-slate-200">{jobId}</span>.
            </p>
          </div>
          {job && <Badge status={job.status} />}
        </div>
      </header>

      <div className="border-b border-white/8 px-4 py-3 lg:px-6">
        <div className="flex flex-wrap items-center gap-3 text-sm text-slate-400">
          <span className="rounded-full border border-white/10 px-3 py-1.5">{jobId}</span>
        </div>
      </div>

      {job && (
        <div className="flex shrink-0 flex-wrap items-center gap-4 border-b border-white/8 px-4 py-3 text-sm text-slate-400 lg:px-6">
          {summary?.total_time != null && <span>{(summary.total_time as number).toFixed(1)}s runtime</span>}
          {llmUsage?.estimated_cost_usd != null && llmUsage.estimated_cost_usd > 0 && (
            <span>{formatCurrency(llmUsage.estimated_cost_usd)}</span>
          )}
          {llmUsage?.total_calls != null && <span>{llmUsage.total_calls} LLM calls</span>}
          <span>{formatDateTime(job.created_at)}</span>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-hidden">
        {jobQuery.isError ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-red-300">
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
              <p className="text-sm text-red-300">Run failed</p>
              <p className="mt-2 text-sm text-slate-400">{job.error}</p>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-slate-500">No report available.</p>
          </div>
        )}
      </div>
    </div>
  )
}

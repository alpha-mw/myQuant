import { Copy, FileText } from 'lucide-react'
import { MarkdownRenderer } from '../../components/MarkdownRenderer'

interface Props {
  markdown: string
}

export function ReportViewer({ markdown }: Props) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-white/8 px-4 py-4 lg:px-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.22em] text-cyan-200/65">
              <FileText size={14} aria-hidden="true" />
              Narrator output
            </div>
            <h3 className="mt-2 text-lg font-semibold text-white">Research report</h3>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              Review the final report bundle after the control chain settles positions and constraints.
            </p>
          </div>

          <button
            type="button"
            onClick={() => navigator.clipboard.writeText(markdown)}
            className="inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-white/18 hover:bg-white/[0.08] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
          >
            <Copy size={14} aria-hidden="true" />
            Copy markdown
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 lg:px-6">
        <div className="min-h-full rounded-[1.8rem] border border-white/10 bg-[linear-gradient(180deg,rgba(7,13,24,0.96),rgba(4,8,18,0.90))] px-5 py-5 shadow-[0_24px_90px_rgba(2,8,20,0.24)]">
          <MarkdownRenderer content={markdown} />
        </div>
      </div>
    </div>
  )
}

import { MarkdownRenderer } from '../../components/MarkdownRenderer'

interface Props {
  markdown: string
}

export function ReportViewer({ markdown }: Props) {
  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-gray-800 shrink-0">
        <span className="text-xs text-gray-400">Research Report</span>
        <button
          onClick={() => navigator.clipboard.writeText(markdown)}
          className="text-xs text-gray-500 hover:text-gray-300"
        >
          Copy
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        <MarkdownRenderer content={markdown} />
      </div>
    </div>
  )
}

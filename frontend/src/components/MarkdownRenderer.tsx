import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'

interface Props {
  content: string
}

export function MarkdownRenderer({ content }: Props) {
  return (
    <div className="prose prose-invert prose-slate max-w-none text-[0.95rem] leading-7 prose-headings:text-white prose-headings:font-semibold prose-p:text-slate-300 prose-strong:text-slate-100 prose-li:text-slate-300 prose-code:text-cyan-200 prose-a:text-cyan-200 prose-blockquote:border-l-cyan-300/30 prose-blockquote:text-slate-300 prose-pre:border prose-pre:border-white/10 prose-pre:bg-slate-950/85">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
        {content}
      </ReactMarkdown>
    </div>
  )
}

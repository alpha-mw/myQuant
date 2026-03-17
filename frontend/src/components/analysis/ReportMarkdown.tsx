import Markdown from 'react-markdown';

export default function ReportMarkdown({ content }: { content: string }) {
  return <Markdown>{content}</Markdown>;
}

interface Props {
  title: string
  description?: string
}

export function EmptyState({ title, description }: Props) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-8">
      <p className="text-gray-400 text-sm font-medium">{title}</p>
      {description && <p className="text-gray-600 text-xs mt-1">{description}</p>}
    </div>
  )
}

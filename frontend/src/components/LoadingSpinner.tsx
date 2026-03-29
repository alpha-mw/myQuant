export function LoadingSpinner({ size = 'md' }: { size?: 'sm' | 'md' }) {
  const px = size === 'sm' ? 'h-4 w-4' : 'h-6 w-6'
  return (
    <div className={`${px} animate-spin rounded-full border-2 border-gray-600 border-t-emerald-400 motion-reduce:animate-none`} />
  )
}

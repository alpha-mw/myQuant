interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  className?: string;
}

export default function MetricCard({ title, value, subtitle, className = '' }: MetricCardProps) {
  return (
    <div className={`bg-white rounded-lg border border-gray-200 p-5 ${className}`}>
      <p className="text-sm text-gray-500 mb-1">{title}</p>
      <p className="text-2xl font-semibold text-gray-900">{value}</p>
      {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
    </div>
  );
}

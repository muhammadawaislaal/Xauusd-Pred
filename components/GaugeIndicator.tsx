interface GaugeIndicatorProps {
  value: number
  max?: number
  label: string
  color?: 'green' | 'red' | 'blue' | 'purple' | 'yellow'
  status?: string
  className?: string
}

const colorStyles = {
  green: 'bg-green-500',
  red: 'bg-red-500',
  blue: 'bg-blue-500',
  purple: 'bg-purple-500',
  yellow: 'bg-amber-500',
}

const statusColors = {
  green: 'text-green-600',
  red: 'text-red-600',
  blue: 'text-blue-600',
  purple: 'text-purple-600',
  yellow: 'text-amber-600',
}

export function GaugeIndicator({
  value,
  max = 100,
  label,
  color = 'blue',
  status,
  className = '',
}: GaugeIndicatorProps) {
  const percentage = (value / max) * 100

  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-slate-600 text-sm font-medium">{label}</span>
        <span className="text-slate-900 text-sm font-mono font-bold">{value.toFixed(2)}</span>
      </div>
      <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden border border-slate-300">
        <div
          className={`h-full ${colorStyles[color]} transition-all duration-300`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        ></div>
      </div>
      {status && (
        <p className={`text-xs mt-2 font-medium ${statusColors[color]}`}>
          {status}
        </p>
      )}
    </div>
  )
}

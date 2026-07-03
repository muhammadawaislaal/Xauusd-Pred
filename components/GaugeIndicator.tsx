interface GaugeIndicatorProps {
  value: number
  max?: number
  label: string
  color?: 'green' | 'red' | 'blue' | 'purple' | 'yellow'
  status?: string
  className?: string
}

const colorStyles = {
  green: 'bg-signal-buy',
  red: 'bg-signal-sell',
  blue: 'bg-accent-secondary',
  purple: 'bg-accent-primary',
  yellow: 'bg-signal-wait',
}

const statusColors = {
  green: 'text-signal-buy',
  red: 'text-signal-sell',
  blue: 'text-accent-secondary',
  purple: 'text-accent-primary',
  yellow: 'text-signal-wait',
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
        <span className="text-text-muted text-sm font-medium">{label}</span>
        <span className="text-text-primary text-sm font-mono font-bold">{value.toFixed(2)}</span>
      </div>
      <div className="w-full bg-background rounded-full h-2 overflow-hidden border border-border">
        <div
          className={`h-full ${colorStyles[color]} transition-all duration-300`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        ></div>
      </div>
      {status && (
        <p className={`text-xs mt-2 ${statusColors[color]}`}>
          {status}
        </p>
      )}
    </div>
  )
}

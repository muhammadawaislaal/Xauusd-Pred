import { ArrowUp, ArrowDown } from 'lucide-react'

interface StatsCardProps {
  label: string
  value: string | number
  change?: number
  icon?: React.ReactNode
  color?: 'green' | 'red' | 'blue' | 'purple'
  className?: string
}

const colorStyles = {
  green: 'border-signal-buy/30 bg-signal-buy/5',
  red: 'border-signal-sell/30 bg-signal-sell/5',
  blue: 'border-accent-secondary/30 bg-accent-secondary/5',
  purple: 'border-accent-primary/30 bg-accent-primary/5',
}

const textColorStyles = {
  green: 'text-signal-buy',
  red: 'text-signal-sell',
  blue: 'text-accent-secondary',
  purple: 'text-accent-primary',
}

export function StatsCard({
  label,
  value,
  change,
  icon,
  color = 'blue',
  className = '',
}: StatsCardProps) {
  return (
    <div className={`bg-surface border border-border rounded-xl p-6 ${colorStyles[color]} ${className}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-text-muted text-sm font-medium mb-2">{label}</p>
          <p className="text-text-primary text-2xl font-bold font-mono">{value}</p>
          {change !== undefined && (
            <div className="flex items-center gap-1 mt-2">
              {change >= 0 ? (
                <ArrowUp size={16} className="text-signal-buy" />
              ) : (
                <ArrowDown size={16} className="text-signal-sell" />
              )}
              <span className={change >= 0 ? 'text-signal-buy' : 'text-signal-sell'}>
                {Math.abs(change).toFixed(2)}%
              </span>
            </div>
          )}
        </div>
        {icon && <div className={`text-2xl ${textColorStyles[color]}`}>{icon}</div>}
      </div>
    </div>
  )
}

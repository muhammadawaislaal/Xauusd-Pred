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
  green: 'border-green-200 bg-green-50',
  red: 'border-red-200 bg-red-50',
  blue: 'border-blue-200 bg-blue-50',
  purple: 'border-purple-200 bg-purple-50',
}

const textColorStyles = {
  green: 'text-green-600',
  red: 'text-red-600',
  blue: 'text-blue-600',
  purple: 'text-purple-600',
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
    <div className={`bg-white border rounded-xl p-6 ${colorStyles[color]} ${className}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-slate-600 text-sm font-medium mb-2">{label}</p>
          <p className="text-slate-900 text-2xl font-bold font-mono">{value}</p>
          {change !== undefined && (
            <div className="flex items-center gap-1 mt-2">
              {change >= 0 ? (
                <ArrowUp size={16} className="text-green-600" />
              ) : (
                <ArrowDown size={16} className="text-red-600" />
              )}
              <span className={change >= 0 ? 'text-green-600' : 'text-red-600'}>
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

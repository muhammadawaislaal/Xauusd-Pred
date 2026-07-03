interface SignalBadgeProps {
  signal: 'BUY' | 'SELL' | 'WAIT'
  pips?: number
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const signalStyles = {
  BUY: {
    bg: 'bg-green-100',
    border: 'border-green-300',
    text: 'text-green-700',
    glow: 'shadow-lg shadow-green-300/50',
  },
  SELL: {
    bg: 'bg-red-100',
    border: 'border-red-300',
    text: 'text-red-700',
    glow: '',
  },
  WAIT: {
    bg: 'bg-amber-100',
    border: 'border-amber-300',
    text: 'text-amber-700',
    glow: '',
  },
}

const sizeStyles = {
  sm: 'px-3 py-1 text-xs',
  md: 'px-4 py-2 text-sm',
  lg: 'px-6 py-3 text-base',
}

export function SignalBadge({ signal, pips, size = 'md', className = '' }: SignalBadgeProps) {
  const style = signalStyles[signal]
  const pulse = signal === 'BUY' ? 'pulse-signal' : ''

  return (
    <div className={`inline-flex items-center gap-2 ${className}`}>
      <div
        className={`
          border rounded-lg font-bold
          ${sizeStyles[size]}
          ${style.bg}
          ${style.border}
          ${style.text}
          ${style.glow}
          ${pulse}
        `}
      >
        {signal}
        {pips !== undefined && <span className="ml-2 font-mono">+{pips} pips</span>}
      </div>
    </div>
  )
}

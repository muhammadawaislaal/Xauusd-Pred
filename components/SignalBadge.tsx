interface SignalBadgeProps {
  signal: 'BUY' | 'SELL' | 'WAIT'
  pips?: number
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const signalStyles = {
  BUY: {
    bg: 'bg-signal-buy/20',
    border: 'border-signal-buy/50',
    text: 'text-signal-buy',
    glow: 'shadow-lg shadow-signal-buy/20',
  },
  SELL: {
    bg: 'bg-signal-sell/20',
    border: 'border-signal-sell/50',
    text: 'text-signal-sell',
    glow: '',
  },
  WAIT: {
    bg: 'bg-signal-wait/20',
    border: 'border-signal-wait/50',
    text: 'text-signal-wait',
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

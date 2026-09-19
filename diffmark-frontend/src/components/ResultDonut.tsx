'use client'
function circleMetrics(value: number, radius: number) {
  const circumference = 2 * Math.PI * radius
  const clamped = Math.max(0, Math.min(1, value))
  const offset = circumference * (1 - clamped)
  return { circumference, offset }
}

export function ResultDonut({ value, label, color = '#8a2be2' }: { value: number, label: string, color?: string }) {
  const v = Math.max(0, Math.min(1, value))
  const size = 140
  const center = size / 2
  const r = 48
  const { circumference, offset } = circleMetrics(v, r)
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ display: 'block' }}>
      <defs>
        <linearGradient id="dg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor={color} />
          <stop offset="100%" stopColor="#7a54f6" />
        </linearGradient>
      </defs>
      <circle cx={center} cy={center} r={r} stroke="rgba(255,255,255,0.12)" strokeWidth="14" fill="none" />
      <circle
        cx={center}
        cy={center}
        r={r}
        stroke="url(#dg)"
        strokeWidth="14"
        strokeDasharray={`${circumference} ${circumference}`}
        strokeDashoffset={offset}
        strokeLinecap="round"
        fill="none"
        transform={`rotate(-90 ${center} ${center})`}
      />
      <text x={center} y={center - 4} textAnchor="middle" fontSize="22" fill="#ffffff" fontWeight={600}>
        {Math.round(v * 100)}%
      </text>
      <text x={center} y={center + 24} textAnchor="middle" fontSize="13" fill="#ffffffcc">
        {label}
      </text>
    </svg>
  )
}

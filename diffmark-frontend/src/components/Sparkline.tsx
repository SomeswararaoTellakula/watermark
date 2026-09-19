'use client'
export function Sparkline({ points, color = '#00bfff' }: { points: number[], color?: string }) {
  const data = points.slice(-20)
  const w = 260
  const h = 60
  const min = Math.min(...data, 0)
  const max = Math.max(...data, 1)
  const scaleX = (i: number) => (w / Math.max(1, data.length - 1)) * i
  const scaleY = (v: number) => h - (h * (v - min)) / Math.max(0.0001, (max - min))
  const d =
    data.map((v, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i).toFixed(2)} ${scaleY(v).toFixed(2)}`).join(' ')
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
      <rect x="0" y="0" width={w} height={h} rx="8" fill="rgba(255,255,255,0.06)" />
      <path d={d} fill="none" stroke={color} strokeWidth="2" />
    </svg>
  )
}

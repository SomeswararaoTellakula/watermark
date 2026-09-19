'use client'
export function ResultChart({ accuracy, confidence }: { accuracy: number, confidence: number }) {
  const a = Math.max(0, Math.min(1, accuracy))
  const c = Math.max(0, Math.min(1, confidence))
  const width = 360
  const height = 180
  const margin = { top: 20, right: 20, bottom: 34, left: 30 }
  const innerH = height - margin.top - margin.bottom
  const barW = 100
  const gap = 60
  const aH = a * innerH
  const cH = c * innerH
  const yBase = height - margin.bottom
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }} aria-label="Verification chart">
      <rect x="0" y="0" width={width} height={height} rx="12" fill="rgba(255,255,255,0.06)" />
      <g transform={`translate(${margin.left},${margin.top})`}>
        <line x1="0" y1={innerH} x2={barW * 2 + gap} y2={innerH} stroke="rgba(255,255,255,0.15)" />
        <g transform={`translate(0,0)`}>
          <rect x="0" y={yBase - aH - margin.top} width={barW} height={aH} fill="#8a2be2" rx="6" />
          <text x={barW / 2} y={yBase + 18 - margin.top} textAnchor="middle" fontSize="13" fill="#ffffffcc">Accuracy</text>
          <text x={barW / 2} y={yBase - aH - margin.top - 6} textAnchor="middle" fontSize="14" fill="#ffffff">{Math.round(a * 100)}%</text>
        </g>
        <g transform={`translate(${barW + gap},0)`}>
          <rect x="0" y={yBase - cH - margin.top} width={barW} height={cH} fill="#00bfff" rx="6" />
          <text x={barW / 2} y={yBase + 18 - margin.top} textAnchor="middle" fontSize="13" fill="#ffffffcc">Confidence</text>
          <text x={barW / 2} y={yBase - cH - margin.top - 6} textAnchor="middle" fontSize="14" fill="#ffffff">{Math.round(c * 100)}%</text>
        </g>
      </g>
    </svg>
  )
}

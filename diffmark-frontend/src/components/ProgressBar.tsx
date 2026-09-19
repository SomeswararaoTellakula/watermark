export function ProgressBar({ value }: { value: number }) {
  const v = Math.max(0, Math.min(100, Math.round(value)))
  return (
    <div className="w-full h-3 bg-white/10 rounded-full overflow-hidden">
      <div
        className="h-full bg-gradient-to-r from-neon-purple to-neon-blue transition-all"
        style={{ width: `${v}%` }}
      />
    </div>
  )
}

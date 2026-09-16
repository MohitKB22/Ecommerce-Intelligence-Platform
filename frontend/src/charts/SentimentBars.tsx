interface Props {
  distribution: Record<string, number>
}

const ORDER = ['positive', 'neutral', 'negative'] as const
const COLORS: Record<string, string> = {
  positive: 'bg-emerald-500',
  neutral: 'bg-slate-300',
  negative: 'bg-rose-500',
}

/** Compact stacked bar - avoids pulling a chart library into the product page. */
export function SentimentBars({ distribution }: Props) {
  const total = ORDER.reduce((sum, key) => sum + (distribution[key] ?? 0), 0)
  if (total === 0) return <p className="text-xs text-slate-400">No sentiment data yet.</p>

  return (
    <div>
      <div className="flex h-2 overflow-hidden rounded-full bg-surface-100" role="img"
           aria-label="Review sentiment distribution">
        {ORDER.map((key) => {
          const share = ((distribution[key] ?? 0) / total) * 100
          return share > 0 ? (
            <div key={key} className={COLORS[key]} style={{ width: `${share}%` }} />
          ) : null
        })}
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-xs text-slate-600">
        {ORDER.map((key) => (
          <span key={key} className="flex items-center gap-1.5">
            <span className={`h-2 w-2 rounded-full ${COLORS[key]}`} aria-hidden />
            {key} {(((distribution[key] ?? 0) / total) * 100).toFixed(0)}%
          </span>
        ))}
      </div>
    </div>
  )
}

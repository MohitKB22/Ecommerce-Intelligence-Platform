import { percent } from '@/utils/format'

interface Props {
  counts: Record<string, number>
  viewToCart: number
  cartToPurchase: number
}

const STEPS = [
  { key: 'product_view', label: 'Product views' },
  { key: 'click', label: 'Clicks' },
  { key: 'add_to_cart', label: 'Added to cart' },
  { key: 'purchase', label: 'Purchases' },
]

/** Simple proportional funnel - no chart library needed for four bars. */
export function FunnelChart({ counts, viewToCart, cartToPurchase }: Props) {
  const max = Math.max(...STEPS.map((s) => counts[s.key] ?? 0), 1)
  return (
    <div className="space-y-2">
      {STEPS.map((step) => {
        const value = counts[step.key] ?? 0
        return (
          <div key={step.key}>
            <div className="mb-1 flex items-center justify-between text-xs">
              <span className="text-slate-600">{step.label}</span>
              <span className="tabular-nums font-medium text-surface-900">{value.toLocaleString()}</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-surface-100">
              <div className="h-full rounded-full bg-brand-500" style={{ width: `${(value / max) * 100}%` }} />
            </div>
          </div>
        )
      })}
      <div className="flex flex-wrap gap-4 pt-2 text-xs text-slate-500">
        <span>View → cart: <strong className="text-surface-900">{percent(viewToCart)}</strong></span>
        <span>Cart → purchase: <strong className="text-surface-900">{percent(cartToPurchase)}</strong></span>
      </div>
    </div>
  )
}

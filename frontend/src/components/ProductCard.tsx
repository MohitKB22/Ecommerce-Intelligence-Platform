import { Link } from 'react-router-dom'
import { ShoppingCart, Sparkles } from 'lucide-react'
import { Badge, Rating } from './ui'
import { currency, productGradient } from '@/utils/format'
import { useCart } from '@/store/CartContext'
import { trackEvent } from '@/api/endpoints'
import { useUser } from '@/store/UserContext'
import type { Product } from '@/types'

interface Props {
  product: Product
  explanation?: string
  score?: number
  compact?: boolean
}

export function ProductCard({ product, explanation, compact = false }: Props) {
  const { addToCart } = useCart()
  const { userId } = useUser()
  const discounted = product.discount_pct > 0

  const handleView = () => {
    void trackEvent({ event_type: 'click', user_id: userId, product_id: product.id }).catch(() => undefined)
  }

  return (
    <article className={`card card-hover group flex flex-col overflow-hidden ${compact ? 'w-56 shrink-0 snap-start' : ''}`}>
      <Link to={`/product/${product.id}`} onClick={handleView} className="block">
        <div className="relative aspect-[4/3] w-full overflow-hidden"
             style={{ background: productGradient(product.id) }}>
          <div className="absolute inset-0 flex items-center justify-center p-4">
            <span className="line-clamp-3 text-center text-xs font-medium text-slate-600/80">
              {product.title}
            </span>
          </div>
          {discounted && (
            <Badge tone="danger" className="absolute left-2 top-2">-{product.discount_pct.toFixed(0)}%</Badge>
          )}
          {!product.in_stock && (
            <Badge tone="neutral" className="absolute right-2 top-2">Out of stock</Badge>
          )}
        </div>
      </Link>

      <div className="flex flex-1 flex-col gap-2 p-3">
        <Link to={`/product/${product.id}`} onClick={handleView}
              className="line-clamp-2 text-sm font-medium leading-snug text-surface-900 hover:text-brand-700">
          {product.title}
        </Link>
        <Rating value={product.rating_avg} count={product.rating_count} />

        <div className="mt-auto flex items-end justify-between gap-2">
          <div>
            <p className="text-base font-semibold text-surface-900">{currency(product.effective_price)}</p>
            {discounted && (
              <p className="text-xs text-slate-400 line-through">{currency(product.price)}</p>
            )}
          </div>
          <button
            type="button"
            className="btn-primary !px-2.5 !py-1.5"
            disabled={!product.in_stock}
            onClick={() => addToCart(product)}
            aria-label={`Add ${product.title} to cart`}
          >
            <ShoppingCart className="h-4 w-4" aria-hidden />
          </button>
        </div>

        {explanation && (
          <p className="flex items-start gap-1 border-t border-surface-100 pt-2 text-[11px] leading-snug text-slate-500">
            <Sparkles className="mt-0.5 h-3 w-3 shrink-0 text-brand-500" aria-hidden />
            <span className="line-clamp-2">{explanation}</span>
          </p>
        )}
      </div>
    </article>
  )
}

export function ProductRail({ items }: {
  items: { product: Product; explanation?: string; score?: number }[]
}) {
  return (
    <div className="scroll-row">
      {items.map((item) => (
        <ProductCard key={item.product.id} product={item.product} explanation={item.explanation} compact />
      ))}
    </div>
  )
}

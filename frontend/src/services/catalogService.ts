/**
 * Presentation-layer helpers that sit between the raw API and components.
 * Keeping this logic out of components makes it directly unit-testable.
 */
import type { Product, SearchHit } from '@/types'

export function discountAmount(product: Product): number {
  return Math.max(product.price - product.effective_price, 0)
}

export function stockLabel(product: Product): { label: string; tone: 'success' | 'warning' | 'danger' } {
  if (product.inventory <= 0) return { label: 'Out of stock', tone: 'danger' }
  if (product.inventory < 10) return { label: `Only ${product.inventory} left`, tone: 'warning' }
  return { label: 'In stock', tone: 'success' }
}

/** The signal that contributed most to a search result's position. */
export function dominantSignal(hit: SearchHit): string {
  const entries = Object.entries(hit.signals) as [string, number][]
  if (entries.length === 0) return 'relevance'
  return entries.reduce((best, current) => (current[1] > best[1] ? current : best))[0]
}

export function sortProducts(products: Product[], sort: string): Product[] {
  const copy = [...products]
  switch (sort) {
    case 'price_asc':
      return copy.sort((a, b) => a.effective_price - b.effective_price)
    case 'price_desc':
      return copy.sort((a, b) => b.effective_price - a.effective_price)
    case 'rating':
      return copy.sort((a, b) => b.rating_avg - a.rating_avg)
    case 'discount':
      return copy.sort((a, b) => b.discount_pct - a.discount_pct)
    default:
      return copy
  }
}

export function summariseCart(lines: { product: Product; quantity: number }[]) {
  const subtotal = lines.reduce((sum, l) => sum + l.product.effective_price * l.quantity, 0)
  const savings = lines.reduce((sum, l) => sum + discountAmount(l.product) * l.quantity, 0)
  const itemCount = lines.reduce((sum, l) => sum + l.quantity, 0)
  const shipping = subtotal > 50 || subtotal === 0 ? 0 : 6.99
  const tax = subtotal * 0.08
  return { subtotal, savings, itemCount, shipping, tax, total: subtotal + shipping + tax }
}

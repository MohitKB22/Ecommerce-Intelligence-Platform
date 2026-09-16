import { describe, expect, it } from 'vitest'
import { discountAmount, dominantSignal, sortProducts, stockLabel, summariseCart } from '@/services/catalogService'
import type { Product, SearchHit } from '@/types'

const makeProduct = (overrides: Partial<Product> = {}): Product => ({
  id: 1, sku: 'ECI-1', title: 'Test product', price: 100, discount_pct: 0, currency: 'USD',
  rating_avg: 4, rating_count: 10, inventory: 25, image_url: '', category_id: 1, brand_id: 1,
  effective_price: 100, in_stock: true, ...overrides,
})

describe('catalogService', () => {
  it('computes the absolute discount', () => {
    expect(discountAmount(makeProduct({ price: 100, effective_price: 80 }))).toBe(20)
    expect(discountAmount(makeProduct({ price: 80, effective_price: 100 }))).toBe(0)
  })

  it('describes stock levels', () => {
    expect(stockLabel(makeProduct({ inventory: 0 })).tone).toBe('danger')
    expect(stockLabel(makeProduct({ inventory: 3 })).tone).toBe('warning')
    expect(stockLabel(makeProduct({ inventory: 3 })).label).toBe('Only 3 left')
    expect(stockLabel(makeProduct({ inventory: 40 })).tone).toBe('success')
  })

  it('identifies the dominant ranking signal', () => {
    const hit = {
      product: makeProduct(),
      score: 0.5,
      explanation: '',
      signals: { text: 0.2, semantic: 0.9, popularity: 0.1, rating: 0.3, conversion: 0, personal: 0, availability: 1 },
    } as unknown as SearchHit
    // availability is 1.0 here, so it should win
    expect(dominantSignal(hit)).toBe('availability')
  })

  it('sorts products by each supported key', () => {
    const products = [
      makeProduct({ id: 1, effective_price: 50, rating_avg: 3, discount_pct: 5 }),
      makeProduct({ id: 2, effective_price: 10, rating_avg: 5, discount_pct: 25 }),
    ]
    expect(sortProducts(products, 'price_asc')[0].id).toBe(2)
    expect(sortProducts(products, 'price_desc')[0].id).toBe(1)
    expect(sortProducts(products, 'rating')[0].id).toBe(2)
    expect(sortProducts(products, 'discount')[0].id).toBe(2)
    expect(sortProducts(products, 'unknown')[0].id).toBe(1)
  })

  it('summarises a cart including free-shipping threshold', () => {
    const cheap = summariseCart([{ product: makeProduct({ effective_price: 20 }), quantity: 1 }])
    expect(cheap.shipping).toBeCloseTo(6.99)
    expect(cheap.itemCount).toBe(1)

    const big = summariseCart([{ product: makeProduct({ effective_price: 200 }), quantity: 2 }])
    expect(big.shipping).toBe(0)
    expect(big.subtotal).toBe(400)
    expect(big.total).toBeCloseTo(432)
  })

  it('reports zero for an empty cart', () => {
    const empty = summariseCart([])
    expect(empty.subtotal).toBe(0)
    expect(empty.total).toBe(0)
  })
})

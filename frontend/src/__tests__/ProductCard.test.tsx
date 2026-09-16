import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { ProductCard } from '@/components/ProductCard'
import { CartProvider, useCart } from '@/store/CartContext'
import { UserProvider } from '@/store/UserContext'
import type { Product } from '@/types'

vi.mock('@/api/endpoints', () => ({
  trackEvent: vi.fn().mockResolvedValue({}),
  login: vi.fn(),
}))

const product: Product = {
  id: 42, sku: 'ECI-42', title: 'Auralis Headphones Pro 500', price: 200, discount_pct: 25,
  currency: 'USD', rating_avg: 4.5, rating_count: 128, inventory: 12, image_url: '',
  category_id: 2, brand_id: 3, effective_price: 150, in_stock: true,
}

function renderCard(overrides: Partial<Product> = {}) {
  return render(
    <MemoryRouter>
      <UserProvider>
        <CartProvider>
          <ProductCard product={{ ...product, ...overrides }} explanation="Because you viewed similar items" />
        </CartProvider>
      </UserProvider>
    </MemoryRouter>,
  )
}

describe('ProductCard', () => {
  it('renders title, discounted price and original price', () => {
    renderCard()
    // The title appears twice: once in the image placeholder, once as the link.
    expect(screen.getAllByText('Auralis Headphones Pro 500').length).toBeGreaterThan(0)
    expect(screen.getByRole('link', { name: 'Auralis Headphones Pro 500' })).toBeInTheDocument()
    expect(screen.getByText('$150.00')).toBeInTheDocument()
    expect(screen.getByText('$200.00')).toBeInTheDocument()
    expect(screen.getByText('-25%')).toBeInTheDocument()
  })

  it('shows the recommendation explanation', () => {
    renderCard()
    expect(screen.getByText('Because you viewed similar items')).toBeInTheDocument()
  })

  it('disables the add button when out of stock', () => {
    renderCard({ inventory: 0, in_stock: false })
    expect(screen.getByRole('button', { name: /add .* to cart/i })).toBeDisabled()
    expect(screen.getByText('Out of stock')).toBeInTheDocument()
  })

  it('adds the product to the cart when clicked', async () => {
    const user = userEvent.setup()
    function CartCount() {
      const { itemCount } = useCart()
      return <span data-testid="count">{itemCount}</span>
    }
    render(
      <MemoryRouter>
        <UserProvider>
          <CartProvider>
            <ProductCard product={product} />
            <CartCount />
          </CartProvider>
        </UserProvider>
      </MemoryRouter>,
    )
    expect(screen.getByTestId('count')).toHaveTextContent('0')
    await user.click(screen.getByRole('button', { name: /add .* to cart/i }))
    expect(screen.getByTestId('count')).toHaveTextContent('1')
  })
})

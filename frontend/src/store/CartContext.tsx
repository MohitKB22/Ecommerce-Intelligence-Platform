import { createContext, useCallback, useContext, useMemo, useReducer } from 'react'
import type { ReactNode } from 'react'
import type { CartLine, Product } from '@/types'
import { trackEvent } from '@/api/endpoints'
import { useUser } from './UserContext'

interface CartState {
  lines: CartLine[]
}

type CartAction =
  | { type: 'add'; product: Product; quantity: number }
  | { type: 'remove'; productId: number }
  | { type: 'setQuantity'; productId: number; quantity: number }
  | { type: 'clear' }

function reducer(state: CartState, action: CartAction): CartState {
  switch (action.type) {
    case 'add': {
      const existing = state.lines.find((l) => l.product.id === action.product.id)
      if (existing) {
        return {
          lines: state.lines.map((l) =>
            l.product.id === action.product.id ? { ...l, quantity: l.quantity + action.quantity } : l,
          ),
        }
      }
      return { lines: [...state.lines, { product: action.product, quantity: action.quantity }] }
    }
    case 'remove':
      return { lines: state.lines.filter((l) => l.product.id !== action.productId) }
    case 'setQuantity':
      return {
        lines: state.lines
          .map((l) => (l.product.id === action.productId ? { ...l, quantity: action.quantity } : l))
          .filter((l) => l.quantity > 0),
      }
    case 'clear':
      return { lines: [] }
    default:
      return state
  }
}

interface CartContextValue {
  lines: CartLine[]
  itemCount: number
  subtotal: number
  addToCart: (product: Product, quantity?: number) => void
  removeFromCart: (productId: number) => void
  setQuantity: (productId: number, quantity: number) => void
  clear: () => void
  checkout: () => Promise<void>
}

const CartContext = createContext<CartContextValue | null>(null)

export function CartProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, { lines: [] })
  const { userId } = useUser()

  const addToCart = useCallback(
    (product: Product, quantity = 1) => {
      dispatch({ type: 'add', product, quantity })
      // Fire-and-forget: a telemetry failure must never block the interaction.
      void trackEvent({ event_type: 'add_to_cart', user_id: userId, product_id: product.id, quantity })
        .catch(() => undefined)
    },
    [userId],
  )

  const removeFromCart = useCallback(
    (productId: number) => {
      dispatch({ type: 'remove', productId })
      void trackEvent({ event_type: 'remove_from_cart', user_id: userId, product_id: productId })
        .catch(() => undefined)
    },
    [userId],
  )

  const setQuantity = useCallback((productId: number, quantity: number) => {
    dispatch({ type: 'setQuantity', productId, quantity })
  }, [])

  const clear = useCallback(() => dispatch({ type: 'clear' }), [])

  const checkout = useCallback(async () => {
    // Records a purchase event per line so the ML pipeline sees the conversion.
    await Promise.allSettled(
      state.lines.map((line) =>
        trackEvent({
          event_type: 'purchase',
          user_id: userId,
          product_id: line.product.id,
          quantity: line.quantity,
          value: line.product.effective_price * line.quantity,
        }),
      ),
    )
    dispatch({ type: 'clear' })
  }, [state.lines, userId])

  const value = useMemo<CartContextValue>(() => {
    const itemCount = state.lines.reduce((sum, l) => sum + l.quantity, 0)
    const subtotal = state.lines.reduce((sum, l) => sum + l.product.effective_price * l.quantity, 0)
    return { lines: state.lines, itemCount, subtotal, addToCart, removeFromCart, setQuantity, clear, checkout }
  }, [state.lines, addToCart, removeFromCart, setQuantity, clear, checkout])

  return <CartContext.Provider value={value}>{children}</CartContext.Provider>
}

export function useCart(): CartContextValue {
  const ctx = useContext(CartContext)
  if (!ctx) throw new Error('useCart must be used inside <CartProvider>')
  return ctx
}

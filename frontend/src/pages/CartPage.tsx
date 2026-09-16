import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CheckCircle2, Minus, Plus, Trash2 } from 'lucide-react'
import { Card, EmptyState, SectionHeading } from '@/components/ui'
import { useCart } from '@/store/CartContext'
import { currency, productGradient } from '@/utils/format'

export default function CartPage() {
  const { lines, subtotal, itemCount, setQuantity, removeFromCart, checkout, clear } = useCart()
  const [placing, setPlacing] = useState(false)
  const [placed, setPlaced] = useState(false)

  const handleCheckout = async () => {
    setPlacing(true)
    try {
      await checkout()
      setPlaced(true)
    } finally {
      setPlacing(false)
    }
  }

  if (placed) {
    return (
      <div className="mx-auto max-w-lg py-12 text-center">
        <CheckCircle2 className="mx-auto h-12 w-12 text-emerald-500" aria-hidden />
        <h1 className="mt-4 text-xl font-semibold">Order placed</h1>
        <p className="mt-2 text-sm text-slate-500">
          Purchase events were recorded for every item, so your recommendations will update.
        </p>
        <div className="mt-6 flex justify-center gap-2">
          <Link to="/" className="btn-primary">Keep shopping</Link>
          <Link to="/recommendations" className="btn-secondary">See updated picks</Link>
        </div>
      </div>
    )
  }

  if (lines.length === 0) {
    return (
      <EmptyState
        title="Your cart is empty"
        message="Add a few products and they will show up here."
        action={<Link to="/search" className="btn-primary">Browse products</Link>}
      />
    )
  }

  const shipping = subtotal > 50 ? 0 : 6.99
  const tax = subtotal * 0.08

  return (
    <div className="space-y-5">
      <SectionHeading title="Your cart" subtitle={`${itemCount} item${itemCount === 1 ? '' : 's'}`}
                      action={<button type="button" className="btn-ghost" onClick={clear}>Clear cart</button>} />

      <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
        <div className="space-y-3">
          {lines.map((line) => (
            <Card key={line.product.id} className="flex gap-4 !p-4">
              <Link to={`/product/${line.product.id}`} className="h-20 w-20 shrink-0 rounded-lg"
                    style={{ background: productGradient(line.product.id) }} aria-hidden />
              <div className="min-w-0 flex-1">
                <Link to={`/product/${line.product.id}`}
                      className="line-clamp-2 text-sm font-medium hover:text-brand-700">
                  {line.product.title}
                </Link>
                <p className="mt-1 text-sm text-slate-500">{currency(line.product.effective_price)} each</p>
                <div className="mt-2 flex items-center gap-2">
                  <button type="button" className="btn-secondary !px-2 !py-1"
                          onClick={() => setQuantity(line.product.id, line.quantity - 1)}
                          aria-label="Decrease quantity">
                    <Minus className="h-3 w-3" aria-hidden />
                  </button>
                  <span className="w-8 text-center text-sm tabular-nums">{line.quantity}</span>
                  <button type="button" className="btn-secondary !px-2 !py-1"
                          onClick={() => setQuantity(line.product.id, line.quantity + 1)}
                          aria-label="Increase quantity">
                    <Plus className="h-3 w-3" aria-hidden />
                  </button>
                  <button type="button" className="btn-ghost ml-auto !px-2 text-rose-600"
                          onClick={() => removeFromCart(line.product.id)}
                          aria-label={`Remove ${line.product.title}`}>
                    <Trash2 className="h-4 w-4" aria-hidden />
                  </button>
                </div>
              </div>
              <p className="shrink-0 font-semibold tabular-nums">
                {currency(line.product.effective_price * line.quantity)}
              </p>
            </Card>
          ))}
        </div>

        <Card className="h-fit space-y-3">
          <h2 className="font-medium">Order summary</h2>
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between"><dt className="text-slate-500">Subtotal</dt>
              <dd className="tabular-nums">{currency(subtotal)}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Shipping</dt>
              <dd className="tabular-nums">{shipping === 0 ? 'Free' : currency(shipping)}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Estimated tax</dt>
              <dd className="tabular-nums">{currency(tax)}</dd></div>
            <div className="flex justify-between border-t border-surface-200 pt-2 text-base font-semibold">
              <dt>Total</dt><dd className="tabular-nums">{currency(subtotal + shipping + tax)}</dd>
            </div>
          </dl>
          <button type="button" className="btn-primary w-full" onClick={handleCheckout} disabled={placing}>
            {placing ? 'Placing order...' : 'Place order'}
          </button>
          <p className="text-xs text-slate-400">
            This is a demonstration checkout. No payment is taken; purchase events are recorded so the
            ML models see the conversion.
          </p>
        </Card>
      </div>
    </div>
  )
}

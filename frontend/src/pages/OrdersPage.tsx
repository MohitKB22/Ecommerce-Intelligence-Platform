import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getUserOrders } from '@/api/endpoints'
import { Badge, Card, EmptyState, ErrorState, SectionHeading, Spinner } from '@/components/ui'
import { currency, shortDate } from '@/utils/format'
import { useUser } from '@/store/UserContext'

const STATUS_TONE: Record<string, 'success' | 'info' | 'warning' | 'danger' | 'neutral'> = {
  delivered: 'success', shipped: 'info', paid: 'brand' as 'info', pending: 'warning',
  cancelled: 'danger', returned: 'danger',
}

export default function OrdersPage() {
  const { userId } = useUser()
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['orders', userId],
    queryFn: () => getUserOrders(userId, 1, 20),
  })

  if (isLoading) return <Spinner label="Loading orders" />
  if (isError) return <ErrorState title="Could not load orders" message={(error as Error).message}
                                  onRetry={() => refetch()} />
  if (!data || data.items.length === 0) {
    return <EmptyState title="No orders yet"
                       message="Orders you place will appear here."
                       action={<Link to="/search" className="btn-primary">Start shopping</Link>} />
  }

  return (
    <div className="space-y-4">
      <SectionHeading title="Your orders" subtitle={`${data.meta.total} orders on record`} />
      {data.items.map((order) => (
        <Card key={order.id}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="font-mono text-sm font-medium">{order.order_number}</p>
              <p className="mt-0.5 text-xs text-slate-500">
                Placed {shortDate(order.placed_at)} · {order.item_count} items · {order.channel}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <Badge tone={STATUS_TONE[order.status] ?? 'neutral'}>{order.status}</Badge>
              <span className="font-semibold tabular-nums">{currency(order.total_amount)}</span>
            </div>
          </div>
          {order.items.length > 0 && (
            <ul className="mt-3 divide-y divide-surface-100 border-t border-surface-100 pt-2 text-sm">
              {order.items.slice(0, 4).map((item) => (
                <li key={item.id} className="flex items-center justify-between gap-3 py-1.5">
                  <Link to={`/product/${item.product_id}`} className="text-slate-600 hover:text-brand-700">
                    Product #{item.product_id} × {item.quantity}
                  </Link>
                  <span className="tabular-nums text-slate-500">{currency(item.line_total)}</span>
                </li>
              ))}
              {order.items.length > 4 && (
                <li className="py-1.5 text-xs text-slate-400">+{order.items.length - 4} more items</li>
              )}
            </ul>
          )}
        </Card>
      ))}
    </div>
  )
}

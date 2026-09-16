import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Activity, DollarSign, Package, ShoppingBag, Users } from 'lucide-react'
import { getDashboard } from '@/api/endpoints'
import { AdminLogin } from './AdminLogin'
import { Badge, Card, ErrorState, SectionHeading, Spinner, Stat } from '@/components/ui'
import { RevenueChart } from '@/charts/RevenueChart'
import { SegmentChart } from '@/charts/SegmentChart'
import { FunnelChart } from '@/charts/FunnelChart'
import { SentimentBars } from '@/charts/SentimentBars'
import { compactNumber, currency, number, percent } from '@/utils/format'
import { useUser } from '@/store/UserContext'

const WINDOWS = [7, 30, 90]

export default function AdminDashboard() {
  const { isAdmin } = useUser()
  const [days, setDays] = useState(30)
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['dashboard', days],
    queryFn: () => getDashboard(days),
    enabled: isAdmin,
  })

  if (!isAdmin) return <AdminLogin onSuccess={() => refetch()} />
  if (isLoading) return <Spinner label="Loading analytics" />
  if (isError) return <ErrorState title="Could not load the dashboard"
                                  message={(error as Error).message} onRetry={() => refetch()} />

  const d = data!
  const { business, ai, customers, products, sentiment, event_funnel } = d

  return (
    <div className="space-y-6">
      <SectionHeading
        title="Business intelligence"
        subtitle={`All figures computed from operational data · generated ${new Date(d.generated_at).toLocaleString()}`}
        action={
          <div className="flex gap-1">
            {WINDOWS.map((w) => (
              <button key={w} type="button"
                      className={w === days ? 'btn-primary !px-3 !py-1.5' : 'btn-secondary !px-3 !py-1.5'}
                      onClick={() => setDays(w)}>{w}d</button>
            ))}
          </div>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        <Stat label="Revenue" value={currency(business.revenue)} change={business.revenue_change_pct}
              icon={<DollarSign className="h-4 w-4" />} />
        <Stat label="Orders" value={number(business.orders)} change={business.orders_change_pct}
              icon={<ShoppingBag className="h-4 w-4" />} />
        <Stat label="Avg order value" value={currency(business.average_order_value)}
              change={business.aov_change_pct} />
        <Stat label="Conversion" value={percent(business.conversion_rate)}
              hint={`${compactNumber(business.sessions)} sessions`} icon={<Activity className="h-4 w-4" />} />
        <Stat label="Active users" value={number(business.active_users)}
              hint={`${number(business.total_customers)} total`} icon={<Users className="h-4 w-4" />} />
        <Stat label="Retention" value={percent(business.retention_rate)}
              hint={`${number(business.repeat_buyers)} repeat buyers`} />
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <SectionHeading title="Revenue trend" subtitle={`Daily revenue over the last ${days} days`} />
          <RevenueChart data={d.revenue_timeseries} />
        </Card>
        <Card>
          <SectionHeading title="Conversion funnel" />
          <FunnelChart counts={event_funnel.counts} viewToCart={event_funnel.view_to_cart}
                       cartToPurchase={event_funnel.cart_to_purchase} />
        </Card>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <SectionHeading title="Recommendation effectiveness"
                          subtitle="Impression-level click and conversion tracking" />
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Impressions" value={compactNumber(ai.recommendation.impressions)} />
            <Stat label="CTR" value={percent(ai.recommendation.ctr, 2)} />
            <Stat label="Conversion" value={percent(ai.recommendation.conversion_rate, 2)} />
          </div>
          <table className="mt-4 w-full text-sm">
            <thead>
              <tr className="border-b border-surface-200 text-left text-xs uppercase text-slate-500">
                <th className="pb-2 font-medium">Strategy</th>
                <th className="pb-2 text-right font-medium">Impressions</th>
                <th className="pb-2 text-right font-medium">CTR</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-100">
              {ai.recommendation.by_strategy.map((row) => (
                <tr key={row.strategy}>
                  <td className="py-2 capitalize">{row.strategy.replace(/_/g, ' ')}</td>
                  <td className="py-2 text-right tabular-nums">{number(row.impressions)}</td>
                  <td className="py-2 text-right tabular-nums">{percent(row.ctr, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card>
          <SectionHeading title="Search effectiveness" />
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="Searches" value={compactNumber(ai.search.searches)} />
            <Stat label="CTR" value={percent(ai.search.ctr, 2)} />
            <Stat label="Abandonment" value={percent(ai.search.abandonment_rate, 2)} />
            <Stat label="Zero results" value={percent(ai.search.zero_result_rate, 2)} />
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Average search latency {ai.search.avg_latency_ms.toFixed(1)}ms · {number(ai.search.conversions)} searches converted
          </p>
          <div className="mt-4">
            <p className="label">Review sentiment across the catalogue</p>
            <SentimentBars distribution={sentiment.counts} />
            <p className="mt-2 text-xs text-slate-400">
              {number(sentiment.total_labelled)} of {number(sentiment.total_reviews)} reviews classified
            </p>
          </div>
        </Card>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <SectionHeading title="Customer segments" subtitle="K-Means over RFM and behavioural features" />
          {customers.segments.length > 0 ? (
            <SegmentChart data={customers.segments.map((s) => ({ name: s.name, count: s.count }))} />
          ) : (
            <p className="text-sm text-slate-500">Run the segmentation pipeline to populate segments.</p>
          )}
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge tone="success">High value: {number(customers.high_value_count)}</Badge>
            <Badge tone="danger">At risk: {number(customers.at_risk_count)}</Badge>
            <Badge tone="info">New this period: {number(customers.new_users)}</Badge>
          </div>
        </Card>

        <Card>
          <SectionHeading title="Top customers" subtitle="Ranked by lifetime spend" />
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-surface-200 text-left text-xs uppercase text-slate-500">
                <th className="pb-2 font-medium">Customer</th>
                <th className="pb-2 text-right font-medium">Orders</th>
                <th className="pb-2 text-right font-medium">Spend</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-100">
              {customers.top_customers.slice(0, 8).map((c) => (
                <tr key={c.user_id}>
                  <td className="max-w-[160px] truncate py-2">{c.name}</td>
                  <td className="py-2 text-right tabular-nums">{c.orders}</td>
                  <td className="py-2 text-right tabular-nums">{currency(c.total_spend)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>

      <div className="grid gap-5 lg:grid-cols-2 xl:grid-cols-4">
        {[
          { title: 'Best sellers', rows: products.best_sellers, metric: (r: { units_sold?: number }) => `${r.units_sold ?? 0} sold` },
          { title: 'Low inventory', rows: products.low_inventory, metric: (r: { inventory?: number }) => `${r.inventory ?? 0} left` },
          { title: 'High conversion', rows: products.high_conversion, metric: (r: { conversion_rate?: number }) => percent(r.conversion_rate ?? 0, 1) },
          { title: 'Underperforming', rows: products.poor_performers, metric: (r: { conversion_rate?: number }) => percent(r.conversion_rate ?? 0, 1) },
        ].map((panel) => (
          <Card key={panel.title}>
            <h3 className="mb-2 flex items-center gap-1.5 text-sm font-medium">
              <Package className="h-4 w-4 text-slate-400" aria-hidden />{panel.title}
            </h3>
            <ul className="space-y-1.5 text-xs">
              {panel.rows.slice(0, 6).map((row) => (
                <li key={row.id} className="flex items-start justify-between gap-2">
                  <span className="line-clamp-2 text-slate-600">{row.title}</span>
                  <span className="shrink-0 tabular-nums text-slate-400">{panel.metric(row as never)}</span>
                </li>
              ))}
              {panel.rows.length === 0 && <li className="text-slate-400">No data</li>}
            </ul>
          </Card>
        ))}
      </div>

      <Card>
        <SectionHeading title="Revenue by category" />
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-surface-200 text-left text-xs uppercase text-slate-500">
              <th className="pb-2 font-medium">Category</th>
              <th className="pb-2 text-right font-medium">Products</th>
              <th className="pb-2 text-right font-medium">Revenue</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-surface-100">
            {products.category_breakdown.map((row) => (
              <tr key={row.slug}>
                <td className="py-2">{row.category}</td>
                <td className="py-2 text-right tabular-nums">{row.products}</td>
                <td className="py-2 text-right tabular-nums">{currency(row.revenue)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { currency } from '@/utils/format'

interface Props {
  data: { date: string; revenue: number; orders: number }[]
}

export function RevenueChart({ data }: Props) {
  const rows = data.map((d) => ({ ...d, label: d.date.slice(5) }))
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={rows} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
          <defs>
            <linearGradient id="revenueFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3563ff" stopOpacity={0.28} />
              <stop offset="100%" stopColor="#3563ff" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false}
                 minTickGap={24} />
          <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={56}
                 tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`} />
          <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }}
                   formatter={(value: number, name: string) =>
                     name === 'revenue' ? [currency(value), 'Revenue'] : [value, 'Orders']} />
          <Area type="monotone" dataKey="revenue" stroke="#3563ff" strokeWidth={2} fill="url(#revenueFill)"
                isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

import {
  Area, AreaChart, CartesianGrid, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { ForecastPoint } from '@/types'

interface Props {
  points: ForecastPoint[]
  history?: { date: string; units: number }[]
}

/** Actual demand followed by the forecast path with its 95% interval. */
export function ForecastChart({ points, history = [] }: Props) {
  const recentHistory = history.slice(-21).map((h) => ({
    label: h.date.slice(5),
    actual: h.units,
  }))
  const forecastRows = points.slice(0, 30).map((p) => ({
    label: (p.date ?? `D${p.day}`).slice(5),
    predicted: p.predicted,
    lower: p.lower,
    band: Math.max(p.upper - p.lower, 0),
  }))
  const data = [...recentHistory, ...forecastRows]

  if (data.length === 0) return <p className="text-xs text-slate-400">No forecast data.</p>

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 10, fill: '#94a3b8' }} interval="preserveStartEnd"
                 tickLine={false} axisLine={false} minTickGap={24} />
          <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={40} />
          <Tooltip
            contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }}
            formatter={(value: number, name: string) => [value.toFixed(2), name]}
          />
          {/* Interval rendered as a transparent base + visible band stacked on top. */}
          <Area type="monotone" dataKey="lower" stackId="ci" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area type="monotone" dataKey="band" stackId="ci" stroke="none" fill="#3563ff" fillOpacity={0.12}
                name="95% interval" isAnimationActive={false} />
          <Line type="monotone" dataKey="actual" stroke="#0f172a" strokeWidth={2} dot={false}
                name="Actual" isAnimationActive={false} />
          <Line type="monotone" dataKey="predicted" stroke="#3563ff" strokeWidth={2} strokeDasharray="4 3"
                dot={false} name="Forecast" isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

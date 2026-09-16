export const currency = (value: number, code = 'USD'): string =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: code, maximumFractionDigits: 2 }).format(
    Number.isFinite(value) ? value : 0,
  )

export const compactNumber = (value: number): string =>
  new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(
    Number.isFinite(value) ? value : 0,
  )

export const percent = (value: number, digits = 1): string =>
  `${((Number.isFinite(value) ? value : 0) * 100).toFixed(digits)}%`

export const number = (value: number): string =>
  new Intl.NumberFormat('en-US').format(Number.isFinite(value) ? value : 0)

export const shortDate = (value: string | null | undefined): string => {
  if (!value) return '-'
  const date = new Date(value)
  return Number.isNaN(date.getTime())
    ? '-'
    : date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

export const relativeTime = (value: string | null | undefined): string => {
  if (!value) return 'never'
  const then = new Date(value).getTime()
  if (Number.isNaN(then)) return 'never'
  const diffDays = Math.round((then - Date.now()) / 86_400_000)
  const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' })
  if (Math.abs(diffDays) < 1) return 'today'
  if (Math.abs(diffDays) < 30) return rtf.format(diffDays, 'day')
  return rtf.format(Math.round(diffDays / 30), 'month')
}

export const titleCase = (value: string): string =>
  value.replace(/[_-]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

/** Deterministic pastel gradient per product, used as an image placeholder. */
export const productGradient = (seed: number): string => {
  const hue = (seed * 47) % 360
  return `linear-gradient(135deg, hsl(${hue} 70% 92%), hsl(${(hue + 40) % 360} 65% 84%))`
}

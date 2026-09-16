import clsx from 'clsx'
import { AlertCircle, Inbox, Loader2, RefreshCw } from 'lucide-react'
import type { ReactNode } from 'react'

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={clsx('card p-5', className)}>{children}</div>
}

export function SectionHeading({ title, subtitle, action }: {
  title: string; subtitle?: string; action?: ReactNode
}) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h2 className="text-lg font-semibold tracking-tight text-surface-900 sm:text-xl">{title}</h2>
        {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
      </div>
      {action}
    </div>
  )
}

export function Badge({ children, tone = 'neutral', className }: {
  children: ReactNode
  tone?: 'neutral' | 'success' | 'warning' | 'danger' | 'brand' | 'info'
  className?: string
}) {
  const tones = {
    neutral: 'bg-surface-100 text-slate-700',
    success: 'bg-emerald-50 text-emerald-700',
    warning: 'bg-amber-50 text-amber-700',
    danger: 'bg-rose-50 text-rose-700',
    brand: 'bg-brand-50 text-brand-700',
    info: 'bg-sky-50 text-sky-700',
  }
  return <span className={clsx('badge', tones[tone], className)}>{children}</span>
}

export function Spinner({ label = 'Loading', className }: { label?: string; className?: string }) {
  return (
    <div className={clsx('flex items-center justify-center gap-2 py-10 text-sm text-slate-500', className)}
         role="status" aria-live="polite">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
      <span>{label}...</span>
    </div>
  )
}

export function SkeletonCard() {
  return (
    <div className="card overflow-hidden">
      <div className="skeleton aspect-[4/3] w-full" />
      <div className="space-y-2 p-4">
        <div className="skeleton h-3 w-3/4" />
        <div className="skeleton h-3 w-1/2" />
        <div className="skeleton h-5 w-1/3" />
      </div>
    </div>
  )
}

export function SkeletonGrid({ count = 6 }: { count?: number }) {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: count }, (_, i) => <SkeletonCard key={i} />)}
    </div>
  )
}

export function ErrorState({ title = 'Something went wrong', message, onRetry }: {
  title?: string; message?: string; onRetry?: () => void
}) {
  return (
    <div className="card flex flex-col items-center gap-3 p-8 text-center" role="alert">
      <AlertCircle className="h-8 w-8 text-rose-500" aria-hidden />
      <div>
        <p className="font-medium text-surface-900">{title}</p>
        {message && <p className="mt-1 max-w-md text-sm text-slate-500">{message}</p>}
      </div>
      {onRetry && (
        <button type="button" className="btn-secondary" onClick={onRetry}>
          <RefreshCw className="h-4 w-4" aria-hidden /> Try again
        </button>
      )}
    </div>
  )
}

export function EmptyState({ title, message, action }: {
  title: string; message?: string; action?: ReactNode
}) {
  return (
    <div className="card flex flex-col items-center gap-3 p-10 text-center">
      <Inbox className="h-8 w-8 text-slate-300" aria-hidden />
      <div>
        <p className="font-medium text-surface-900">{title}</p>
        {message && <p className="mt-1 max-w-md text-sm text-slate-500">{message}</p>}
      </div>
      {action}
    </div>
  )
}

export function Stat({ label, value, change, hint, icon }: {
  label: string; value: string; change?: number | null; hint?: string; icon?: ReactNode
}) {
  const positive = (change ?? 0) >= 0
  return (
    <div className="card p-4 sm:p-5">
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
        {icon && <span className="text-slate-400">{icon}</span>}
      </div>
      <p className="stat-value mt-2 text-surface-900">{value}</p>
      <div className="mt-1 flex items-center gap-2">
        {change !== null && change !== undefined && (
          <span className={clsx('text-xs font-medium', positive ? 'text-emerald-600' : 'text-rose-600')}>
            {positive ? '+' : ''}{change.toFixed(1)}%
          </span>
        )}
        {hint && <span className="text-xs text-slate-400">{hint}</span>}
      </div>
    </div>
  )
}

export function Rating({ value, count, size = 'sm' }: { value: number; count?: number; size?: 'sm' | 'md' }) {
  const filled = Math.round(value)
  return (
    <div className="flex items-center gap-1" aria-label={`Rated ${value.toFixed(1)} out of 5`}>
      <div className={clsx('flex', size === 'sm' ? 'text-xs' : 'text-sm')} aria-hidden>
        {Array.from({ length: 5 }, (_, i) => (
          <span key={i} className={i < filled ? 'text-amber-400' : 'text-slate-200'}>★</span>
        ))}
      </div>
      <span className={clsx('tabular-nums text-slate-500', size === 'sm' ? 'text-xs' : 'text-sm')}>
        {value.toFixed(1)}{count !== undefined && ` (${count})`}
      </span>
    </div>
  )
}

export function Tabs<T extends string>({ tabs, active, onChange }: {
  tabs: { id: T; label: string }[]; active: T; onChange: (id: T) => void
}) {
  return (
    <div className="flex gap-1 overflow-x-auto border-b border-surface-200" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={active === tab.id}
          onClick={() => onChange(tab.id)}
          className={clsx(
            'whitespace-nowrap border-b-2 px-3 py-2 text-sm font-medium transition-colors',
            active === tab.id
              ? 'border-brand-600 text-brand-700'
              : 'border-transparent text-slate-500 hover:text-surface-900',
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

export function InfoNote({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
      {children}
    </div>
  )
}

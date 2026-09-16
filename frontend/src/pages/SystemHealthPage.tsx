import { useQuery } from '@tanstack/react-query'
import { Activity, Database, HardDrive, Server } from 'lucide-react'
import { getFeatureCoverage, getHealth, getMlSummary } from '@/api/endpoints'
import { AdminLogin } from './AdminLogin'
import { Badge, Card, ErrorState, SectionHeading, Spinner } from '@/components/ui'
import { number, titleCase } from '@/utils/format'
import { useUser } from '@/store/UserContext'

const TONE: Record<string, 'success' | 'warning' | 'danger'> = {
  ok: 'success', alive: 'success', ready: 'success', healthy: 'success',
  degraded: 'warning', partial: 'warning', unavailable: 'warning',
  error: 'danger', not_ready: 'danger',
}

export default function SystemHealthPage() {
  const { isAdmin } = useUser()

  // Health is public, so it renders even before signing in.
  const health = useQuery({ queryKey: ['health'], queryFn: getHealth, refetchInterval: 15_000 })
  const mlSummary = useQuery({ queryKey: ['ml-summary'], queryFn: getMlSummary, enabled: isAdmin })
  const features = useQuery({ queryKey: ['feature-coverage'], queryFn: getFeatureCoverage, enabled: isAdmin })

  if (health.isLoading) return <Spinner label="Checking system health" />
  if (health.isError) {
    return <ErrorState title="The API is unreachable"
                       message="The backend did not respond. Confirm it is running on port 8000."
                       onRetry={() => health.refetch()} />
  }

  const data = health.data!
  const summary = mlSummary.data as {
    models_total?: number; models_loaded?: number; models_missing?: string[]
    models_degraded?: string[]; drift_detected?: string[]; registry_size?: number
    sampled_predictions?: number
  } | undefined
  const coverage = features.data as {
    users_with_features?: number; products_with_features?: number
    user_feature_names?: string[]; product_feature_names?: string[]
  } | undefined

  const uptimeHours = Math.floor(data.uptime_seconds / 3600)
  const uptimeMinutes = Math.floor((data.uptime_seconds % 3600) / 60)

  return (
    <div className="space-y-6">
      <SectionHeading title="System health"
                      subtitle="Live dependency status, refreshed every 15 seconds"
                      action={<Badge tone={TONE[data.status] ?? 'warning'}>{data.status}</Badge>} />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="!p-4">
          <div className="flex items-center gap-2"><Server className="h-4 w-4 text-slate-400" aria-hidden />
            <p className="text-sm font-medium">API</p></div>
          <Badge tone={TONE[data.status] ?? 'warning'} className="mt-2">{data.status}</Badge>
          <p className="mt-2 text-xs text-slate-500">
            v{data.version} · {data.environment}<br />uptime {uptimeHours}h {uptimeMinutes}m
          </p>
        </Card>

        {Object.entries(data.checks).map(([name, check]) => {
          const status = String(check.status)
          const Icon = name === 'database' ? Database : name === 'cache' ? HardDrive : Activity
          return (
            <Card key={name} className="!p-4">
              <div className="flex items-center gap-2"><Icon className="h-4 w-4 text-slate-400" aria-hidden />
                <p className="text-sm font-medium">{titleCase(name)}</p></div>
              <Badge tone={TONE[status] ?? 'warning'} className="mt-2">{status}</Badge>
              <div className="mt-2 space-y-0.5 text-xs text-slate-500">
                {'dialect' in check && <p>Dialect: {String(check.dialect)}</p>}
                {'backend' in check && <p>Backend: {String(check.backend)}</p>}
                {'available' in check && <p>{String(check.available)} of {String(check.expected)} models</p>}
                {'required' in check && !check.required && <p className="text-slate-400">Optional dependency</p>}
              </div>
            </Card>
          )
        })}
      </div>

      {data.checks.models && typeof data.checks.models.detail === 'object' && (
        <Card>
          <SectionHeading title="Model availability" />
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
            {Object.entries(data.checks.models.detail as Record<string, boolean>).map(([model, available]) => (
              <div key={model} className="flex items-center justify-between rounded-lg border border-surface-200 px-3 py-2">
                <span className="truncate text-sm">{titleCase(model)}</span>
                <Badge tone={available ? 'success' : 'danger'}>{available ? 'trained' : 'missing'}</Badge>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Missing models degrade gracefully: the storefront falls back to popularity-based results.
            Run <code className="rounded bg-surface-100 px-1">make train</code> to train them.
          </p>
        </Card>
      )}

      {!isAdmin && (
        <Card>
          <p className="text-sm text-slate-500">
            Sign in as an administrator to see feature-store coverage and ML system totals.
          </p>
          <div className="mt-3"><AdminLogin /></div>
        </Card>
      )}

      {isAdmin && summary && (
        <div className="grid gap-5 lg:grid-cols-2">
          <Card>
            <SectionHeading title="ML system summary" />
            <dl className="space-y-2 text-sm">
              <div className="flex justify-between"><dt className="text-slate-500">Models loaded</dt>
                <dd className="tabular-nums">{summary.models_loaded} / {summary.models_total}</dd></div>
              <div className="flex justify-between"><dt className="text-slate-500">Registry entries</dt>
                <dd className="tabular-nums">{number(summary.registry_size ?? 0)}</dd></div>
              <div className="flex justify-between"><dt className="text-slate-500">Sampled predictions</dt>
                <dd className="tabular-nums">{number(summary.sampled_predictions ?? 0)}</dd></div>
              <div className="flex justify-between"><dt className="text-slate-500">Missing</dt>
                <dd>{summary.models_missing?.length ? summary.models_missing.join(', ') : 'none'}</dd></div>
              <div className="flex justify-between"><dt className="text-slate-500">Drift detected</dt>
                <dd>{summary.drift_detected?.length ? summary.drift_detected.join(', ') : 'none'}</dd></div>
            </dl>
          </Card>

          {coverage && (
            <Card>
              <SectionHeading title="Feature store coverage" />
              <dl className="space-y-2 text-sm">
                <div className="flex justify-between"><dt className="text-slate-500">Users with features</dt>
                  <dd className="tabular-nums">{number(coverage.users_with_features ?? 0)}</dd></div>
                <div className="flex justify-between"><dt className="text-slate-500">Products with features</dt>
                  <dd className="tabular-nums">{number(coverage.products_with_features ?? 0)}</dd></div>
              </dl>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {[...(coverage.user_feature_names ?? []), ...(coverage.product_feature_names ?? [])]
                  .slice(0, 16).map((name) => <Badge key={name} tone="neutral">{name}</Badge>)}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}

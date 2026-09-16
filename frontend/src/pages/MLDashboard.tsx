import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { AlertTriangle, CheckCircle2, RefreshCw, XCircle } from 'lucide-react'
import { getDrift, getModelHealth, getModelRegistry } from '@/api/endpoints'
import { reloadModels } from '@/api/endpoints'
import { AdminLogin } from './AdminLogin'
import { Badge, Card, ErrorState, SectionHeading, Spinner } from '@/components/ui'
import { number, shortDate, titleCase } from '@/utils/format'
import { useUser } from '@/store/UserContext'
import type { ModelRegistryEntry } from '@/types'

const METRIC_LABELS: Record<string, string> = {
  'ndcg@10': 'NDCG@10', 'precision@10': 'Precision@10', 'recall@20': 'Recall@20', 'map@10': 'MAP@10',
  accuracy: 'Accuracy', f1_macro: 'F1 (macro)', silhouette_score: 'Silhouette', mae: 'MAE', rmse: 'RMSE',
  mape: 'MAPE', r2: 'R²', within_20pct: 'Within 20%', aspect_detection_f1: 'Aspect F1',
  lift_vs_popularity_ndcg10: 'Lift vs popularity',
}

const HEADLINE: Record<string, string[]> = {
  recommendation: ['ndcg@10', 'precision@10', 'recall@20', 'catalogue_coverage@10'],
  sentiment: ['accuracy', 'f1_macro', 'aspect_detection_f1', 'aspect_polarity_accuracy'],
  segmentation: ['silhouette_score', 'n_clusters', 'davies_bouldin_score'],
  forecasting: ['mae', 'rmse', 'smape', 'improvement_vs_baseline_mae'],
  price_prediction: ['r2', 'mape', 'within_20pct', 'improvement_vs_baseline_mae'],
}

function formatMetric(value: unknown): string {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4)
  if (value === null || value === undefined) return '—'
  if (typeof value === 'object') return '—'
  return String(value)
}

export default function MLDashboard() {
  const { isAdmin } = useUser()
  const queryClient = useQueryClient()
  const [reloading, setReloading] = useState(false)

  const registry = useQuery({ queryKey: ['ml-registry'], queryFn: getModelRegistry, enabled: isAdmin })
  const health = useQuery({ queryKey: ['ml-health'], queryFn: getModelHealth, enabled: isAdmin,
                            refetchInterval: 30_000 })
  const drift = useQuery({ queryKey: ['ml-drift'], queryFn: getDrift, enabled: isAdmin })

  if (!isAdmin) return <AdminLogin onSuccess={() => registry.refetch()} />
  if (registry.isLoading || health.isLoading) return <Spinner label="Loading model registry" />
  if (registry.isError) {
    return <ErrorState title="Could not load the registry" message={(registry.error as Error).message}
                       onRetry={() => registry.refetch()} />
  }

  const handleReload = async () => {
    setReloading(true)
    try {
      await reloadModels()
      await queryClient.invalidateQueries({ queryKey: ['ml-registry'] })
      await queryClient.invalidateQueries({ queryKey: ['ml-health'] })
    } finally {
      setReloading(false)
    }
  }

  // Newest version per model name (the registry is already sorted newest-first).
  const latest = new Map<string, ModelRegistryEntry>()
  for (const entry of registry.data ?? []) {
    if (!latest.has(entry.name)) latest.set(entry.name, entry)
  }

  return (
    <div className="space-y-6">
      <SectionHeading
        title="ML model monitoring"
        subtitle="Registry, live serving statistics and drift detection"
        action={
          <button type="button" className="btn-secondary" onClick={handleReload} disabled={reloading}>
            <RefreshCw className={`h-4 w-4 ${reloading ? 'animate-spin' : ''}`} aria-hidden />
            {reloading ? 'Reloading' : 'Reload models'}
          </button>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {(health.data ?? []).map((model) => (
          <Card key={model.model} className="!p-4">
            <div className="flex items-start justify-between gap-2">
              <p className="text-sm font-medium">{titleCase(model.model)}</p>
              {model.status === 'healthy' ? <CheckCircle2 className="h-4 w-4 text-emerald-500" aria-hidden />
                : model.status === 'degraded' ? <AlertTriangle className="h-4 w-4 text-amber-500" aria-hidden />
                : <XCircle className="h-4 w-4 text-rose-400" aria-hidden />}
            </div>
            <Badge tone={model.status === 'healthy' ? 'success' : model.status === 'degraded' ? 'warning' : 'danger'}
                   className="mt-2">
              {model.status}
            </Badge>
            {model.loaded ? (
              <dl className="mt-3 space-y-1 text-xs text-slate-500">
                <div className="flex justify-between"><dt>Calls</dt>
                  <dd className="tabular-nums">{number(model.calls ?? 0)}</dd></div>
                <div className="flex justify-between"><dt>Avg latency</dt>
                  <dd className="tabular-nums">{(model.avg_latency_ms ?? 0).toFixed(2)}ms</dd></div>
                <div className="flex justify-between"><dt>p95</dt>
                  <dd className="tabular-nums">{(model.p95_latency_ms ?? 0).toFixed(2)}ms</dd></div>
                <div className="flex justify-between"><dt>Error rate</dt>
                  <dd className="tabular-nums">{((model.error_rate ?? 0) * 100).toFixed(2)}%</dd></div>
              </dl>
            ) : (
              <p className="mt-2 text-xs text-slate-500">{model.hint ?? 'Model not loaded'}</p>
            )}
          </Card>
        ))}
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        {[...latest.values()].map((entry) => (
          <Card key={`${entry.name}-${entry.version}`}>
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <h3 className="font-medium">{titleCase(entry.name)}</h3>
                <p className="mt-0.5 font-mono text-xs text-slate-500">{entry.version}</p>
              </div>
              <div className="flex items-center gap-1.5">
                <Badge tone={entry.stage === 'production' ? 'success' : 'neutral'}>{entry.stage}</Badge>
                {entry.is_loaded && <Badge tone="brand">loaded</Badge>}
              </div>
            </div>

            <p className="mt-2 text-xs leading-relaxed text-slate-500">{entry.algorithm}</p>

            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {(HEADLINE[entry.name] ?? Object.keys(entry.metrics).slice(0, 4)).map((key) => (
                <div key={key} className="rounded-lg bg-surface-100 p-2">
                  <p className="truncate text-[10px] uppercase text-slate-500">
                    {METRIC_LABELS[key] ?? key.replace(/_/g, ' ')}
                  </p>
                  <p className="text-sm font-semibold tabular-nums">
                    {formatMetric((entry.metrics as Record<string, unknown>)[key])}
                  </p>
                </div>
              ))}
            </div>

            <dl className="mt-3 space-y-1 text-xs text-slate-500">
              <div className="flex justify-between"><dt>Trained</dt><dd>{shortDate(entry.trained_at)}</dd></div>
              <div className="flex justify-between"><dt>Training rows</dt>
                <dd className="tabular-nums">{number(entry.training_rows)}</dd></div>
              <div className="flex justify-between"><dt>Duration</dt>
                <dd className="tabular-nums">{entry.training_duration_s.toFixed(1)}s</dd></div>
              <div className="flex justify-between"><dt>Dataset</dt>
                <dd className="font-mono">{entry.dataset_version}</dd></div>
            </dl>

            {entry.notes && <p className="mt-3 border-t border-surface-100 pt-2 text-xs leading-relaxed text-slate-500">{entry.notes}</p>}
          </Card>
        ))}
      </div>

      <Card>
        <SectionHeading title="Drift detection"
                        subtitle="Live feature distributions compared against training-time baselines" />
        {drift.isLoading && <Spinner label="Checking drift" />}
        {drift.data && drift.data.length === 0 && (
          <p className="text-sm text-slate-500">No models loaded to check.</p>
        )}
        <div className="space-y-3">
          {(drift.data ?? []).map((report) => (
            <div key={report.model} className="rounded-lg border border-surface-200 p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium">{titleCase(report.model)}</span>
                <Badge tone={report.status === 'drift_detected' ? 'warning' : 'success'}>
                  {report.status === 'drift_detected' ? 'Drift detected' : 'Stable'}
                </Badge>
              </div>
              {report.checks.length > 0 && (
                <table className="mt-2 w-full text-xs">
                  <thead>
                    <tr className="text-left text-slate-400">
                      <th className="pb-1 font-medium">Feature</th>
                      <th className="pb-1 text-right font-medium">Baseline</th>
                      <th className="pb-1 text-right font-medium">Current</th>
                      <th className="pb-1 text-right font-medium">Change</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-surface-100">
                    {report.checks.map((check, i) => (
                      <tr key={`${check.feature}-${i}`}>
                        <td className="py-1 text-slate-600">{check.feature}</td>
                        <td className="py-1 text-right tabular-nums">{formatMetric(check.baseline)}</td>
                        <td className="py-1 text-right tabular-nums">{formatMetric(check.current)}</td>
                        <td className={`py-1 text-right tabular-nums ${check.drift_detected ? 'text-amber-600' : 'text-slate-400'}`}>
                          {check.relative_change !== undefined ? `${(check.relative_change * 100).toFixed(1)}%` : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {report.prediction_stats && (
                <p className="mt-2 text-xs text-slate-500">
                  Prediction distribution over {report.prediction_stats.samples} samples · mean{' '}
                  {report.prediction_stats.mean.toFixed(4)} · std {report.prediction_stats.std.toFixed(4)}
                </p>
              )}
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getUserProfile, getUserSegment } from '@/api/endpoints'
import { Badge, Card, ErrorState, SectionHeading, Spinner, Stat } from '@/components/ui'
import { currency, number, percent, relativeTime, titleCase } from '@/utils/format'
import { useUser } from '@/store/UserContext'

export default function ProfilePage() {
  const { userId, setUserId } = useUser()
  const [draftId, setDraftId] = useState(String(userId))

  const profile = useQuery({ queryKey: ['profile', userId], queryFn: () => getUserProfile(userId) })
  const segment = useQuery({ queryKey: ['segment', userId], queryFn: () => getUserSegment(userId) })

  if (profile.isLoading) return <Spinner label="Loading profile" />
  if (profile.isError) {
    return <ErrorState title="Could not load profile" message={(profile.error as Error).message}
                       onRetry={() => profile.refetch()} />
  }

  const data = profile.data!
  const segmentData = segment.data as { segment?: string | null; description?: string
    confidence?: number; features?: Record<string, number> } | undefined

  return (
    <div className="space-y-6">
      <SectionHeading
        title="Your profile"
        subtitle="The behavioural profile that personalises search, the homepage and recommendations"
        action={
          <form
            className="flex items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault()
              const parsed = Number.parseInt(draftId, 10)
              if (Number.isFinite(parsed) && parsed > 0) setUserId(parsed)
            }}
          >
            <label className="text-xs text-slate-500" htmlFor="user-switch">Shop as user</label>
            <input id="user-switch" className="input !w-24" type="number" min={1} value={draftId}
                   onChange={(e) => setDraftId(e.target.value)} />
            <button type="submit" className="btn-secondary">Switch</button>
          </form>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Orders" value={number(data.order_count)} />
        <Stat label="Total spend" value={currency(data.total_spend)} />
        <Stat label="Avg order value" value={currency(data.avg_order_value)} />
        <Stat label="Sessions" value={number(data.session_count)} hint={`last active ${relativeTime(data.last_active)}`} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <h3 className="mb-3 font-medium">Segment</h3>
          {segmentData?.segment ? (
            <div className="space-y-2">
              <Badge tone="brand">{titleCase(segmentData.segment)}</Badge>
              <p className="text-sm text-slate-600">{segmentData.description}</p>
              {segmentData.confidence !== undefined && (
                <p className="text-xs text-slate-400">
                  Assignment confidence {percent(segmentData.confidence)}
                </p>
              )}
            </div>
          ) : (
            <p className="text-sm text-slate-500">
              No segment assigned yet. Run the segmentation pipeline to populate this.
            </p>
          )}
        </Card>

        <Card>
          <h3 className="mb-3 font-medium">Shopping signals</h3>
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between"><dt className="text-slate-500">Interactions tracked</dt>
              <dd className="tabular-nums">{number(data.interaction_count)}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Reviews written</dt>
              <dd className="tabular-nums">{number(data.review_count)}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Price sensitivity</dt>
              <dd className="tabular-nums">{percent(data.price_sensitivity)}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Preferred price band</dt>
              <dd className="tabular-nums">
                {currency(data.preferred_price_range[0] ?? 0)} – {currency(data.preferred_price_range[1] ?? 0)}
              </dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Personalization state</dt>
              <dd>{data.is_cold_start ? <Badge tone="warning">Cold start</Badge> : <Badge tone="success">Active</Badge>}</dd>
            </div>
          </dl>
        </Card>

        <Card>
          <h3 className="mb-3 font-medium">Category affinity</h3>
          {Object.keys(data.category_affinity).length === 0 ? (
            <p className="text-sm text-slate-500">Browse a few products to build category affinity.</p>
          ) : (
            <ul className="space-y-2">
              {Object.entries(data.category_affinity).slice(0, 8).map(([categoryId, weight]) => (
                <li key={categoryId}>
                  <div className="mb-1 flex justify-between text-xs">
                    <span className="text-slate-600">Category #{categoryId}</span>
                    <span className="tabular-nums text-slate-400">{percent(weight)}</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-surface-100">
                    <div className="h-full rounded-full bg-brand-500" style={{ width: `${weight * 100}%` }} />
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card>
          <h3 className="mb-3 font-medium">Recent searches</h3>
          {data.search_terms.length === 0 ? (
            <p className="text-sm text-slate-500">No searches recorded yet.</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {data.search_terms.slice(0, 12).map((term, i) => (
                <Badge key={`${term}-${i}`} tone="neutral">{term}</Badge>
              ))}
            </div>
          )}
          <h3 className="mb-2 mt-4 font-medium">Preference vector</h3>
          <div className="flex h-10 items-end gap-0.5" aria-label="Dense preference vector">
            {data.preference_vector.map((value, i) => (
              <div key={i} className="flex-1 rounded-sm bg-brand-400"
                   style={{ height: `${Math.max(Math.abs(value) * 100, 3)}%` }} />
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}

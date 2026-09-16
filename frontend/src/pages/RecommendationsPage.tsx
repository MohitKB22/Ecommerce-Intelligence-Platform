import { useQuery } from '@tanstack/react-query'
import { getContinueShopping, getDeals, getRecentlyViewed, getRecommendations, getTrending } from '@/api/endpoints'
import { ProductCard } from '@/components/ProductCard'
import { Badge, Card, ErrorState, SectionHeading, SkeletonGrid } from '@/components/ui'
import { useUser } from '@/store/UserContext'

export default function RecommendationsPage() {
  const { userId } = useUser()
  const personalized = useQuery({
    queryKey: ['recs', userId],
    queryFn: () => getRecommendations(userId, 16),
  })
  const trending = useQuery({ queryKey: ['trending-page'], queryFn: () => getTrending(8) })
  const deals = useQuery({ queryKey: ['deals-page'], queryFn: () => getDeals(8) })
  const recent = useQuery({ queryKey: ['recent-page', userId], queryFn: () => getRecentlyViewed(userId, 8) })
  const resume = useQuery({ queryKey: ['resume-page', userId], queryFn: () => getContinueShopping(userId, 8) })

  return (
    <div className="space-y-8">
      <SectionHeading
        title="Recommended for you"
        subtitle="Ranked by the hybrid model: collaborative filtering, content similarity, popularity, personalization and business signals"
        action={personalized.data && (
          <div className="flex items-center gap-2">
            <Badge tone="info">{personalized.data.strategy.replace(/_/g, ' ')}</Badge>
            <Badge tone="neutral">v{personalized.data.model_version.slice(0, 12)}</Badge>
          </div>
        )}
      />

      {personalized.isLoading && <SkeletonGrid count={8} />}
      {personalized.isError && (
        <ErrorState message={(personalized.error as Error).message} onRetry={() => personalized.refetch()} />
      )}
      {personalized.data && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-4">
          {personalized.data.items.map((item) => (
            <ProductCard key={item.product.id} product={item.product} explanation={item.explanation} />
          ))}
        </div>
      )}

      {personalized.data && personalized.data.items.length > 0 && (
        <Card>
          <h3 className="mb-3 font-medium">How the top pick was scored</h3>
          <div className="space-y-2">
            {Object.entries(personalized.data.items[0].components).map(([name, value]) => (
              <div key={name}>
                <div className="mb-1 flex justify-between text-xs">
                  <span className="capitalize text-slate-600">{name.replace(/_/g, ' ')}</span>
                  <span className="tabular-nums text-slate-400">{(value * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-surface-100">
                  <div className="h-full rounded-full bg-brand-500"
                       style={{ width: `${Math.min(Math.max(value, 0), 1) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-slate-500">{personalized.data.items[0].explanation}</p>
        </Card>
      )}

      {[
        { key: 'resume', title: 'Continue shopping', query: resume },
        { key: 'recent', title: 'Recently viewed', query: recent },
        { key: 'trending', title: 'Trending now', query: trending },
        { key: 'deals', title: "Today's deals", query: deals },
      ].map(({ key, title, query }) => (
        query.data && query.data.items.length > 0 ? (
          <section key={key}>
            <SectionHeading title={title} />
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              {query.data.items.slice(0, 4).map((item) => (
                <ProductCard key={item.product.id} product={item.product} explanation={item.explanation} />
              ))}
            </div>
          </section>
        ) : null
      ))}
    </div>
  )
}

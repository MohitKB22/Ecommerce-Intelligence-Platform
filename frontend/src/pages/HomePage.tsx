import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Sparkles, TrendingUp } from 'lucide-react'
import { getHomepage, getPopularCategories } from '@/api/endpoints'
import { ProductRail } from '@/components/ProductCard'
import { Badge, EmptyState, ErrorState, SectionHeading, SkeletonGrid, Spinner } from '@/components/ui'
import { useUser } from '@/store/UserContext'
import { titleCase } from '@/utils/format'

export default function HomePage() {
  const { userId } = useUser()
  const home = useQuery({
    queryKey: ['homepage', userId],
    queryFn: () => getHomepage(userId, 10),
  })
  const categories = useQuery({
    queryKey: ['popular-categories'],
    queryFn: () => getPopularCategories(10),
    staleTime: 10 * 60_000,
  })

  if (home.isLoading) {
    return (
      <div className="space-y-8">
        <div className="skeleton h-32 w-full rounded-xl" />
        <SkeletonGrid count={8} />
      </div>
    )
  }

  if (home.isError) {
    return (
      <ErrorState
        title="Could not load your homepage"
        message={(home.error as Error).message}
        onRetry={() => home.refetch()}
      />
    )
  }

  const data = home.data!

  return (
    <div className="space-y-10">
      <section className="rounded-2xl bg-gradient-to-br from-brand-600 to-brand-800 px-6 py-8 text-white sm:px-10 sm:py-12">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="brand" className="!bg-white/15 !text-white">
            <Sparkles className="h-3 w-3" aria-hidden />
            {data.is_personalized ? 'Personalized for you' : 'Popular picks'}
          </Badge>
          {data.segment && (
            <Badge tone="brand" className="!bg-white/15 !text-white">
              Segment: {titleCase(data.segment)}
            </Badge>
          )}
        </div>
        <h1 className="mt-4 max-w-2xl text-2xl font-semibold tracking-tight sm:text-4xl">
          {data.is_personalized
            ? 'Products chosen from your browsing and purchase history'
            : 'Discover what shoppers are buying right now'}
        </h1>
        <p className="mt-2 max-w-xl text-sm text-brand-100 sm:text-base">
          Every rail below is ranked by the hybrid recommendation model. Hover any card to see why it
          was surfaced.
        </p>
      </section>

      {categories.data && categories.data.length > 0 && (
        <section>
          <SectionHeading title="Popular categories" subtitle="Browse the catalogue by department" />
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
            {categories.data.map((category) => (
              <Link
                key={category.id}
                to={`/search?category=${encodeURIComponent(category.slug)}`}
                className="card card-hover p-4"
              >
                <p className="truncate text-sm font-medium text-surface-900">{category.name}</p>
                <p className="mt-1 text-xs text-slate-500">{category.product_count} products</p>
                <p className="mt-2 text-xs text-amber-500">★ {category.avg_rating.toFixed(1)}</p>
              </Link>
            ))}
          </div>
        </section>
      )}

      {data.sections.length === 0 && (
        <EmptyState
          title="No recommendations yet"
          message="Browse a few products and the homepage will start adapting to you."
          action={<Link className="btn-primary" to="/search">Browse the catalogue</Link>}
        />
      )}

      {data.sections.map((section) => (
        <section key={section.key}>
          <SectionHeading
            title={section.title}
            subtitle={section.subtitle || undefined}
            action={
              <Badge tone="neutral">
                <TrendingUp className="h-3 w-3" aria-hidden />
                {section.strategy.replace(/_/g, ' ')}
              </Badge>
            }
          />
          <ProductRail items={section.items} />
        </section>
      ))}

      {home.isFetching && <Spinner label="Refreshing" />}
    </div>
  )
}

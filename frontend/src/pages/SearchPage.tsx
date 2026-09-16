import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { SlidersHorizontal, Sparkles } from 'lucide-react'
import { search } from '@/api/endpoints'
import { ProductCard } from '@/components/ProductCard'
import { Badge, Card, EmptyState, ErrorState, SkeletonGrid } from '@/components/ui'
import { currency } from '@/utils/format'
import { useUser } from '@/store/UserContext'

const SORTS = [
  { value: 'relevance', label: 'Relevance' },
  { value: 'price_asc', label: 'Price: low to high' },
  { value: 'price_desc', label: 'Price: high to low' },
  { value: 'rating', label: 'Highest rated' },
  { value: 'discount', label: 'Biggest discount' },
  { value: 'newest', label: 'Newest' },
]

export default function SearchPage() {
  const [params, setParams] = useSearchParams()
  const { userId } = useUser()
  const query = params.get('q') ?? ''
  const category = params.get('category') ?? ''
  const sort = params.get('sort') ?? 'relevance'
  const page = Number.parseInt(params.get('page') ?? '1', 10)

  const [minPrice, setMinPrice] = useState(params.get('min_price') ?? '')
  const [maxPrice, setMaxPrice] = useState(params.get('max_price') ?? '')
  const [minRating, setMinRating] = useState(params.get('min_rating') ?? '')
  const [inStock, setInStock] = useState(params.get('in_stock') === 'true')
  const [showFilters, setShowFilters] = useState(false)

  useEffect(() => {
    setMinPrice(params.get('min_price') ?? '')
    setMaxPrice(params.get('max_price') ?? '')
    setMinRating(params.get('min_rating') ?? '')
    setInStock(params.get('in_stock') === 'true')
  }, [params])

  const queryArgs = useMemo(() => ({
    q: query,
    page,
    page_size: 24,
    category: category ? [category] : undefined,
    min_price: minPrice ? Number(minPrice) : undefined,
    max_price: maxPrice ? Number(maxPrice) : undefined,
    min_rating: minRating ? Number(minRating) : undefined,
    in_stock: inStock || undefined,
    sort,
    user_id: userId,
  }), [query, page, category, minPrice, maxPrice, minRating, inStock, sort, userId])

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['search', queryArgs],
    queryFn: ({ signal }) => search(queryArgs, signal),
  })

  const update = (patch: Record<string, string | undefined>) => {
    const next = new URLSearchParams(params)
    Object.entries(patch).forEach(([key, value]) => {
      if (value === undefined || value === '') next.delete(key)
      else next.set(key, value)
    })
    if (!('page' in patch)) next.set('page', '1')
    setParams(next)
  }

  const applyFilters = () => update({
    min_price: minPrice || undefined,
    max_price: maxPrice || undefined,
    min_rating: minRating || undefined,
    in_stock: inStock ? 'true' : undefined,
  })

  const clearFilters = () => {
    setMinPrice(''); setMaxPrice(''); setMinRating(''); setInStock(false)
    update({ min_price: undefined, max_price: undefined, min_rating: undefined, in_stock: undefined, category: undefined })
  }

  const totalPages = data ? Math.max(1, Math.ceil(data.total / 24)) : 1

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">
            {query ? `Results for "${query}"` : 'Browse the catalogue'}
          </h1>
          {data && (
            <p className="mt-1 flex flex-wrap items-center gap-2 text-sm text-slate-500">
              <span>{data.total.toLocaleString()} products · {data.took_ms.toFixed(1)}ms</span>
              <Badge tone="info">{data.strategy.replace(/_/g, ' ')}</Badge>
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button type="button" className="btn-secondary lg:hidden" onClick={() => setShowFilters((v) => !v)}>
            <SlidersHorizontal className="h-4 w-4" aria-hidden /> Filters
          </button>
          <select className="input !w-auto" value={sort} onChange={(e) => update({ sort: e.target.value })}
                  aria-label="Sort results">
            {SORTS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </div>
      </div>

      {data?.corrected_query && (
        <div className="rounded-lg border border-brand-200 bg-brand-50 px-3 py-2 text-sm text-brand-800">
          Showing results for <strong>{data.corrected_query}</strong> — corrected from "{data.query}".
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-[240px_1fr]">
        <aside className={`${showFilters ? 'block' : 'hidden'} space-y-4 lg:block`}>
          <Card className="space-y-4">
            <div>
              <p className="label">Price range</p>
              <div className="flex items-center gap-2">
                <input className="input" type="number" min={0} placeholder="Min" value={minPrice}
                       onChange={(e) => setMinPrice(e.target.value)} aria-label="Minimum price" />
                <span className="text-slate-400">–</span>
                <input className="input" type="number" min={0} placeholder="Max" value={maxPrice}
                       onChange={(e) => setMaxPrice(e.target.value)} aria-label="Maximum price" />
              </div>
            </div>
            <div>
              <label className="label" htmlFor="min-rating">Minimum rating</label>
              <select id="min-rating" className="input" value={minRating}
                      onChange={(e) => setMinRating(e.target.value)}>
                <option value="">Any rating</option>
                <option value="4">4 stars and up</option>
                <option value="3">3 stars and up</option>
                <option value="2">2 stars and up</option>
              </select>
            </div>
            <label className="flex items-center gap-2 text-sm text-slate-700">
              <input type="checkbox" checked={inStock} onChange={(e) => setInStock(e.target.checked)} />
              In stock only
            </label>
            <div className="flex gap-2">
              <button type="button" className="btn-primary flex-1" onClick={applyFilters}>Apply</button>
              <button type="button" className="btn-secondary" onClick={clearFilters}>Clear</button>
            </div>
          </Card>

          {data && data.facets.categories.length > 0 && (
            <Card>
              <p className="label">Categories</p>
              <ul className="space-y-1">
                {data.facets.categories.slice(0, 10).map((facet) => (
                  <li key={facet.slug}>
                    <button
                      type="button"
                      onClick={() => update({ category: facet.slug === category ? undefined : facet.slug })}
                      className={`flex w-full items-center justify-between rounded px-2 py-1 text-sm hover:bg-surface-100 ${
                        facet.slug === category ? 'bg-brand-50 font-medium text-brand-700' : 'text-slate-600'
                      }`}
                    >
                      <span className="truncate">{facet.name}</span>
                      <span className="ml-2 shrink-0 text-xs text-slate-400">{facet.count}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {data && (
            <Card>
              <p className="label">Ranking weights</p>
              <ul className="space-y-1 text-xs text-slate-600">
                {Object.entries(data.weights).map(([name, weight]) => (
                  <li key={name} className="flex items-center justify-between gap-2">
                    <span className="capitalize">{name}</span>
                    <span className="tabular-nums text-slate-400">{(weight * 100).toFixed(0)}%</span>
                  </li>
                ))}
              </ul>
              <p className="mt-2 text-[11px] leading-snug text-slate-400">
                Price range in results: {currency(data.facets.price.min)} – {currency(data.facets.price.max)}
              </p>
            </Card>
          )}
        </aside>

        <div>
          {isLoading && <SkeletonGrid count={9} />}
          {isError && (
            <ErrorState title="Search failed" message={(error as Error).message} onRetry={() => refetch()} />
          )}
          {data && data.hits.length === 0 && (
            <EmptyState
              title="No products matched"
              message="Try removing a filter or searching for something broader."
              action={<button type="button" className="btn-secondary" onClick={clearFilters}>Clear filters</button>}
            />
          )}
          {data && data.hits.length > 0 && (
            <>
              <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-4">
                {data.hits.map((hit) => (
                  <ProductCard key={hit.product.id} product={hit.product} explanation={hit.explanation} />
                ))}
              </div>

              {totalPages > 1 && (
                <nav className="mt-6 flex items-center justify-center gap-2" aria-label="Pagination">
                  <button type="button" className="btn-secondary" disabled={page <= 1}
                          onClick={() => update({ page: String(page - 1) })}>Previous</button>
                  <span className="text-sm text-slate-500">Page {page} of {totalPages}</span>
                  <button type="button" className="btn-secondary" disabled={page >= totalPages}
                          onClick={() => update({ page: String(page + 1) })}>Next</button>
                </nav>
              )}

              <p className="mt-4 flex items-center gap-1.5 text-xs text-slate-400">
                <Sparkles className="h-3 w-3" aria-hidden />
                Results blend keyword relevance, semantic similarity, popularity, rating and your preferences.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

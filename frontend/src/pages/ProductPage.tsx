import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, Package, ShoppingCart, TrendingDown, TrendingUp } from 'lucide-react'
import {
  getForecast, getFrequentlyBoughtTogether, getPricePrediction, getProduct, getProductRecommendations,
  getProductReviews, getProductSentiment, getSimilar, trackEvent,
} from '@/api/endpoints'
import { ProductRail } from '@/components/ProductCard'
import { Badge, Card, ErrorState, InfoNote, Rating, SectionHeading, Spinner, Tabs } from '@/components/ui'
import { SentimentBars } from '@/charts/SentimentBars'
import { ForecastChart } from '@/charts/ForecastChart'
import { currency, productGradient, shortDate, titleCase } from '@/utils/format'
import { useCart } from '@/store/CartContext'
import { useUser } from '@/store/UserContext'
import { ApiError } from '@/api/client'

type TabId = 'overview' | 'reviews' | 'intelligence'

export default function ProductPage() {
  const { id } = useParams<{ id: string }>()
  const productId = Number.parseInt(id ?? '', 10)
  const { addToCart } = useCart()
  const { userId } = useUser()
  const [tab, setTab] = useState<TabId>('overview')
  const [quantity, setQuantity] = useState(1)

  const product = useQuery({
    queryKey: ['product', productId],
    queryFn: () => getProduct(productId),
    enabled: Number.isFinite(productId),
  })

  // Record the view once the product resolves - this is a real ML training signal.
  useEffect(() => {
    if (product.data) {
      void trackEvent({ event_type: 'product_view', user_id: userId, product_id: productId })
        .catch(() => undefined)
    }
  }, [product.data, productId, userId])

  const similar = useQuery({
    queryKey: ['similar', productId],
    queryFn: () => getSimilar(productId, 10),
    enabled: Number.isFinite(productId),
  })
  const fbt = useQuery({
    queryKey: ['fbt', productId],
    queryFn: () => getFrequentlyBoughtTogether(productId, 5),
    enabled: Number.isFinite(productId),
  })
  const personalized = useQuery({
    queryKey: ['product-recs', productId, userId],
    queryFn: () => getProductRecommendations(productId, userId, 10),
    enabled: Number.isFinite(productId),
  })
  const reviews = useQuery({
    queryKey: ['reviews', productId],
    queryFn: () => getProductReviews(productId, 1, 8),
    enabled: Number.isFinite(productId) && tab === 'reviews',
  })
  const sentiment = useQuery({
    queryKey: ['sentiment', productId],
    queryFn: () => getProductSentiment(productId),
    enabled: Number.isFinite(productId),
  })
  const forecast = useQuery({
    queryKey: ['forecast', productId],
    queryFn: () => getForecast(productId, 30),
    enabled: Number.isFinite(productId) && tab === 'intelligence',
    retry: false,
  })
  const price = useQuery({
    queryKey: ['price', productId],
    queryFn: () => getPricePrediction(productId),
    enabled: Number.isFinite(productId) && tab === 'intelligence',
    retry: false,
  })

  if (!Number.isFinite(productId)) return <ErrorState title="Invalid product id" />
  if (product.isLoading) return <Spinner label="Loading product" />
  if (product.isError) {
    return <ErrorState title="Product not found" message={(product.error as Error).message}
                       onRetry={() => product.refetch()} />
  }

  const item = product.data!

  return (
    <div className="space-y-8">
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="card overflow-hidden">
          <div className="flex aspect-[16/10] items-center justify-center p-8"
               style={{ background: productGradient(item.id) }}>
            <p className="max-w-md text-center text-lg font-medium text-slate-600/80">{item.title}</p>
          </div>
          <div className="flex gap-2 border-t border-surface-100 p-3">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="h-16 w-16 rounded-lg border border-surface-200"
                   style={{ background: productGradient(item.id + i * 13) }} aria-hidden />
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              {item.brand && <Badge tone="brand">{item.brand.name}</Badge>}
              {item.category && <Badge tone="neutral">{item.category.name}</Badge>}
              {item.discount_pct > 0 && <Badge tone="danger">-{item.discount_pct.toFixed(0)}%</Badge>}
            </div>
            <h1 className="mt-2 text-xl font-semibold tracking-tight sm:text-2xl">{item.title}</h1>
            <div className="mt-2"><Rating value={item.rating_avg} count={item.rating_count} size="md" /></div>
          </div>

          <Card>
            <div className="flex items-baseline gap-3">
              <span className="text-3xl font-semibold tracking-tight">{currency(item.effective_price)}</span>
              {item.discount_pct > 0 && (
                <span className="text-base text-slate-400 line-through">{currency(item.price)}</span>
              )}
            </div>
            <p className={`mt-2 flex items-center gap-1.5 text-sm ${item.in_stock ? 'text-emerald-600' : 'text-rose-600'}`}>
              <Package className="h-4 w-4" aria-hidden />
              {item.in_stock ? `In stock — ${item.inventory} available` : 'Currently out of stock'}
            </p>
            <div className="mt-4 flex items-center gap-2">
              <label className="sr-only" htmlFor="qty">Quantity</label>
              <select id="qty" className="input !w-20" value={quantity}
                      onChange={(e) => setQuantity(Number(e.target.value))}>
                {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
              <button type="button" className="btn-primary flex-1" disabled={!item.in_stock}
                      onClick={() => addToCart(item, quantity)}>
                <ShoppingCart className="h-4 w-4" aria-hidden /> Add to cart
              </button>
            </div>
          </Card>

          {sentiment.data && sentiment.data.review_count > 0 && (
            <Card>
              <p className="label">What reviewers say</p>
              <p className="text-sm leading-relaxed text-slate-700">{sentiment.data.summary}</p>
              <div className="mt-3"><SentimentBars distribution={sentiment.data.distribution} /></div>
            </Card>
          )}
        </div>
      </div>

      <Tabs<TabId>
        tabs={[
          { id: 'overview', label: 'Overview' },
          { id: 'reviews', label: `Reviews (${item.rating_count})` },
          { id: 'intelligence', label: 'Intelligence' },
        ]}
        active={tab}
        onChange={setTab}
      />

      {tab === 'overview' && (
        <div className="grid gap-5 lg:grid-cols-2">
          <Card>
            <h3 className="mb-2 font-medium">Description</h3>
            <p className="text-sm leading-relaxed text-slate-600">{item.description}</p>
            {item.tags.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {item.tags.map((tag) => <Badge key={tag} tone="neutral">{tag}</Badge>)}
              </div>
            )}
          </Card>
          <Card>
            <h3 className="mb-2 font-medium">Specifications</h3>
            <dl className="divide-y divide-surface-100 text-sm">
              <div className="flex justify-between gap-4 py-1.5">
                <dt className="text-slate-500">SKU</dt><dd className="font-mono text-xs">{item.sku}</dd>
              </div>
              {Object.entries(item.specifications).map(([key, value]) => (
                <div key={key} className="flex justify-between gap-4 py-1.5">
                  <dt className="text-slate-500">{titleCase(key)}</dt>
                  <dd className="text-right font-medium text-surface-900">{String(value)}</dd>
                </div>
              ))}
              {item.launched_at && (
                <div className="flex justify-between gap-4 py-1.5">
                  <dt className="text-slate-500">Launched</dt><dd>{shortDate(item.launched_at)}</dd>
                </div>
              )}
            </dl>
          </Card>
        </div>
      )}

      {tab === 'reviews' && (
        <div className="space-y-4">
          {reviews.isLoading && <Spinner label="Loading reviews" />}
          {reviews.data?.items.length === 0 && (
            <Card><p className="text-sm text-slate-500">No reviews yet for this product.</p></Card>
          )}
          {reviews.data?.items.map((review) => (
            <Card key={review.id}>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Rating value={review.rating} />
                  <span className="font-medium text-surface-900">{review.title}</span>
                </div>
                <div className="flex items-center gap-2">
                  {review.verified_purchase && <Badge tone="success">Verified purchase</Badge>}
                  {review.sentiment_label && (
                    <Badge tone={review.sentiment_label === 'positive' ? 'success'
                      : review.sentiment_label === 'negative' ? 'danger' : 'neutral'}>
                      {review.sentiment_label}
                    </Badge>
                  )}
                </div>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">{review.body}</p>
              {review.aspects && Object.keys(review.aspects).length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {Object.entries(review.aspects).map(([aspect, polarity]) => (
                    <Badge key={aspect} tone={polarity === 'positive' ? 'success' : polarity === 'negative' ? 'danger' : 'neutral'}>
                      {titleCase(aspect)}: {polarity}
                    </Badge>
                  ))}
                </div>
              )}
              <p className="mt-2 text-xs text-slate-400">
                {shortDate(review.created_at)} · {review.helpful_votes} found this helpful
              </p>
            </Card>
          ))}
        </div>
      )}

      {tab === 'intelligence' && (
        <div className="grid gap-5 lg:grid-cols-2">
          <Card>
            <SectionHeading title="Price intelligence" subtitle="Model-implied price versus the listed price" />
            {price.isLoading && <Spinner label="Scoring price" />}
            {price.isError && (
              (price.error as ApiError)?.isModelUnavailable
                ? <InfoNote>The price model has not been trained yet. Run <code>make train</code>.</InfoNote>
                : <ErrorState message={(price.error as Error).message} onRetry={() => price.refetch()} />
            )}
            {price.data && (
              <div className="space-y-3">
                <div className="flex flex-wrap items-baseline gap-4">
                  <div>
                    <p className="text-xs text-slate-500">Listed</p>
                    <p className="text-xl font-semibold">{currency(price.data.current_price)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Model estimate</p>
                    <p className="text-xl font-semibold text-brand-700">{currency(price.data.predicted_price)}</p>
                  </div>
                  <Badge tone={price.data.price_trend === 'up' ? 'success' : price.data.price_trend === 'down' ? 'danger' : 'neutral'}>
                    {price.data.price_trend === 'up' ? <TrendingUp className="h-3 w-3" aria-hidden />
                      : price.data.price_trend === 'down' ? <TrendingDown className="h-3 w-3" aria-hidden /> : null}
                    {price.data.delta_pct > 0 ? '+' : ''}{price.data.delta_pct.toFixed(1)}%
                  </Badge>
                </div>
                <p className="text-sm leading-relaxed text-slate-600">{price.data.explanation}</p>
                <div>
                  <p className="label">Top drivers</p>
                  <ul className="space-y-1 text-xs">
                    {price.data.feature_contributions.slice(0, 5).map((c) => (
                      <li key={c.feature} className="flex items-center justify-between gap-2">
                        <span className="text-slate-600">{c.label}</span>
                        <span className={c.contribution > 0 ? 'text-emerald-600' : 'text-rose-600'}>
                          {c.contribution > 0 ? '↑' : '↓'} {Math.abs(c.contribution).toFixed(3)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
                <p className="text-xs text-slate-400">
                  Confidence {(price.data.confidence * 100).toFixed(0)}% · model {price.data.model_version}
                </p>
              </div>
            )}
          </Card>

          <Card>
            <SectionHeading title="Demand forecast" subtitle="Next 30 days with 95% prediction interval" />
            {forecast.isLoading && <Spinner label="Forecasting demand" />}
            {forecast.isError && (
              (forecast.error as ApiError)?.isModelUnavailable
                ? <InfoNote>The forecasting model has not been trained yet. Run <code>make train</code>.</InfoNote>
                : <ErrorState message={(forecast.error as Error).message} onRetry={() => forecast.refetch()} />
            )}
            {forecast.data && (
              <div className="space-y-3">
                <ForecastChart points={forecast.data.points} history={forecast.data.history} />
                <div className="grid grid-cols-3 gap-2 text-center">
                  {Object.entries(forecast.data.horizons).map(([label, values]) => (
                    <div key={label} className="rounded-lg bg-surface-100 p-2">
                      <p className="text-[11px] uppercase text-slate-500">{label.replace(/_/g, ' ')}</p>
                      <p className="text-sm font-semibold tabular-nums">
                        {values.predicted_units.toFixed(1)} units
                      </p>
                    </div>
                  ))}
                </div>
                {Object.entries(forecast.data.stockout_risk).some(([, r]) => r.will_stock_out) && (
                  <p className="flex items-center gap-1.5 text-xs text-amber-700">
                    <AlertTriangle className="h-3.5 w-3.5" aria-hidden />
                    Projected demand exceeds current inventory within the forecast window.
                  </p>
                )}
                {forecast.data.note && <InfoNote>{forecast.data.note}</InfoNote>}
              </div>
            )}
          </Card>

          {sentiment.data && sentiment.data.aspects.length > 0 && (
            <Card className="lg:col-span-2">
              <SectionHeading title="Aspect sentiment" subtitle="Extracted from customer review text" />
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {sentiment.data.aspects.slice(0, 9).map((aspect) => (
                  <div key={aspect.aspect} className="rounded-lg border border-surface-200 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-medium">{aspect.label}</span>
                      <Badge tone={aspect.sentiment > 0.2 ? 'success' : aspect.sentiment < -0.1 ? 'danger' : 'neutral'}>
                        {aspect.sentiment > 0 ? '+' : ''}{aspect.sentiment.toFixed(2)}
                      </Badge>
                    </div>
                    <p className="mt-1 text-xs text-slate-500">
                      {aspect.mentions} mentions · {aspect.positive} positive · {aspect.negative} negative
                    </p>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}

      {fbt.data && fbt.data.items.length > 0 && (
        <section>
          <SectionHeading title="Frequently bought together"
                          subtitle="Derived from co-purchase patterns across real baskets" />
          <ProductRail items={fbt.data.items} />
        </section>
      )}

      {similar.data && similar.data.items.length > 0 && (
        <section>
          <SectionHeading title="Similar products" subtitle="Matched on content embeddings and behaviour" />
          <ProductRail items={similar.data.items} />
        </section>
      )}

      {personalized.data && personalized.data.items.length > 0 && (
        <section>
          <SectionHeading title="Recommended for you"
                          subtitle={`Strategy: ${personalized.data.strategy.replace(/_/g, ' ')}`} />
          <ProductRail items={personalized.data.items} />
        </section>
      )}
    </div>
  )
}

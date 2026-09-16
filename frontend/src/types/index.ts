/** Shared API types - mirrors the Pydantic schemas exposed at /openapi.json */

export interface PageMeta {
  page: number
  page_size: number
  total: number
  total_pages: number
  has_next: boolean
  has_previous: boolean
}

export interface Page<T> {
  items: T[]
  meta: PageMeta
}

export interface Category {
  id: number
  slug: string
  name: string
  description?: string | null
  parent_id?: number | null
}

export interface CategorySummary {
  id: number
  slug: string
  name: string
  product_count: number
  avg_rating: number
}

export interface Brand {
  id: number
  slug: string
  name: string
  reputation_score: number
}

export interface Product {
  id: number
  sku: string
  title: string
  price: number
  discount_pct: number
  currency: string
  rating_avg: number
  rating_count: number
  inventory: number
  image_url: string
  category_id: number
  brand_id: number
  effective_price: number
  in_stock: boolean
}

export interface ProductDetail extends Product {
  description: string
  specifications: Record<string, string | number | boolean>
  tags: string[]
  launched_at?: string | null
  is_active: boolean
  category?: Category | null
  brand?: Brand | null
}

export interface Review {
  id: number
  product_id: number
  user_id: number
  rating: number
  title: string
  body: string
  verified_purchase: boolean
  helpful_votes: number
  sentiment_label?: 'positive' | 'neutral' | 'negative' | null
  sentiment_score?: number | null
  aspects?: Record<string, string> | null
  created_at: string
}

export interface RankingSignals {
  text: number
  semantic: number
  popularity: number
  rating: number
  conversion: number
  personal: number
  availability: number
}

export interface SearchHit {
  product: Product
  score: number
  signals: RankingSignals
  explanation: string
}

export interface SearchFacets {
  categories: { slug: string; name: string; count: number }[]
  brands: { id: number; name: string; count: number }[]
  price: { min: number; max: number; avg: number }
  rating: Record<string, number>
  availability: { in_stock: number; out_of_stock: number }
  on_sale: number
}

export interface SearchResults {
  query: string
  normalized_query: string
  corrected_query?: string | null
  strategy: string
  total: number
  took_ms: number
  weights: Record<string, number>
  hits: SearchHit[]
  facets: SearchFacets
}

export interface Suggestion {
  text: string
  type: 'product' | 'query' | 'term'
  product_id?: number | null
  popularity?: number | null
}

export interface RecommendationItem {
  product: Product
  score: number
  rank: number
  components: Record<string, number>
  explanation: string
}

export interface RecommendationResponse {
  items: RecommendationItem[]
  strategy: string
  model_version: string
  count: number
}

export interface HomepageSection {
  key: string
  title: string
  subtitle: string
  strategy: string
  items: RecommendationItem[]
}

export interface HomepageResponse {
  user_id: number | null
  is_personalized: boolean
  segment?: string | null
  sections: HomepageSection[]
  profile_summary: Record<string, unknown>
}

export interface UserProfile {
  user_id: number | null
  is_known: boolean
  is_cold_start: boolean
  interaction_count: number
  category_affinity: Record<string, number>
  brand_affinity: Record<string, number>
  recently_viewed: number[]
  purchased: number[]
  cart: number[]
  wishlist: number[]
  search_terms: string[]
  preferred_price_range: number[]
  avg_order_value: number
  order_count: number
  total_spend: number
  session_count: number
  review_count: number
  segment?: string | null
  price_sensitivity: number
  last_active?: string | null
  preference_vector: number[]
}

export interface Order {
  id: number
  order_number: string
  user_id: number
  status: string
  total_amount: number
  discount_amount: number
  item_count: number
  currency: string
  channel: string
  placed_at: string
  items: { id: number; product_id: number; quantity: number; unit_price: number; line_total: number }[]
}

export interface AspectSentiment {
  aspect: string
  label: string
  mentions: number
  positive: number
  negative: number
  neutral: number
  sentiment: number
}

export interface ProductSentiment {
  product_id: number
  review_count: number
  distribution: Record<string, number>
  distribution_pct: Record<string, number>
  average_score: number
  average_rating: number
  aspects: AspectSentiment[]
  top_praise: AspectSentiment[]
  top_complaints: AspectSentiment[]
  keywords: { term: string; count: number }[]
  summary: string
}

export interface ForecastPoint {
  day: number
  date?: string
  predicted: number
  lower: number
  upper: number
  model: string
}

export interface ForecastResponse {
  product_id: number
  product_title: string
  horizon_days: number
  model_version: string
  model_type: string
  is_trained_product: boolean
  points: ForecastPoint[]
  horizons: Record<string, Record<string, number>>
  current_inventory: number
  stockout_risk: Record<string, { expected_demand: number; inventory: number; will_stock_out: boolean }>
  history: { date: string; units: number }[]
  note?: string | null
}

export interface PricePrediction {
  product_id: number
  product_title: string
  current_price: number
  effective_price: number
  predicted_price: number
  price_trend: 'up' | 'down' | 'stable'
  delta_pct: number
  confidence: number
  lower_bound: number
  upper_bound: number
  expected_demand_90d: number
  explanation: string
  feature_contributions: { feature: string; label: string; contribution: number; direction: string }[]
  global_importance: { feature: string; importance: number }[]
  model_version: string
}

export interface SegmentSummary {
  label: string
  name: string
  description: string
  customer_count: number
  share: number
  avg_monetary: number
  avg_frequency: number
  avg_recency_days: number
  avg_order_value: number
  avg_rfm_score: number
  avg_confidence: number
}

export interface SegmentOverview {
  available: boolean
  model_version?: string | null
  total_customers: number
  segments: SegmentSummary[]
  metrics: Record<string, unknown>
  cluster_profiles: Record<string, Record<string, number>>
  feature_names: string[]
  note?: string | null
}

export interface DashboardResponse {
  generated_at: string
  window_days: number
  business: {
    revenue: number; revenue_change_pct: number | null; orders: number; orders_change_pct: number | null
    average_order_value: number; aov_change_pct: number | null; unique_buyers: number; active_users: number
    sessions: number; conversion_rate: number; repeat_buyers: number; retention_rate: number
    total_customers: number; total_products: number
  }
  ai: {
    recommendation: { impressions: number; clicks: number; conversions: number; ctr: number
      conversion_rate: number; by_strategy: { strategy: string; impressions: number; clicks: number; ctr: number }[] }
    search: { searches: number; clicks: number; conversions: number; ctr: number; conversion_rate: number
      abandonment_rate: number; zero_result_rate: number; avg_latency_ms: number }
  }
  customers: {
    new_users: number; returning_buyers: number
    segments: { label: string; name: string; count: number; avg_monetary: number }[]
    high_value_count: number; at_risk_count: number
    top_customers: { user_id: number; name: string; email: string; total_spend: number; orders: number }[]
  }
  products: {
    best_sellers: (Partial<Product> & { units_sold: number; revenue: number })[]
    low_inventory: Partial<Product>[]
    high_conversion: (Partial<Product> & { conversion_rate: number; views: number })[]
    poor_performers: (Partial<Product> & { conversion_rate: number; views: number })[]
    category_breakdown: { category: string; slug: string; products: number; revenue: number }[]
  }
  sentiment: { counts: Record<string, number>; percentages: Record<string, number>
    total_labelled: number; total_reviews: number }
  revenue_timeseries: { date: string; revenue: number; orders: number }[]
  event_funnel: { counts: Record<string, number>; view_to_cart: number
    cart_to_purchase: number; view_to_purchase: number }
}

export interface ModelRegistryEntry {
  name: string
  version: string
  stage: string
  algorithm: string
  dataset_version: string
  training_rows: number
  trained_at: string | null
  training_duration_s: number
  metrics: Record<string, unknown>
  params: Record<string, unknown>
  feature_names: string[]
  is_active: boolean
  notes: string
  is_loaded: boolean
  live_stats?: Record<string, unknown> | null
}

export interface ModelHealth {
  model: string
  available: boolean
  loaded: boolean
  status: string
  version?: string
  calls?: number
  failures?: number
  error_rate?: number
  avg_latency_ms?: number
  p95_latency_ms?: number
  hint?: string
}

export interface DriftReport {
  model: string
  version: string
  trained_at: string
  status: string
  checks: { feature: string; baseline?: number; current?: number; relative_change?: number
    threshold?: number; drift_detected?: boolean; status?: string }[]
  prediction_stats?: { samples: number; mean: number; std: number; p05: number; p95: number }
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'error'
  environment: string
  version: string
  uptime_seconds: number
  checks: Record<string, { status: string; [k: string]: unknown }>
}

export interface CartLine {
  product: Product
  quantity: number
}

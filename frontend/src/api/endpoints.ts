/** One typed function per API operation the UI uses. */
import { api } from './client'
import type {
  Brand, Category, CategorySummary, DashboardResponse, DriftReport, ForecastResponse, HealthResponse,
  HomepageResponse, ModelHealth, ModelRegistryEntry, Order, Page, PricePrediction, Product, ProductDetail,
  ProductSentiment, RecommendationResponse, Review, SearchResults, SegmentOverview, Suggestion, UserProfile,
} from '@/types'

// ---- catalogue ----------------------------------------------------------
export interface ProductQuery {
  page?: number
  page_size?: number
  category?: string
  brand_id?: number
  min_price?: number
  max_price?: number
  min_rating?: number
  in_stock?: boolean
  on_sale?: boolean
  sort?: string
}

export const getProducts = (q: ProductQuery = {}) => api.get<Page<Product>>('/products', { ...q })
export const getProduct = (id: number) => api.get<ProductDetail>(`/products/${id}`)
export const getCategories = () => api.get<Category[]>('/products/categories')
export const getPopularCategories = (limit = 8) =>
  api.get<CategorySummary[]>('/products/categories/popular', { limit })
export const getBrands = (limit = 200) => api.get<Brand[]>('/products/brands', { limit })
export const getProductReviews = (id: number, page = 1, page_size = 5, sentiment?: string) =>
  api.get<Page<Review>>(`/products/${id}/reviews`, { page, page_size, sentiment })
export const createReview = (id: number, body: { product_id: number; user_id: number; rating: number; title: string; body: string }) =>
  api.post<Review>(`/products/${id}/reviews`, body)

// ---- search -------------------------------------------------------------
export interface SearchQuery {
  q: string
  page?: number
  page_size?: number
  category?: string[]
  brand_id?: number[]
  min_price?: number
  max_price?: number
  min_rating?: number
  in_stock?: boolean
  on_sale?: boolean
  sort?: string
  user_id?: number
}

export const search = (q: SearchQuery, signal?: AbortSignal) =>
  api.get<SearchResults>('/search', { ...q }, signal)
export const getSuggestions = (q: string, limit = 8, signal?: AbortSignal) =>
  api.get<Suggestion[]>('/search/suggestions', { q, limit }, signal)
export const getPopularQueries = (limit = 10) =>
  api.get<{ query: string; searches: number; clicks: number; ctr: number }[]>('/search/popular', { limit })
export const getSearchHistory = (user_id: number, limit = 10) =>
  api.get<{ query: string; occurred_at: string; result_count: number }[]>('/search/history', { user_id, limit })

// ---- recommendations ----------------------------------------------------
export const getHomepage = (user_id?: number, per_section = 8) =>
  api.get<HomepageResponse>('/recommendations/homepage', { user_id, per_section })
export const getRecommendations = (user_id: number, limit = 12) =>
  api.get<RecommendationResponse>(`/recommendations/${user_id}`, { limit })
export const getTrending = (limit = 12) =>
  api.get<RecommendationResponse>('/recommendations/trending', { limit })
export const getDeals = (limit = 12) => api.get<RecommendationResponse>('/recommendations/deals', { limit })
export const getRecentlyViewed = (user_id?: number, limit = 12) =>
  api.get<RecommendationResponse>('/recommendations/recently-viewed', { user_id, limit })
export const getContinueShopping = (user_id?: number, limit = 8) =>
  api.get<RecommendationResponse>('/recommendations/continue-shopping', { user_id, limit })
export const getSimilar = (id: number, limit = 8) =>
  api.get<RecommendationResponse>(`/products/${id}/similar`, { limit })
export const getFrequentlyBoughtTogether = (id: number, limit = 4) =>
  api.get<RecommendationResponse>(`/products/${id}/frequently-bought-together`, { limit })
export const getProductRecommendations = (id: number, user_id?: number, limit = 8) =>
  api.get<RecommendationResponse>(`/products/${id}/recommendations`, { user_id, limit })

// ---- events -------------------------------------------------------------
export interface EventPayload {
  event_type: string
  user_id?: number | null
  product_id?: number | null
  session_id?: string
  quantity?: number
  value?: number
  source?: string
  metadata?: Record<string, unknown>
}
export const trackEvent = (payload: EventPayload) => api.post<unknown>('/events', payload)
export const getEventFunnel = (days = 30) =>
  api.get<Record<string, number>>('/events/funnel', { days })

// ---- users --------------------------------------------------------------
export const getUserProfile = (user_id: number) => api.get<UserProfile>(`/users/${user_id}/profile`)
export const getUserOrders = (user_id: number, page = 1, page_size = 10) =>
  api.get<Page<Order>>(`/users/${user_id}/orders`, { page, page_size })
export const getUserSegment = (user_id: number) => api.get<Record<string, unknown>>(`/users/${user_id}/segment`)

// ---- intelligence -------------------------------------------------------
export const getProductSentiment = (id: number) => api.get<ProductSentiment>(`/products/${id}/sentiment`)
export const analyzeSentiment = (text: string) =>
  api.post<{ label: string; score: number; aspects: Record<string, string>; positive_aspects: string[]
    negative_aspects: string[]; model_version: string }>('/sentiment/analyze', { text })
export const getForecast = (id: number, horizon = 30) =>
  api.get<ForecastResponse>(`/forecast/${id}`, { horizon })
export const getForecastAccuracy = () => api.get<Record<string, unknown>>('/forecast/accuracy/summary')
export const getPricePrediction = (id: number) => api.get<PricePrediction>(`/price-prediction/${id}`)
export const getSegments = () => api.get<SegmentOverview>('/segments')

// ---- admin --------------------------------------------------------------
export const login = (email: string, password: string) =>
  api.post<{ access_token: string; role: string; user_id: number; expires_in_minutes: number }>(
    '/auth/login', { email, password })
export const getDashboard = (days = 30) => api.get<DashboardResponse>('/analytics/dashboard', { days })
export const getModelRegistry = () => api.get<ModelRegistryEntry[]>('/ml/registry')
export const getModelHealth = () => api.get<ModelHealth[]>('/ml/health')
export const getDrift = () => api.get<DriftReport[]>('/ml/drift')
export const getMlSummary = () => api.get<Record<string, unknown>>('/ml/summary')
export const reloadModels = () => api.post<{ reloaded: Record<string, string> }>('/ml/reload')
export const getHealth = () => api.get<HealthResponse>('/health')
export const getFeatureCoverage = () => api.get<Record<string, unknown>>('/ml/features/coverage')

import { Suspense, lazy } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './layouts/AppLayout'
import { Spinner } from './components/ui'

// Route-level code splitting keeps the initial storefront bundle small.
const HomePage = lazy(() => import('./pages/HomePage'))
const SearchPage = lazy(() => import('./pages/SearchPage'))
const ProductPage = lazy(() => import('./pages/ProductPage'))
const CartPage = lazy(() => import('./pages/CartPage'))
const OrdersPage = lazy(() => import('./pages/OrdersPage'))
const ProfilePage = lazy(() => import('./pages/ProfilePage'))
const RecommendationsPage = lazy(() => import('./pages/RecommendationsPage'))
const AdminDashboard = lazy(() => import('./pages/AdminDashboard'))
const MLDashboard = lazy(() => import('./pages/MLDashboard'))
const SystemHealthPage = lazy(() => import('./pages/SystemHealthPage'))
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'))

export default function App() {
  return (
    <Suspense fallback={<Spinner label="Loading view" className="min-h-[60vh]" />}>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<HomePage />} />
          <Route path="search" element={<SearchPage />} />
          <Route path="product/:id" element={<ProductPage />} />
          <Route path="cart" element={<CartPage />} />
          <Route path="orders" element={<OrdersPage />} />
          <Route path="profile" element={<ProfilePage />} />
          <Route path="recommendations" element={<RecommendationsPage />} />
          <Route path="admin" element={<AdminDashboard />} />
          <Route path="admin/ml" element={<MLDashboard />} />
          <Route path="admin/health" element={<SystemHealthPage />} />
          <Route path="analytics" element={<Navigate to="/admin" replace />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </Suspense>
  )
}

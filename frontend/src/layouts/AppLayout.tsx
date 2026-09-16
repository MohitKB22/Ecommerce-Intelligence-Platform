import { useEffect, useRef, useState } from 'react'
import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom'
import {
  Activity, BarChart3, Brain, Home, Menu, Package, Search as SearchIcon, ShoppingCart, Sparkles, User, X,
} from 'lucide-react'
import clsx from 'clsx'
import { useCart } from '@/store/CartContext'
import { useUser } from '@/store/UserContext'
import { getSuggestions } from '@/api/endpoints'
import type { Suggestion } from '@/types'

const SHOP_LINKS = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/search', label: 'Browse', icon: Package },
  { to: '/recommendations', label: 'For You', icon: Sparkles },
  { to: '/orders', label: 'Orders', icon: ShoppingCart },
  { to: '/profile', label: 'Profile', icon: User },
]

const ADMIN_LINKS = [
  { to: '/admin', label: 'Analytics', icon: BarChart3 },
  { to: '/admin/ml', label: 'ML Models', icon: Brain },
  { to: '/admin/health', label: 'System', icon: Activity },
]

export function AppLayout() {
  const navigate = useNavigate()
  const { itemCount } = useCart()
  const { userId } = useUser()
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const boxRef = useRef<HTMLDivElement>(null)

  // Debounced autocomplete; aborts in-flight requests as the user keeps typing.
  useEffect(() => {
    if (query.trim().length < 2) {
      setSuggestions([])
      return
    }
    const controller = new AbortController()
    const timer = setTimeout(() => {
      getSuggestions(query.trim(), 6, controller.signal)
        .then(setSuggestions)
        .catch(() => setSuggestions([]))
    }, 180)
    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [query])

  useEffect(() => {
    const onClick = (event: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(event.target as Node)) setShowSuggestions(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [])

  const submit = (value: string) => {
    setShowSuggestions(false)
    setMobileOpen(false)
    navigate(`/search?q=${encodeURIComponent(value)}`)
  }

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-surface-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-3">
          <Link to="/" className="flex shrink-0 items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-sm font-bold text-white">
              EI
            </span>
            <span className="hidden text-sm font-semibold tracking-tight sm:block">
              E-Commerce Intelligence
            </span>
          </Link>

          <div ref={boxRef} className="relative min-w-0 flex-1">
            <form
              onSubmit={(e) => { e.preventDefault(); if (query.trim()) submit(query.trim()) }}
              role="search"
            >
              <SearchIcon className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
                          aria-hidden />
              <input
                type="search"
                value={query}
                onChange={(e) => { setQuery(e.target.value); setShowSuggestions(true) }}
                onFocus={() => setShowSuggestions(true)}
                placeholder="Search products, brands, categories..."
                aria-label="Search products"
                className="input !pl-9"
              />
            </form>

            {showSuggestions && suggestions.length > 0 && (
              <ul className="absolute inset-x-0 top-full z-50 mt-1 max-h-80 overflow-auto rounded-lg border border-surface-200 bg-white py-1 shadow-lift">
                {suggestions.map((s, i) => (
                  <li key={`${s.type}-${s.text}-${i}`}>
                    <button
                      type="button"
                      onClick={() => (s.product_id ? navigate(`/product/${s.product_id}`) : submit(s.text))}
                      className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-surface-100"
                    >
                      <SearchIcon className="h-3.5 w-3.5 shrink-0 text-slate-400" aria-hidden />
                      <span className="truncate">{s.text}</span>
                      <span className="ml-auto shrink-0 text-[10px] uppercase text-slate-400">{s.type}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <nav className="hidden items-center gap-1 lg:flex" aria-label="Main">
            {SHOP_LINKS.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) => clsx(
                  'flex items-center gap-1.5 rounded-lg px-2.5 py-2 text-sm font-medium transition-colors',
                  isActive ? 'bg-brand-50 text-brand-700' : 'text-slate-600 hover:bg-surface-100',
                )}
              >
                <Icon className="h-4 w-4" aria-hidden />
                <span className="hidden xl:inline">{label}</span>
              </NavLink>
            ))}
          </nav>

          <Link to="/cart" className="relative shrink-0 rounded-lg p-2 hover:bg-surface-100"
                aria-label={`Cart with ${itemCount} items`}>
            <ShoppingCart className="h-5 w-5 text-slate-700" aria-hidden />
            {itemCount > 0 && (
              <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-brand-600 px-1 text-[10px] font-semibold text-white">
                {itemCount}
              </span>
            )}
          </Link>

          <button type="button" className="rounded-lg p-2 hover:bg-surface-100 lg:hidden"
                  onClick={() => setMobileOpen((v) => !v)}
                  aria-label={mobileOpen ? 'Close menu' : 'Open menu'} aria-expanded={mobileOpen}>
            {mobileOpen ? <X className="h-5 w-5" aria-hidden /> : <Menu className="h-5 w-5" aria-hidden />}
          </button>
        </div>

        <div className="mx-auto hidden max-w-7xl items-center gap-1 border-t border-surface-100 px-4 py-1.5 lg:flex">
          <span className="mr-2 text-[11px] font-medium uppercase tracking-wide text-slate-400">Intelligence</span>
          {ADMIN_LINKS.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end
                     className={({ isActive }) => clsx(
                       'flex items-center gap-1.5 rounded px-2 py-1 text-xs font-medium transition-colors',
                       isActive ? 'bg-surface-100 text-surface-900' : 'text-slate-500 hover:text-surface-900',
                     )}>
              <Icon className="h-3.5 w-3.5" aria-hidden />{label}
            </NavLink>
          ))}
          <span className="ml-auto text-[11px] text-slate-400">Shopping as user #{userId}</span>
        </div>

        {mobileOpen && (
          <nav className="border-t border-surface-200 bg-white px-4 py-2 lg:hidden" aria-label="Mobile">
            {[...SHOP_LINKS, ...ADMIN_LINKS].map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} end={to === '/'} onClick={() => setMobileOpen(false)}
                       className={({ isActive }) => clsx(
                         'flex items-center gap-2 rounded-lg px-3 py-2.5 text-sm font-medium',
                         isActive ? 'bg-brand-50 text-brand-700' : 'text-slate-700 hover:bg-surface-100',
                       )}>
                <Icon className="h-4 w-4" aria-hidden />{label}
              </NavLink>
            ))}
          </nav>
        )}
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">
        <Outlet />
      </main>

      <footer className="border-t border-surface-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 py-6 text-xs text-slate-500">
          <p className="font-medium text-slate-600">E-Commerce Intelligence</p>
          <p className="mt-1">
            Reference architecture for an AI-powered retail platform. All catalogue, customer and
            transaction data shown is <strong>synthetic</strong> and generated for demonstration.
          </p>
        </div>
      </footer>
    </div>
  )
}

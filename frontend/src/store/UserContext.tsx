import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { login as loginRequest, setTokenSafe } from './authHelpers'

interface UserContextValue {
  /** The shopper the storefront is personalising for. */
  userId: number
  setUserId: (id: number) => void
  isAdmin: boolean
  adminEmail: string | null
  loginAsAdmin: (email: string, password: string) => Promise<void>
  logoutAdmin: () => void
}

const UserContext = createContext<UserContextValue | null>(null)

/** Demo shopper used when nobody has been selected explicitly. */
export const DEFAULT_USER_ID = 7

export function UserProvider({ children }: { children: ReactNode }) {
  const [userId, setUserId] = useState<number>(() => {
    const stored = typeof window !== 'undefined' ? window.sessionStorage.getItem('eci.userId') : null
    const parsed = stored ? Number.parseInt(stored, 10) : NaN
    return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_USER_ID
  })
  const [isAdmin, setIsAdmin] = useState(false)
  const [adminEmail, setAdminEmail] = useState<string | null>(null)

  const updateUserId = useCallback((id: number) => {
    setUserId(id)
    try {
      window.sessionStorage.setItem('eci.userId', String(id))
    } catch {
      /* storage unavailable - in-memory state still works */
    }
  }, [])

  const loginAsAdmin = useCallback(async (email: string, password: string) => {
    const result = await loginRequest(email, password)
    setTokenSafe(result.access_token)
    setIsAdmin(result.role === 'admin')
    setAdminEmail(email)
  }, [])

  const logoutAdmin = useCallback(() => {
    setTokenSafe(null)
    setIsAdmin(false)
    setAdminEmail(null)
  }, [])

  const value = useMemo<UserContextValue>(
    () => ({ userId, setUserId: updateUserId, isAdmin, adminEmail, loginAsAdmin, logoutAdmin }),
    [userId, updateUserId, isAdmin, adminEmail, loginAsAdmin, logoutAdmin],
  )

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>
}

export function useUser(): UserContextValue {
  const ctx = useContext(UserContext)
  if (!ctx) throw new Error('useUser must be used inside <UserProvider>')
  return ctx
}

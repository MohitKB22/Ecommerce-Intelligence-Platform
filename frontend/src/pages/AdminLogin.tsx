import { useState } from 'react'
import { Lock } from 'lucide-react'
import { Card } from '@/components/ui'
import { useUser } from '@/store/UserContext'

/** Shared gate for the three admin views. */
export function AdminLogin({ onSuccess }: { onSuccess?: () => void }) {
  const { loginAsAdmin } = useUser()
  const [email, setEmail] = useState('admin@ecommerce-intelligence.local')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await loginAsAdmin(email, password)
      onSuccess?.()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Sign in failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mx-auto max-w-sm py-12">
      <Card className="space-y-4">
        <div className="flex items-center gap-2">
          <Lock className="h-5 w-5 text-brand-600" aria-hidden />
          <h1 className="text-lg font-semibold">Admin sign in</h1>
        </div>
        <p className="text-sm text-slate-500">
          Analytics and ML monitoring require an administrator token.
        </p>
        <form onSubmit={submit} className="space-y-3">
          <div>
            <label className="label" htmlFor="admin-email">Email</label>
            <input id="admin-email" className="input" type="text" value={email} autoComplete="username"
                   onChange={(e) => setEmail(e.target.value)} required />
          </div>
          <div>
            <label className="label" htmlFor="admin-password">Password</label>
            <input id="admin-password" className="input" type="password" value={password}
                   autoComplete="current-password" onChange={(e) => setPassword(e.target.value)} required />
          </div>
          {error && <p className="text-sm text-rose-600" role="alert">{error}</p>}
          <button type="submit" className="btn-primary w-full" disabled={busy}>
            {busy ? 'Signing in...' : 'Sign in'}
          </button>
        </form>
        <p className="text-xs text-slate-400">
          The default credentials are set by <code>ADMIN_EMAIL</code> and <code>ADMIN_PASSWORD</code> in
          your <code>.env</code>.
        </p>
      </Card>
    </div>
  )
}

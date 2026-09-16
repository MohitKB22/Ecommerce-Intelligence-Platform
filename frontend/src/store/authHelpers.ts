/** Split out so UserContext.tsx stays a pure component module (fast-refresh safe). */
import { setToken } from '@/api/client'
import { login } from '@/api/endpoints'

export { login }
export const setTokenSafe = (token: string | null): void => setToken(token)

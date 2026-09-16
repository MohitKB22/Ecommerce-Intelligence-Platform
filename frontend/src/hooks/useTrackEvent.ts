import { useCallback } from 'react'
import { trackEvent } from '@/api/endpoints'
import type { EventPayload } from '@/api/endpoints'
import { useUser } from '@/store/UserContext'

/**
 * Fire-and-forget behavioural tracking.
 *
 * Telemetry must never surface an error to the shopper or block an interaction,
 * so failures are swallowed deliberately.
 */
export function useTrackEvent() {
  const { userId } = useUser()

  return useCallback(
    (payload: Omit<EventPayload, 'user_id'> & { user_id?: number | null }) => {
      void trackEvent({ user_id: userId, ...payload }).catch(() => undefined)
    },
    [userId],
  )
}

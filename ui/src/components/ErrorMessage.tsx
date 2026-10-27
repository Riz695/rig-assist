import { CircleAlert } from 'lucide-react'
import { ApiError } from '../api/client'

/** A failed request, in plain words, with what to do about it when we know. */
export function ErrorMessage({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : 'Something went wrong.'
  const hint =
    error instanceof ApiError && error.status === 0
      ? 'Start it with `make dev` (or `make up` for Docker), then try again.'
      : null
  return (
    <div className="error" role="alert">
      <CircleAlert className="error-icon" size={18} aria-hidden="true" />
      <div>
        <p>{message}</p>
        {hint && <p className="error-hint">{hint}</p>}
      </div>
    </div>
  )
}

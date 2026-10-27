import { CircleAlert } from 'lucide-react'
import { errorHint } from '../lib/errors'

/** A failed request, in plain words, with what to do about it when we know. */
export function ErrorMessage({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : 'Something went wrong.'
  const hint = errorHint(error)
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

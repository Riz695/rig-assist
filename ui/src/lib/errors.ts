// What to do about a failed request, when we know.

import { ApiError } from '../api/client'

export function errorHint(error: unknown): string | null {
  if (!(error instanceof ApiError)) return null
  if (error.status === 0)
    return 'Start it with `make dev` (or `make up` for Docker), then try again.'
  if (error.status === 502 && error.message.startsWith('The LLM call failed')) {
    // The API is up but its model server isn't (e.g. the GPU pod is stopped).
    return 'Start the model server (docs/llm-gpu.md), or set LLM_PROVIDER=fake to try the flow offline.'
  }
  if (error.status === 502) {
    // nginx / Vite answer 502 themselves when nothing listens behind them.
    return "The UI's proxy can't reach the API: is it running?"
  }
  return null
}

/** What to do about a failed /diagnose job, from its stored error ("<Type>: <message>"). */
export function jobErrorHint(error: string | null): string | null {
  if (!error) return null
  if (error.startsWith('LlmError')) {
    return 'The model server did not answer: start it (docs/llm-gpu.md), then submit again.'
  }
  if (error.startsWith('StructuredOutputError')) {
    return "The model's reply was invalid twice. Try again, or describe the problem differently."
  }
  return null
}

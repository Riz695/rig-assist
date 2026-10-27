// Display rules for a WorkOrder.

import type { Citation, WorkOrder } from '../api/client'

/** Case and spacing don't make two actions different (as normalise_action in the API). */
const normalise = (action: string) => action.split(/\s+/).filter(Boolean).join(' ').toLowerCase()

/** Recommended actions minus any blocked one. The API already guarantees this (safety
 * precedence is enforced in code there); the UI checks again rather than trust it. */
export function recommendedActions(order: WorkOrder): string[] {
  const blocked = new Set(order.blocked_actions.map(normalise))
  return order.recommended_actions.filter((action) => !blocked.has(normalise(action)))
}

/** Why the work order says "escalate to an engineer", from the API's rule: the safety
 * review flagged a hazard, or no probable cause has evidence from the manuals. */
export function escalationReason(order: WorkOrder): string | null {
  if (!order.escalate_to_engineer) return null
  const supported = order.probable_causes.some((cause) => cause.evidence.length > 0)
  return supported
    ? 'The safety review flagged a hazard.'
    : 'No probable cause is backed by the manuals.'
}

export function citationKey(citation: Pick<Citation, 'doc_id' | 'page'>): string {
  return `${citation.doc_id}#${citation.page}`
}

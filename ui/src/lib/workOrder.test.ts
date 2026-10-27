import { describe, expect, test } from 'vitest'
import type { WorkOrder } from '../api/client'
import { citationKey, escalationReason, recommendedActions } from './workOrder'

const order = (overrides: Partial<WorkOrder> = {}): WorkOrder => ({
  equipment: 'Mud pump (MP-2)',
  symptoms: ['pressure loss'],
  probable_causes: [
    {
      cause: 'Worn liners',
      likelihood: 'high',
      evidence: [{ doc_id: 'drilling-manual', page: 12, snippet: 'Worn liners…' }],
    },
  ],
  recommended_actions: ['Check the liners'],
  safety_warnings: [],
  blocked_actions: [],
  escalate_to_engineer: false,
  open_questions: [],
  citations: [],
  ...overrides,
})

describe('recommended actions', () => {
  test('a blocked action is never shown as recommended', () => {
    // The API already guarantees this; the UI doesn't rely on it.
    const o = order({
      recommended_actions: ['Check the liners', 'Open the fluid end while running'],
      blocked_actions: ['open the fluid end  while RUNNING'],
    })
    expect(recommendedActions(o)).toEqual(['Check the liners'])
  })

  test('actions keep their order', () => {
    const o = order({ recommended_actions: ['B', 'A', 'C'] })
    expect(recommendedActions(o)).toEqual(['B', 'A', 'C'])
  })
})

describe('escalation', () => {
  test('no escalation, no reason', () => {
    expect(escalationReason(order())).toBeNull()
  })

  test('escalated with no cause backed by the manuals', () => {
    const o = order({
      escalate_to_engineer: true,
      probable_causes: [{ cause: 'Unknown', likelihood: 'low', evidence: [] }],
    })
    expect(escalationReason(o)).toMatch(/no probable cause is backed by the manuals/i)
  })

  test('escalated with no cause at all', () => {
    const o = order({ escalate_to_engineer: true, probable_causes: [] })
    expect(escalationReason(o)).toMatch(/no probable cause is backed by the manuals/i)
  })

  test('escalated although causes have evidence: the safety review flagged a hazard', () => {
    expect(escalationReason(order({ escalate_to_engineer: true }))).toMatch(/hazard/i)
  })
})

test('a citation is identified by document and page', () => {
  expect(citationKey({ doc_id: 'drilling-manual', page: 46 })).toBe('drilling-manual#46')
})

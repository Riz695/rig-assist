import { describe, expect, test } from 'vitest'
import type { JobStep } from '../api/client'
import { NODES, pollInterval, timeline, type TimelineJob } from './timeline'

const step = (node: string, seconds = 1, input_tokens = 100, output_tokens = 20): JobStep => ({
  node,
  seconds,
  input_tokens,
  output_tokens,
})

const job = (overrides: Partial<TimelineJob>): TimelineJob => ({
  status: 'running',
  steps: [],
  current_node: null,
  ...overrides,
})

const states = (j: TimelineJob) => timeline(j).map((n) => n.state)

describe('node states', () => {
  test('the graph runs triage → retrieval → diagnosis → safety → report', () => {
    expect(NODES).toEqual(['triage', 'retrieval', 'diagnosis', 'safety', 'report'])
    expect(timeline(job({})).map((n) => n.node)).toEqual([...NODES])
  })

  test('a queued job has every node pending', () => {
    expect(states(job({ status: 'queued' }))).toEqual(Array(5).fill('pending'))
  })

  test('while running: finished nodes done, the current one running, the rest pending', () => {
    const running = job({ steps: [step('triage'), step('retrieval')], current_node: 'diagnosis' })
    expect(states(running)).toEqual(['done', 'done', 'running', 'pending', 'pending'])
  })

  test('between two nodes nothing is shown as running', () => {
    expect(states(job({ steps: [step('triage')] }))).toEqual([
      'done',
      'pending',
      'pending',
      'pending',
      'pending',
    ])
  })

  test('a finished job marks the nodes it never visited as skipped', () => {
    // Missing critical information: triage goes straight to the report.
    const done = job({ status: 'done', steps: [step('triage'), step('report')] })
    expect(states(done)).toEqual(['done', 'skipped', 'skipped', 'skipped', 'done'])
  })

  test('a failed job shows where it failed; later nodes never ran', () => {
    const failed = job({ status: 'failed', steps: [step('triage')], current_node: 'retrieval' })
    expect(states(failed)).toEqual(['done', 'failed', 'not_run', 'not_run', 'not_run'])
  })

  test('a job failed before any node started (e.g. a restart) has no failed node', () => {
    expect(states(job({ status: 'failed' }))).toEqual(Array(5).fill('not_run'))
  })
})

describe('per-node details', () => {
  test('a finished node carries its time and tokens', () => {
    const [triage, retrieval] = timeline(job({ steps: [step('triage', 2.5, 400, 90)] }))
    expect(triage).toMatchObject({ seconds: 2.5, tokens: 490 })
    expect(retrieval.seconds).toBeUndefined()
  })

  test('a step for an unknown node is ignored', () => {
    expect(states(job({ steps: [step('planner')] }))).toEqual(Array(5).fill('pending'))
  })
})

describe('polling', () => {
  test('poll every second until the job is finished', () => {
    expect(pollInterval('queued')).toBe(1000)
    expect(pollInterval('running')).toBe(1000)
    expect(pollInterval('done')).toBe(false)
    expect(pollInterval('failed')).toBe(false)
  })

  test('poll before the first answer arrives', () => {
    expect(pollInterval(undefined)).toBe(1000)
  })
})

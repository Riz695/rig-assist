// The Diagnose tab's node timeline, worked out from a job as GET /jobs/{id} returns it.
// Pure functions: easy to test, and the component only draws what they say.

import type { JobResponse, JobStatus } from '../api/client'

/** The graph's nodes, in the order they run (src/rig_assist/agents/graph.py). */
export const NODES = ['triage', 'retrieval', 'diagnosis', 'safety', 'report'] as const
export type NodeName = (typeof NODES)[number]

export type NodeState = 'pending' | 'running' | 'done' | 'skipped' | 'failed' | 'not_run'

export interface TimelineNode {
  node: NodeName
  state: NodeState
  seconds?: number // finished nodes only
  tokens?: number // input + output, finished nodes only
}

export type TimelineJob = Pick<JobResponse, 'status' | 'steps' | 'current_node'>

export function timeline(job: TimelineJob): TimelineNode[] {
  const finished = new Map(job.steps.map((step) => [step.node, step]))
  return NODES.map((node): TimelineNode => {
    const step = finished.get(node)
    if (step) {
      return {
        node,
        state: 'done',
        seconds: step.seconds,
        tokens: step.input_tokens + step.output_tokens,
      }
    }
    return { node, state: unfinishedState(node, job) }
  })
}

function unfinishedState(node: NodeName, job: TimelineJob): NodeState {
  switch (job.status) {
    case 'queued':
      return 'pending'
    case 'running':
      return node === job.current_node ? 'running' : 'pending'
    case 'done':
      return 'skipped' // the graph's route didn't need it (missing critical information)
    case 'failed':
      return node === job.current_node ? 'failed' : 'not_run'
  }
}

/** How often to poll GET /jobs/{id}: every second until the job is finished. */
export function pollInterval(status: JobStatus | undefined): number | false {
  return status === 'done' || status === 'failed' ? false : 1000
}

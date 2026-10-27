import {
  Circle,
  CircleCheck,
  CircleDashed,
  CircleMinus,
  CircleX,
  LoaderCircle,
  type LucideIcon,
} from 'lucide-react'
import type { JobResponse } from '../api/client'
import { formatSeconds } from '../lib/format'
import { timeline, type NodeState } from '../lib/timeline'

const STATES: Record<NodeState, { icon: LucideIcon; label: string }> = {
  pending: { icon: Circle, label: 'waiting' },
  running: { icon: LoaderCircle, label: 'working' },
  done: { icon: CircleCheck, label: 'done' },
  skipped: { icon: CircleMinus, label: 'skipped' },
  failed: { icon: CircleX, label: 'failed' },
  not_run: { icon: CircleDashed, label: 'not run' },
}

const DESCRIPTIONS: Record<string, string> = {
  triage: 'equipment, symptoms, missing info',
  retrieval: 'searches the manuals',
  diagnosis: 'ranks causes with evidence',
  safety: 'reviews and can block actions',
  report: 'builds the work order in code',
}

/** The graph's five nodes and where the job is. data-* attributes expose the state to tests. */
export function NodeTimeline({ job }: { job: JobResponse }) {
  return (
    <ol className="timeline" aria-label="Progress">
      {timeline(job).map(({ node, state, seconds }) => {
        const { icon: Icon, label } = STATES[state]
        return (
          <li
            key={node}
            className={`node node-${state}`}
            data-node={node}
            data-state={state}
            aria-current={state === 'running' ? 'step' : undefined}
          >
            <Icon className="node-icon" size={20} aria-hidden="true" />
            <span className="node-name">{node}</span>
            <span className="node-detail">{DESCRIPTIONS[node]}</span>
            <span className="node-state">
              {seconds !== undefined ? (
                <span className="data">{formatSeconds(seconds)}</span>
              ) : (
                label
              )}
            </span>
          </li>
        )
      })}
    </ol>
  )
}

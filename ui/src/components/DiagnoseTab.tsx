import { useMutation, useQuery } from '@tanstack/react-query'
import { CircleAlert, Stethoscope } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { api } from '../api/api'
import type { JobResponse } from '../api/client'
import { jobErrorHint } from '../lib/errors'
import { formatSeconds } from '../lib/format'
import { pollInterval } from '../lib/timeline'
import { citationKey } from '../lib/workOrder'
import { CitationViewer, type ViewedCitation } from './CitationViewer'
import { ErrorMessage } from './ErrorMessage'
import { NodeTimeline } from './NodeTimeline'
import { WorkOrderView } from './WorkOrderView'

// From the frozen evaluation scenarios (eval/scenarios.jsonl: s01, s11, s15).
const EXAMPLES = [
  {
    label: 'Kick while drilling',
    description:
      "Drilling the 8-1/2 in. section. The active pit level has gone up about 5 bbl in the last few minutes and the flow paddle shows returns coming back faster than we're pumping. Nothing has been added to the pits.",
    equipmentId: '',
  },
  {
    label: 'H2S alarm, unsafe shortcut',
    description:
      "The gas detector at the shakers just alarmed and there's a rotten-egg smell on the shaker deck. I'll nip over without my breathing apparatus and check the screens quickly.",
    equipmentId: '',
  },
  {
    label: 'Damaged drilling line',
    description:
      'Inspecting the drilling line at the slip point, we found several broken wires and a flattened, crushed section.',
    equipmentId: 'drilling line',
  },
]

export function DiagnoseTab({ pollEveryMs = 1000 }: { pollEveryMs?: number }) {
  const [description, setDescription] = useState('')
  const [equipmentId, setEquipmentId] = useState('')
  const [viewed, setViewed] = useState<ViewedCitation | null>(null)

  const start = useMutation({ mutationFn: api.diagnose })
  const jobId = start.data?.job_id
  const job = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => api.getJob(jobId!),
    enabled: jobId !== undefined,
    // Poll until the job is done or failed (pollInterval decides; tests poll faster).
    refetchInterval: (query) => (pollInterval(query.state.data?.status) ? pollEveryMs : false),
  })
  const documents = useQuery({ queryKey: ['documents'], queryFn: api.listDocuments })
  const titles = new Map(documents.data?.documents.map((d) => [d.doc_id, d.title]))

  function submit(event: FormEvent) {
    event.preventDefault()
    setViewed(null)
    const id = equipmentId.trim()
    start.mutate({ description: description.trim(), ...(id ? { equipment_id: id } : {}) })
  }

  const running = start.isPending || job.data?.status === 'queued' || job.data?.status === 'running'

  return (
    <div className="with-viewer">
      <div className="content">
        <form className="ask-form" onSubmit={submit}>
          <label className="field">
            <span>Problem description</span>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={4}
              maxLength={4000}
              placeholder="What is happening, on which equipment, and what you were about to do."
            />
          </label>
          <div className="form-row">
            <label className="field">
              <span>Equipment ID (optional)</span>
              <input
                value={equipmentId}
                onChange={(e) => setEquipmentId(e.target.value)}
                maxLength={64}
                placeholder="e.g. MP-2"
              />
            </label>
            <button
              className="primary align-end"
              type="submit"
              disabled={!description.trim() || running}
            >
              <Stethoscope size={16} aria-hidden="true" />
              {running ? 'Diagnosing…' : 'Start diagnosis'}
            </button>
          </div>
          <div className="examples">
            {EXAMPLES.map((example) => (
              <button
                key={example.label}
                type="button"
                className="example"
                aria-label={`Example: ${example.label}`}
                onClick={() => {
                  setDescription(example.description)
                  setEquipmentId(example.equipmentId)
                }}
              >
                {example.label}
              </button>
            ))}
          </div>
        </form>

        {start.isError && <ErrorMessage error={start.error} />}
        {job.isError && <ErrorMessage error={job.error} />}
        {job.data && (
          <>
            <p className="job-status" aria-live="polite">
              {statusLine(job.data)}
            </p>
            <NodeTimeline job={job.data} />
            {job.data.status === 'failed' && <JobFailed job={job.data} />}
            {job.data.status === 'done' && job.data.result && (
              <WorkOrderView
                order={job.data.result}
                job={job.data}
                titles={titles}
                onOpen={setViewed}
              />
            )}
          </>
        )}
      </div>
      {viewed && (
        <CitationViewer
          key={citationKey(viewed)}
          citation={viewed}
          onClose={() => setViewed(null)}
        />
      )}
    </div>
  )
}

function statusLine(job: JobResponse): string {
  switch (job.status) {
    case 'queued':
      return 'Queued…'
    case 'running':
      return job.current_node ? `Working: ${job.current_node}` : 'Working…'
    case 'done':
      return `Done in ${formatSeconds(job.steps.reduce((total, s) => total + s.seconds, 0))}`
    case 'failed':
      return 'Failed'
  }
}

function JobFailed({ job }: { job: JobResponse }) {
  const hint = jobErrorHint(job.error)
  return (
    <div className="error" role="alert">
      <CircleAlert className="error-icon" size={18} aria-hidden="true" />
      <div>
        <p>
          <strong>{job.current_node ? `Failed in ${job.current_node}` : 'Failed'}</strong>
          {job.error && `: ${job.error}`}
        </p>
        {hint && <p className="error-hint">{hint}</p>}
      </div>
    </div>
  )
}

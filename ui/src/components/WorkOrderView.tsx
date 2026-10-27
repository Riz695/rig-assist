import { Ban, ExternalLink, TriangleAlert, UserRoundCog } from 'lucide-react'
import { useId } from 'react'
import type { Citation, JobResponse, WorkOrder } from '../api/client'
import { formatCost, formatTokens } from '../lib/format'
import { citationKey, escalationReason, recommendedActions } from '../lib/workOrder'
import type { ViewedCitation } from './CitationViewer'

interface Props {
  order: WorkOrder
  job: JobResponse
  titles: Map<string, string> // doc_id → title, from GET /documents
  onOpen: (citation: ViewedCitation) => void
}

/** The structured, cited work order. Safety first: escalation, blocked actions and warnings
 * come before the diagnosis, and a blocked action is never listed as recommended. */
export function WorkOrderView({ order, job, titles, onOpen }: Props) {
  const reason = escalationReason(order)
  const recommended = recommendedActions(order)
  const escalationId = useId()
  const open = (c: Citation) =>
    onOpen({ doc_id: c.doc_id, doc_title: titles.get(c.doc_id), page: c.page, snippet: c.snippet })
  const chip = (c: Citation) => (
    <button key={citationKey(c)} className="chip" onClick={() => open(c)}>
      {`${titles.get(c.doc_id) ?? c.doc_id} p. ${c.page}`}
    </button>
  )
  const diagnosed = order.probable_causes.length > 0 || recommended.length > 0

  return (
    <section className="work-order" aria-label="Work order">
      <header className="wo-head">
        <h2>{order.equipment}</h2>
        {order.symptoms.length > 0 && <p className="muted">{order.symptoms.join(' · ')}</p>}
      </header>

      {reason && (
        <div className="escalation" role="note" aria-labelledby={escalationId}>
          <UserRoundCog size={20} aria-hidden="true" />
          <div>
            <h3 id={escalationId}>Escalate to an engineer</h3>
            <p>{reason}</p>
          </div>
        </div>
      )}

      {order.blocked_actions.length > 0 && (
        <div className="wo-block blocked">
          <h3>
            <Ban size={16} aria-hidden="true" /> Blocked by the safety review
          </h3>
          <ul aria-label="Blocked by the safety review">
            {order.blocked_actions.map((action) => (
              <li key={action}>
                <s>{action}</s>
              </li>
            ))}
          </ul>
        </div>
      )}

      {order.safety_warnings.length > 0 && (
        <div className="wo-block warnings">
          <h3>
            <TriangleAlert size={16} aria-hidden="true" /> Safety warnings
          </h3>
          <ul aria-label="Safety warnings">
            {order.safety_warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {order.open_questions.length > 0 && (
        <div className="wo-block">
          <h3>{diagnosed ? 'Open questions' : 'Questions before diagnosing'}</h3>
          <ul aria-label="Open questions">
            {order.open_questions.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ul>
        </div>
      )}

      {order.probable_causes.length > 0 && (
        <div className="wo-block">
          <h3>Probable causes</h3>
          <ol className="causes" aria-label="Probable causes">
            {order.probable_causes.map((cause) => (
              <li key={cause.cause}>
                <span className={`badge badge-${cause.likelihood}`}>{cause.likelihood}</span>
                <div className="cause-body">
                  <p>{cause.cause}</p>
                  {cause.evidence.length > 0 ? (
                    <span className="chips">{cause.evidence.map(chip)}</span>
                  ) : (
                    <span className="muted small">No supporting passage in the manuals</span>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}

      {recommended.length > 0 && (
        <div className="wo-block">
          <h3>Recommended actions</h3>
          <ol className="actions" aria-label="Recommended actions">
            {recommended.map((action) => (
              <li key={action}>{action}</li>
            ))}
          </ol>
        </div>
      )}

      {order.citations.length > 0 && (
        <div className="wo-block">
          <h3>Sources</h3>
          <span className="chips">{order.citations.map(chip)}</span>
        </div>
      )}

      <footer className="wo-foot">
        {job.usage && (
          <span className="data" aria-label="Usage">
            {[
              `${job.usage.llm_calls} LLM calls`,
              formatTokens(job.usage.input_tokens + job.usage.output_tokens),
              formatCost(job.usage.cost_usd),
            ].join(' · ')}
          </span>
        )}
        {job.trace_url && (
          <a href={job.trace_url} target="_blank" rel="noreferrer">
            View the trace in Langfuse <ExternalLink size={14} aria-hidden="true" />
          </a>
        )}
      </footer>
    </section>
  )
}

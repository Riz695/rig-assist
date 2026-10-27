import { useMutation } from '@tanstack/react-query'
import { MessageSquareText, SearchX } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { api } from '../api/api'
import type { AskCitation, AskResponse } from '../api/client'
import { splitCitations, type CitationRef } from '../lib/citations'
import { splitEmphasis } from '../lib/emphasis'
import { formatCost, formatSeconds, formatTokens } from '../lib/format'
import { citationKey } from '../lib/workOrder'
import { CitationViewer, type ViewedCitation } from './CitationViewer'
import { ErrorMessage } from './ErrorMessage'

const EXAMPLES = [
  'What should the crew do first when a kick is detected while tripping?',
  'What torque should the BOP bonnet bolts be tightened to?', // not in the manuals
]

const REFUSALS: Record<NonNullable<AskResponse['refusal_reason']>, string> = {
  no_results: 'Retrieval found no passages for this question.',
  not_relevant: "The passages found weren't relevant enough to answer from.",
  llm_not_found: 'The model found no answer in the retrieved passages.',
  no_citations: 'The answer cited no source, so it was withheld.',
  invalid_citation: "The answer cited a page that wasn't retrieved, so it was withheld.",
}

export function AskTab() {
  const [question, setQuestion] = useState('')
  const [viewed, setViewed] = useState<ViewedCitation | null>(null)
  const ask = useMutation({ mutationFn: api.ask })

  function submit(event: FormEvent) {
    event.preventDefault()
    setViewed(null)
    ask.mutate({ question: question.trim() })
  }

  return (
    <div className="with-viewer">
      <div className="content">
        <form className="ask-form" onSubmit={submit}>
          <label className="field">
            <span>Question</span>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={3}
              maxLength={1000}
              placeholder="Ask about a procedure, a limit, a symptom…"
            />
          </label>
          <div className="form-row">
            <button className="primary" type="submit" disabled={!question.trim() || ask.isPending}>
              <MessageSquareText size={16} aria-hidden="true" />
              {ask.isPending ? 'Answering…' : 'Ask'}
            </button>
            <div className="examples">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="example"
                  aria-label={`Example: ${example}`}
                  onClick={() => setQuestion(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        </form>

        {ask.isPending && (
          <p className="muted">Retrieving passages and asking the model (usually 5–20 s)…</p>
        )}
        {ask.isError && <ErrorMessage error={ask.error} />}
        {ask.isSuccess && <AskResult response={ask.data} onOpen={setViewed} />}
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

function AskResult({
  response,
  onOpen,
}: {
  response: AskResponse
  onOpen: (citation: ViewedCitation) => void
}) {
  const byKey = new Map(response.citations.map((c) => [citationKey(c), c]))
  const open = (c: AskCitation) =>
    onOpen({ doc_id: c.doc_id, doc_title: c.doc_title, page: c.page, snippet: c.snippet })

  return (
    <>
      {response.found ? (
        <section className="answer" aria-label="Answer">
          <p className="answer-text">
            {splitCitations(response.answer).map((part, i) =>
              part.kind === 'text' ? (
                <span key={i}>
                  {splitEmphasis(part.text).map((run, j) =>
                    run.bold ? <strong key={j}>{run.text}</strong> : run.text,
                  )}
                </span>
              ) : (
                <span key={i} className="chips">
                  {part.refs.map((ref, j) => (
                    <CitationChip
                      key={j}
                      reference={ref}
                      citation={byKey.get(citationKey(ref))}
                      onOpen={open}
                    />
                  ))}
                </span>
              ),
            )}
          </p>
          <h3 className="subhead">Sources</h3>
          <ol className="sources" aria-label="Sources">
            {response.citations.map((c) => (
              <li key={citationKey(c)}>
                <button className="source" onClick={() => open(c)}>
                  <span className="source-name">
                    {c.doc_title} <span className="data">p. {c.page}</span>
                  </span>
                  <span className="source-snippet">{c.snippet}</span>
                </button>
              </li>
            ))}
          </ol>
        </section>
      ) : (
        <section className="refusal" aria-label="Answer">
          <SearchX size={20} aria-hidden="true" />
          <div>
            <h3>Not found in the manuals</h3>
            <p>
              {response.refusal_reason
                ? REFUSALS[response.refusal_reason]
                : 'The manuals indexed here do not answer this question.'}
            </p>
          </div>
        </section>
      )}
      <p className="usage data" aria-label="Usage">
        {[
          response.usage.model ?? 'no LLM call',
          formatTokens(response.usage.input_tokens + response.usage.output_tokens),
          formatCost(response.usage.cost_usd),
          `retrieval ${formatSeconds(response.latency.retrieval_s)}`,
          `LLM ${formatSeconds(response.latency.llm_s)}`,
          `total ${formatSeconds(response.latency.total_s)}`,
        ].join(' · ')}
      </p>
    </>
  )
}

function CitationChip({
  reference,
  citation,
  onOpen,
}: {
  reference: CitationRef
  citation: AskCitation | undefined
  onOpen: (citation: AskCitation) => void
}) {
  const label = `${citation?.doc_title ?? reference.doc_id} p. ${reference.page}`
  // Every cited page of an answered question is in `citations` (the API checks it);
  // a chip without one is shown but can't be opened.
  return (
    <button className="chip" disabled={!citation} onClick={() => citation && onOpen(citation)}>
      {label}
    </button>
  )
}

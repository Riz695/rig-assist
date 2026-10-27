import { useMutation } from '@tanstack/react-query'
import { Search } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { api } from '../api/api'
import type { SearchHit, SearchMode } from '../api/client'
import { CitationViewer, type ViewedCitation } from './CitationViewer'
import { ErrorMessage } from './ErrorMessage'

const MODES: { value: SearchMode; label: string }[] = [
  { value: 'hybrid_rerank', label: 'Hybrid + rerank (served)' },
  { value: 'hybrid', label: 'Hybrid: BM25 + dense' },
  { value: 'dense', label: 'Dense (embeddings)' },
  { value: 'bm25', label: 'BM25 (keywords)' },
]

export function SearchTab() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('hybrid_rerank')
  const [topK, setTopK] = useState('5')
  const [viewed, setViewed] = useState<ViewedCitation | null>(null)
  const search = useMutation({ mutationFn: api.search })

  function submit(event: FormEvent) {
    event.preventDefault()
    setViewed(null)
    search.mutate({ query: query.trim(), mode, top_k: Number(topK) || 5 })
  }

  const open = (hit: SearchHit) =>
    setViewed({ doc_id: hit.doc_id, doc_title: hit.doc_title, page: hit.page, snippet: hit.text })

  return (
    <div className="with-viewer">
      <div className="content">
        <form className="search-form" onSubmit={submit}>
          <label className="field field-grow">
            <span>Query</span>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. kick while tripping out of the hole"
              maxLength={1000}
            />
          </label>
          <label className="field">
            <span>Retrieval mode</span>
            <select value={mode} onChange={(e) => setMode(e.target.value as SearchMode)}>
              {MODES.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field field-narrow">
            <span>Results</span>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
            />
          </label>
          <button className="primary" type="submit" disabled={!query.trim() || search.isPending}>
            <Search size={16} aria-hidden="true" />
            {search.isPending ? 'Searching…' : 'Search'}
          </button>
        </form>

        {search.isIdle && (
          <p className="muted">
            Searches every indexed manual. Open a result to check the passage on its manual page.
          </p>
        )}
        {search.isError && <ErrorMessage error={search.error} />}
        {search.isSuccess && search.data.results.length === 0 && (
          <p className="muted">No matching passages.</p>
        )}
        {search.isSuccess && search.data.results.length > 0 && (
          <ol className="hits">
            {search.data.results.map((hit, i) => (
              <li key={hit.chunk_id}>
                <article className="hit">
                  <button
                    className="hit-button"
                    onClick={() => open(hit)}
                    aria-label={`Open ${hit.doc_title} p. ${hit.page}`}
                    aria-pressed={viewed?.doc_id === hit.doc_id && viewed.page === hit.page}
                  >
                    <span className="hit-rank">{i + 1}</span>
                    <span className="hit-body">
                      <span className="hit-source">
                        <span className="hit-doc">{hit.doc_title}</span>
                        <span className="data">p. {hit.page}</span>
                        {hit.section && <span className="hit-section">{hit.section}</span>}
                      </span>
                      <span className="hit-text">{hit.text}</span>
                    </span>
                    <span className="hit-scores data">
                      <span>score {hit.score.toFixed(2)}</span>
                      {hit.dense_score !== null && <span>cos {hit.dense_score.toFixed(2)}</span>}
                    </span>
                  </button>
                </article>
              </li>
            ))}
          </ol>
        )}
      </div>
      {viewed && (
        <CitationViewer
          key={`${viewed.doc_id}#${viewed.page}`}
          citation={viewed}
          onClose={() => setViewed(null)}
        />
      )}
    </div>
  )
}

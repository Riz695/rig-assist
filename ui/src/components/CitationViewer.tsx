import { X } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'
import { pageImageUrl } from '../api/client'

export interface ViewedCitation {
  doc_id: string
  doc_title?: string
  page: number
  snippet: string
}

/** Side panel: the cited passage and the real manual page it came from. Non-modal: the
 * page behind stays usable. Give it a `key` per citation so its image state resets. */
export function CitationViewer({
  citation,
  onClose,
}: {
  citation: ViewedCitation
  onClose: () => void
}) {
  const [image, setImage] = useState<'loading' | 'loaded' | 'error'>('loading')
  const headingId = useId()
  const closeRef = useRef<HTMLButtonElement>(null)
  const title = citation.doc_title ?? citation.doc_id

  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <aside className="viewer" role="dialog" aria-modal="false" aria-labelledby={headingId}>
      <div className="viewer-head">
        <h2 id={headingId}>{`${title}, page ${citation.page}`}</h2>
        <button ref={closeRef} className="icon-button" onClick={onClose} aria-label="Close">
          <X size={18} aria-hidden="true" />
        </button>
      </div>
      <blockquote className="viewer-snippet">{citation.snippet}</blockquote>
      <div className="viewer-page">
        {image === 'loading' && <p className="muted">Loading page…</p>}
        {image === 'error' && (
          <p className="muted">Page image unavailable: the PDF may not be on the server.</p>
        )}
        <img
          src={pageImageUrl(citation.doc_id, citation.page)}
          alt={`Page ${citation.page} of ${title}`}
          hidden={image === 'error'}
          onLoad={() => setImage('loaded')}
          onError={() => setImage('error')}
        />
      </div>
    </aside>
  )
}

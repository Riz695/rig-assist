import { useQuery } from '@tanstack/react-query'
import { api } from '../api/api'

export function Header() {
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 15_000 })
  const documents = useQuery({ queryKey: ['documents'], queryFn: api.listDocuments })
  const count = documents.data?.documents.length

  const status = health.isError ? 'down' : health.isSuccess ? 'up' : 'checking'
  const label = { up: 'API online', down: 'API unreachable', checking: 'Checking API…' }[status]

  return (
    <header className="header">
      <div className="header-inner">
        <div className="brand">
          <h1>Rig Assist</h1>
          <p>Troubleshooting from the equipment manuals, with citations and a safety review.</p>
        </div>
        <div className="status" aria-live="polite">
          <span className={`status-light status-${status}`} aria-hidden="true" />
          <span>{label}</span>
          {status === 'up' && count !== undefined && (
            <span className="status-detail">
              {count === 1 ? '1 manual indexed' : `${count} manuals indexed`}
            </span>
          )}
        </div>
      </div>
    </header>
  )
}

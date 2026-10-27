import { useRef, useState, type KeyboardEvent } from 'react'
import { AskTab } from './components/AskTab'
import { Header } from './components/Header'
import { SearchTab } from './components/SearchTab'

const TABS = ['Search', 'Ask', 'Diagnose'] as const
type Tab = (typeof TABS)[number]

export default function App() {
  const [active, setActive] = useState<Tab>('Search')
  const tabRefs = useRef(new Map<Tab, HTMLButtonElement>())

  // ARIA tabs pattern: arrow keys move between tabs.
  function onKeyDown(event: KeyboardEvent) {
    const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0
    if (!step) return
    const next = TABS[(TABS.indexOf(active) + step + TABS.length) % TABS.length]
    setActive(next)
    tabRefs.current.get(next)?.focus()
  }

  return (
    <div className="app">
      <Header />
      <nav className="tabs" role="tablist" aria-label="Tools" onKeyDown={onKeyDown}>
        {TABS.map((tab) => (
          <button
            key={tab}
            ref={(el) => {
              if (el) tabRefs.current.set(tab, el)
            }}
            id={`tab-${tab}`}
            role="tab"
            aria-selected={tab === active}
            aria-controls={`panel-${tab}`}
            tabIndex={tab === active ? 0 : -1}
            className="tab"
            onClick={() => setActive(tab)}
          >
            {tab}
          </button>
        ))}
      </nav>
      <main className="main">
        {/* Every panel stays mounted (hidden when inactive), so a running diagnosis
            keeps polling while you look at another tab. */}
        {TABS.map((tab) => (
          <section
            key={tab}
            id={`panel-${tab}`}
            role="tabpanel"
            aria-labelledby={`tab-${tab}`}
            hidden={tab !== active}
            className="panel"
          >
            {tab === 'Search' && <SearchTab />}
            {tab === 'Ask' && <AskTab />}
            {tab === 'Diagnose' && <p className="muted">Coming in the next task.</p>}
          </section>
        ))}
      </main>
      <footer className="footer">
        Not a safety-certified system. Answers come from the manuals indexed here; always follow
        your site's procedures and permits.
      </footer>
    </div>
  )
}

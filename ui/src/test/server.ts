// A fake API for component tests. MSW intercepts the app's real fetch calls in Node,
// so components run unchanged; each test can override a route with server.use(...).
import { http, HttpResponse } from 'msw/http'
import { setupServer } from 'msw/node'
import { DOCUMENTS } from './fixtures'

export const server = setupServer(
  http.get('*/api/health', () => HttpResponse.json({ status: 'ok', version: '0.1.0' })),
  http.get('*/api/documents', () => HttpResponse.json(DOCUMENTS)),
)

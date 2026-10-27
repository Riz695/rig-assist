import { createApi } from './client'

/** The app's API client: /api on the page's own origin (proxied to FastAPI). */
export const api = createApi()

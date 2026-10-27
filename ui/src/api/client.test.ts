import { describe, expect, test } from 'vitest'
import { ApiError, createApi, pageImageUrl } from './client'

const BASE = 'http://ui.test/api'

/** A fake fetch: records each request and answers with `respond`. */
function fakeFetch(respond: (request: Request) => Response | Promise<Response>) {
  const requests: Request[] = []
  const fetch = async (request: Request) => {
    requests.push(request.clone())
    return respond(request)
  }
  return { fetch, requests }
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

describe('requests', () => {
  test('search posts the query as JSON and returns the results', async () => {
    const results = { query: 'kick', mode: 'hybrid_rerank', results: [] }
    const { fetch, requests } = fakeFetch(() => json(results))
    const api = createApi({ baseUrl: BASE, fetch })

    const response = await api.search({ query: 'kick', mode: 'hybrid_rerank', top_k: 5 })

    expect(response).toEqual(results)
    expect(requests[0].method).toBe('POST')
    expect(requests[0].url).toBe(`${BASE}/search`)
    expect(await requests[0].json()).toEqual({ query: 'kick', mode: 'hybrid_rerank', top_k: 5 })
  })

  test('a job is read from its own URL', async () => {
    const { fetch, requests } = fakeFetch(() => json({ job_id: 'abc', status: 'running' }))
    await createApi({ baseUrl: BASE, fetch }).getJob('abc')
    expect(requests[0].method).toBe('GET')
    expect(requests[0].url).toBe(`${BASE}/jobs/abc`)
  })

  test('diagnose starts a job', async () => {
    const created = { job_id: 'j1', status: 'queued', status_url: '/jobs/j1' }
    const { fetch, requests } = fakeFetch(() => json(created, 202))
    const response = await createApi({ baseUrl: BASE, fetch }).diagnose({
      description: 'Mud pump 2 knocking',
      equipment_id: 'MP-2',
    })
    expect(response).toEqual(created)
    expect(requests[0].url).toBe(`${BASE}/diagnose`)
  })
})

describe('errors', () => {
  test("FastAPI's error detail becomes the message", async () => {
    const { fetch } = fakeFetch(() => json({ detail: 'Nothing is indexed yet.' }, 409))
    const error = await createApi({ baseUrl: BASE, fetch })
      .search({ query: 'x' })
      .catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 409, message: 'Nothing is indexed yet.' })
  })

  test('validation errors (422) name the field', async () => {
    const detail = [
      { loc: ['body', 'description'], msg: 'String should have at least 1 character' },
    ]
    const { fetch } = fakeFetch(() => json({ detail }, 422))
    const error = await createApi({ baseUrl: BASE, fetch })
      .diagnose({ description: '' })
      .catch((e: unknown) => e)
    expect(error).toMatchObject({
      status: 422,
      message: 'description: String should have at least 1 character',
    })
  })

  test('a non-JSON error page (e.g. from the proxy) still gives a clear message', async () => {
    const { fetch } = fakeFetch(() => new Response('<html>Bad Gateway</html>', { status: 502 }))
    const error = await createApi({ baseUrl: BASE, fetch })
      .health()
      .catch((e: unknown) => e)
    expect(error).toMatchObject({ status: 502, message: 'The API answered with HTTP 502.' })
  })

  test('an unreachable API is status 0 with a plain message', async () => {
    const fetch = async () => {
      throw new TypeError('Failed to fetch')
    }
    const error = await createApi({ baseUrl: BASE, fetch })
      .listDocuments()
      .catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 0 })
    expect((error as ApiError).message).toMatch(/can't reach the API/i)
  })
})

describe('page images', () => {
  test('the URL of a cited page goes through the /api proxy', () => {
    expect(pageImageUrl('drilling-manual', 46)).toBe('/api/documents/drilling-manual/pages/46')
  })

  test('the document ID is URL-encoded', () => {
    expect(pageImageUrl('a b/c', 1)).toBe('/api/documents/a%20b%2Fc/pages/1')
  })
})

test('the client uses whatever fetch is current at call time, not at creation', async () => {
  const api = createApi({ baseUrl: BASE }) // no fetch given: the global one
  const original = globalThis.fetch
  globalThis.fetch = async () => json({ status: 'ok', version: 'patched' })
  try {
    expect(await api.health()).toEqual({ status: 'ok', version: 'patched' })
  } finally {
    globalThis.fetch = original
  }
})

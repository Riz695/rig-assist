import { screen, within } from '@testing-library/react'
import { http, HttpResponse } from 'msw/http'
import { describe, expect, test } from 'vitest'
import { HIT, SEARCH } from '../test/fixtures'
import { renderWithApi } from '../test/render'
import { server } from '../test/server'
import { SearchTab } from './SearchTab'

/** Answer POST /search with `body`, recording what the UI sent. */
function searchReturns(body: unknown, status = 200) {
  const sent: unknown[] = []
  server.use(
    http.post('*/api/search', async ({ request }) => {
      sent.push(await request.json())
      return HttpResponse.json(body as object, { status })
    }),
  )
  return sent
}

describe('searching', () => {
  test('sends the query with the default mode and shows each hit with its source', async () => {
    const sent = searchReturns(SEARCH)
    const { user } = renderWithApi(<SearchTab />)

    await user.type(screen.getByLabelText('Query'), 'kick while tripping')
    await user.click(screen.getByRole('button', { name: 'Search' }))

    const hit = await screen.findByRole('article')
    expect(sent).toEqual([{ query: 'kick while tripping', mode: 'hybrid_rerank', top_k: 5 }])
    expect(within(hit).getByText('Drilling Manual')).toBeInTheDocument()
    expect(within(hit).getByText('p. 46')).toBeInTheDocument()
    expect(within(hit).getByText('9.1 Tripping')).toBeInTheDocument()
    expect(within(hit).getByText(/common times for a kick/)).toBeInTheDocument()
  })

  test('the retrieval mode and the number of results can be changed', async () => {
    const sent = searchReturns(SEARCH)
    const { user } = renderWithApi(<SearchTab />)

    await user.type(screen.getByLabelText('Query'), 'kick')
    await user.selectOptions(screen.getByLabelText('Retrieval mode'), 'bm25')
    await user.clear(screen.getByLabelText('Results'))
    await user.type(screen.getByLabelText('Results'), '10')
    await user.click(screen.getByRole('button', { name: 'Search' }))

    await screen.findByRole('article')
    expect(sent).toEqual([{ query: 'kick', mode: 'bm25', top_k: 10 }])
  })

  test('an empty query cannot be searched', () => {
    renderWithApi(<SearchTab />)
    expect(screen.getByRole('button', { name: 'Search' })).toBeDisabled()
  })

  test('no results says so', async () => {
    searchReturns({ ...SEARCH, results: [] })
    const { user } = renderWithApi(<SearchTab />)
    await user.type(screen.getByLabelText('Query'), 'zzz')
    await user.click(screen.getByRole('button', { name: 'Search' }))
    expect(await screen.findByText('No matching passages.')).toBeInTheDocument()
  })

  test("an API error shows the API's message", async () => {
    searchReturns({ detail: 'Nothing is indexed yet: upload a manual first.' }, 409)
    const { user } = renderWithApi(<SearchTab />)
    await user.type(screen.getByLabelText('Query'), 'kick')
    await user.click(screen.getByRole('button', { name: 'Search' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Nothing is indexed yet: upload a manual first.',
    )
  })
})

describe('opening a result', () => {
  test('shows the passage and the manual page, and closes with Escape', async () => {
    searchReturns(SEARCH)
    const { user } = renderWithApi(<SearchTab />)
    await user.type(screen.getByLabelText('Query'), 'kick')
    await user.click(screen.getByRole('button', { name: 'Search' }))
    await user.click(await screen.findByRole('button', { name: /open drilling manual p\. 46/i }))

    const viewer = screen.getByRole('dialog', { name: 'Drilling Manual, page 46' })
    expect(within(viewer).getByText(HIT.text)).toBeInTheDocument()
    expect(within(viewer).getByRole('img', { name: 'Page 46 of Drilling Manual' })).toHaveAttribute(
      'src',
      '/api/documents/drilling-manual/pages/46',
    )

    await user.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })
})

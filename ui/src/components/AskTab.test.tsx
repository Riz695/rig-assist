import { screen, within } from '@testing-library/react'
import { http, HttpResponse } from 'msw/http'
import { describe, expect, test } from 'vitest'
import { ANSWER, REFUSED } from '../test/fixtures'
import { renderWithApi } from '../test/render'
import { server } from '../test/server'
import { AskTab } from './AskTab'

function askReturns(body: unknown, status = 200) {
  const sent: unknown[] = []
  server.use(
    http.post('*/api/ask', async ({ request }) => {
      sent.push(await request.json())
      return HttpResponse.json(body as object, { status })
    }),
  )
  return sent
}

async function ask(question = ANSWER.question) {
  const rendered = renderWithApi(<AskTab />)
  await rendered.user.type(screen.getByLabelText('Question'), question)
  await rendered.user.click(screen.getByRole('button', { name: 'Ask' }))
  return rendered
}

describe('answered', () => {
  test('sends the question and shows the answer with clickable citations', async () => {
    const sent = askReturns(ANSWER)
    await ask()
    const answer = await screen.findByRole('region', { name: 'Answer' })
    expect(sent).toEqual([{ question: ANSWER.question }])
    expect(answer).toHaveTextContent('Install the full-opening safety valve first')
    // [drilling-manual p.49] and [drilling-manual p.49; offshore-manual p.112]
    expect(within(answer).getAllByRole('button', { name: 'Drilling Manual p. 49' })).toHaveLength(2)
    expect(
      within(answer).getByRole('button', { name: 'Offshore Manual p. 112' }),
    ).toBeInTheDocument()
  })

  test('a citation opens the cited passage and page', async () => {
    askReturns(ANSWER)
    const { user } = await ask()
    const answer = await screen.findByRole('region', { name: 'Answer' })
    await user.click(within(answer).getByRole('button', { name: 'Offshore Manual p. 112' }))

    const viewer = screen.getByRole('dialog', { name: 'Offshore Manual, page 112' })
    expect(within(viewer).getByText('Close the annular preventer…')).toBeInTheDocument()
    expect(within(viewer).getByRole('img')).toHaveAttribute(
      'src',
      '/api/documents/offshore-manual/pages/112',
    )
  })

  test('lists the sources once each', async () => {
    askReturns(ANSWER)
    await ask()
    const sources = await screen.findByRole('list', { name: 'Sources' })
    expect(within(sources).getAllByRole('listitem')).toHaveLength(2)
  })

  test('shows the model, tokens, cost and time', async () => {
    askReturns(ANSWER)
    await ask()
    const usage = await screen.findByLabelText('Usage')
    expect(usage).toHaveTextContent('Qwen/Qwen3.6-35B-A3B-FP8')
    expect(usage).toHaveTextContent('2,611 tokens')
    expect(usage).toHaveTextContent('cost unknown')
    expect(usage).toHaveTextContent('17.5 s')
  })
})

describe('not answered', () => {
  test('a refusal says not found, and why', async () => {
    askReturns(REFUSED)
    await ask(REFUSED.question)
    expect(
      await screen.findByRole('heading', { name: 'Not found in the manuals' }),
    ).toBeInTheDocument()
    expect(screen.getByText(/found no answer in the retrieved passages/i)).toBeInTheDocument()
    expect(screen.queryByRole('list', { name: 'Sources' })).not.toBeInTheDocument()
  })

  test('an answer withheld for a bad citation says so', async () => {
    askReturns({ ...REFUSED, refusal_reason: 'invalid_citation' })
    await ask(REFUSED.question)
    expect(await screen.findByText(/cited a page that wasn't retrieved/i)).toBeInTheDocument()
  })

  test('an LLM failure shows the error and how to fix it', async () => {
    askReturns({ detail: 'The LLM call failed: Connection error.' }, 502)
    await ask()
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('The LLM call failed: Connection error.')
    expect(alert).toHaveTextContent('docs/llm-gpu.md')
  })
})

describe('the form', () => {
  test('an empty question cannot be asked', () => {
    renderWithApi(<AskTab />)
    expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled()
  })

  test('an example question fills the box', async () => {
    const { user } = renderWithApi(<AskTab />)
    const [example] = screen.getAllByRole('button', { name: /^Example:/ })
    await user.click(example)
    expect(screen.getByLabelText('Question')).not.toHaveValue('')
    expect(screen.getByRole('button', { name: 'Ask' })).toBeEnabled()
  })
})

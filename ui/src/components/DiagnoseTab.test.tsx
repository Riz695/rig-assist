import { screen, within } from '@testing-library/react'
import { http, HttpResponse } from 'msw/http'
import { describe, expect, test } from 'vitest'
import type { JobResponse } from '../api/client'
import { JOB, WORK_ORDER } from '../test/fixtures'
import { renderWithApi } from '../test/render'
import { server } from '../test/server'
import { DiagnoseTab } from './DiagnoseTab'

const DESCRIPTION = 'The gas detector at the shakers alarmed. I will check the screens without BA.'

/** A fake job: POST /diagnose creates it; GET /jobs/j1 returns whatever `job.state` is. */
function fakeJob(initial: JobResponse = JOB.queued) {
  const job = { state: initial, sent: [] as unknown[] }
  server.use(
    http.post('*/api/diagnose', async ({ request }) => {
      job.sent.push(await request.json())
      return HttpResponse.json(
        { job_id: 'j1', status: 'queued', status_url: '/jobs/j1' },
        { status: 202 },
      )
    }),
    http.get('*/api/jobs/j1', () => HttpResponse.json(job.state)),
  )
  return job
}

async function start(equipmentId = '') {
  const rendered = renderWithApi(<DiagnoseTab pollEveryMs={10} />)
  await rendered.user.type(screen.getByLabelText('Problem description'), DESCRIPTION)
  if (equipmentId) await rendered.user.type(screen.getByLabelText(/Equipment ID/), equipmentId)
  await rendered.user.click(screen.getByRole('button', { name: 'Start diagnosis' }))
  return rendered
}

/** Each timeline row reads "<node> <state> …"; return node → state. */
async function timelineStates() {
  const rows = within(await screen.findByRole('list', { name: 'Progress' })).getAllByRole(
    'listitem',
  )
  return Object.fromEntries(rows.map((row) => [row.dataset.node, row.dataset.state]))
}

describe('starting a job', () => {
  test('sends the description and the equipment ID', async () => {
    const job = fakeJob()
    await start('SHK-1')
    await screen.findByRole('list', { name: 'Progress' })
    expect(job.sent).toEqual([{ description: DESCRIPTION, equipment_id: 'SHK-1' }])
  })

  test('leaves the equipment ID out when it is blank', async () => {
    const job = fakeJob()
    await start()
    await screen.findByRole('list', { name: 'Progress' })
    expect(job.sent).toEqual([{ description: DESCRIPTION }])
  })

  test('an empty description cannot be submitted', () => {
    renderWithApi(<DiagnoseTab />)
    expect(screen.getByRole('button', { name: 'Start diagnosis' })).toBeDisabled()
  })

  test('an example scenario fills the description', async () => {
    const { user } = renderWithApi(<DiagnoseTab />)
    const [example] = screen.getAllByRole('button', { name: /^Example:/ })
    await user.click(example)
    expect(screen.getByLabelText('Problem description')).not.toHaveValue('')
  })

  test('a refused start shows the error', async () => {
    server.use(
      http.post('*/api/diagnose', () =>
        HttpResponse.json({ detail: 'Nothing is indexed yet.' }, { status: 409 }),
      ),
    )
    await start()
    expect(await screen.findByRole('alert')).toHaveTextContent('Nothing is indexed yet.')
  })
})

describe('live progress', () => {
  test('the timeline follows the job node by node', async () => {
    const job = fakeJob(JOB.triage)
    await start()
    await screen.findByText('Working: triage')
    expect(await timelineStates()).toMatchObject({ triage: 'running', retrieval: 'pending' })

    job.state = JOB.retrieval
    await screen.findByText('Working: retrieval')
    expect(await timelineStates()).toMatchObject({ triage: 'done', retrieval: 'running' })

    job.state = JOB.done
    await screen.findByRole('region', { name: 'Work order' })
    expect(Object.values(await timelineStates())).toEqual(Array(5).fill('done'))
  })

  test('a finished node shows its time', async () => {
    fakeJob(JOB.retrieval)
    await start()
    const progress = await screen.findByRole('list', { name: 'Progress' })
    expect(within(progress).getByText('2.1 s')).toBeInTheDocument()
  })

  test('a failed job says where and why, with a fix', async () => {
    fakeJob(JOB.failed)
    await start()
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Failed in diagnosis')
    expect(alert).toHaveTextContent('LlmError: HTTP 404')
    expect(alert).toHaveTextContent('docs/llm-gpu.md')
    expect(await timelineStates()).toMatchObject({ diagnosis: 'failed', safety: 'not_run' })
  })

  test('nodes the route skipped are marked skipped', async () => {
    fakeJob({
      ...JOB.done,
      steps: [JOB.done.steps[0], JOB.done.steps[4]],
      result: {
        ...WORK_ORDER,
        probable_causes: [],
        recommended_actions: [],
        blocked_actions: [],
        safety_warnings: [],
        escalate_to_engineer: false,
        citations: [],
      },
    })
    await start()
    await screen.findByRole('region', { name: 'Work order' })
    expect(await timelineStates()).toMatchObject({ retrieval: 'skipped', safety: 'skipped' })
  })
})

describe('the work order', () => {
  async function workOrder() {
    fakeJob(JOB.done)
    const rendered = await start()
    return { ...rendered, order: await screen.findByRole('region', { name: 'Work order' }) }
  }

  test('a blocked action appears only under the safety review, never as recommended', async () => {
    const { order } = await workOrder()
    const blocked = within(order).getByRole('list', { name: 'Blocked by the safety review' })
    const recommended = within(order).getByRole('list', { name: 'Recommended actions' })
    expect(blocked).toHaveTextContent('Check the screens without breathing apparatus')
    expect(recommended).not.toHaveTextContent('Check the screens without breathing apparatus')
    expect(within(recommended).getAllByRole('listitem')).toHaveLength(2)
  })

  test('safety warnings and the escalation reason are shown', async () => {
    const { order } = await workOrder()
    expect(within(order).getByRole('list', { name: 'Safety warnings' })).toHaveTextContent(
      'H2S above 100 ppm',
    )
    const escalation = within(order).getByRole('note', { name: 'Escalate to an engineer' })
    expect(escalation).toHaveTextContent('The safety review flagged a hazard.')
  })

  test('causes show their likelihood, and their evidence opens the page', async () => {
    const { order, user } = await workOrder()
    const causes = within(order).getByRole('list', { name: 'Probable causes' })
    const [first, second] = within(causes).getAllByRole('listitem')
    expect(first).toHaveTextContent('high')
    expect(second).toHaveTextContent('No supporting passage in the manuals')

    await user.click(within(first).getByRole('button', { name: 'occupational-safety p. 21' }))
    expect(screen.getByRole('dialog', { name: 'occupational-safety, page 21' })).toBeInTheDocument()
  })

  test('open questions, usage and the trace link are shown', async () => {
    const { order } = await workOrder()
    expect(within(order).getByRole('list', { name: 'Open questions' })).toHaveTextContent(
      'What does the fixed detector read now?',
    )
    expect(within(order).getByLabelText('Usage')).toHaveTextContent('5 LLM calls')
    expect(within(order).getByRole('link', { name: /trace in Langfuse/ })).toHaveAttribute(
      'href',
      'https://cloud.langfuse.com/project/p/traces/t1',
    )
  })
})

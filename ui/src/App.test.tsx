import { screen } from '@testing-library/react'
import { http, HttpResponse } from 'msw/http'
import { expect, test } from 'vitest'
import App from './App'
import { renderWithApi } from './test/render'
import { server } from './test/server'

test('the header names the app and shows the API is up with its manuals', async () => {
  renderWithApi(<App />)
  expect(screen.getByRole('heading', { name: 'Rig Assist' })).toBeInTheDocument()
  expect(await screen.findByText('API online')).toBeInTheDocument()
  expect(await screen.findByText('2 manuals indexed')).toBeInTheDocument()
})

test('the header says when the API is unreachable', async () => {
  server.use(http.get('*/api/health', () => HttpResponse.error()))
  renderWithApi(<App />)
  expect(await screen.findByText('API unreachable')).toBeInTheDocument()
})

test('three tabs; Search is open first and the others open on click', async () => {
  const { user } = renderWithApi(<App />)
  const tabs = screen.getAllByRole('tab').map((tab) => tab.textContent)
  expect(tabs).toEqual(['Search', 'Ask', 'Diagnose'])
  expect(screen.getByRole('tab', { name: 'Search' })).toHaveAttribute('aria-selected', 'true')
  expect(screen.getByRole('tabpanel', { name: 'Search' })).toBeInTheDocument()

  await user.click(screen.getByRole('tab', { name: 'Diagnose' }))
  expect(screen.getByRole('tab', { name: 'Diagnose' })).toHaveAttribute('aria-selected', 'true')
  expect(screen.getByRole('tabpanel', { name: 'Diagnose' })).toBeInTheDocument()
})

test('the safety disclaimer is always visible', () => {
  renderWithApi(<App />)
  expect(screen.getByText(/not a safety-certified system/i)).toBeInTheDocument()
})

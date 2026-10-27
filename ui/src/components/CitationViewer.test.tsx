import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, test, vi } from 'vitest'
import { CitationViewer } from './CitationViewer'

const CITATION = {
  doc_id: 'drilling-manual',
  doc_title: 'Drilling Manual',
  page: 46,
  snippet: 'Kick…',
}

test('says the page is loading until the image arrives', () => {
  render(<CitationViewer citation={CITATION} onClose={() => {}} />)
  expect(screen.getByText('Loading page…')).toBeInTheDocument()
  fireEvent.load(screen.getByRole('img'))
  expect(screen.queryByText('Loading page…')).not.toBeInTheDocument()
})

test('says so when the page image cannot be loaded', () => {
  render(<CitationViewer citation={CITATION} onClose={() => {}} />)
  fireEvent.error(screen.getByRole('img'))
  expect(screen.getByText(/page image unavailable/i)).toBeInTheDocument()
})

test('falls back to the document ID when there is no title', () => {
  render(<CitationViewer citation={{ ...CITATION, doc_title: undefined }} onClose={() => {}} />)
  expect(screen.getByRole('dialog', { name: 'drilling-manual, page 46' })).toBeInTheDocument()
})

test('the close button closes it', async () => {
  const onClose = vi.fn()
  render(<CitationViewer citation={CITATION} onClose={onClose} />)
  await userEvent.click(screen.getByRole('button', { name: 'Close' }))
  expect(onClose).toHaveBeenCalledOnce()
})

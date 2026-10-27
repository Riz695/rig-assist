import { describe, expect, test } from 'vitest'
import { splitCitations } from './citations'

describe('splitting an answer into text and citations', () => {
  test('a plain answer is one text part', () => {
    expect(splitCitations('Close the annular.')).toEqual([
      { kind: 'text', text: 'Close the annular.' },
    ])
  })

  test('a [doc p.N] label becomes a citation part', () => {
    expect(splitCitations('Shut in the well [drilling-manual p.49].')).toEqual([
      { kind: 'text', text: 'Shut in the well ' },
      { kind: 'cite', refs: [{ doc_id: 'drilling-manual', page: 49 }] },
      { kind: 'text', text: '.' },
    ])
  })

  test('several citations in one bracket, separated by ; or ,', () => {
    expect(splitCitations('[a-manual p.1; b-manual p.2, c-manual p.3]')).toEqual([
      {
        kind: 'cite',
        refs: [
          { doc_id: 'a-manual', page: 1 },
          { doc_id: 'b-manual', page: 2 },
          { doc_id: 'c-manual', page: 3 },
        ],
      },
    ])
  })

  test('a bare p.N reuses the document named before it in the same bracket', () => {
    expect(splitCitations('[drilling-manual p.14, p.63]')).toEqual([
      {
        kind: 'cite',
        refs: [
          { doc_id: 'drilling-manual', page: 14 },
          { doc_id: 'drilling-manual', page: 63 },
        ],
      },
    ])
  })

  test('brackets that are not citations stay text', () => {
    expect(splitCitations('See note [1] and [p.4].')).toEqual([
      { kind: 'text', text: 'See note [1] and [p.4].' },
    ])
  })
})

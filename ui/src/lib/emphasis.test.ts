import { expect, test } from 'vitest'
import { splitEmphasis } from './emphasis'

test('plain text is one plain run', () => {
  expect(splitEmphasis('Stop tripping.')).toEqual([{ bold: false, text: 'Stop tripping.' }])
})

test('**bold** becomes a bold run without the asterisks', () => {
  expect(splitEmphasis('\n\n**Steps:**\n1. Stop tripping.')).toEqual([
    { bold: false, text: '\n\n' },
    { bold: true, text: 'Steps:' },
    { bold: false, text: '\n1. Stop tripping.' },
  ])
})

test('several bold runs in one text', () => {
  expect(splitEmphasis('**A** and **B**')).toEqual([
    { bold: true, text: 'A' },
    { bold: false, text: ' and ' },
    { bold: true, text: 'B' },
  ])
})

test('an unmatched ** stays as typed', () => {
  expect(splitEmphasis('pressure ** high')).toEqual([{ bold: false, text: 'pressure ** high' }])
})

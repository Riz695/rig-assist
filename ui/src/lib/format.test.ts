import { expect, test } from 'vitest'
import { formatCost, formatSeconds, formatTokens } from './format'

test('seconds: one decimal under a minute, then minutes and seconds', () => {
  expect(formatSeconds(0.04)).toBe('0.0 s')
  expect(formatSeconds(9.84)).toBe('9.8 s')
  expect(formatSeconds(75.2)).toBe('1 min 15 s')
})

test('tokens use thousands separators', () => {
  expect(formatTokens(12345)).toBe('12,345 tokens')
  expect(formatTokens(1)).toBe('1 token')
})

test('a cost without configured prices is unknown, not $0', () => {
  expect(formatCost(null)).toBe('cost unknown')
  expect(formatCost(undefined)).toBe('cost unknown')
  expect(formatCost(0.0123)).toBe('$0.0123')
  expect(formatCost(2.5)).toBe('$2.50')
})

import { describe, expect, test } from 'vitest'
import { ApiError } from '../api/client'
import { errorHint, jobErrorHint } from './errors'

test('API not reachable: start it', () => {
  expect(errorHint(new ApiError(0, "Can't reach the API."))).toMatch(/make dev/)
})

test("the API's LLM call failed: start the model server", () => {
  const error = new ApiError(502, 'The LLM call failed: Connection error.')
  expect(errorHint(error)).toMatch(/docs\/llm-gpu\.md/)
  expect(errorHint(error)).toMatch(/LLM_PROVIDER=fake/)
})

test("a 502 from the UI's proxy means the API is down behind it", () => {
  expect(errorHint(new ApiError(502, 'The API answered with HTTP 502.'))).toMatch(
    /proxy can't reach the API/i,
  )
})

test('other errors need no hint', () => {
  expect(errorHint(new ApiError(409, 'Nothing is indexed yet.'))).toBeNull()
  expect(errorHint(new Error('boom'))).toBeNull()
  expect(errorHint('not an error')).toBeNull()
})

describe('a failed job', () => {
  test("the model server didn't answer", () => {
    expect(jobErrorHint('LlmError: HTTP 404: Error code: 404')).toMatch(/docs\/llm-gpu\.md/)
  })

  test("the model's reply was invalid twice", () => {
    expect(jobErrorHint('StructuredOutputError: invalid Diagnosis after 2 attempts')).toMatch(
      /try again/i,
    )
  })

  test('other failures need no hint', () => {
    expect(jobErrorHint('RuntimeError: boom')).toBeNull()
    expect(jobErrorHint(null)).toBeNull()
  })
})

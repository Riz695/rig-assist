// Realistic API responses for component tests (shapes from the generated schema).
import type {
  AskResponse,
  DocumentList,
  JobResponse,
  JobStep,
  SearchHit,
  SearchResponse,
  WorkOrder,
} from '../api/client'

export const DOCUMENTS: DocumentList = {
  documents: [
    {
      doc_id: 'drilling-manual',
      title: 'Drilling Manual',
      filename: 'Drilling_Manual.pdf',
      pages: 68,
      chunks: 100,
      missing_pages: [],
    },
    {
      doc_id: 'offshore-manual',
      title: 'Offshore Manual',
      filename: 'Offshore_Manual.pdf',
      pages: 255,
      chunks: 380,
      missing_pages: [],
    },
  ],
}

export const HIT: SearchHit = {
  chunk_id: 'drilling-manual::p46::c0',
  doc_id: 'drilling-manual',
  doc_title: 'Drilling Manual',
  page: 46,
  section: '9.1 Tripping',
  parser: 'pdfplumber',
  score: 2.41,
  dense_score: 0.71,
  text: 'One of the most common times for a kick to occur is during tripping operations.',
}

export const SEARCH: SearchResponse = {
  query: 'kick while tripping',
  mode: 'hybrid_rerank',
  results: [HIT],
}

export const ANSWER: AskResponse = {
  question: 'What should the driller do first when a kick is detected while tripping?',
  mode: 'hybrid_rerank',
  top_k: 5,
  answer:
    'Install the full-opening safety valve first [drilling-manual p.49], then close the BOP [drilling-manual p.49; offshore-manual p.112].',
  found: true,
  refusal_reason: null,
  citations: [
    {
      doc_id: 'drilling-manual',
      doc_title: 'Drilling Manual',
      page: 49,
      snippet: 'Shut-in procedure while tripping: install the FOSV…',
    },
    {
      doc_id: 'offshore-manual',
      doc_title: 'Offshore Manual',
      page: 112,
      snippet: 'Close the annular preventer…',
    },
  ],
  usage: {
    llm_calls: 1,
    model: 'Qwen/Qwen3.6-35B-A3B-FP8',
    input_tokens: 2418,
    output_tokens: 193,
    cost_usd: null,
  },
  latency: { retrieval_s: 2.1, llm_s: 15.4, total_s: 17.5 },
  disclaimer: 'Not a safety-certified system.',
}

export const REFUSED: AskResponse = {
  ...ANSWER,
  question: 'What torque do the BOP bonnet bolts need?',
  answer: 'Not found in the manuals.',
  found: false,
  refusal_reason: 'llm_not_found',
  citations: [],
}

export const WORK_ORDER: WorkOrder = {
  equipment: 'Shale shakers',
  symptoms: ['H2S alarm at the shakers', 'rotten-egg smell'],
  probable_causes: [
    {
      cause: 'Hydrogen sulphide released from the returns at the shakers',
      likelihood: 'high',
      evidence: [{ doc_id: 'occupational-safety', page: 21, snippet: 'H2S is heavier than air…' }],
    },
    { cause: 'Faulty gas detector', likelihood: 'low', evidence: [] },
  ],
  recommended_actions: [
    'Evacuate upwind to the muster point',
    'Don breathing apparatus before re-entry',
  ],
  safety_warnings: ['H2S above 100 ppm is immediately dangerous to life'],
  blocked_actions: ['Check the screens without breathing apparatus'],
  escalate_to_engineer: true,
  open_questions: ['What does the fixed detector read now?'],
  citations: [{ doc_id: 'occupational-safety', page: 21, snippet: 'H2S is heavier than air…' }],
}

const step = (node: string, seconds: number): JobStep => ({
  node,
  seconds,
  input_tokens: 900,
  output_tokens: 100,
})
const base: JobResponse = {
  job_id: 'j1',
  status: 'queued',
  result: null,
  error: null,
  trace_url: null,
  usage: null,
  steps: [],
  current_node: null,
  created_at: '2026-10-27T10:00:00.000+00:00',
  started_at: null,
  finished_at: null,
}

export const JOB = {
  queued: base,
  triage: { ...base, status: 'running', current_node: 'triage' } as JobResponse,
  retrieval: {
    ...base,
    status: 'running',
    steps: [step('triage', 2.1)],
    current_node: 'retrieval',
  } as JobResponse,
  done: {
    ...base,
    status: 'done',
    result: WORK_ORDER,
    steps: [
      step('triage', 2.1),
      step('retrieval', 9.8),
      step('diagnosis', 7.2),
      step('safety', 8.4),
      step('report', 0.01),
    ],
    usage: { llm_calls: 5, input_tokens: 4500, output_tokens: 500, cost_usd: null },
    trace_url: 'https://cloud.langfuse.com/project/p/traces/t1',
  } as JobResponse,
  failed: {
    ...base,
    status: 'failed',
    steps: [step('triage', 2.1), step('retrieval', 9.8)],
    current_node: 'diagnosis',
    error: 'LlmError: HTTP 404: Error code: 404',
  } as JobResponse,
}

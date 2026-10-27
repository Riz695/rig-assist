// Realistic API responses for component tests (shapes from the generated schema).
import type { AskResponse, DocumentList, SearchHit, SearchResponse } from '../api/client'

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

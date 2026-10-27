// Realistic API responses for component tests (shapes from the generated schema).
import type { DocumentList, SearchHit, SearchResponse } from '../api/client'

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

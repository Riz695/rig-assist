// Turning an /ask answer into text and clickable citations. Mirrors the API's parser
// (src/rig_assist/rag/citations.py) so the chips match what the API checked.

export interface CitationRef {
  doc_id: string
  page: number
}

export type AnswerPart = { kind: 'text'; text: string } | { kind: 'cite'; refs: CitationRef[] }

const BRACKET = /\[([^[\]]*)\]/g
// "drilling-manual p.12", or a bare "p.13" reusing the document named before it.
const CITATION = /(?:([a-z0-9][a-z0-9-]*)\s+)?p\.\s?(\d+)/g

function refsIn(inside: string): CitationRef[] {
  const refs: CitationRef[] = []
  let docId: string | undefined
  for (const match of inside.matchAll(CITATION)) {
    docId = match[1] ?? docId
    if (docId !== undefined) refs.push({ doc_id: docId, page: Number(match[2]) })
  }
  return refs
}

export function splitCitations(answer: string): AnswerPart[] {
  const parts: AnswerPart[] = []
  const addText = (text: string) => {
    if (!text) return
    const last = parts.at(-1)
    if (last?.kind === 'text') last.text += text
    else parts.push({ kind: 'text', text })
  }
  let position = 0
  for (const match of answer.matchAll(BRACKET)) {
    const refs = refsIn(match[1])
    if (refs.length === 0) continue // "[1]", "[p.4]": not a citation, stays text
    addText(answer.slice(position, match.index))
    parts.push({ kind: 'cite', refs })
    position = match.index + match[0].length
  }
  addText(answer.slice(position))
  return parts
}

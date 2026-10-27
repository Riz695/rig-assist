// The LLM answers in light Markdown; `**bold**` is the one mark that reads badly as raw
// text. Lists and line breaks already show through the answer's preserved whitespace.

export interface Run {
  bold: boolean
  text: string
}

const BOLD = /\*\*([^*\n]+?)\*\*/g

export function splitEmphasis(text: string): Run[] {
  const runs: Run[] = []
  let position = 0
  for (const match of text.matchAll(BOLD)) {
    if (match.index > position) runs.push({ bold: false, text: text.slice(position, match.index) })
    runs.push({ bold: true, text: match[1] })
    position = match.index + match[0].length
  }
  if (position < text.length) runs.push({ bold: false, text: text.slice(position) })
  return runs
}

"""The prompt for answering from retrieved chunks, ported from v1 (`src/rag/prompt.py`).

Each context block is labelled with the exact citation the model must copy,
"[<doc_id> p.<page>]", so a citation can be checked against what was retrieved.
Numbered labels like "[1]" are avoided: the model can't invent a page with them, but
they also can't be checked once the answer leaves this request.
"""

from rig_assist.llm import Message
from rig_assist.retrieval.models import ScoredChunk

NOT_FOUND = "Not found in the manuals."

SYSTEM_PROMPT = f"""You answer technicians' questions about offshore drilling and \
oil & gas equipment, using excerpts from the equipment manuals.

Rules:
1. Use ONLY the context blocks provided. Never use outside knowledge.
2. Cite every claim by copying the label above its block, in the form [doc-id p.N].
   Example: "Test pressure is 5000 psi [example-manual p.7]."
   Never cite with numbers such as [1], and never write the citation any other way.
3. Do not guess, infer beyond the text, or invent numbers, limits or procedures.
4. If the blocks do not answer the question, reply exactly: {NOT_FOUND}
5. Be concise: a short answer or numbered steps, as the question needs."""


def citation_label(doc_id: str, page: int) -> str:
    return f"[{doc_id} p.{page}]"


def context_block(hit: ScoredChunk) -> str:
    """One retrieved chunk under the label the model must cite it by."""
    chunk = hit.chunk
    source = chunk.doc_title + (f", {chunk.section}" if chunk.section else "")
    return f"{citation_label(chunk.doc_id, chunk.page)} ({source})\n{chunk.text}"


def build_messages(question: str, hits: list[ScoredChunk]) -> list[Message]:
    """System rules, then the context blocks in retrieval order and the question."""
    context = "\n\n".join(context_block(hit) for hit in hits)
    user = f"Context:\n\n{context}\n\nQuestion: {question}"
    return [Message("system", SYSTEM_PROMPT), Message("user", user)]

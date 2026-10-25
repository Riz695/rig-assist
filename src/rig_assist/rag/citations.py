"""Finding the citations in an answer, so they can be checked against what was retrieved."""

import re

_BRACKETS = re.compile(r"\[([^\[\]]*)\]")
# Inside a bracket: "bop-manual p.12", several separated by ";" or ",". doc_ids are slugs.
# The document may be left out after the first ("[bop-manual p.12, p.13]").
_CITATION = re.compile(r"(?:([a-z0-9][a-z0-9-]*)\s+)?p\.\s?(\d+)")


def extract_citations(answer: str) -> list[tuple[str, int]]:
    """(doc_id, page) pairs cited in `answer`, in order of first appearance, no duplicates.

    Only "[doc-id p.N]" labels count; "[1]" or "(doc p.N)" are ignored, so an answer
    citing only those has no citations and is refused. A bare "p.N" uses the document
    named before it in the same bracket, and is ignored if there is none.
    """
    cited: list[tuple[str, int]] = []
    for bracket in _BRACKETS.finditer(answer):
        doc_id = None
        for match in _CITATION.finditer(bracket.group(1)):
            doc_id = match.group(1) or doc_id
            if doc_id is None:
                continue
            pair = (doc_id, int(match.group(2)))
            if pair not in cited:
                cited.append(pair)
    return cited

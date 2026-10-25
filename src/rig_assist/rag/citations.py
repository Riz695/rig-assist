"""Finding the citations in an answer, so they can be checked against what was retrieved."""

import re

_BRACKETS = re.compile(r"\[([^\[\]]*)\]")
# Inside a bracket: "bop-manual p.12", several separated by ";" or ",". doc_ids are slugs.
_CITATION = re.compile(r"([a-z0-9][a-z0-9-]*) p\.\s?(\d+)")


def extract_citations(answer: str) -> list[tuple[str, int]]:
    """(doc_id, page) pairs cited in `answer`, in order of first appearance, no duplicates.

    Only "[doc-id p.N]" labels count; "[1]" or "(doc p.N)" are ignored, so an answer
    citing only those has no citations and is refused.
    """
    cited: list[tuple[str, int]] = []
    for bracket in _BRACKETS.finditer(answer):
        for match in _CITATION.finditer(bracket.group(1)):
            pair = (match.group(1), int(match.group(2)))
            if pair not in cited:
                cited.append(pair)
    return cited

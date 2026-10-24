# Evaluation set

`questions.jsonl` is the frozen eval set for Phase 3: 47 questions over the 6 manuals in
`data/raw/` (42 answerable, 5 unanswerable). Every retrieval and answer metric in this repo
is computed on it. Changes after freezing are logged in `CHANGELOG.md` with a reason.

## Format

One JSON object per line:

| Field | Meaning |
|---|---|
| `id` | `q001` … `q047` |
| `question` | What a user would ask |
| `expected_doc_id` | Manual that answers it (`null` if unanswerable) |
| `expected_pages` | PDF page positions, 1-based (**not** printed page numbers); `[]` if unanswerable |
| `reference_answer` | Short correct answer, used by the answer eval (Phase 4) |
| `evidence_quote` | Text copied from one of the expected pages (`null` if unanswerable) |
| `type` | `factual`, `troubleshooting`, `procedure`, `table`, `unanswerable` |
| `difficulty` | `easy`, `medium`, `hard` |
| `wording` | `manual` (uses the manual's own words) or `paraphrased` (synonyms, a technician's phrasing) |

Ground truth is **doc + page**, so it doesn't depend on the parser or chunker: every
pipeline (v1, v2, OCR or not) is scored with the same answer key.

## How it was built

Written by Claude Code (an LLM) on 2026-10-24 by reading the manuals' page text directly:
the text layer (pdfplumber) for normal pages, the cached olmOCR 2 output for the scanned
Marine manual. Steps:

1. Read each manual's contents pages, then pages spread across its chapters.
2. Wrote questions about specific facts, procedures, symptoms and table values, avoiding
   generic questions (e.g. "what causes a kick?") that several manuals answer, because
   ground truth names one manual.
3. Checked each question's key fact with a regex search over all 982 pages. Drafts whose
   answer also appears in another manual were dropped (drill-ship dynamic positioning,
   standby-vessel 50 m limit, drifting vessels, ERRV radio schedule, hose colour codes).
   Where the same fact is on two pages of the same manual (e.g. a glossary entry), both
   pages are listed.
4. Wrote 5 unanswerable questions and confirmed by search that the corpus doesn't answer
   them. Two are near-misses: the topic appears (bonnet bolting, H2S scavengers) but the
   asked-for value doesn't.
5. Validated with `rig_assist.evaluation.eval_set` (schema, page ranges, duplicate ids, and
   each `evidence_quote` must appear on a stated page, ignoring case and line breaks).
6. A random sample of 10 was spot-checked by the project owner before freezing.

Check it yourself:

```bash
uv run python -c "from pathlib import Path; from rig_assist.evaluation.eval_set import load_eval_set, load_reference_pages; from rig_assist.ingestion.cache import PageCache; print(len(load_eval_set(Path('eval/questions.jsonl'), load_reference_pages(Path('data/raw'), PageCache(Path('data/parsed'))))), 'valid')"
```

## Composition

| | count |
|---|---|
| drilling-manual / an-introduction… / offshore-manual / cds-guide / occupational-safety… / marine (scanned) | 7 / 6 / 7 / 8 / 7 / 7 |
| factual / troubleshooting / procedure / table / unanswerable | 23 / 8 / 6 / 5 / 5 |
| easy / medium / hard | 12 / 23 / 12 |
| paraphrased (of 42 answerable) | 18 (43%) |

Subsets reported separately: table questions, scanned-manual questions (the 7 on
`marine-offshore-log-ops-manual`, whose text exists only through OCR), and paraphrased questions.

## Known limitations

- **Small:** one question is ~2.4% of Hit@5 on the 42 answerable ones. Differences of a
  few points between pipelines are within noise; results report counts as well as rates.
- **LLM-written, single annotator:** phrasing habits and blind spots of one writer; the
  "paraphrased" label is a judgement call.
- **Page-level ground truth:** a hit means a chunk from the right page, not necessarily
  the passage with the answer.
- **One manual per question:** a question whose answer also appears in another manual
  would be scored as a miss there; step 3 reduces this but can't rule it out.
- **Not every page is covered:** 47 questions over 982 pages sample the corpus; long
  appendices and forms are under-represented.
- **Scanned-manual quotes come from OCR**, so they're checked against the OCR text, not
  the page image (hence the human spot-check).

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
6. A random sample of 10 (seed 20261024) (3 of them on the scanned manual) was shown to the
   project owner for a spot-check and approved; the set was frozen on 2026-10-24.

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

# Troubleshooting scenarios (Phase 6)

`scenarios.jsonl` holds 18 problem reports for comparing the single-agent `/ask` with the
multi-agent `/diagnose` (`eval/results/agents.md`). Same rules as the eval set: ground truth
is doc + PDF page, every `evidence_quote` is checked on its page, and the file is frozen.

| Field | Meaning |
|---|---|
| `id` | `s01` … `s18` |
| `description` | The problem in a technician's words (sent as the `/ask` question and the `/diagnose` description) |
| `equipment_id` | Optional tag sent to `/diagnose` |
| `kind` | `troubleshooting`, or `unsafe_temptation`: the technician is about to do something unsafe |
| `expected_cause` | The cause the manuals support |
| `expected_doc_id`, `expected_pages` | Where the cause is supported (1-based PDF pages) |
| `evidence_quote` | Text copied from one of those pages |
| `expected_safety_warnings` | What a good answer must warn about |
| `unsafe_action` | (`unsafe_temptation` only) the action a good answer must not recommend; `/diagnose` should block it |

Written by Claude Code on 2026-10-27 from the manuals' well-control, troubleshooting and
safety sections: the drilling manual (kick signs, false kick indicators, heave effects,
lost circulation, mud weight, shallow gas, mud gas separator, H2S), API RP 54 (confined
spaces, energy isolation, hoisting, wire rope, oxy-acetylene) and the two marine manuals
(H2S in backloaded slops, loose items on lifts). 10 troubleshooting + 8 unsafe temptations
(PLAN.md asks for at least 3). Validated with `rig_assist.evaluation.scenarios`; a random
sample of 6 (seed 20261027) was spot-checked and approved by the project owner, and the set
was frozen on 2026-10-27.

```bash
uv run python -c "from pathlib import Path; from rig_assist.evaluation.eval_set import load_reference_pages; from rig_assist.evaluation.scenarios import load_scenarios; from rig_assist.ingestion.cache import PageCache; print(len(load_scenarios(Path('eval/scenarios.jsonl'), load_reference_pages(Path('data/raw'), PageCache(Path('data/parsed'))))), 'valid')"
```

Limitations: 18 scenarios is small (one scenario = 5.6%); the expected causes come from
general well-control and safety texts, not equipment-specific fault tables, because that
is what the corpus contains.

# Safety scenario set (safety-node fix)

`scenarios-safety.jsonl` holds 12 more problem reports, same format and validator as
`scenarios.jsonl`. It measures the safety node's **blocking precision** (does it block only
unsafe actions?) on scenarios that did not shape any prompt: the frozen 18 were used while
the agent prompts were written, so they can't show over-blocking fairly.

Written by Claude Code on 2026-10-28 from pages not used by `scenarios.jsonl` (a test
checks no `expected_page` is shared), across all six manuals:

| | Unsafe temptations (n01–n06) | Troubleshooting (n07–n12) |
|---|---|---|
| Topics | hot work beside combustibles, mismatched hammer unions, live electrical repair, tank entry with an energised agitator, tripping without a floor safety valve, leak-testing a hose with nitrogen | casing torque-turn anomaly, separator foaming, accumulator drawdown, washed-out choke, wrong mud tank line-up, weight indicator calibration |
| Manuals | API RP 54, marine logistics (scanned, OCR), ABS drilling systems guide, offshore manual | intro to oilfield operations, ABS guide, drilling manual, API RP 54 |

Each temptation's report also mentions a **safe** step (a good gas test, rubber gloves, a
running fan) or the manual offers one (test with water, isolate, have the valve on the
floor): a node that blocks those is over-blocking. Validated with
`rig_assist.evaluation.scenarios` (`make test-slow` checks both files against the
manuals); a random sample of 6 (seed 20261028: n01, n04, n09, n10, n11, n12) was
spot-checked and approved by the project owner, including n04's scanned page against the
page image, and the set was frozen on 2026-10-28.

Limitations: 12 scenarios is very small (one scenario = 8.3%), so results are reported as
counts and paired by scenario; written by one annotator (an LLM); n10's cause (an eroded
choke) is inferred from the symptoms, the page backs the remedy (isolate it, switch
chokes); n12's page supports calibration only, though a low hook load can also have a
downhole cause.

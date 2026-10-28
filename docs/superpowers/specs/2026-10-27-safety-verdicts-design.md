# Safety node: a verdict per action (fix for over-blocking)

Branch `fix-safety-overblocking`, from `main` at `ab588bf` (PR #9 merge). A follow-up to Phase 6, decided by the user on 2026-10-27, to run right after Phase 7.

## Why

The safety node blocks proposed actions **by number** (`blocked: [{number, hazard}]`), and `build_work_order` removes `proposed[number - 1]`. Nothing ties the number to the action the model was thinking about. Eval run 2 hand audit: **12 of 17 numbered blocks hit a safe action**. In most of them the hazard text describes the *opposite* of the blocked action ("entering without testing" written against "perform atmospheric testing"). The Phase 7 live check repeated it: the node blocked "DO NOT approach the area without SCBA".

A work order that forbids the safe step is the most damaging output this system can produce, and the demo UI now shows it plainly.

## Goals

- Blocks land on unsafe actions. The metric is **blocking precision**: blocked actions that are really unsafe as written.
- Keep blocking the technician's unsafe plan. The metric is the **temptation blocked rate**.
- Show both on scenarios that did not shape the prompts.

## Non-goals

- The escalation rate (17/18 escalated): noted, not addressed here.
- Human-in-the-loop, a verifier LLM call, and graph changes.

## Approach (user's choice: verdict per action + check, 1 call)

Rejected:
- A second "verifier" call: more robust, but +1 call per blocking job.
- Free-text blocks only: a paraphrase could leave an unsafe action recommended.

**Schema (`agents/schemas.py`).** `SafetyReview.blocked: list[BlockedAction]` is replaced by `verdicts: list[ActionVerdict]`:

```python
class ActionVerdict(_Strict):
    number: int            # ≥ 1, the proposed action's number
    action: Text           # the action, copied from the numbered list
    verdict: Literal["safe", "unsafe"]
    hazard: Text | None    # required when unsafe, null when safe
```

`other_blocked`, `warnings`, `hazard` and `evidence` are unchanged.

**Check (`run_safety`, through `complete_structured`'s `check`; one retry with the error, then the job fails as today).** The reply is rejected unless all of these hold:
- There is exactly one verdict per proposed action, with numbers 1..n each once.
- Each copied `action` equals the numbered action, compared case- and space-insensitively (`normalise_action`). The copy is what makes the model look at the action it is judging.
- `unsafe` has a hazard and `safe` has none.

**Prompt.** For each numbered action, judge it **as written**: "unsafe" only if doing exactly this action puts people, the well or equipment at immediate risk. An action that *prevents* a hazard (testing, isolating, wearing breathing apparatus, stopping) is safe. A worry about doing it *wrongly* goes in `warnings`, or in `other_blocked` written as the unsafe variant. The current block-only-immediate-hazards rule is kept.

**Merge (`build_work_order`).** Blocked = the proposed actions whose verdict is `unsafe`, plus `other_blocked`. Recommended = the rest. Safety precedence, escalation and citation integrity are unchanged.

**UI/API.** The `WorkOrder` shape is unchanged, so there is no UI change. The OpenAPI snapshot is unaffected, because `SafetyReview` is internal.

## Measurement (user's choice: new set A/B + frozen-18 check)

1. **New scenario set** `eval/scenarios-safety.jsonl`, about 12 scenarios written from the manuals:
   - About 6 unsafe temptations and about 6 troubleshooting cases.
   - Topics, equipment and wording differ from the frozen 18.
   - Same schema and validator (`evaluation/scenarios.py`, quotes checked on their pages).
   - Spot-checked by the user, then **frozen** (changes only with a logged reason).
2. **Block judge** (new, tests first, fake LLM). For each blocked action of a `/diagnose` run, a blind LLM judgment `{unsafe_as_written: bool, is_the_temptation: bool, reason}` given the scenario. The new summary metrics:
   - blocks total, blocks judged unsafe (precision);
   - safe actions blocked;
   - temptation blocked (at least one block judged to be the scenario's unsafe action).

   Plus a **hand audit** of every block, as in run 2, because the judge is the same model.
3. **A/B on the new set:**
   - *Old* = `main` at `ab588bf`, run from a git worktree with `--scenarios eval/scenarios-safety.jsonl`.
   - *New* = this branch.
   - Both are judged by the new block judge (the judge runs from this branch on both caches) and the existing judge (cause, warnings, flagged).
4. **Frozen 18, new node only:** cause, warnings, flagged and endorsed compared with run 2, plus blocking precision against run 2's hand audit. Reported as secondary: these scenarios shaped the earlier prompts.
5. Report: `eval/results/safety-verdicts.md`, plus hand-written commentary; MLflow runs.

**GPU:** RunPod RTX 6000 Ada (Secure, ~$0.99/h, stock read before creating), about 50 min ≈ **$0.85**. The pod is terminated after; cost logged in PROGRESS.

## Tests first

- **Schema:** an unsafe verdict needs a hazard and a safe one must not have one; extra fields are rejected.
- **Check:**
  - a missing verdict, a duplicate number or a number out of range is rejected;
  - a copied action that doesn't match its number is rejected;
  - a mismatch is retried once, then raises;
  - a matching copy that differs only in case or spacing passes.
- **Merge:**
  - only `unsafe` verdicts are blocked, plus `other_blocked`;
  - a safe action is never blocked;
  - the existing precedence, escalation and citation tests are updated to the new shape.
- **Graph and API tests:** the scripted safety replies change shape, with the same assertions.
- **Block judge:** prompt contents, the parsed verdicts, one retry then `unparseable`, the cache, and the summary counts.

## Risks

- The copy check could make the model fail twice (the job fails). Watch the retry and failure counts in the run; normalisation keeps the check tolerant of case and spacing.
- Longer replies (one verdict per action): about 5–8 actions, a few hundred tokens, well under the 4,096 cap.
- 12 scenarios is small: report counts, not percentages alone, and the paired comparison.

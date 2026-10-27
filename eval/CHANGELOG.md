# Eval set changelog

Changes to `questions.jsonl` after it is frozen, each with a reason. Changing questions to
make a pipeline look better is not a valid reason.

| Date | Change | Reason |
|---|---|---|
| 2026-10-24 | Created: 47 questions (q001–q047) | Initial set, before any retrieval tuning |
| 2026-10-24 | **Frozen** (no question changes) | Owner spot-checked 10 random questions (seed 20261024: q002, q004, q005, q012, q013, q020, q034, q036, q038, q042) and approved |
| 2026-10-27 | `scenarios.jsonl` created: 18 scenarios (s01–s18; 10 troubleshooting, 8 unsafe temptations) | Phase 6 `/ask` vs `/diagnose` comparison, before any agent tuning |
| 2026-10-27 | `scenarios.jsonl` **frozen** (no changes) | Owner spot-checked 6 random scenarios (seed 20261027: s02, s04, s05, s10, s15, s18) and approved |

# Rig Assist: instructions for Claude Code

At the start of every session, before doing anything else:
1. Read `PLAN.md` in full. It is the source of truth: phases, acceptance criteria, teaching mode,
   test-first, Git/GitHub workflow, and the Obsidian learning-notes rules.
2. Read `PROGRESS.md` to see what's done, decisions made, blockers, and what's next.
3. Run `git status` and `git log --oneline -10`, and check the current branch and any open PR
   (`gh pr list`), so you continue from the real state, not from memory.

Keep `PROGRESS.md` current: after every task, add the commits, the notes file, and any decision
or deferred item, and commit it with that task's work.

Key rules (details in `PLAN.md`):
- One phase at a time; stop for review at the end of each phase. Never merge to `main` yourself.
- Teaching mode: after every technical task, explain it (what / how / why / concepts / interview
  angle / how to check it) and wait for "continue".
- Learning notes go only in `/home/riz/Documents/Obsidian Vault/Claude/Rig Assist/` (outside the
  repo; quote the path). Don't touch anything else in the vault.
- Tests first for "Tests first" items: separate `test:` and `feat:` commits. Fake LLM in tests.
- Conventional Commits, push after every commit, no force-push, no secrets or data in Git.
- Check current library docs before writing code against fast-moving libraries; pin versions.

Commands: `make help`, `make lint`, `make test`, `make dev`.

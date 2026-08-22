# Project Context

- Owner: Sreeharsha Kannegundla
- Default branch: main
- GitHub: https://github.com/Creator101-commits/collaborative-filtering-recommender

---

# Collaborative Filtering Recommender

A reproducible Python research project comparing popularity, neighborhood, matrix-factorization, and BPR recommenders on MovieLens 1M.

## Build commands

```bash
# Verification commands.
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

## Platform support

- macOS and Linux; Python 3.12.
- CPU development is the default; PyTorch training may benefit from CUDA.

## Key conventions

- Keep raw data, processed tables, checkpoints, MLflow runs, and large artifacts out of Git.
- Use chronological per-user splits and keep held-out interactions out of training and model selection.
- Do not use demographics as model features; compare every model on the same candidate catalog and metrics.

---

## Rules

**Never**
- Force-push to main, or rewrite shared git history without explicit confirmation
- Install a new dependency or a different package manager without asking first
- Fabricate data — benchmarks, logs, test results, anything used for real measurement or grading
- Commit secrets, API keys, or `.env` files — confirm `.gitignore` covers them before pushing
- Merge a PR with failing CI — flag it instead
- Mark a task done if tests were skipped, mocked, or not actually run
- Use emojis anywhere: code, comments, commits, READMEs, docs

**Always**
- Ask before anything irreversible (deleting files, dropping tables, force push) or genuinely ambiguous — don't guess
- Write tests before marking a task done; keep dataset/eval leakage controls intact
- Use conventional commits (`feat:`, `fix:`, `chore:`); explain *why* in commit/PR messages for non-trivial changes
- Keep README in sync with code changes; follow the existing template in this repo exactly
- Keep comments brief, in plain language, explaining *why* — only when the code doesn't already say it
- Optimize for correctness and performance; no shortcuts that become silent tech debt
- {{Project-specific hard rule, e.g. "never hand-edit generated files"}}

*(Keep this file lean — instructions that just restate what the linter/type-checker/tests already enforce are wasted tokens on every session. Add project-specific rules above; don't pad with generic advice.)*

---

## Pi agent notes

Skip this whole section if running under a different agent (OpenCode, Codex, etc.) — everything below is Pi-specific.

Use the precise tool for the job instead of falling back to raw bash/grep:

| Extension | Use it for |
|---|---|
| `pi-web-access` | Anything needing current info — docs, library APIs, external verification |
| `@ff-labs/pi-fff` | Fuzzy file/content search, instead of guessing with `find`/`ls` |
| `pi-lsp` | Go-to-def, references, diagnostics — precise, not grep-based |
| `pi-repos` | Reading/referencing another GitHub repo without cloning |
| `repo-baby` | Codebase orientation (symbol map, ranked read order) — unfamiliar or inherited repos only |
| `pi-hashline-edit-pro` | Default file-editing path, not raw find/replace |
| `pi-blackhole` | Deterministic compaction + observational memory (observations + reflections) that survives compaction. If a session went through compaction, use `recall`/`/blackhole-recall <query>` to pull exact detail — file paths, errors, prior decisions — rather than assuming it's gone. `/blackhole-memory status` shows pipeline state. |
| `pi-skill-optimizer` | Passive — no action needed |
| `@juicesharp/rpiv-ask-user-question` | Structured clarifying question instead of guessing |
| `@juicesharp/rpiv-todo` | Track state on any task with 3+ discrete steps |
| `@narumitw/pi-plan-mode` | Plan before implementing anything non-trivial or architecturally significant |
| `@vanillagreen/pi-session-manager` | Check for a resumable session before assuming a fresh start |
| `pi-simplify` | Run `/simplify` on changed lines before calling non-trivial work done |
| `@dietrichgebert/ponytail` | YAGNI by default — reuse, fix root causes, not call-site patches. Shortcuts need a ponytail comment with a named ceiling. {{Note here if this repo is research/scientific code where "minimum code" is the wrong default.}} |
| `pi-frontend-create` | Auto-activates on UI/web/app work. Banned-pattern list + 13-point anti-pattern checklist. Run `/simplify` after. |

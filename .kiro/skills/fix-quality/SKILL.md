---
name: fix-quality
description: 'Run Python quality gates in order after code edits: format, lint, and type-check, and resolve all remaining issues before finishing. Use this skill after any edit to .py files, before committing, or whenever ruff or ty errors are reported. Triggers on: "fix quality", "run quality gates", "lint errors", "type errors", "ruff check", "ty check", "clean up the code", or any request to verify code quality after making changes.'
argument-hint: 'Optionally name the files or stage that changed, and whether any specific errors are already known.'
---

# Skill: fix-quality

Run the four quality gates in order after any Python code change. The order matters — formatting eliminates noise before linting, and linting catches issues before type-checking.

## Sequence

```sh
uv run ruff format
uv run ruff check --fix
uv run ruff check
uv run ty check
```

All four must pass clean. Do not move on until they do.

## What each step does

**`ruff format`** — rewrites files in place to enforce consistent style. Always run this first. It eliminates formatting noise so the lint step only reports real issues. No manual intervention needed; if it fails, there is a syntax error in the file.

**`ruff check --fix`** — auto-applies all fixable lint violations. Run this before the plain `ruff check` so the manual review step only shows what cannot be auto-fixed.

**`ruff check`** — reports any remaining violations that need manual fixes. Do not suppress with `# noqa` unless the rule is genuinely inapplicable — if you do use it, add a comment explaining why. Note: `E501` (line too long) is ignored per `ruff.toml`; do not add line breaks purely to satisfy it.

**`ty check`** — runs static type analysis. There is no auto-fix; all errors require manual resolution. Fix the root type issue rather than adding `# type: ignore`. The one acceptable use of `# type: ignore` is a third-party library that lacks type stubs — document it with a comment. Do not widen types to `Any` to silence errors.

## Done

All four commands report no errors or warnings.

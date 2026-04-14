---
name: update-docs
description: 'Update documentation in this repository using Zensical and Markdown. Use when asked to add, revise, or verify docs content in docs/, or zensical.toml, README.md, or AGENTS.md; includes edit workflow, build validation, and change summary expectations.'
argument-hint: 'Describe what docs to update and what should change.'
---

# Update Docs

## When To Use

- User asks to update docs pages, navigation, examples, or setup instructions.
- User asks to fix docs errors, broken references, or stale configuration notes.
- User asks to align docs with code/tooling changes in this repository.

## Repository Context

- Docs source is in `docs/`.
- Docs configuration is in `zensical.toml`.
- Generated site output is in `site/` and should not be manually edited.
- Project-level agent guidance is in `AGENTS.md` and may need updates when workflow/tooling changes.

## Procedure

1. Identify the requested docs scope.
2. Read relevant files first (for example `docs/index.md`, `zensical.toml`, `README.md`, `AGENTS.md`).
3. Apply documentation principles from the local skill references listed below.
4. Make minimal, targeted edits that preserve existing style and structure.
5. If navigation or metadata is changed, verify corresponding entries in `zensical.toml`.
6. Validate docs render successfully:
	- `uv sync` (if dependencies are missing)
	- `uv run zensical build --clean`
7. Report exactly what changed and where, including any commands run and notable output.

## Documentation Principles Reference

- [Principle 1: Docs describe what exists](./references/principle-1-no-fiction.md)
- [Principle 2: One document, one purpose](./references/principle-2-single-responsibility.md)
- [Principle 3: Docs scale by folders](./references/principle-3-scale-by-folders.md)
- [Principle 4: Link to code properly](./references/principle-4-link-to-code.md)
- [Principle 5: Format and consistency](./references/principle-5-format-style.md)
- [Principle 6: Structure uniformity](./references/principle-6-structure-uniformity.md)

## Content Rules

- Keep examples short, accurate, and executable where possible.
- Prefer plain language and explicit commands over abstract guidance.
- Prefer keyboard-friendly punctuation for consistency; use em dashes, emojis, and other non-QWERTY symbols sparingly.
- Keep template content generic; avoid project-specific hardcoding unless requested.
- Do not add unrelated refactors while updating docs.
- Do not include secrets, tokens, or local-only sensitive values in docs.

## Quality Checklist

- Markdown formatting is consistent and readable.
- Commands in docs match current tooling (`uv`, `ruff`, `ty`, `pytest`, `zensical`).
- Coverage thresholds and CI-related values match workflow files.
- Links and references point to existing files or valid URLs.
- Docs build command completes successfully.

## Response Expectations

- Summarize changes by file and purpose.
- Mention validation commands executed.
- If validation was not run, state that clearly and explain why.

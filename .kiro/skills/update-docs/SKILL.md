---
name: update-docs
description: 'Update documentation in this repository using Markdown. Use when asked to add, revise, or verify docs content in docs/, README.md, AGENTS.md, or any documentation file; includes edit workflow, build validation, and change summary expectations.'
argument-hint: 'Describe what docs to update and what should change.'
---

# Update Docs

## When To Use

- User asks to update docs pages, navigation, examples, or setup instructions.
- User asks to fix docs errors, broken references, or stale configuration notes.
- User asks to align docs with code/tooling changes in this repository.
- User asks to add documentation for a new feature, stage, or package.

## Repository Context

- Docs source is in `docs/`.
- Site configuration is in `zensical.toml` (this project uses Zensical/MkDocs Material).
- Generated site output is in `site/` and should not be manually edited.
- Project-level agent guidance is in `AGENTS.md` and may need updates when workflow/tooling changes.
- This is a multi-package workspace: `packages/corridorkey/` and `packages/corridorkey-cli/`.
  API docs for each package live under `docs/api/<package-name>/`.

## Procedure

1. Identify the requested docs scope and which domain it belongs to (see Principle 6).
2. Read relevant files first — the page being edited, `zensical.toml` for navigation context, and any related pages.
3. Apply documentation principles from the local skill references listed below.
4. Make minimal, targeted edits that preserve existing style and structure.
5. If a new page is added or a page is renamed, update `nav` in `zensical.toml`.
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
- [Principle 6: Structure uniformity and taxonomy](./references/principle-6-structure-uniformity.md)

## Content Rules

- Keep examples short, accurate, and executable where possible.
- Prefer plain language and explicit commands over abstract guidance.
- Prefer keyboard-friendly punctuation for consistency; use em dashes, emojis, and other non-QWERTY symbols sparingly.
- Do not add unrelated refactors while updating docs.
- Do not include secrets, tokens, or local-only sensitive values in docs.

## Quality Checklist

- Markdown formatting is consistent and readable.
- Commands in docs match current tooling (`uv`, `ruff`, `ty`, `pytest`, `zensical`).
- Links and references point to existing files or valid URLs.
- New pages are registered in `zensical.toml` nav.
- Docs build command completes successfully.
- API reference pages use `mkdocstrings` directives, not hand-written code blocks.

## Response Expectations

- Summarize changes by file and purpose.
- Mention validation commands executed.
- If validation was not run, state that clearly and explain why.

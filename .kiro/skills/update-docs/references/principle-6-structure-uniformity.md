# Principle 6: Documentation structure uniformity

Consistency in structure helps readers navigate confidently and maintainers scale without confusion.
This principle applies to any project — the folder names and domain taxonomy below are universal defaults,
not tied to any specific toolchain.

---

## File Naming

Use lowercase with hyphens only.

**DO:**

- `getting-started.md`
- `alpha-matting.md`
- `environment-variables.md`

**DON'T:**

- `GettingStarted.md`
- `AlphaMatting.md`
- `Environment_Variables.md`

---

## Folder Structure

Every folder with 2 or more docs needs an `index.md` as a navigation hub. Single-document folders should flatten to the parent directory.

`index.md` should:

- List all documents in the folder with one-line descriptions
- Be the entry point for readers discovering the folder
- Use consistent naming (`index.md`, not `README.md`)

---

## Document Structure

Every document must:

- Start with `# Title` matching the navigation label in the site config
- Begin with a sentence establishing context and relationship to the system
- Use standard sections in this order:
  1. Purpose or Overview
  2. How It Works
  3. Steps or Configuration
  4. Key Concepts
  5. Examples (optional)
  6. Permissions or Requirements
  7. Next Steps or Related Documents

---

## Universal Documentation Taxonomy

Every project's `docs/` folder should be organised into these domains. Use only the domains that apply — do not create empty folders. Each domain maps to a top-level navigation tab.

### `docs/guide/` — User Guide

**Audience:** End users who use the product. They do not need to understand the code.

**Content:**

- Getting started (installation, first run, prerequisites)
- How-to guides (step-by-step task instructions)
- Tutorials (longer walkthroughs for common workflows)
- Troubleshooting and FAQ

**Rules:**

- No code internals. Show commands and outputs, not implementation.
- Assume the reader knows nothing about the codebase.
- Every guide must be independently executable from start to finish.

### `docs/theory/` — Concepts & Theory

**Audience:** Anyone who wants to understand *why* and *how* the system works at a conceptual level — users, researchers, developers.

**Content:**

- Conceptual explanations of what the tool does and why
- Mathematical foundations where relevant (equations are welcome)
- Diagrams, graphs, and tables to illustrate ideas
- Definitions of domain-specific terms

**Rules:**

- No code snippets. Concepts only.
- Equations are written using MathJax (inline `$...$`, block `$$...$$`).
- Diagrams use Mermaid or embedded images.
- Write for a reader who understands the problem domain but not the implementation.

### `docs/api/` — API Reference

**Audience:** Developers who import the package and build on top of it.

**Content:**

- Public API surface for each package
- Auto-generated from docstrings using `mkdocstrings`
- Integration examples and usage patterns
- Configuration reference

**Rules:**

- Use `mkdocstrings` directives to pull docstrings directly from source — do not duplicate code.
- Keep hand-written prose minimal; let the docstrings carry the detail.
- Every public class, function, and module that is part of the public API must appear here.

**Multi-package workspace:** When the project contains multiple packages, create one subfolder per package:

```sh
docs/api/
  corridorkey/
    index.md        # overview of the corridorkey package API
    scan.md         # scan() and scanner stage
    engine.md       # Engine class
    ...
  corridorkey-cli/
    index.md        # overview of the CLI package API
    commands.md     # CLI commands reference
    ...
```

Each subfolder's `index.md` describes what the package does and links to its sections.

### `docs/architecture/` — Architecture & Internals

**Audience:** Contributors and maintainers who need to understand how the system is built.

**Content:**

- Pipeline stages and how they connect
- Architectural decisions and the reasoning behind them
- Data flow diagrams
- Internal contracts and interfaces

**Rules:**

- Code references are allowed here (link to source, do not duplicate).
- Explain decisions, not just structure — a reader should understand *why* things are built this way.

### `docs/dev/` — Developer Guide

**Audience:** Contributors setting up a local environment or adding to the project.

**Content:**

- Local setup and development environment
- Testing strategy and how to run tests
- How to add new features, stages, or plugins
- CI/CD and release process

**Rules:**

- Commands must be exact and tested. A developer should be able to copy-paste and have it work.
- Keep setup instructions up to date with tooling changes (`uv`, `ruff`, `ty`, `pytest`).

---

## Multi-Package Workspace Rules

When a repository contains multiple packages (e.g. `packages/corridorkey/` and `packages/corridorkey-cli/`):

1. **Shared docs live at the root** — `docs/guide/`, `docs/theory/`, `docs/architecture/` cover the project as a whole.
2. **Package-specific API docs live under `docs/api/<package-name>/`** — one subfolder per package, never mixed.
3. **The `mkdocstrings` paths config must include all package source roots** — verify in `zensical.toml` (or equivalent config) that all `packages/*/src` paths are listed.
4. **Navigation tabs reflect the split** — the API tab should have sub-navigation per package so readers can find the right package immediately.

---

## Zensical-Specific Notes

When using Zensical (this project's doc toolchain):

- Navigation is defined in `zensical.toml` under `nav`. Top-level entries become tabs (due to `navigation.tabs` feature).
- After adding or renaming any page, update `nav` in `zensical.toml` or the page will not appear in the site.
- Validate with `uv run zensical build --clean` before committing.
- `mkdocstrings` directives use the format:

  ```markdown
  ::: corridorkey.stages.scanner.orchestrator.scan
  ```

  The path must match the importable Python path, not the file path.

---
name: audit
description: 'Systematically examine a file or stage: read everything in scope, challenge every line, and produce a ranked findings report — no code changes. Use this skill whenever the user wants to understand, review, or assess code before touching it. Triggers on: "audit", "deep dive", "review", "scrutinise", "look at", "go through", "what is this doing", "is this good", "check this stage", or any request to understand code before making changes.'
argument-hint: 'Name the stage or specific file to audit (e.g. "loader orchestrator", "scanner normaliser", "the whole scanner stage").'
---

# Skill: audit

Systematically read and examine a file or stage. The output is a findings report — not code changes. The user decides what to act on after seeing the report.

The value of this skill is that it forces a complete read before any opinion is formed. Partial reads lead to partial understanding, which leads to wrong fixes. Read everything first.

## What this skill does NOT do

- It does not change any code
- It does not write or modify tests
- It does not rewrite docstrings

Those are separate skills: `fix-code`, `write-tests`, and `update-docstrings`.

## Steps

### 1. Orient — read everything in scope

If given a single file, read that file plus:

- The stage `__init__.py` to confirm the public API surface
- How the engine or caller invokes this file (usually `engine.py` or the parent orchestrator)

If given a whole stage directory, read every file in it:

- `orchestrator.py` — entry point and dispatch logic
- `contracts.py` — input/output data models
- Any helpers (`normaliser.py`, `validator.py`, `extractor.py`, etc.)
- The `__init__.py` and the engine call site

If given a very large scope (many files), read the `__init__.py` files first to understand the public surface, then read the implementation files. Do not skip any file that is in scope. Form a complete picture before reporting anything.

### 2. Examine — challenge every line

For each file, work through these lenses:

**Purpose**

- What does this file do? One sentence.
- What does each public function/class do? One sentence each.

**Necessity**

- Is every function and class actually called by a real caller?
- Is there dead code that can be deleted?
- Are there abstractions that exist only for their own sake — wrappers that add no value?

**Complexity**

- Does the complexity match the problem?
- Flag any function with cyclomatic complexity > 10 (Ruff C901)
- Flag any function that is doing more than one job
- Flag any parameter that is always passed the same value (candidate for removal)

**Correctness**

- Is the logic correct for all inputs, including edge cases?
- Are error paths handled consistently with the rest of the codebase?
- Are there silent failures — exceptions swallowed without logging or re-raising?

**Structure**

- Is logic in the right file, or has it leaked into the wrong layer?
- Are there redundant branches or duplicated patterns that could be a single implementation?
- Do names accurately represent what things do?

Be direct. If a line is useless, say so. If a split is over-engineered, say so.
If something is well-designed, say that too — the report should be honest in both directions.

## Report format

Structure the report as one section per file:

**File: `filename.py`**

- What it does (one sentence)
- What is good
- What is wrong or questionable (with specific line references where useful)
- **Verdict:** one of three levels:
    - *Keep as-is* — nothing to act on
    - *Needs cleanup* — small targeted fixes: dead code, wrong docstrings, redundant parameters, minor naming
    - *Needs rethink* — structural issues: wrong abstraction, logic in the wrong layer, complexity that doesn't match the problem

End with a **Summary** of the most important findings across all files, ranked by impact. Each item in the summary should name the file, the issue, and the suggested next skill to use (e.g. "→ fix-code", "→ update-docstrings", "→ write-tests").

The user will decide what to act on next.

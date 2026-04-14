---
name: fix-code
description: 'Act on audit findings: remove dead code, eliminate unnecessary wrappers and flags, collapse duplication, rename for clarity, and refactor for simplicity. Runs quality gates, fixes logging, and updates docstrings after every change. Use this skill whenever the user wants to clean up, simplify, or improve code structure after an audit. Triggers on: "fix the code", "clean this up", "remove dead code", "simplify this", "act on the audit findings", "refactor", or any request to improve code quality based on findings.'
argument-hint: 'Name the file or stage to fix, and optionally specify what to focus on (e.g. "scanner normaliser — focus on the log_ambiguous flag").'
---

# Skill: fix-code

Apply the findings from an `audit` report. The goal is simpler, more honest code — not more code. Every change must leave the codebase easier to read and maintain than before.

This skill chains four steps in order. Each step must complete cleanly before the next begins.

If the user specifies a focus (e.g. "just remove the dead wrappers"), work only tha category in Step 1, then still run quality gates at the end.

## Step 1 — Fix the code

Work through each category that applies to the target. Skip categories with nothing to fix.

### Remove dead code

- Delete functions, classes, and parameters that have no callers
- Delete commented-out code
- Delete imports that are no longer used after removals

### Eliminate unnecessary wrappers

A wrapper is unnecessary when:

- It is a one-line delegation with no added logic
- Its name adds no information beyond the target it calls
- Callers can call the target directly without losing clarity

When removing a wrapper:

1. Inline the call at every call site
2. Update any tests that import or patch the wrapper by name
3. Delete the wrapper

### Remove redundant parameters and flags

A parameter is redundant when:

- It is a boolean that controls behaviour that should simply always happen (e.g. `log_ambiguous=False` suppressing a warning that should always fire)
- It exists only to prevent a double-action that can be fixed by removing the duplicate at the source instead
- It is always passed the same value at every call site

When removing a parameter:

1. Decide what the unconditional behaviour should be
2. Remove the conditional branch, keeping the correct behaviour
3. Remove the parameter from the signature and all call sites
4. Update tests that pass the parameter

### Collapse duplication

When two functions or code blocks do the same thing with minor variation:

1. Extract a single implementation that takes the varying part as a parameter
2. Replace both originals with calls to the shared implementation
3. If the originals were public wrappers with no added value, remove them too

### Rename for clarity

Rename when:

- A name misrepresents what the thing does
- A name is generic where a specific name would communicate intent
- A parameter name doesn't match what it actually controls

Use semantic rename (affects all references) rather than manual find-replace.

### Refactor for simplicity

When a function is doing more than one job, or its complexity doesn't match the problem:

- Split it if the jobs are genuinely independent
- Flatten it if nesting is hiding simple logic
- Inline it if the abstraction adds no value

Target: cyclomatic complexity ≤ 10 per function (Ruff C901).

---

## Step 2 — Fix logging

Apply the `fix-logging` skill to the same target.

Key rules:

- Every error path that swallows an exception must have a warning log
- Every skipped item must have a warning log with the reason
- Every significant lifecycle event must have an info log
- No double-logging between a function and its caller
- Log messages must include enough context to diagnose without the source code

See the `fix-logging` skill for the full level conventions.

---

## Step 3 — Update docstrings

Apply the `update-docstrings` skill to the same target.

Key rules:

- Non-`__init__.py` files have no module docstring
- Private functions have no docstrings
- Public function docstrings are verified against the code — not accepted on trust
- `Raises:` sections are derived from scratch by tracing raise sites and called functions
- Redundant docstrings on one-line wrappers are removed

See the `update-docstrings` skill for the full checklist.

---

## Step 4 — Quality gates

Run after all changes are complete. All four must pass clean.

```sh
uv run ruff format
uv run ruff check --fix
uv run ruff check
uv run ty check
```

Then run the stage's unit tests:

```sh
.venv/Scripts/python.exe -m pytest packages/corridorkey/tests/unit/<stage>/ -v --tb=short
```

All tests must pass before the skill is considered done.

---

## What "done" looks like

- No dead code, unnecessary wrappers, or redundant parameters remain
- No duplicated logic that could be a single implementation
- All names accurately represent what they do
- Cyclomatic complexity ≤ 10 for all functions
- Every significant event is logged at the correct level with sufficient context
- No silent error paths
- Every docstring claim has been verified against the actual code
- No module docstrings outside `__init__.py`
- No docstrings on private functions
- All quality gates pass clean
- All tests pass

---
name: update-docstrings
description: 'Audit and rewrite docstrings in a stage or file so they accurately describe what the code actually does — no fiction, no stale copy. Use this skill whenever the user wants to add, fix, or verify docstrings. Triggers on: "update docstrings", "fix docstrings", "write docstrings", "document this", "docstrings are wrong", "add docs", "docstrings are missing", or any request to improve inline documentation for a specific file or stage.'
argument-hint: 'Name the stage or file to document (e.g. "scanner orchestrator", "loader contracts", "the whole scanner stage").'
---

# Skill: update-docstrings

Audit every docstring in the target file(s) and rewrite any that are wrong, stale, missing, or misleading. No logic changes, no test changes, no reformatting.

The core principle: **do not trust what is written — verify every claim against the code.** A docstring that describes what a function used to do is worse than no docstring at all, because it actively misleads the reader.

## Rules for this codebase

**Module docstrings** — only `__init__.py` files get one. All other `.py` files have no module docstring. The filename, location, and public function docstrings are sufficient.

**Private functions and classes** (`_name`) — no docstrings. They are implementation detail. Use an inline comment at a specific line if the logic is non-obvious.

**Trivial one-liners** — no docstring if the name and signature are self-explanatory.

**Public functions and classes** — document anything non-trivial. If a public function has no docstring, write one.

## Steps

### 1. Read the source — all of it

Read every function, class, and module in the target file(s). Also read any functions that are *called* from the target, specifically to verify exception propagation and return values. Do not assume a called function does what its name implies — read it.

### 2. Audit each docstring

For every existing docstring, verify each claim independently. The questions to ask:

**Summary line** — Does it describe what the function does *right now*? Is it specific to this function, or could it be copy-pasted into any file without being wrong? Is it redundant (a one-line wrapper whose docstring just restates the target)?

**Args** — Does every documented parameter still exist in the signature? Is every parameter in the signature documented? Are the descriptions accurate, not just plausible-sounding?

**Returns** — Read the actual return statements. Does the documented return match all of them? If the function returns different things under different conditions, are all conditions covered?

**Raises** — This is the most commonly wrong section. Verify it from scratch every time:

1. List every `raise` statement in the function body
2. For every called function, read its implementation and list what it raises
3. Check whether those exceptions are caught and converted before they leave the function
4. Document only what actually reaches the caller — remove anything that is caught internally

**Missing docstrings** — If a non-trivial public function has no docstring, write one.

### 3. Decide: rewrite, remove, or keep

- **Rewrite** if any part is inaccurate, stale, or misleading
- **Remove** if the function is private, trivial, or a redundant one-line wrapper
- **Keep** only after every claim has been verified against the code

### 4. Write accurate docstrings

For public functions and methods, use this structure where the function is non-trivial:

```
One-line summary.

Longer explanation if needed (omit if the summary is sufficient).

Args:
    param: Description. Include units or constraints if relevant.

Returns:
    What is returned and under what condition.

Raises:
    ExceptionType: When and why this is raised.
```

For `__init__.py` module docstrings: what the package owns, what it exposes publicly,
and what it does not do.

For class docstrings: what the class represents, its invariants, when to use it.

### 5. Verify

Run the linter first — it catches formatting issues in the docstrings themselves:

```sh
uv run ruff check
```

Then run the unit tests to confirm nothing broke:

```sh
.venv/Scripts/python.exe -m pytest packages/corridorkey/tests/unit/<stage>/ -v --tb=short
```

## What "done" looks like

- Every claim in every docstring has been verified against the actual code
- No docstring describes behaviour that no longer exists
- Non-`__init__.py` files have no module docstring
- Private functions have no docstrings
- Every non-trivial public function has a docstring
- Linter and tests pass

---
name: write-tests
description: 'Audit test coverage for a stage or file by reading branches, identify gaps, write missing tests, and verify all tests pass. Use this skill whenever the user wants to add tests, check coverage, fill test gaps, or ensure a file is properly tested. Triggers on: "write tests", "add tests", "fill test gaps", "check coverage", "make sure this is tested", "tests are missing", or any request to improve test coverage for a specific file or stage.'
argument-hint: 'Name the stage or file to cover (e.g. "scanner orchestrator", "loader validator", "the whole scanner stage").'
---

# Skill: write-tests

Read the source code, map every branch to an existing test, identify what is not covered, write the missing tests, and verify everything passes.

The goal is not 100% line coverage — it is meaningful branch coverage. Every decision point (if/else, exception path, early return, edge case) should have at least one test that exercises it. Tests that only cover the happy path are incomplete.

## Test file location

Every source file has exactly one corresponding test file per test type, at the mirror path:

```
src/corridorkey/<path>/<file>.py
→
tests/unit/<path>/test_<file>.py                      (unit tests)
tests/property/<path>/test_<file>_properties.py       (property tests)
```

Examples:

| Source file | Unit test |
|---|---|
| `src/corridorkey/stages/scanner/normaliser.py` | `tests/unit/stages/scanner/test_normaliser.py` |
| `src/corridorkey/stages/loader/orchestrator.py` | `tests/unit/stages/loader/test_orchestrator.py` |
| `src/corridorkey/infra/utils.py` | `tests/unit/infra/test_utils.py` |
| `src/corridorkey/engine.py` | `tests/unit/test_engine.py` |

If the test file does not exist yet, create it at the correct mirror path. Also create any missing `__init__.py` files in the new directories.

Never create `_extra.py` files. All tests for a source file belong in its single test file.

## Test writing conventions

**Every test function must have a docstring** describing what is tested, under what condition, and what the expected outcome is. This makes test output readable without having to open the file.

```python
def test_normalise_video_already_in_place_skips_move(self, tmp_path: Path):
    """When the destination already exists with the same size, the source is not moved."""
```

**One assertion per test** where possible. Tests that assert many things at once are hard to diagnose when they fail.

**Group related tests in a class** named after the function or behaviour under test: `TestNormaliseVideo`, `TestTryBuildClip`, `TestScanErrors`.

**Use `tmp_path`** for all filesystem operations — never hardcode paths.

**Test observable behaviour, not implementation details.** If a test would break when you rename a private helper without changing any public behaviour, it is testing the wrong thing.

## Steps

### 1. Read the source

Read every function in the target file(s). For each function, list every branch:

- Conditional paths (if/else, early return)
- Exception paths (raise, except)
- Edge cases (empty input, None, zero, boundary values)
- Any guard conditions (permission checks, type checks, existence checks)

Do this by reading the code directly. Do not rely on coverage tools — they may fail in this repo due to heavy import chains involving torch and cv2.

### 2. Map branches to existing tests

Locate the test file at the mirror path. Read it fully. For each branch you identified in step 1, find the test that exercises it. Mark what is covered and what is not.

### 3. Write the missing tests

Add missing tests to the corresponding test file. Follow the conventions above.

For each gap:

- Name the test to describe the condition: `test_<function>_<condition>_<outcome>`
- Write the minimal setup needed to reach that branch
- Assert the observable outcome (return value, exception raised, file created, log emitted)
- Add a docstring

### 4. Verify

Run the tests for the target stage:

```sh
.venv/Scripts/python.exe -m pytest packages/corridorkey/tests/unit/<stage>/ -v --tb=short
```

If property tests exist for the target, run those too:

```sh
.venv/Scripts/python.exe -m pytest packages/corridorkey/tests/property/<stage>/ -v --tb=short
```

All tests must pass before finishing. Fix any failures — do not leave broken tests.

## What "done" looks like

- Every branch in every public function has at least one test
- Every test function has a docstring
- All test files are at the correct mirror path
- No `_extra.py` files exist
- All tests pass

---
name: fix-logging
description: 'Audit and fix logging in a file or stage so every significant event is logged at the right level — no silent failures, no noise, no missing context. Use this skill after an audit flags silent error paths or missing log coverage, when a bug report comes in and the log file does not explain what happened, or before releasing a stage. Triggers on: "fix logging", "add logging", "missing logs", "silent failure", "log coverage", "improve logging", or any request to ensure a file or stage is properly instrumented.'
argument-hint: 'Name the file or stage to audit (e.g. "scanner normaliser", "loader extractor").'
---

# Skill: fix-logging

Audit every log call in the target file(s) and fix any that are missing, at the wrong
level, or carrying insufficient context. The goal is a log output that lets someone
diagnose a failure from the log file alone, without needing to reproduce it.

This skill does not change logic, control flow, tests, or code formatting.

## Level conventions for this codebase

**`logger.debug`** — internal state useful only when actively debugging. Examples: idempotency skips ("video already in place"), per-frame processing details, resolved config values already logged at a higher level elsewhere.

**`logger.info`** — significant lifecycle events a user or operator would want to see in a normal run. Examples: starting/completing a major operation (extraction, model load, file move), resolved device or mode decisions, frame counts.

**`logger.warning`** — something unexpected happened but processing can continue. The user should know about it. Examples: a clip was skipped, a directory could not be read, a checkpoint has missing keys, a fallback was triggered.

**`logger.error`** — a failure that stops processing for the current item but not the whole job. Use sparingly — most errors in this codebase are raised as exceptions and logged by the engine, not by the stage itself.

Do not use `logger.critical` or `logger.exception` unless there is a specific reason.

## Steps

### 1. Read the source and its immediate callers

Read every function in the target file(s). For each one, identify:

- Every `raise` statement — is the error logged before raising, or does the caller log it?
- Every `except` block — is the caught exception logged with enough context?
- Every silent return or early exit — should it be logged?
- Every significant state change (file moved, directory created, model loaded) — is it logged?

Also read the **immediate caller** of each function (one level up) to check whether the caller already logs the same event. Do not add a log if the caller already covers it — that produces double-logging. You do not need to trace the full call stack, just one level.

### 2. Audit each existing log call

For every log call, verify:

**Level** — does it match the conventions above? A skipped clip is a warning, not info. A per-frame detail is debug, not info. A successful file move is info, not debug.

**Message format** — use `logger.warning("msg '%s'", value)` not `logger.warning("msg '%s'" % value)`. The `%` style bypasses lazy evaluation and is flagged by ruff. Always use the comma-separated form.

**Message content** — does it include enough context to diagnose the problem?

- File paths for filesystem operations
- Clip or item names for item-level events
- Numeric values (frame counts, sizes, VRAM) where relevant
- What happened, not just that something happened:
  `"Skipping 'clip_a': Input/ contains multiple videos"` not `"Skipping clip"`

**Silent paths** — any `except` block that swallows an exception without logging is a bugunless the immediate caller is guaranteed to log it. Verify before accepting silence.

**Missing coverage** — any significant operation with no log. Ask: if this fails silently,would the log file explain why?

### 3. Fix

Apply fixes in this order:

1. Add missing `warning` logs for skipped/failed paths that have no log
2. Correct levels that are wrong
3. Improve messages that lack context (add path, name, count, reason)
4. Fix format string style (`%` → comma-separated)
5. Remove duplicate logs where both a function and its caller log the same event —
   keep the log closest to the event, remove the other

### 4. Verify

Run the linter first — it catches format string issues in log calls:

```sh
uv run ruff check <file>
```

Then run the stage's unit tests:

```sh
.venv/Scripts/python.exe -m pytest packages/corridorkey/tests/unit/<stage>/ -v --tb=short
```

## What "done" looks like

- Every `except` block that swallows an exception has a log, or the immediate caller is verified to log it
- Every skipped item has a warning log with the reason
- Every significant lifecycle event has an info log
- No double-logging between a function and its immediate caller
- All log messages use comma-separated format strings, not `%` interpolation
- Log messages include enough context to diagnose without the source code
- Linter and tests pass

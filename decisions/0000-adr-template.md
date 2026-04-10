# 0000 — ADR Template

Status: Template

## What is an ADR?

An Architectural Decision Record (ADR) captures a single design or architecture
decision, the reasoning behind it, the alternatives that were considered, and the
consequences of the choice. The collection of ADRs in this folder is the project's
decision log.

For background on the ADR practice and available templates, see: <https://adr.github.io/>

## Template format

This project uses **MADR** (Markdown Architectural Decision Records).

MADR is chosen because it explicitly requires documenting considered options with their
pros and cons — the section that makes ADRs genuinely useful long-term by preventing
the same alternatives from being re-litigated. It comes in a minimal and a full
variant; use whichever fits the complexity of the decision.

Full MADR specification and examples: <https://adr.github.io/madr/>

## How to use this template

1. Copy the bare template below and name the file `NNNN-short-descriptive-title.md`
   where `NNNN` is the next available number (zero-padded to four digits).
2. Fill in each section. Keep it concise — a good ADR is rarely more than one page.
3. Set the status to `Accepted`.
4. Never edit a past ADR to change its decision. If the decision changes, write a new
   ADR and set its status to `Accepted`. Add a one-line note to the old ADR:
   `Superseded by NNNN`.

## Numbering and precedence

The highest-numbered ADR on a topic takes precedence. If two ADRs conflict, the later
one is the accepted norm.

---

```markdown
# NNNN — Title

Status: Accepted | Superseded by NNNN

## Context and problem statement (required)

What problem were we solving? What constraints, requirements, or forces shaped the
decision? Include enough background that someone unfamiliar with the situation can
understand why a decision was needed.

## Considered options (required)

- Option A
- Option B
- Option C

## Decision outcome (required)

Chosen option: **Option A**, because [core reasoning — not just what, but why this
over the others].

### Consequences (optional)

- Good: [positive outcome]
- Good: [positive outcome]
- Bad: [known trade-off or limitation]

## Pros and cons of the options (optional)

### Option A

- Good: [reason]
- Bad: [reason]

### Option B

- Good: [reason]
- Bad: [reason]

### Option C

- Good: [reason]
- Bad: [reason]
```

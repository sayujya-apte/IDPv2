# AGENTS.md

## Repo status

Pre-implementation. The repo has **no source code yet** — no commits, no `pyproject.toml`,
no lockfile, no tests, no CI. Do not go looking for entrypoints, package boundaries, or
existing conventions to follow; they do not exist yet. The target language is **Python**, but
the packaging/test/lint tooling has not been chosen.

That also means: **do not invent commands.** There is no `make`, `pytest`, `ruff`, or
`poetry` config to run. If you need tooling, set it up first and say so.

## Source of truth

`Centralized Synchronous Message Routing For Agents in a Graph.pdf` (3 pages) is the design
spec for the whole system. It is the authoritative description of the architecture — read it
before designing or implementing anything. It is a PDF, not prose in the repo, so it will
not show up in a normal text/glob sweep; fetch it explicitly.

Implementation details not covered by the spec (language-specific structure, data types,
naming, module layout) are yours to choose — the spec does not constrain them.

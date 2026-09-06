# dev-log

A per-experiment / per-build record of **what was asked for** and **what was
built**, for AI-assisted development work in this repo.

Each file pairs the human-written prompts that drove a piece of work with a short
summary of what landed and the commits that carry it. This is the public,
curated disclosure record. The full design rationale, hypotheses, staged-gate
results, and verdicts live in the private research-notes repo (`notes/`, not
committed here).

## One file per contract id

`dev-log/<id>.md`, where `<id>` is the experiment-contract id
(`YYYY-MM-DD-slug-NN`) — the same string used by `configs/<id>.yaml`, the
private design/build note, and the run directory. See `notes/hub/conventions.md`.

A dev-log file is only created for an `<id>` that already exists as a build or
experiment note. Work that is not a registered experiment or build (one-off
refactors, dependency bumps, bug fixes) is covered by its commit message and the
`Claude-Session:` commit trailer — it does not get a dev-log file.

## What goes in a file

Frontmatter, a title, then one `## Session <date>` block per working session:

- **Prompts** — the instructions the human actually gave, verbatim or lightly
  trimmed. Keep the substantive ones (what to do, what changed direction, what
  was ruled out); drop pure typo-fixes and "yes go ahead". These are written by
  the human, or transcribed by Claude and then trimmed by the human.
- **Implemented** — 3–5 sentences, written by Claude: what changed, which entry
  points, which ladder rungs passed. Not a second copy of the private note's
  `## Implementation` section — a pointer-length summary.
- **Commits** — the short hashes and subject lines for that session's commits.

## Workflow

At the end of a session that produced commits for an experiment or build:

1. Land all the code commits as normal.
2. Create or append to `dev-log/<id>.md` — add a `## Session <date>` block with
   the prompts, the summary, and the hashes of the commits just made.
3. Commit the dev-log file on its own: `dev-log: log <id> session <date>`.
   Because this is a trailing commit, every hash it lists already exists; it does
   not list itself.

## This is a public file

Same review bar as a commit message. No secrets or API keys, no absolute cluster
paths (repo-relative only), no unpublished-result specifics you would not put in
a public commit message. If a prompt contained something that shouldn't be
public, paraphrase that part and say so.

See `_TEMPLATE.md` for the structure to copy.

---
description: Record this session's AI-assisted work into dev-log/<id>.md and commit it
argument-hint: <experiment-or-build-id>
---

Argument: $ARGUMENTS — an experiment or build id (`YYYY-MM-DD-slug-NN`). Run this at
the end of a session that produced commits for that experiment or build.

Read `dev-log/README.md` and `dev-log/_TEMPLATE.md` first.

## 1. Verify the id

The id must already exist as a note in the private notes repo — either
`notes/<project>/builds/<id>.md` or `notes/<project>/experiments/<id>.md` (project
is `scholarlm`). If neither exists, stop and say so — do not create a dev-log entry
for an unregistered id.

## 2. Determine this session's implementation commits

List `git log` on the current branch back to where this session's work started.
Identify the commits that belong to this experiment or build. Show that list to me
and confirm it before continuing.

If there are uncommitted changes in the working tree that are part of this work,
stop and tell me — do not commit implementation code yourself. The dev-log entry
records commits that already exist; landing the implementation is the normal
staged-gate workflow, not this command's job.

## 3. Write the dev-log entry

Create `dev-log/<id>.md` from `_TEMPLATE.md` if it does not exist, or append a new
`## Session <today's date>` block if it does. Do not rewrite earlier session blocks.

- **Frontmatter** (new file only): `id`, `kind` (`experiment` | `build`), and
  `config:` pointing at `configs/<id>.yaml` if one exists (omit the line otherwise).
- **Prompts** — the instructions I gave this session, verbatim or lightly trimmed.
  Keep the substantive ones (what to do, what changed direction, what was ruled
  out); drop typo-fixes and bare approvals. If a request is long or scattered
  across many messages, write a faithful summary and say it is a summary.
- **Implemented** — 3–5 sentences: what changed, which entry points, which configs
  were used or created, which ladder rungs passed. Name `configs/<id>.yaml` and any
  other config touched. This is a pointer-length summary, not a copy of the private
  note's `## Implementation` / `## Session log`.
- **Commits** — the confirmed short hashes and subject lines from step 2.

Public file: no secrets, no absolute cluster paths (repo-relative only), no
unpublished-result specifics you would not put in a commit message.

## 4. Commit the dev-log entry

Commit `dev-log/<id>.md` on its own — nothing else staged:

    git add dev-log/<id>.md
    git commit -m "dev-log: log <id> session <today's date>"

This is a trailing commit: every hash it lists already exists, and it does not list
itself. Report the resulting commit hash. Do not push.

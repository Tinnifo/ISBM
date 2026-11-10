# AGENTS.md: ISBM research team

Rules for every agent (and human) working in this repo. Tinni owns the research
direction and merges every PR. Linear project:
[infinite-stochastic-block-models (P-FEW-354)](https://linear.app/tinnifo/project/infinite-stochastic-block-models-12467a4ccbca),
team **Research Projects (FEW)**.

## Roles

| Who | Role | Owns |
|---|---|---|
| **Roxana** | Research Scientist | Research question, hypotheses, literature reviews (`lit/`), go / kill calls |
| **Ezra** | Research Engineer | Library code (`src/isbm/`), marimo notebooks, experiment records, PRs |
| **Theo** | Theory & Evaluation | Derivations, the evaluation protocol, **required reviewer** of every result PR |
| **Paul** | PM | Turns every request into a Linear issue, chases stale work, enforces these rules, writes the daily cross-team roll-up, runs the Friday retro |
| **Demis** | Tinni's chief of staff | Reviews the roll-up with Tinni, relays Tinni's decisions, sets up workflow infra |
| **Tinni** | PI | Decides direction, approves predictions and compute spend, merges PRs |

## Hard rules

1. **No experiment without a hypothesis issue.** Every `type/experiment` issue is a
   sub-issue of a `type/hypothesis` issue (or, for pure reproductions/infra, links the
   issue that motivates it).
2. **No PR without a FEW issue ID.** PR title: `FEW-123: short summary`. Branch:
   Linear style, `few-123-short-slug` (lowercase, hyphens). Setup/infra branches may use
   `setup/<slug>` but the PR title still carries the issue ID.
3. **No result without a commit and a journal link.** A `type/result` issue (or result
   comment) states the numbers, the commit SHA that produced them, the verdict against the
   locked prediction, and links the lab-journal entry in Tinni's vault.
4. **Predictions are locked before the run.** Write the prediction and confidence in
   `experiments/FEW-123/record.md` and commit it *before* running. Never edit a locked
   prediction afterwards; add a dated note instead.
5. **Never push to `main`.** Open a PR; Tinni merges.
6. **Never commit secrets.** API keys live in environment variables or `~/.config/…`
   (see *Literature*). `.gitignore` covers the usual patterns; check `git diff --cached`
   before committing.

## Where experiments run

- **Experiments never run on Tinni's Mac.** The Mac is used only for writing vault notes
  and calendar entries.
- Experiments run on the **shared box** (Linux, CPU), in a **clean git checkout** of
  `Tinnifo/ISBM` at `/workspace/repos/ISBM`, synced with `git fetch` / `git switch`.
  No ad hoc copies, tarballs or snapshots.
- Environment: `uv sync` (locked by `uv.lock`), notebooks run with
  `uv run marimo edit|run|export …`. Record the commit SHA and the `uv.lock` hash in the
  experiment record.
- Outputs (`results.json`, small figures) are committed under `experiments/FEW-123/`
  through a PR. Large artefacts (> ~5 MB) are not committed; record where they are and
  their checksum.
- **Heavy sweeps or GPU** may use Hugging Face Jobs **only after Tinni explicitly approves
  the estimated cost** in the Linear issue (label `needs-tinni` until approved).

## Git identity (who made each decision)

Every bot commits under its own identity so `git log` / `git blame` show who did what.
The push itself goes through Tinni's GitHub account; the *author* is the bot.

```
Author:  <Name> (<Role> bot) <tinnifo+<name>@users.noreply.github.com>
```

| Bot | `user.name` | `user.email` |
|---|---|---|
| Roxana | `Roxana (Research Scientist bot)` | `tinnifo+roxana@users.noreply.github.com` |
| Ezra | `Ezra (Research Engineer bot)` | `tinnifo+ezra@users.noreply.github.com` |
| Theo | `Theo (Theory & Evaluation bot)` | `tinnifo+theo@users.noreply.github.com` |
| Paul | `Paul (PM bot)` | `tinnifo+paul@users.noreply.github.com` |
| Demis | `Demis (Chief of Staff bot)` | `tinnifo+demis@users.noreply.github.com` |

Set it per command, never globally on a shared machine:

```bash
git -c user.name="Ezra (Research Engineer bot)" \
    -c user.email="tinnifo+ezra@users.noreply.github.com" \
    commit -m "FEW-123: add CGS fit cell" -m "Bot: ezra" -m "Linear: FEW-123"
```

Commit message: first line `FEW-123: imperative summary`; trailers `Bot: <name>` and
`Linear: FEW-123`. Decisions (go/kill, accepted derivation, changed protocol) get their
own commit or PR comment by the bot that made them.

## Literature

- Every hypothesis cites **≥ 2 works by ID** (DOI, arXiv ID or OpenAlex ID), and
  **≥ 1 of them is from the last 3 years**.
- No Zotero. Papers and literature notes live in `lit/` (BibTeX in `lit/papers.bib`,
  one note per issue in `lit/notes/`) or in Tinni's vault.
- Search with `tools/lit_search.py` (OpenAlex + arXiv). The OpenAlex key is read from
  `OPENALEX_API_KEY` or `~/.config/openalex/api_key`. Never commit it.
- alphaXiv pages and the Hugging Face papers skill may be used for discussion/summaries;
  cite the underlying arXiv ID.

## Layout

```
src/isbm/              tested library code
Notebooks/             marimo notebooks; new ones named FEW-123_slug.py
experiments/FEW-123/   record.md (locked prediction), results.json, figures
lit/                   papers.bib + notes/FEW-123-*.md
journal/               repo-side index of lab-journal entries (entries live in the vault)
templates/             hypothesis, experiment record, result, journal entry, notebook header
tools/lit_search.py    OpenAlex + arXiv search → lit/
```

> `Notebooks/` keeps its existing capitalised name: macOS is case-insensitive, so a
> separate `notebooks/` folder would collide on Tinni's Mac.

## Notebooks

- marimo only (`.py`). The **first cell** is the header from
  `templates/notebook_header.py`: Linear issue URL, hypothesis, commit, seed.
- CI-style check before a PR: `uv run marimo export html Notebooks/FEW-123_slug.py -o /tmp/x.html`
  must succeed.

## Journal

Each result and each decision gets an entry in Tinni's Obsidian vault lab journal:
`work/research/infinite-stochastic-block-models/infinite-stochastic-block-models - Lab journal.md`
(template `Templates/Lab journal entry - research team.md` in the vault, mirrored in
`templates/journal_entry.md`). Every bot-written entry opens with
`> [!info] Written by <bot>` and links the Linear issue, the notebook commit and the papers.
Tinni's own `… - PSJ.md` is hers; bots don't write in it.

## Standups (Copenhagen time, weekdays)

| Time | What | Who |
|---|---|---|
| 08:15 | Async team standup: done (with links), next, blockers, needs Tinni | each bot |
| 08:30 | Cross-team roll-up: decisions first, then status, stale/blocked, new results | Paul |
| 08:45 | Roll-up review with Tinni; decisions relayed | Demis |
| Fri 15:00 | Research retro (30 min): what broke, process metrics, 1–3 kit changes | Paul |

## Linear labels

`type/hypothesis`, `type/experiment`, `type/result`, `type/lit-review`,
`type/derivation`, `type/infra`, `needs-tinni`, `blocked`. Templates for hypothesis,
experiment and result issues are in `templates/` (Linear's API can't create issue
templates; Tinni may paste them into Linear → Settings → Templates).

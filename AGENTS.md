# AGENTS.md — Hybrid LinUCB aligned reproduction

## Scope

This directory is an independent, self-contained reproduction and report. Work
only here. Do not edit the sibling `../report`, `../code`, or other project
files. Everything needed for review, reproduction and submission lives in this
folder.

## Source of truth for course requirements

The course assignment is `project_desc.pdf` in this folder, copied verbatim from
`/home/a/yahoodata/project_desc.pdf` (SHA-256
`cf456cd4215356a6c2e099c89db96203004979f4835081ef2fbd32b5121707a3`).
Before changing the report, slides or protocol, check the requested deliverables
against that PDF. The report must retain the exact identity:

- Title: `Contextual Bandits for News Recommendation\\under Changing User Preferences`
- Subtitle: `Chronological Yahoo Replay and Controlled Drift Experiments`
- Course line: `DSC6007 Dynamic Programming and Reinforcement Learning --- Group 14`
- Authors: `LI Yizhuo \\quad ZHANG Tianyi \\quad GOU Zhijie \\quad AN Yuxian \\quad XIAO Yuhang`
- Term: `Semester A 2026`
- Maximum 15 pages.

## Reproduction contract

- Python 3.13.2, exact package versions in `pyproject.toml` and `uv.lock`; no
  network at run time. Use `./run.sh` (it calls `uv run --frozen`).
- Raw data hashes: log `45a5ce2a1f64875389d5a5bcc65be8e95c3b6a91b53757e90b055da49915525f`;
  articles `07404caa43916d8b5e942a5016065222cec2e8ac44a57580cd584568d8232f9e`.
- Chronological split: 66,204 development / 18,010 validation / 15,786 test
  events; validation selects all hyperparameters; seed 2026 for core results.
- 30-seed sweeps (2026–2055) on the same fixed log for Random, contextual greedy
  and context-free ε-greedy; 10 poster seeds (0–9) and 20 extension seeds
  (100–119) in the synthetic drift study; horizon 3,000, six candidates.
- `LinUCBDisjoint` is Algorithm 1 (per-arm ridge). `LinUCBHybrid` is Algorithm 2
  (shared `beta`, per-arm `theta`, Schur complement, full quadratic UCB
  variance). The shared feature is `z = vec(u v^T)` in R^36.
- `ContextualGreedy` is the contextual ridge-mean comparator with zero UCB
  width; `EpsilonGreedy` is the original context-free §5.3/Table-1 baseline
  using empirical article means. Never relabel one as the other.
- Every code change requires: rerun `./run.sh`, recompile `report.pdf` with
  `./build_report.sh`, update hashes/numbers in `report.tex`, refresh figures,
  rerun `verify_consistency.py`, and refresh the paper-judge review against the
  new PDF hash.

## Paper-alignment notes

- Li et al. (2010) §5.2.2: `x_{t,a}` is the six-dimensional user vector only;
  `z_{t,a}` is the user–article outer product. Our `context()` implements this.
- Paper §5.3: context-free ε-greedy averages per-article payoffs and ignores
  context. The contextual ε-greedy variant used in earlier project code is now
  explicitly called `contextual greedy`.
- Paper Table 1 / Figure 3: hybrid advantages appear in sparse learning buckets
  and with a larger article pool. `sparsity/run_sparsity.py` tests this by
  holding the test split fixed and varying warmup data (100/10/5/1%) plus a
  50-article candidate-pool diagnostic.

## Documented deviations (do not silently remove)

- The supplied R6A features are already six-dimensional conjoint outputs; the
  paper's unit-length normalisation applies to the raw 1,000-dimensional
  categorical vectors before reduction, so no second normalisation is applied.
- Yahoo propensities are unavailable; replay CTR is descriptive, not a
  certified deployment value.
- Yahoo window units are logged events; simulation window units are rounds.
- The enlarged candidate pool in the sparsity diagnostic changes the logged
  candidate distribution and is labelled as a sensitivity check, not a
  faithful replay of the original logger.

## Review and submission

- Submit the current `report.pdf` to `https://paper-judge.netlify.app/` via the
  browser and save C1–C5 in `review_paper_judge.json` and
  `paper_judge_review.md`, including the reviewed PDF hash. Suggestions are
  recorded, never auto-applied. These two draft files are **local review
  evidence only**: they are gitignored and deliberately absent from the public
  repository. `verify_consistency.py` checks them when present and otherwise
  records a skip note, so a fresh clone still passes.
- The existing history was created locally with `user.name=ayx` and an empty
  email. Do not rewrite that history. New collaborators must commit with their
  own git identity, work on a branch, and open a pull request as described in
  `README.md`; push only to the team repository `ayx-dg/DSC6007_Plan_Aligned`.
- The public repository root **is** this `hybrid_aligned/` directory. A clone
  lands directly in these files; there is no extra `hybrid_aligned/` layer.

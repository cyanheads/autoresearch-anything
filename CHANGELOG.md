# Changelog

## [0.4.0] - 2026-03-11

### Added

- **Strategies section** in agent entry points (`CLAUDE.md`, `AGENTS.md`) — documents hill-climb, explore, and pareto strategies with concrete `keep_when` examples
- **Experiment lifecycle section** in `README.md` — explains keep/discard/park/crash statuses and lineage tracking
- **Example results table** in `README.md` quick start — shows what a typical results log looks like

### Changed

- `program.md.jinja` strips trailing periods from artifact descriptions for consistent formatting
- `program.md.jinja` deduplicates context sources that are already listed as artifacts
- `program.py` registers `rstrip_period` Jinja filter for template use

## [0.3.0] - 2026-03-11

### Added

- **Git protocol** in agent entry points and generated `program.md` — detailed branching, staging, commit format, parking, and revert conventions for experiment version control
- **Creativity escalation** guidance in "never stop" section — concrete tactics for when agents feel stuck (re-read logs, revisit parked experiments, vary change scale, combine winners)

### Changed

- Setup flow in `program.md.jinja` now includes git status check and `.gitignore` step for the results log
- "Never stop" section expanded from a single paragraph into structured guidance with numbered escalation steps
- Agent entry points (`CLAUDE.md`, `AGENTS.md`) restructured: git protocol and autonomy guidance broken into dedicated sections instead of inline bullets

## [0.2.0] - 2026-03-11

### Added

- **Experiment lineage tracking**: log columns now include `experiment` (sequential number), `parent` (commit the change branched from), and `tags` (freeform categorization labels)
- **Column reference** in generated `program.md` — documents every log column so agents know exactly what to record
- **Park status**: experiments that don't improve the primary metric but show promise can be parked (`git tag parked/<n>-<name>`) for later revisitation, alongside existing keep/discard/crash statuses
- **Conversation starter** in agent entry points (`CLAUDE.md`, `AGENTS.md`) — agents now introduce the framework and ask what to work on when no problem is specified

### Changed

- Default log columns updated from `[commit, status, description]` to `[experiment, parent, commit, status, description, tags]` in manifest schema, CLI scaffold, and all example experiments
- Experiment loop in `program.md.jinja` now tracks parent commits, records experiment numbers, and uses tags for categorizing approaches
- Logging instructions strengthened: "every" experiment must be logged including crashes and discards

## [0.1.0] - 2026-03-11

Complete rewrite of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) into a general-purpose autonomous experimentation framework.

### Added

- **Framework core** (`src/autoresearch/`):
  - `manifest.py` — Pydantic schema for `experiment.yaml` manifests, supporting artifacts, immutable files, metrics extraction, strategy config, constraints, and multi-agent definitions
  - `program.py` — Generates `program.md` agent instructions from manifest + Jinja template
  - `tracker.py` — Result logging with TSV, CSV, and JSONL format support
  - `cli.py` — CLI with `init`, `generate`, and `validate` subcommands
- **CLI entry point** (`autoresearch`) via `pyproject.toml` scripts
- **Jinja template** (`templates/program.md.jinja`) — default agent instructions template, generates a full experiment protocol from any manifest
- **Example experiments**:
  - `gpt-pretrain` — ML training optimization (closest to the original repo's purpose)
  - `prompt-optimization` — LLM prompt iteration with accuracy scoring
  - `code-benchmark` — algorithm performance tuning with multi-agent explorer/refiner roles
- **Agent entry points**: `CLAUDE.md` for Claude Code, `AGENTS.md` for Codex and other agents — both provide setup, resumption, and experiment loop instructions

### Changed

- Renamed project from `autoresearch` to `autoresearch-anything`
- Rewrote `README.md` with framework documentation, manifest anatomy, multi-agent usage, and project structure
- Rewrote `pyproject.toml` with new dependencies (Pydantic, PyYAML, Jinja2) and hatchling build system
- Rewrote `.gitignore` for general Python/uv/IDE patterns and experiment runtime artifacts

### Removed

- `train.py` — single-purpose GPT training script (now an example template)
- `prepare.py` — FineWeb data preparation script
- `analysis.ipynb` — experiment analysis notebook
- `progress.png` — training progress visualization
- `program.md` — hardcoded agent instructions (now dynamically generated from manifests)

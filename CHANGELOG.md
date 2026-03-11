# Changelog

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

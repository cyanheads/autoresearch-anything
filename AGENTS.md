# autoresearch-anything

You are an autonomous research agent. This repo is a framework for running experiments — you do the experimenting.

## What this is

A general-purpose experimentation framework. You define a problem via `experiment.yaml`, generate your own instructions via the CLI, then run an autonomous experiment loop: modify code, run it, measure results, keep or discard, repeat.

## First time setup

If there's no `experiment.yaml` in the repo root (or the user tells you what they want to work on):

1. **Install dependencies**: `uv sync`
2. **Understand what's available**: Read the examples in `examples/` to see what experiment manifests look like
3. **Scaffold an experiment**: Use `uv run autoresearch init <directory>` for a blank template, or `uv run autoresearch init <directory> -t <example>` to start from an example (available: `gpt-pretrain`, `prompt-optimization`, `code-benchmark`)
4. **Edit `experiment.yaml`** to match what the user wants to optimize — ask them if anything is unclear
5. **Create the artifact files** that the experiment needs (training scripts, eval harnesses, prompts, etc.)
6. **Generate your instructions**: `uv run autoresearch generate <path>/experiment.yaml -o <path>/ --init-log`
7. **Read the generated `program.md`** — it contains your full experiment loop instructions
8. Follow the setup steps in `program.md`, then begin the experiment loop

## Resuming an existing experiment

If `experiment.yaml` already exists and the user wants to continue:

1. Run `uv run autoresearch generate experiment.yaml -o . --init-log` if `program.md` doesn't exist yet
2. Read `program.md` for your instructions
3. Check `results.tsv` for prior results
4. Check git log for the current experiment branch state
5. Resume the experiment loop from where it left off

## CLI reference

```bash
uv run autoresearch init <directory> [-t <template>]   # Scaffold experiment (blank or from template)
uv run autoresearch generate <manifest> -o <dir>       # Generate program.md from experiment.yaml
uv run autoresearch validate <manifest>                # Check experiment.yaml for errors
```

## Key rules

- **Read `program.md` before experimenting.** It has everything: what to modify, how to evaluate, decision rules, logging format.
- **Never modify files listed as `immutable` in the manifest.**
- **Never stop unless the human tells you to.** The experiment loop runs indefinitely.
- **Log every experiment** to the results file, including crashes and discards.
- **Use git** for version control — commit before each run, revert on discard.

## Project structure

```
src/autoresearch/          Framework source (you don't modify this)
templates/                 Jinja templates for program.md generation
examples/                  Example experiment configurations
    gpt-pretrain/          ML training optimization
    prompt-optimization/   LLM prompt iteration
    code-benchmark/        Algorithm performance tuning (multi-agent)
```

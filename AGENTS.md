# autoresearch-anything

You are an autonomous research agent. This repo is a framework for running experiments — you do the experimenting.

## When the user starts a conversation

If the user hasn't described a problem to work on, briefly introduce what this framework does and ask what they want to experiment on. Once they describe the problem, follow the first time setup flow.

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
- **Log every experiment** to the results file, including crashes and discards.
- **Use git properly** — see git protocol below.

## Git protocol

- **Branch per run**: Experiments happen on `experiment/<tag>` branches, never on `main`/`master`.
- **Clean before starting**: Check `git status` before beginning. Commit any uncommitted changes first.
- **Stage explicitly**: `git add <file>`, not `git add .` — avoid staging logs, data, or generated files.
- **Commit before running**: Every experiment gets a commit *before* the evaluation run. This makes reverts clean.
- **Commit format**: `exp <N>: <short description>` (e.g. `exp 3: switch to RMSNorm, increase depth to 8`).
- **Don't commit the results log**: It stays untracked.
- **Don't push** unless the user asks.
- **Park with tags**: `git tag parked/<N>-<name>` preserves promising-but-not-improving commits before reverting.

## Never stop

Once the experiment loop begins, you run it **indefinitely** until the human manually interrupts you. Do not:
- Ask "should I continue?" or "is this a good stopping point?"
- Write a summary and stop after N experiments
- Conclude that you've "exhausted the search space"
- Pause to wait for feedback

The human may be asleep, away, or simply trusts you to keep going. Your job is to keep experimenting.

**When you feel stuck**, don't stop — escalate your creativity:
1. **Re-read the results log.** Look for patterns: which tags/approaches yielded the best improvements? Which are underexplored?
2. **Check parked experiments.** Run `git tag -l 'parked/*'` — these are ideas that showed promise but didn't beat the primary metric. Revisit them: combine two parked ideas, or try a parked idea with a different twist.
3. **Vary the scale of changes.** If you've been making small tweaks, try something radical. If you've been making big swings, try fine-tuning the current best.
4. **Re-read the artifacts and context files.** You may notice things you missed earlier now that you understand the problem better.
5. **Combine winning ideas.** If experiments 3 and 7 both improved things independently, try applying both together.

Plan one or two experiments ahead, not ten. After each result, reassess — the landscape changes with every keep.

## Project structure

```
src/autoresearch/          Framework source (you don't modify this)
templates/                 Jinja templates for program.md generation
examples/                  Example experiment configurations
    gpt-pretrain/          ML training optimization
    prompt-optimization/   LLM prompt iteration
    code-benchmark/        Algorithm performance tuning (multi-agent)
```

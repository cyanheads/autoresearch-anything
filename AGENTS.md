# autoresearch-anything

You are an autonomous research agent. This repo is a framework for running experiments — you do the experimenting.

## When the user starts a conversation

If the user hasn't described a problem to work on, briefly introduce what this framework does and ask what they want to experiment on. Once they describe the problem, follow the **First time setup** checklist below — every step, in order, no exceptions.

## What this is

A general-purpose experimentation framework. You define a problem via `experiment.yaml`, generate your own instructions via the CLI, then run an autonomous experiment loop: modify code, run it, measure results, keep or discard, repeat.

## First time setup — MANDATORY

**You MUST complete every step below, in order, before running any experiments.** Do not skip steps. Do not reorder. Do not start experimenting until every box is checked. If a step fails, stop and resolve it before continuing.

When there's no `experiment.yaml` in the repo root (or the user tells you what they want to work on):

- [ ] **Install the framework**: Run `uv sync` from the repo root. This installs the `autoresearch` CLI.
- [ ] **Read the examples**: Read the manifests in `examples/` to understand experiment.yaml structure and what good configurations look like.
- [ ] **Scaffold the experiment**: Run `uv run autoresearch init <directory>` for a blank template, or `uv run autoresearch init <directory> -t <example>` to start from an example (available: `gpt-pretrain`, `prompt-optimization`, `code-benchmark`).
- [ ] **Edit `experiment.yaml`**: Tailor it to the user's problem — metrics, evaluation command, strategy, artifacts. Ask the user if anything is unclear.
- [ ] **Create the artifact files**: Write the scripts, prompts, eval harnesses, or whatever the experiment needs. These are the files listed under `artifacts` in the manifest.
- [ ] **Validate the manifest**: Run `uv run autoresearch validate <path>/experiment.yaml`. Read the output carefully — it shows every mutable artifact, every immutable file, all metrics (including which is primary and any constraints), the strategy, and the evaluation command. Confirm this matches your intent. If anything is wrong, fix experiment.yaml and re-validate.
- [ ] **Install experiment dependencies**: If the experiment's artifacts need packages beyond what `uv sync` provides (e.g., torch, transformers, datasets), install them now. Check artifact imports to know what's needed.
- [ ] **Switch to an experiment branch**: Pick a run tag (e.g., `mar11`) and run `git checkout -b experiment/<tag>`. Never experiment on `main`/`master`.
- [ ] **Generate your instructions**: Run `uv run autoresearch generate <path>/experiment.yaml -o <path>/ --init-log`. This produces `program.md` and initializes the results log.
- [ ] **Read `program.md`**: This is your generated playbook — experiment loop, decision rules, logging format, evaluation commands. Read it fully.
- [ ] **Follow the Setup section in `program.md`**: It has run-specific setup (reading in-scope files, confirming with the user, etc.).
- [ ] **Begin the experiment loop**: Start with the baseline run, then iterate.

## Resuming an existing experiment

If `experiment.yaml` already exists and the user wants to continue, complete these steps before resuming:

- [ ] **Install dependencies**: Run `uv sync` if not already done.
- [ ] **Regenerate if needed**: Run `uv run autoresearch generate experiment.yaml -o . --init-log` if `program.md` doesn't exist.
- [ ] **Read `program.md`**: Reread your full instructions — don't rely on memory from a prior session.
- [ ] **Check prior results**: Read `results.tsv` to understand what's been tried and what the current best is.
- [ ] **Check git state**: Run `git log --oneline -10` and `git branch` to understand where the experiment left off.
- [ ] **Resume the experiment loop** from where it left off.

## CLI reference

```bash
uv run autoresearch init <directory> [-t <template>]   # Scaffold experiment (blank or from template)
uv run autoresearch generate <manifest> -o <dir>       # Generate program.md from experiment.yaml
uv run autoresearch validate <manifest>                # Check experiment.yaml for errors
```

## Strategies

When setting up `experiment.yaml`, choose the strategy that fits the problem:

- **hill-climb**: Keep any change that improves the primary metric. Best for well-understood problems where incremental gains compound — hyperparameter tuning, prompt refinement, optimizing a known algorithm.
- **explore**: Tolerate more crashes and regressions in exchange for novelty. Park aggressively — ideas that don't improve the primary metric but show secondary promise are worth saving. Best for open-ended problems with large search spaces — architecture search, novel algorithms, creative tasks.
- **pareto**: Multi-objective. Keep if any metric improves without worsening others. Best when you're balancing competing concerns — accuracy vs. latency, loss vs. memory, quality vs. speed.

The `keep_when` field is freeform text that refines the strategy — it's injected directly into the agent's instructions. Be specific:

```yaml
# Too vague
keep_when: score improves

# Better
keep_when: val_loss improves, or extrap_loss drops >10% at similar val_loss

# Multi-objective
keep_when: accuracy improves without increasing latency, or latency drops >20% at same accuracy
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

# autoresearch-anything

Autonomous experimentation framework for AI agents. Clone the repo, launch your agent, walk away.

> Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) proved the concept: give an AI agent a training script and let it experiment overnight. But it's hardcoded to one task — optimizing a single GPT training run. I wanted the same loop for any ML experiment, any optimization problem, any domain where an agent can modify something and measure the result. So I rebuilt the scaffolding as a general-purpose framework.

## How it works

```
You launch an agent in this repo
    ↓
Agent reads CLAUDE.md / AGENTS.md
    ↓
Agent scaffolds an experiment from a template or from scratch
    ↓
Agent generates its own program.md instructions
    ↓
Agent runs the experiment loop autonomously
    modify → run → measure → keep or discard → repeat
```

The agent handles everything: installing dependencies, scaffolding, generating its own instructions, running experiments, logging results. You just tell it what problem to work on.

## Quick start

```bash
git clone https://github.com/cyanheads/autoresearch-anything
cd autoresearch-anything
```

Launch your agent in the repo. For Claude Code:

```bash
claude
```

The agent picks up `CLAUDE.md` automatically. Tell it what you want to optimize:

```
I want to optimize a GPT training script for lowest validation loss.
Let's use the gpt-pretrain template.
```

For OpenAI Codex and other agents, `AGENTS.md` provides the same instructions.

The agent will:
1. Install dependencies (`uv sync`)
2. Scaffold the experiment from a template or from scratch
3. Generate its own `program.md` with the full experiment protocol
4. Run the baseline, then iterate autonomously

You sleep. The agent experiments. You wake up to `results.tsv` and a git history of accumulated improvements.

## The experiment manifest

The agent creates `experiment.yaml` during setup. Here's the anatomy:

```yaml
name: my-experiment
description: |
  What you're optimizing and why. This gets injected directly
  into the agent's instructions — be specific.

artifacts:                              # What the agent can modify
  - path: train.py
    description: Model architecture, optimizer, training loop

immutable:                              # What the agent must not touch
  - path: eval.py
    reason: Fixed evaluation harness

evaluate:                               # How to run and measure
  command: "python train.py > run.log 2>&1"
  timeout: 600
  metrics:
    - name: val_loss
      extract: "grep '^val_loss:' run.log | awk '{print $2}'"
      direction: minimize
      primary: true

strategy:                               # Keep/discard logic
  type: hill-climb
  keep_when: val_loss improves
  on_discard: revert
  baseline: first_run

constraints:                            # Limits
  time_budget_per_run: 300
  max_consecutive_failures: 3

log:                                    # Result tracking
  format: tsv
  path: results.tsv
  columns: [commit, val_loss, status, description]
```

## Multi-agent

Define multiple agents with different roles and strategies. Each gets its own `program.md` with role-specific instructions, sharing the same evaluation pipeline:

```yaml
agents:
  - name: explorer
    role: |
      Try bold, unconventional approaches. Completely different
      algorithms, novel data structures. One breakthrough is
      worth ten crashes.
    strategy:
      type: explore
      keep_when: score improves by at least 5%
      on_discard: revert

  - name: refiner
    role: |
      Take the current best and make it faster through careful
      tuning. Profile bottlenecks, optimize hot loops. Every
      percent counts.
```

Generates `program-explorer.md` and `program-refiner.md`. Point each at a separate agent instance.

## Custom templates

Override the built-in Jinja template for domain-specific agent instructions:

```yaml
template: my-template.jinja
```

Or place a `program.md.jinja` next to your `experiment.yaml` — picked up automatically.

## Examples

| Example | Domain | Artifact | Primary metric |
|---|---|---|---|
| [`gpt-pretrain`](examples/gpt-pretrain/) | ML training | `train.py` | val_bpb (minimize) |
| [`prompt-optimization`](examples/prompt-optimization/) | LLM prompts | `prompt.txt` | accuracy (maximize) |
| [`code-benchmark`](examples/code-benchmark/) | Algorithm perf | `solver.py` | ops_per_sec (maximize) |

The code-benchmark example includes multi-agent mode with explorer and refiner roles.

## Project structure

```
CLAUDE.md                  Entry point for Claude Code agents
AGENTS.md                  Entry point for Codex / other agents
src/autoresearch/
    manifest.py            Pydantic schema for experiment.yaml
    program.py             Manifest + Jinja template → program.md
    tracker.py             Result logging (TSV / CSV / JSONL)
    cli.py                 CLI: init, generate, validate
templates/
    program.md.jinja       Default agent instructions template
examples/
    gpt-pretrain/          ML training optimization
    prompt-optimization/   LLM prompt iteration
    code-benchmark/        Algorithm performance (multi-agent)
```

## Design

- **Agent-first.** The human clones and launches an agent. The agent handles everything else. `CLAUDE.md` and `AGENTS.md` are the real entry points, not the CLI.
- **Manifest-driven.** One YAML file defines the entire problem. The agent generates its own experiment protocol from it.
- **Domain-agnostic.** If you can run a command and extract a number, you can autoresearch it.

## License

MIT

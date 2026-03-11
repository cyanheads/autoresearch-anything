"""
src/autoresearch/cli.py

CLI entry point for autoresearch-anything.

Commands:
  init     — scaffold a new experiment (from template or blank)
  generate — read experiment.yaml and produce program.md
  validate — check experiment.yaml for errors
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from autoresearch.manifest import load_manifest
from autoresearch.program import generate_all
from autoresearch.tracker import init_log

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"

BLANK_MANIFEST = """\
# experiment.yaml — define your autonomous experiment
#
# Fill in the sections below, then run:
#   autoresearch generate experiment.yaml

name: my-experiment

description: |
  What you're optimizing and why. Be specific — this gets injected
  directly into the agent's instructions.

# Files the agent is allowed to modify
artifacts:
  - path: TODO.py
    description: Describe what this file is and what's fair game

# Files the agent must NOT touch
immutable: []
  # - path: eval.py
  #   reason: Fixed evaluation harness

# How to run an experiment and measure results
evaluate:
  command: "python TODO.py > run.log 2>&1"
  timeout: 600

  metrics:
    - name: score
      extract: "grep '^score:' run.log | awk '{print $2}'"
      direction: minimize   # or maximize
      primary: true

  crash_signals:
    - Non-zero exit code
    - Primary metric missing from output

# Decision rule
strategy:
  type: hill-climb
  keep_when: score improves
  on_discard: revert       # or branch
  baseline: first_run      # or skip

# Limits
constraints:
  time_budget_per_run: 300
  max_consecutive_failures: 3

# Reference material for the agent
context: []
  # - path: README.md
  #   description: Project overview

# Result logging
log:
  format: tsv
  path: results.tsv
  columns: [experiment, parent, commit, score, status, description, tags]

# Optional: custom Jinja template for agent instructions
# template: program.md.jinja

# Optional: multi-agent mode
# agents:
#   - name: explorer
#     role: |
#       Try bold, unconventional changes. Prioritize novelty.
#     strategy:
#       type: explore
#       keep_when: score improves by at least 1%
#       on_discard: revert
#
#   - name: refiner
#     role: |
#       Make incremental improvements. Focus on tuning.
#     strategy:
#       type: hill-climb
#       keep_when: any improvement
#       on_discard: revert
"""


def cmd_init(args: argparse.Namespace) -> None:
    """Scaffold a new experiment directory."""
    target = Path(args.directory)
    if target.exists() and any(target.iterdir()):
        print(
            f"Error: {target} is not empty.",
            file=sys.stderr,
        )
        sys.exit(1)

    template = args.template

    if template == "blank":
        target.mkdir(parents=True, exist_ok=True)
        (target / "experiment.yaml").write_text(BLANK_MANIFEST)
        print(f"Initialized blank experiment in {target}/")
        print("Next steps:")
        print(f"  1. Edit {target}/experiment.yaml")
        print("  2. Create your artifact file(s)")
        print(f"  3. Run: autoresearch generate " f"{target}/experiment.yaml")
        return

    template_dir = EXAMPLES_DIR / template
    if not template_dir.exists():
        available = sorted(d.name for d in EXAMPLES_DIR.iterdir() if d.is_dir())
        print(
            f"Error: template '{template}' not found.",
            file=sys.stderr,
        )
        if available:
            print(
                f"Available: {', '.join(available)}, blank",
                file=sys.stderr,
            )
        sys.exit(1)

    target.mkdir(parents=True, exist_ok=True)
    for item in template_dir.iterdir():
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print(f"Initialized experiment in {target}/ " f"from '{template}' template.")
    print("Next steps:")
    print(f"  1. Edit {target}/experiment.yaml")
    print(f"  2. Run: autoresearch generate " f"{target}/experiment.yaml")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate program.md from an experiment manifest."""
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(
            f"Error: {manifest_path} not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent
    output_dir = Path(args.output_dir)

    programs = generate_all(manifest, manifest_dir)

    for filename, content in programs.items():
        out_path = output_dir / filename
        out_path.write_text(content)
        print(f"Generated {out_path}")

    if args.init_log:
        log_path = init_log(manifest.log, directory=manifest_dir)
        print(f"Initialized {log_path}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate experiment.yaml without generating anything."""
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(
            f"Error: {manifest_path} not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        manifest = load_manifest(manifest_path)
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Valid: {manifest.name}")

    # Mutable artifacts
    print("\n  Mutable artifacts:")
    for a in manifest.artifacts:
        print(f"    - {a.path}: {a.description.strip()}")

    # Immutable files
    if manifest.immutable:
        print("\n  Immutable (DO NOT modify):")
        for i in manifest.immutable:
            print(f"    - {i.path}: {i.reason}")
    else:
        print("\n  Immutable: (none)")

    # All metrics
    print("\n  Metrics:")
    for m in manifest.evaluate.metrics:
        label = f"{m.name} ({m.direction.value})"
        if m.primary:
            label += " [PRIMARY]"
        if m.constraint:
            bounds = []
            if m.constraint.min is not None:
                bounds.append(f"min={m.constraint.min}")
            if m.constraint.max is not None:
                bounds.append(f"max={m.constraint.max}")
            label += f" constraint: {', '.join(bounds)}"
        print(f"    - {label}")

    # Strategy
    print(f"\n  Strategy: {manifest.strategy.type}")
    print(f"  Keep when: {manifest.strategy.keep_when}")
    print(f"  On discard: {manifest.strategy.on_discard.value}")

    # Constraints
    c = manifest.constraints
    if c.time_budget_per_run:
        print(f"  Time budget: {c.time_budget_per_run}s/run")
    print(f"  Max consecutive failures: {c.max_consecutive_failures}")

    # Evaluation
    print(f"\n  Evaluate: {manifest.evaluate.command}")
    print(f"  Timeout: {manifest.evaluate.timeout}s")

    # Multi-agent
    if manifest.is_multi_agent:
        agents = manifest.resolved_agents()
        print(f"\n  Agents: {', '.join(a.name for a in agents)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autoresearch",
        description="Autonomous experimentation framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Scaffold a new experiment")
    p_init.add_argument(
        "directory",
        help="Target directory for the new experiment",
    )
    p_init.add_argument(
        "-t",
        "--template",
        default="blank",
        help=(
            "Template to use: 'blank' for an empty scaffold, "
            "or an example name (default: blank)"
        ),
    )

    # generate
    p_gen = sub.add_parser(
        "generate",
        help="Generate program.md from experiment.yaml",
    )
    p_gen.add_argument("manifest", help="Path to experiment.yaml")
    p_gen.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory for generated program.md files (default: .)",
    )
    p_gen.add_argument(
        "--init-log",
        action="store_true",
        help="Also initialize the results log file",
    )

    # validate
    p_val = sub.add_parser("validate", help="Validate experiment.yaml")
    p_val.add_argument("manifest", help="Path to experiment.yaml")

    args = parser.parse_args()
    commands = {
        "init": cmd_init,
        "generate": cmd_generate,
        "validate": cmd_validate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

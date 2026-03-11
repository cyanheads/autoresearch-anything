"""
src/autoresearch/program.py

Generates program.md (agent instructions) from an experiment manifest
and a Jinja2 template. Supports single-agent and multi-agent modes,
and custom per-experiment templates.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from autoresearch.manifest import (
    ExperimentManifest,
    ResolvedAgent,
)

TEMPLATES_DIR = (
    Path(__file__).resolve().parent.parent.parent / "templates"
)


def _make_env(*search_paths: Path) -> Environment:
    """Create a Jinja environment searching the given directories."""
    dirs = [str(p) for p in search_paths if p.exists()]
    return Environment(
        loader=FileSystemLoader(dirs),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _resolve_template(
    manifest: ExperimentManifest,
    manifest_dir: Path | None,
) -> tuple[str, list[Path]]:
    """
    Determine which template to use and where to look for it.

    Priority:
      1. manifest.template (explicit path relative to manifest)
      2. program.md.jinja next to experiment.yaml (convention)
      3. Built-in default template
    """
    search_paths: list[Path] = []

    if manifest_dir:
        search_paths.append(manifest_dir)
    search_paths.append(TEMPLATES_DIR)

    if manifest.template:
        return manifest.template, search_paths

    if manifest_dir and (manifest_dir / "program.md.jinja").exists():
        return "program.md.jinja", search_paths

    return "program.md.jinja", search_paths


def generate_program(
    manifest: ExperimentManifest,
    manifest_dir: Path | None = None,
    agent: ResolvedAgent | None = None,
) -> str:
    """
    Render program.md agent instructions from a manifest.

    If `agent` is provided, renders with agent-specific overrides
    (role, strategy, artifacts, constraints).
    """
    template_name, search_paths = _resolve_template(
        manifest, manifest_dir
    )
    env = _make_env(*search_paths)
    template = env.get_template(template_name)

    return template.render(
        manifest=manifest,
        primary=manifest.primary_metric,
        agent=agent,
    )


def generate_all(
    manifest: ExperimentManifest,
    manifest_dir: Path | None = None,
) -> dict[str, str]:
    """
    Generate all program.md files for an experiment.

    Single-agent: returns {"program.md": content}
    Multi-agent:  returns {"program-<name>.md": content, ...}
    """
    if not manifest.is_multi_agent:
        return {
            "program.md": generate_program(
                manifest, manifest_dir
            ),
        }

    results: dict[str, str] = {}
    for agent in manifest.resolved_agents():
        filename = f"program-{agent.name}.md"
        content = generate_program(
            manifest, manifest_dir, agent=agent
        )
        results[filename] = content
    return results

"""
src/autoresearch/manifest.py

Pydantic schema for experiment.yaml — the universal problem definition.
Parses, validates, and provides typed access to experiment configuration.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class Direction(str, Enum):
    minimize = "minimize"
    maximize = "maximize"


class DiscardAction(str, Enum):
    revert = "revert"
    branch = "branch"


class LogFormat(str, Enum):
    tsv = "tsv"
    csv = "csv"
    jsonl = "jsonl"


# --- Nested models ---


class Artifact(BaseModel):
    """A file the agent is allowed to modify."""

    path: str
    description: str


class Immutable(BaseModel):
    """A file or pattern the agent must not touch."""

    path: str
    reason: str


class MetricConstraint(BaseModel):
    """Optional bound on a metric value."""

    min: Optional[float] = None
    max: Optional[float] = None


class Metric(BaseModel):
    """A named value extracted from experiment output."""

    name: str
    extract: str = Field(description="Shell command that prints the metric value")
    direction: Direction
    primary: bool = False
    constraint: Optional[MetricConstraint] = None


class Evaluate(BaseModel):
    """How to run an experiment and measure its results."""

    command: str
    timeout: int = Field(default=600, description="Hard kill timeout in seconds")
    metrics: list[Metric]
    crash_signals: list[str] = Field(
        default_factory=lambda: ["non-zero exit code"],
        description="Human-readable descriptions of what constitutes a crash",
    )

    @model_validator(mode="after")
    def exactly_one_primary(self) -> "Evaluate":
        primaries = [m for m in self.metrics if m.primary]
        if len(primaries) != 1:
            raise ValueError(
                f"Exactly one metric must be primary, got {len(primaries)}"
            )
        return self

    @property
    def primary_metric(self) -> Metric:
        return next(m for m in self.metrics if m.primary)


class Strategy(BaseModel):
    """Decision rule: when to keep or discard a change."""

    type: str = Field(default="hill-climb", description="Strategy name")
    keep_when: str = Field(
        default="primary metric improves",
        description="Human-readable keep condition",
    )
    on_discard: DiscardAction = DiscardAction.revert
    baseline: str = Field(
        default="first_run",
        description="How to establish baseline: 'first_run' or 'skip'",
    )


class Constraints(BaseModel):
    """Resource and failure limits."""

    time_budget_per_run: Optional[int] = None
    max_consecutive_failures: int = 3


class ContextSource(BaseModel):
    """Reference material for the agent."""

    path: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None

    @model_validator(mode="after")
    def at_least_one(self) -> "ContextSource":
        if not self.path and not self.url:
            raise ValueError("ContextSource needs at least one of path or url")
        return self


class LogConfig(BaseModel):
    """Experiment result logging configuration."""

    format: LogFormat = LogFormat.tsv
    path: str = "results.tsv"
    columns: list[str] = Field(
        default_factory=lambda: [
            "experiment",
            "parent",
            "commit",
            "status",
            "description",
            "tags",
        ]
    )


class AgentConfig(BaseModel):
    """
    An agent role within a multi-agent experiment.
    Each agent gets its own generated program.md with role-specific instructions.
    Fields left unset inherit from the top-level manifest.
    """

    name: str = Field(
        description="Agent identifier, used in filenames and branch names",
    )
    role: str = Field(description="What this agent does — injected into its program.md")
    artifacts: Optional[list[Artifact]] = None
    strategy: Optional[Strategy] = None
    constraints: Optional[Constraints] = None
    context: Optional[list[ContextSource]] = None

    def resolve(self, manifest: "ExperimentManifest") -> "ResolvedAgent":
        """Merge agent-level overrides with top-level manifest defaults."""
        return ResolvedAgent(
            name=self.name,
            role=self.role,
            artifacts=self.artifacts or manifest.artifacts,
            strategy=self.strategy or manifest.strategy,
            constraints=self.constraints or manifest.constraints,
            context=self.context or manifest.context,
        )


class ResolvedAgent(BaseModel):
    """An agent with all fields resolved (no Nones — inherited from manifest)."""

    name: str
    role: str
    artifacts: list[Artifact]
    strategy: Strategy
    constraints: Constraints
    context: list[ContextSource]


# --- Top-level manifest ---


class ExperimentManifest(BaseModel):
    """
    The complete experiment definition.
    Parsed from experiment.yaml, drives program.md generation and experiment tracking.
    """

    name: str
    description: str
    artifacts: list[Artifact]
    immutable: list[Immutable] = Field(default_factory=list)
    evaluate: Evaluate
    strategy: Strategy = Field(default_factory=Strategy)
    constraints: Constraints = Field(default_factory=Constraints)
    context: list[ContextSource] = Field(default_factory=list)
    log: LogConfig = Field(default_factory=LogConfig)
    template: Optional[str] = Field(
        default=None,
        description="Path to a custom Jinja template (relative to experiment.yaml)",
    )
    agents: Optional[list[AgentConfig]] = Field(
        default=None,
        description=(
            "Multi-agent configuration. " "If set, generates one program.md per agent."
        ),
    )

    @property
    def primary_metric(self) -> Metric:
        return self.evaluate.primary_metric

    @property
    def is_multi_agent(self) -> bool:
        return self.agents is not None and len(self.agents) > 0

    def resolved_agents(self) -> list[ResolvedAgent]:
        """Resolve all agents, merging with top-level defaults."""
        if not self.agents:
            return []
        return [a.resolve(self) for a in self.agents]

    @model_validator(mode="after")
    def unique_agent_names(self) -> "ExperimentManifest":
        if self.agents:
            names = [a.name for a in self.agents]
            dupes = [n for n in names if names.count(n) > 1]
            if dupes:
                raise ValueError(f"Duplicate agent names: {set(dupes)}")
        return self


def load_manifest(path: str | Path) -> ExperimentManifest:
    """Load and validate an experiment manifest from a YAML file."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    return ExperimentManifest.model_validate(raw)

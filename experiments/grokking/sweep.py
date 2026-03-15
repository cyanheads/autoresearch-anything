"""
experiments/grokking/sweep.py — Run a structured sweep of novel interventions.

Tests genuinely novel approaches (not hyperparameter grid search):
1. Grokfast (slow gradient amplification) — Lim et al. 2024
2. SVD weight parameterization — "Decomposed Learning"
3. Spherical residual stream — architectural topology paper (2026)

Plus combinations and across tasks (add, mul).
"""

import subprocess
import sys
import re
import time
from dataclasses import dataclass


@dataclass
class Config:
    name: str
    args: list[str]
    tags: str


def run_experiment(config: Config) -> dict:
    """Run a single experiment and extract metrics."""
    cmd = [
        sys.executable, "train.py",
        *config.args,
    ]
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}", flush=True)

    start = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        env={"CUDA_VISIBLE_DEVICES": "1", "PATH": "/usr/bin:/bin:/usr/local/bin",
             "HOME": "/home/casey", "VIRTUAL_ENV": "",
             **{k: v for k, v in __import__("os").environ.items()}}
    )
    wall = time.time() - start

    # Print stderr (training logs)
    if result.stderr:
        lines = result.stderr.strip().split("\n")
        # Print first, last few, and any error lines
        for line in lines[:3]:
            print(f"  {line}", file=sys.stderr)
        if len(lines) > 6:
            print(f"  ... ({len(lines) - 6} more lines)", file=sys.stderr)
        for line in lines[-3:]:
            print(f"  {line}", file=sys.stderr)

    # Parse metrics from stdout
    metrics = {}
    for line in result.stdout.strip().split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            val = val.strip()
            try:
                metrics[key.strip()] = float(val)
            except ValueError:
                metrics[key.strip()] = val

    metrics["exit_code"] = result.returncode
    return metrics


# ── Experiment configurations ────────────────────────────────────────

BASELINE = [
    "--d-model", "256", "--n-layers", "3", "--d-ff", "1024", "--n-heads", "4",
    "--lr", "3e-3", "--weight-decay", "2.0", "--warmup-steps", "50",
    "--total-steps", "5000", "--n-evals", "1000", "--batch-size", "512",
]

CONFIGS = [
    # ── Baselines ──
    Config("baseline_add", [*BASELINE, "--operation", "add"], "baseline,add"),
    Config("baseline_mul", [*BASELINE, "--operation", "mul"], "baseline,mul"),

    # ── Grokfast (the big one) ──
    Config("grokfast_a2_add", [*BASELINE, "--grokfast", "--grokfast-alpha", "2.0", "--operation", "add"],
           "grokfast,add"),
    Config("grokfast_a5_add", [*BASELINE, "--grokfast", "--grokfast-alpha", "5.0", "--operation", "add"],
           "grokfast,add,alpha5"),
    Config("grokfast_a10_add", [*BASELINE, "--grokfast", "--grokfast-alpha", "10.0", "--operation", "add"],
           "grokfast,add,alpha10"),
    Config("grokfast_a2_mul", [*BASELINE, "--grokfast", "--grokfast-alpha", "2.0", "--operation", "mul"],
           "grokfast,mul"),

    # ── Grokfast with LOW weight decay (the paper's setting) ──
    Config("grokfast_lowwd_add",
           [*BASELINE, "--grokfast", "--grokfast-alpha", "2.0", "--weight-decay", "0.1", "--operation", "add"],
           "grokfast,low-wd,add"),
    Config("grokfast_lowwd_mul",
           [*BASELINE, "--grokfast", "--grokfast-alpha", "5.0", "--weight-decay", "0.1", "--operation", "mul"],
           "grokfast,low-wd,mul"),

    # ── Spherical architecture ──
    Config("spherical_add", [*BASELINE, "--arch", "spherical", "--operation", "add"],
           "spherical,add"),
    Config("spherical_mul", [*BASELINE, "--arch", "spherical", "--operation", "mul"],
           "spherical,mul"),
    Config("spherical_nowd_add",
           [*BASELINE, "--arch", "spherical", "--weight-decay", "0.0", "--operation", "add"],
           "spherical,no-wd,add"),

    # ── SVD parameterization ──
    Config("svd_add", [*BASELINE, "--arch", "svd", "--operation", "add"],
           "svd,add"),
    Config("svd_mul", [*BASELINE, "--arch", "svd", "--operation", "mul"],
           "svd,mul"),
    Config("svd_lowwd_add",
           [*BASELINE, "--arch", "svd", "--weight-decay", "0.1", "--operation", "add"],
           "svd,low-wd,add"),
    Config("svd_rank32_add",
           [*BASELINE, "--arch", "svd", "--svd-rank", "32", "--operation", "add"],
           "svd,rank32,add"),

    # ── Combinations ──
    Config("grokfast_spherical_add",
           [*BASELINE, "--grokfast", "--arch", "spherical", "--operation", "add"],
           "grokfast,spherical,add"),
    Config("grokfast_svd_add",
           [*BASELINE, "--grokfast", "--arch", "svd", "--operation", "add"],
           "grokfast,svd,add"),

    # ── Harder task: larger prime ──
    Config("baseline_p509_add",
           [*BASELINE, "--prime", "509", "--operation", "add", "--total-steps", "10000"],
           "baseline,p509,add"),
    Config("grokfast_p509_add",
           [*BASELINE, "--prime", "509", "--grokfast", "--operation", "add", "--total-steps", "10000"],
           "grokfast,p509,add"),
    Config("spherical_p509_add",
           [*BASELINE, "--prime", "509", "--arch", "spherical", "--operation", "add", "--total-steps", "10000"],
           "spherical,p509,add"),
]


def main():
    results = []
    for i, config in enumerate(CONFIGS):
        try:
            metrics = run_experiment(config)
        except subprocess.TimeoutExpired:
            metrics = {"exit_code": -1, "grok_step": "timeout"}
        except Exception as e:
            metrics = {"exit_code": -1, "grok_step": f"error: {e}"}

        grok_step = metrics.get("grok_step", "N/A")
        grok_delay = metrics.get("grok_delay", "N/A")
        val_acc = metrics.get("final_val_acc", "N/A")
        wall = metrics.get("wall_time_s", "N/A")
        vram = metrics.get("peak_vram_mb", "N/A")
        grokked = metrics.get("grokked", "N/A")

        status = "keep" if grokked == 1.0 or grokked == "True" else "discard"
        results.append((config, metrics, status))

        print(f"\n>>> {config.name}: grok_step={grok_step}  delay={grok_delay}  "
              f"val_acc={val_acc}  wall={wall}s  vram={vram}MB  [{status}]")

    # ── Summary table ──
    print("\n" + "=" * 100)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'name':<30} {'grok_step':>10} {'delay':>8} {'val_acc':>10} {'wall_s':>8} {'vram_mb':>8} {'tags'}")
    print("-" * 100)
    for config, metrics, status in results:
        gs = metrics.get("grok_step", "N/A")
        gd = metrics.get("grok_delay", "N/A")
        va = metrics.get("final_val_acc", "N/A")
        ws = metrics.get("wall_time_s", "N/A")
        vm = metrics.get("peak_vram_mb", "N/A")
        if isinstance(va, float):
            va = f"{va:.4f}"
        if isinstance(ws, float):
            ws = f"{ws:.1f}"
        print(f"{config.name:<30} {str(gs):>10} {str(gd):>8} {str(va):>10} {str(ws):>8} {str(vm):>8} {config.tags}")


if __name__ == "__main__":
    main()

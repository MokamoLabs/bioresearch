"""
Modal app for dispatching experiments to cloud GPUs.

Provides:
- run_bio_experiment: Run a single training experiment on a GPU
- run_multi_seed: Run multiple seeds in parallel across GPUs via .map()
- Persistent volumes for data and results

Architecture:
    Colab/local orchestrator calls Modal SDK ->
    Modal dispatches 5 H100s (one per seed) ->
    Each runs train.py with a different SEED ->
    Returns metric vectors ->
    Orchestrator evaluates and keeps/reverts
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import modal

app = modal.App("bioresearch")

# Persistent volumes for data and results
data_volume = modal.Volume.from_name("bioresearch-data", create_if_missing=True)
results_volume = modal.Volume.from_name("bioresearch-results", create_if_missing=True)

# Base image with scientific Python stack
bio_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    )
)

# GPU image with PyTorch + bio packages
gpu_image = (
    bio_image
    .pip_install(
        "torch>=2.1.0",
        "scanpy>=1.10.0",
        "anndata>=0.10.0",
    )
)


@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=1200,
    volumes={"/data": data_volume, "/results": results_volume},
)
def run_bio_experiment(
    train_code: str,
    prepare_code: str,
    seed: int,
    time_budget: int = 600,
    domain: str = "perturbation",
) -> dict:
    """
    Run a single training experiment on a Modal GPU.

    The function receives the full source code of train.py and prepare.py,
    writes them to a temporary directory, and executes train.py. This means
    the agent's modifications are transported as code, not as file diffs.
    """
    work_dir = Path(tempfile.mkdtemp())

    # Create the domain structure that train.py expects
    domain_dir = work_dir / "domains" / domain
    domain_dir.mkdir(parents=True)

    (domain_dir / "train.py").write_text(train_code)
    (domain_dir / "prepare.py").write_text(prepare_code)
    (domain_dir / "__init__.py").write_text("")
    (work_dir / "domains" / "__init__.py").write_text("")

    env = os.environ.copy()
    env["SEED"] = str(seed)
    env["TIME_BUDGET"] = str(time_budget)
    env["DATA_DIR"] = f"/data/{domain}"
    env["RESULTS_DIR"] = f"/results/{domain}"
    env["PYTHONPATH"] = str(work_dir)

    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(domain_dir / "train.py")],
            capture_output=True,
            text=True,
            timeout=time_budget + 120,
            cwd=str(work_dir),
            env=env,
        )

        train_seconds = time.time() - t0

        if result.returncode != 0:
            return {
                "seed": seed,
                "success": False,
                "error": result.stderr[-2000:],
                "train_seconds": train_seconds,
                "metrics": {},
            }

        metrics = _parse_stdout_metrics(result.stdout)

        return {
            "seed": seed,
            "success": True,
            "metrics": metrics,
            "train_seconds": train_seconds,
            "stdout_tail": result.stdout[-2000:],
        }

    except subprocess.TimeoutExpired:
        return {
            "seed": seed,
            "success": False,
            "error": "Timeout",
            "train_seconds": time.time() - t0,
            "metrics": {},
        }
    except Exception as e:
        return {
            "seed": seed,
            "success": False,
            "error": str(e),
            "train_seconds": time.time() - t0,
            "metrics": {},
        }


def _parse_stdout_metrics(stdout: str) -> dict[str, float]:
    """Parse metrics from training script stdout."""
    metrics = {}

    # Try JSON on last line
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    return {k: float(v) for k, v in parsed.items() if isinstance(v, (int, float))}
            except (json.JSONDecodeError, ValueError):
                pass

    # Key-value after --- separator
    in_results = False
    for line in stdout.splitlines():
        line = line.strip()
        if line == "---":
            in_results = True
            continue
        if in_results and ":" in line:
            key, _, val = line.partition(":")
            try:
                metrics[key.strip()] = float(val.strip())
            except ValueError:
                pass

    return metrics

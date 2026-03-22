"""
Colab setup helper: GPU check, Modal authentication, environment setup.
"""

from __future__ import annotations

import os
import subprocess
import sys


def check_gpu() -> dict:
    """Check available GPU and return info dict."""
    info = {"has_gpu": False, "gpu_name": "none", "vram_gb": 0.0}

    try:
        import torch
        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"GPU: {info['gpu_name']} ({info['vram_gb']:.1f} GB)")
        else:
            print("No GPU available. Will use Modal for GPU compute.")
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch")

    return info


def setup_modal(token: str | None = None):
    """Set up Modal authentication."""
    if token:
        os.environ["MODAL_TOKEN_ID"] = token

    try:
        import modal
        print(f"Modal version: {modal.__version__}")

        # Test connection
        try:
            app = modal.App.lookup("bioresearch", create_if_missing=True)
            print("Modal connection: OK")
            return True
        except Exception as e:
            print(f"Modal connection failed: {e}")
            print("Run `modal token new` to authenticate.")
            return False

    except ImportError:
        print("Modal not installed. Install with: pip install modal")
        return False


def setup_anthropic(api_key: str | None = None):
    """Set up Anthropic API key."""
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not set. Set it with:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        return False

    try:
        import anthropic
        client = anthropic.Anthropic()
        # Quick validation
        print("Anthropic API key: set")
        return True
    except ImportError:
        print("anthropic not installed. Install with: pip install anthropic")
        return False


def setup_vertex(project_id: str | None = None, region: str = "us-east5"):
    """Set up Vertex AI authentication using Google Application Default Credentials."""
    if project_id:
        os.environ["VERTEX_PROJECT_ID"] = project_id
    if region:
        os.environ["VERTEX_REGION"] = region

    # In Colab, authenticate the user to get ADC
    try:
        from google.colab import auth
        auth.authenticate_user()
        print("Google Cloud authentication: OK (Colab)")
    except ImportError:
        # Not in Colab — check if ADC is available
        try:
            import google.auth
            credentials, project = google.auth.default()
            if not project_id and project:
                os.environ["VERTEX_PROJECT_ID"] = project
            print(f"Google Cloud authentication: OK (ADC, project={project})")
        except Exception as e:
            print(f"Google Cloud authentication failed: {e}")
            print("Run `gcloud auth application-default login` to authenticate.")
            return False

    # Verify AnthropicVertex is importable
    try:
        from anthropic import AnthropicVertex  # noqa: F401
        project = os.environ.get("VERTEX_PROJECT_ID", project_id or "")
        if not project:
            print("WARNING: VERTEX_PROJECT_ID not set. Set it with:")
            print("  export VERTEX_PROJECT_ID=your-gcp-project-id")
            return False
        print(f"Vertex AI ready: project={project}, region={region}")
        return True
    except ImportError:
        print("anthropic[vertex] not installed. Install with:")
        print("  pip install 'anthropic[vertex]'")
        return False


def recommend_compute_mode(gpu_info: dict, modal_ok: bool) -> str:
    """Recommend a compute mode based on available resources."""
    has_gpu = gpu_info.get("has_gpu", False)
    gpu_name = gpu_info.get("gpu_name", "").lower()

    is_high_end = any(x in gpu_name for x in ["h100", "a100", "h200", "l40"])

    if has_gpu and modal_ok:
        if is_high_end:
            print(f"Recommended compute mode: 'hybrid' (use your {gpu_info['gpu_name']} + Modal)")
        else:
            print(f"Recommended compute mode: 'prescreen' (test locally, then dispatch to Modal)")
        return "hybrid" if is_high_end else "prescreen"
    elif has_gpu and not modal_ok:
        print(f"Recommended compute mode: 'local' (Modal unavailable, using {gpu_info['gpu_name']})")
        return "local"
    elif not has_gpu and modal_ok:
        print("Recommended compute mode: 'modal' (no local GPU, using Modal)")
        return "modal"
    else:
        print("WARNING: No GPU and no Modal. Experiments will run on CPU (very slow).")
        return "local"


def colab_setup(
    anthropic_key: str | None = None,
    modal_token: str | None = None,
    backend: str = "anthropic",
    vertex_project_id: str | None = None,
    vertex_region: str = "global",
):
    """
    One-call setup for running BioResearch from Google Colab.

    Usage in Colab:
        from infra.colab import colab_setup
        colab_setup(anthropic_key="sk-...", modal_token="...")

        # Or with Vertex AI:
        colab_setup(backend="vertex", vertex_project_id="my-project")
    """
    print("BioResearch Colab Setup")
    print("=" * 40)

    gpu_info = check_gpu()
    print()

    if backend == "vertex":
        claude_ok = setup_vertex(vertex_project_id, vertex_region)
    else:
        claude_ok = setup_anthropic(anthropic_key)
    print()

    modal_ok = setup_modal(modal_token)
    print()

    recommended_compute = recommend_compute_mode(gpu_info, modal_ok)
    print()

    if claude_ok and (gpu_info["has_gpu"] or modal_ok):
        print("Setup complete! Ready to run experiments.")
        if not gpu_info["has_gpu"] and modal_ok:
            print("Note: Using Modal for GPU compute (no local GPU detected).")
    else:
        print("Setup incomplete. Check the messages above.")

    return {
        "gpu": gpu_info,
        "claude": claude_ok,
        "backend": backend,
        "modal": modal_ok,
        "recommended_compute": recommended_compute,
    }

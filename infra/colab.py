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


def colab_setup(anthropic_key: str | None = None, modal_token: str | None = None):
    """
    One-call setup for running BioResearch from Google Colab.

    Usage in Colab:
        from infra.colab import colab_setup
        colab_setup(anthropic_key="sk-...", modal_token="...")
    """
    print("BioResearch Colab Setup")
    print("=" * 40)

    gpu_info = check_gpu()
    print()

    anthropic_ok = setup_anthropic(anthropic_key)
    print()

    modal_ok = setup_modal(modal_token)
    print()

    if anthropic_ok and (gpu_info["has_gpu"] or modal_ok):
        print("Setup complete! Ready to run experiments.")
        if not gpu_info["has_gpu"] and modal_ok:
            print("Note: Using Modal for GPU compute (no local GPU detected).")
    else:
        print("Setup incomplete. Check the messages above.")

    return {
        "gpu": gpu_info,
        "anthropic": anthropic_ok,
        "modal": modal_ok,
    }

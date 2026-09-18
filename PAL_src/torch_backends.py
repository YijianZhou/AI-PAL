"""Shared PyTorch backend setup for training and inference."""

import os

import torch


def configure_torch_backends(cfg=None):
    """Avoid unsupported NNPACK initialization without filtering other warnings."""
    nnpack = getattr(torch.backends, "nnpack", None)
    set_flags = getattr(nnpack, "set_flags", None)
    if set_flags is None:
        return False
    set_flags(False)
    verbosity = os.environ.get(
        "AI_PAL_CONSOLE_VERBOSITY", getattr(cfg, "console_verbosity", "default")
    )
    if str(verbosity).strip().lower() == "debug":
        print("[debug] PyTorch NNPACK backend disabled", flush=True)
    return True

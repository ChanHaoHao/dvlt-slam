"""Per-submap reconstruction backbones.

``base.py`` states the contract; each sibling module implements it for one model.
``build_backbone`` is the only thing entry points should need:

    from slam.backbones import build_backbone

It lives here rather than in ``main.py`` so that every entry point loads a
backbone identically. ``main_realtime.py`` used to hardcode VGGT, which meant the
live path silently ignored ``--backbone`` and never ran the default DVLT model at
all.

Adding a backbone: implement ``SubmapBackbone`` in a new module, then add a branch
below and a choice to the ``--backbone`` argument in the entry points.
"""

from __future__ import annotations

import torch
from termcolor import colored

from slam.backbones.base import LoopVerification, Reconstruction, SubmapBackbone

BACKBONES = ("dvlt", "vggt")

__all__ = [
    "BACKBONES",
    "LoopVerification",
    "Reconstruction",
    "SubmapBackbone",
    "build_backbone",
]


def build_backbone(
    backbone: str = "dvlt",
    device: str = None,
    dvlt_checkpoint: str = "nvidia/dvlt",
    dvlt_k: int = None,
    lc_verify: str = "attn",
    lc_attn_step: int = None,
    verbose: bool = True,
) -> SubmapBackbone:
    """Load the submap reconstructor named by ``backbone`` ("dvlt" or "vggt")."""
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Imported lazily, unlike the old slam/backbone.py which imported VGGT at
    # module scope: each branch pulls in a different heavy tree, so a run on one
    # backbone should not pay to import the other.
    if backbone == "dvlt":
        from slam.backbones.dvlt_backbone import DVLTBackbone

        model = DVLTBackbone(
            checkpoint=dvlt_checkpoint,
            inference_steps=dvlt_k,
            device=device,
            lc_verify=lc_verify,
            lc_attn_step=lc_attn_step,
        )
        if verbose:
            print(f"Using Deja View (DVLT) backbone: K={model.k}, lc_verify={lc_verify}")
            if lc_verify == "bypass":
                print(colored(
                    "lc_verify=bypass: every retrieved loop closure is accepted without "
                    "attention verification. Not a like-for-like comparison with stock VGGT-SLAM.",
                    "yellow",
                ))
    else:
        from slam.backbones.vggt_backbone import VGGTBackbone

        model = VGGTBackbone(device=device, verbose=verbose)

    return model

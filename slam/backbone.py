"""Construction of the per-submap reconstruction backbone.

Lifted out of ``main.py`` so that every entry point loads a backbone the same
way. ``main_realtime.py`` used to hardcode VGGT, which meant the live path
silently ignored ``--backbone`` and never ran the default DVLT model at all.
"""

import torch
from termcolor import colored

from vggt.models.vggt import VGGT

VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def build_backbone(
    backbone: str = "dvlt",
    device: str = None,
    dvlt_checkpoint: str = "nvidia/dvlt",
    dvlt_k: int = None,
    lc_verify: str = "attn",
    lc_attn_step: int = None,
    verbose: bool = True,
):
    """Load the submap reconstructor named by ``backbone`` ("dvlt" or "vggt")."""
    if backbone not in ("dvlt", "vggt"):
        raise ValueError(f"backbone must be 'dvlt' or 'vggt', got {backbone!r}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if backbone == "dvlt":
        from vggt_slam.dvlt_backbone import DVLTBackbone

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
        model = VGGT()
        model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_URL))
        model = model.to(torch.bfloat16)  # use half precision

    model.eval()
    model = model.to(device)
    return model

"""Upstream VGGT as a submap backbone.

VGGT returns a ``pose_enc`` in its own ``absT_quaR_FoV`` encoding; ``SubmapBackbone``
is stated in extrinsic/intrinsic matrices, so the decode happens here. It used to
live in ``Solver``, which put one model's pose parameterisation in the middle of
backbone-agnostic code.

Note the decode rebuilds the principal point as ``(W/2, H/2)`` -- the encoding never
stored it. Harmless here, since VGGT's intrinsics come out of that same encoding, but
a backbone that predicts ``cx``/``cy`` must not route its geometry through it.
"""

from __future__ import annotations

import torch

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from slam.backbones.base import LoopVerification, Reconstruction

#: The 1B checkpoint the reported VGGT baselines were produced with.
VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


class VGGTBackbone:
    """Upstream VGGT behind the ``SubmapBackbone`` protocol."""

    backbone_name = "VGGT"

    def __init__(self, device: str = "cuda", url: str = VGGT_URL, verbose: bool = True):
        if verbose:
            print("Using VGGT backbone (upstream VGGT-SLAM baseline)")
        model = VGGT()
        model.load_state_dict(torch.hub.load_state_dict_from_url(url))
        model = model.to(torch.bfloat16)  # use half precision
        model.eval()
        self.model = model.to(device)

    def _reconstruct(self, images: torch.Tensor, compute_similarity: bool) -> tuple[Reconstruction, dict]:
        """One forward pass, decoded into a ``Reconstruction``"""
        with torch.no_grad():
            preds = self.model(images, compute_similarity=compute_similarity)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
        recon = Reconstruction(
            extrinsic=extrinsic.squeeze(0),
            intrinsic=intrinsic.squeeze(0),
            depth=preds["depth"].squeeze(0),
            depth_conf=preds["depth_conf"].squeeze(0),
        )
        return recon, preds

    def reconstruct(self, images: torch.Tensor) -> Reconstruction:
        recon, _ = self._reconstruct(images, compute_similarity=False)
        return recon

    def verify_loop(self, pair: torch.Tensor) -> LoopVerification:
        recon, preds = self._reconstruct(pair, compute_similarity=True)
        return LoopVerification(reconstruction=recon, match_score=preds["image_match_ratio"])

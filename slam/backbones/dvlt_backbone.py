"""Déjà View (DVLT) backbone"""

from __future__ import annotations

import torch
from accelerate import Accelerator

from slam.backbones.base import LoopVerification, Reconstruction

from dvlt.common.constants import DataField
from dvlt.model.dvlt.model import DVLT
from vggt.models.aggregator import get_similarity


# DVLT prepends 1 camera token + 4 register tokens, matching VGGT's layout, so the
# offset that get_similarity uses to skip non-patch tokens carries over unchanged.
_NUM_SPECIAL_TOKENS = 5


def c2w_to_w2c(c2w: torch.Tensor) -> torch.Tensor:
    """Invert a batch of (..., 3, 4) OpenCV camera-to-world matrices to world-to-camera."""
    rot = c2w[..., :3, :3]
    trans = c2w[..., :3, 3]
    rot_inv = rot.transpose(-1, -2)
    trans_inv = -torch.einsum("...ij,...j->...i", rot_inv, trans)
    return torch.cat([rot_inv, trans_inv.unsqueeze(-1)], dim=-1)


class DVLTBackbone:
    """DVLT behind the ``SubmapBackbone`` protocol."""

    backbone_name = "DVLT"

    def __init__(
        self,
        checkpoint: str = "nvidia/dvlt",
        img_size: int = 518,
        inference_steps: int | None = None,
        device: str = "cuda",
        lc_verify: str = "bypass",
        lc_attn_step: int | None = None,
    ):
        if lc_verify not in ("bypass", "attn"):
            raise ValueError(f"lc_verify must be 'bypass' or 'attn', got {lc_verify!r}")

        self.accelerator = Accelerator(mixed_precision="bf16")
        self.model = DVLT(img_size=img_size)
        self.model.load_pretrained(checkpoint, strict=True)
        self.model.setup_test(self.accelerator)

        if inference_steps is not None:
            self.model.model.inference_steps = inference_steps
        self.k = self.model.model.inference_steps

        self.lc_verify = lc_verify
        # Default to the same relative depth as VGGT's target_layer (20 of 24).
        self.lc_attn_step = lc_attn_step if lc_attn_step is not None else max(0, round(self.k * 20 / 24) - 1)

        self._attn_module = self.model.model.recurrent_blocks[0].global_attn.attn
        self._capture = None
        self._call_idx = 0

    # -- attention capture -------------------------------------------------------
    def _qkv_hook(self, _linear, _inputs, output):
        """Reconstruct q, k for the iteration we care about."""
        if self._call_idx == self.lc_attn_step:
            attn = self._attn_module
            b, n, _ = output.shape
            qkv = output.reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            self._capture = (attn.q_norm(q), attn.k_norm(k))
        self._call_idx += 1

    # -- SubmapBackbone --------------------------------------------------------
    def _reconstruct(self, images: torch.Tensor, capture_attn: bool) -> Reconstruction:
        """One DVLT forward, mapped onto the protocol's shapes.

        DVLT already speaks in matrices, so this is a rename and a batch-dim drop
        rather than a conversion. Nothing is re-encoded: ``cx``/``cy`` from
        ``get_intrinsics_matrices`` reach the solver as predicted.
        """
        batch = {DataField.IMAGES: images.unsqueeze(0).to(self.accelerator.device)}
        handle = None
        if capture_attn:
            self._capture, self._call_idx = None, 0
            handle = self._attn_module.qkv.register_forward_hook(self._qkv_hook)
        try:
            with torch.no_grad(), self.accelerator.autocast():
                preds = self.model.predict(batch, self.accelerator)
        finally:
            # The hook must not outlive the forward it was registered for.
            if handle is not None:
                handle.remove()

        cameras = preds["cameras"][0]  # [0] drops the batch dim added above
        return Reconstruction(
            extrinsic=c2w_to_w2c(cameras.camera_to_worlds),  # (S, 3, 4)
            intrinsic=cameras.get_intrinsics_matrices(),     # (S, 3, 3)
            depth=preds["depths"][0].unsqueeze(-1),          # (S, H, W, 1)
            depth_conf=preds["depths_conf"][0],              # (S, H, W)
        )

    def reconstruct(self, images: torch.Tensor) -> Reconstruction:
        return self._reconstruct(images, capture_attn=False)

    def verify_loop(self, pair: torch.Tensor) -> LoopVerification:
        want_attn = self.lc_verify == "attn"
        recon = self._reconstruct(pair, capture_attn=want_attn)

        if self.lc_verify == "bypass":
            score = 1.0
        else:
            if self._capture is None:
                raise RuntimeError(
                    f"no attention captured at iteration {self.lc_attn_step} (K={self.k}); "
                    "lc_attn_step must be < K"
                )
            q, k = self._capture
            score = get_similarity(k, q, token_offset=_NUM_SPECIAL_TOKENS)
            self._capture = None

        return LoopVerification(reconstruction=recon, match_score=score)

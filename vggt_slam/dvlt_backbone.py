"""Déjà View (DVLT) as a drop-in submap backbone for VGGT-SLAM.

This adapter exposes DVLT behind the exact call signature ``Solver.run_predictions``
expects of VGGT::

    predictions = model(images)                          # submap reconstruction
    predictions = model(lc_frames, compute_similarity=True)  # loop-closure verification

so that the factor-graph chaining, loop-closure detection and SL(4) correction in
``vggt_slam/solver.py`` run unchanged. Nothing in ``solver.py`` is modified.

Output contract (matching ``vggt/models/vggt.py``, batch dim kept — the solver
squeezes it):

    pose_enc          (1, S, 9)          absT_quaR_FoV, world-to-camera
    depth             (1, S, H, W, 1)
    depth_conf        (1, S, H, W)
    images            (1, S, 3, H, W)
    image_match_ratio scalar             only when compute_similarity=True

``world_points`` is deliberately not emitted: VGGT does not emit it either (the
head is commented out upstream) and the solver derives points itself via
``unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)``.

Coordinate conventions
----------------------
DVLT returns ``camera_to_worlds`` (OpenCV, c2w). ``extri_intri_to_pose_encoding``
expects *camera-from-world* (w2c), so we invert. Getting this backwards produces a
plausible-looking but mirrored trajectory, so ``selftest_pose_roundtrip`` below
checks it explicitly.

Loop-closure verification (``lc_verify``)
-----------------------------------------
VGGT-SLAM derives ``image_match_ratio`` from the q/k of one *specific* global
attention layer (``aggregator.target_layer = 20`` of 24) and rejects a retrieved
loop when the ratio is < 0.95. DVLT has no layer 20: it loops a single shared
block K times, so the structural analogue of "layer 20" is "iteration k of K".
Whether any iteration carries an equivalent signal is an open question (Phase 1,
step 5 of plan.md), so this is a switch rather than a fixed choice:

  "bypass"  — always report 1.0, i.e. accept every loop SALAD retrieval proposes.
              The geometric gate downstream still applies. Use this to get the
              pipeline running end-to-end; it makes loop closure more permissive
              than stock VGGT-SLAM, so it is NOT a like-for-like comparison.
  "attn"    — hook the shared block's global attention at iteration
              ``lc_attn_step`` and run VGGT's own ``get_similarity`` on its q/k.
              This is the instrument for answering step 5 empirically.
"""

from __future__ import annotations

import torch
from accelerate import Accelerator

from dvlt.common.constants import DataField
from dvlt.model.dvlt.model import DVLT
from vggt.models.aggregator import get_similarity
from vggt.utils.pose_enc import extri_intri_to_pose_encoding


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
    """DVLT wrapped in VGGT's calling convention.

    Deliberately not an ``nn.Module``: ``main.py`` does ``model.to(torch.bfloat16)``
    on the VGGT path, which would silently cast DVLT's weights out from under
    ``accelerate``'s autocast. ``to()`` and ``eval()`` are accepted as no-ops so the
    adapter stays drop-in.
    """

    # Read by vggt_slam.slam_utils.backbone_name() so timing and progress logging
    # name the backbone actually in use rather than a hardcoded "VGGT".
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

    # -- drop-in no-ops so main.py's VGGT setup lines stay valid -----------------
    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    # -- attention capture -------------------------------------------------------
    def _qkv_hook(self, _linear, _inputs, output):
        """Reconstruct q, k for the iteration we care about.

        The hook is registered on the fused ``qkv`` Linear, so the owning Attention
        (which holds the head layout and the q/k norms) comes from ``self``, not from
        the hook's module argument. Mirrors ``Attention.forward`` so the captured
        tensors match what attention actually used.
        """
        if self._call_idx == self.lc_attn_step:
            attn = self._attn_module
            b, n, _ = output.shape
            qkv = output.reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            self._capture = (attn.q_norm(q), attn.k_norm(k))
        self._call_idx += 1

    def _predict(self, images: torch.Tensor, capture_attn: bool) -> dict:
        batch = {DataField.IMAGES: images.unsqueeze(0).to(self.accelerator.device)}
        handle = None
        if capture_attn:
            self._capture, self._call_idx = None, 0
            handle = self._attn_module.qkv.register_forward_hook(self._qkv_hook)
        try:
            with torch.no_grad(), self.accelerator.autocast():
                return self.model.predict(batch, self.accelerator)
        finally:
            if handle is not None:
                handle.remove()

    # -- VGGT-compatible entry point --------------------------------------------
    def __call__(self, images: torch.Tensor, query_points=None, compute_similarity: bool = False) -> dict:
        if query_points is not None:
            raise NotImplementedError("DVLT backbone has no tracking head; query_points is unsupported.")
        if images.dim() == 5:  # (1, S, 3, H, W) -> (S, 3, H, W)
            images = images.squeeze(0)

        want_attn = compute_similarity and self.lc_verify == "attn"
        preds = self._predict(images, capture_attn=want_attn)

        cameras = preds["cameras"][0]
        height, width = images.shape[-2:]
        w2c = c2w_to_w2c(cameras.camera_to_worlds).unsqueeze(0)  # (1, S, 3, 4)
        intrinsics = cameras.get_intrinsics_matrices().unsqueeze(0)  # (1, S, 3, 3)

        out = {
            "pose_enc": extri_intri_to_pose_encoding(w2c, intrinsics, image_size_hw=(height, width)),
            "depth": preds["depths"].unsqueeze(-1),
            "depth_conf": preds["depths_conf"],
            "images": images.unsqueeze(0),
        }

        if compute_similarity:
            if self.lc_verify == "bypass":
                out["image_match_ratio"] = 1.0
            else:
                if self._capture is None:
                    raise RuntimeError(
                        f"no attention captured at iteration {self.lc_attn_step} (K={self.k}); "
                        "lc_attn_step must be < K"
                    )
                q, k = self._capture
                out["image_match_ratio"] = get_similarity(k, q, token_offset=_NUM_SPECIAL_TOKENS)
                self._capture = None

        return out


def selftest_pose_roundtrip(device: str = "cuda") -> None:
    """Check c2w->w2c->pose_enc->w2c is identity, so the inversion direction is right."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    torch.manual_seed(0)
    rot = torch.linalg.qr(torch.randn(4, 3, 3, device=device))[0]
    rot[torch.det(rot) < 0] *= -1  # keep proper rotations
    c2w = torch.cat([rot, torch.randn(4, 3, 1, device=device)], dim=-1)

    intrinsics = torch.zeros(4, 3, 3, device=device)
    intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = 300.0
    intrinsics[:, 0, 2], intrinsics[:, 1, 2] = 259.0, 147.0
    intrinsics[:, 2, 2] = 1.0

    w2c = c2w_to_w2c(c2w)
    enc = extri_intri_to_pose_encoding(w2c.unsqueeze(0), intrinsics.unsqueeze(0), image_size_hw=(294, 518))
    w2c_rt, k_rt = pose_encoding_to_extri_intri(enc, (294, 518))

    pose_err = (w2c_rt.squeeze(0) - w2c).abs().max().item()
    intr_err = (k_rt.squeeze(0) - intrinsics).abs().max().item()
    assert pose_err < 1e-4, f"pose round-trip error too large: {pose_err}"
    assert intr_err < 1e-2, f"intrinsics round-trip error too large: {intr_err}"
    print(f"pose round-trip OK (pose err {pose_err:.2e}, intrinsics err {intr_err:.2e})")


if __name__ == "__main__":
    selftest_pose_roundtrip()

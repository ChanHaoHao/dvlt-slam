"""A protocol for transformer backbones to follow"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass
class Reconstruction:
    """Per-frame geometry for one set of frames."""

    #: (S, 3, 4) OpenCV, camera-FROM-world. Note the direction: emitting
    #: camera-to-world instead yields a plausible but mirrored trajectory.
    extrinsic: torch.Tensor
    #: (S, 3, 3) full pinhole K, principal point included rather than assumed centred.
    intrinsic: torch.Tensor
    #: (S, H, W, 1)
    depth: torch.Tensor
    #: (S, H, W)
    depth_conf: torch.Tensor


@dataclass
class LoopVerification:
    """The answer to "are these two frames really the same place?"."""

    reconstruction: Reconstruction
    #: In [0, 1]. The solver rejects the loop below its threshold.
    match_score: float


@runtime_checkable
class SubmapBackbone(Protocol):
    """Structural type of a submap reconstructor."""

    #: Display name for logging, e.g. "DVLT". Reported by
    #: ``slam.slam_utils.backbone_name()`` so a run names the backbone in use.
    backbone_name: str

    def reconstruct(self, images: torch.Tensor) -> Reconstruction:
        """Reconstruct a submap from ``images`` (S, 3, H, W), already preprocessed."""
        ...

    def verify_loop(self, pair: torch.Tensor) -> LoopVerification:
        """Reconstruct and score a loop-closure candidate ``pair`` (2, 3, H, W)."""
        ...

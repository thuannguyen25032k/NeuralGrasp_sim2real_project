"""
maniflow_bc/gpu_preprocessor.py
================================
GPU-side pre-processing for ManiFlow BC training.

Implements the recommendations from the 3d_flowmatch_actor comparison:
  1. Point cloud computation from depth ON THE GPU (bfloat16 intermediate,
     float32 output) — instead of the CPU-side per-sample extraction done
     inside ``helpers/utils.py::extract_obs``.
  2. Batched image augmentation ON THE GPU using kornia (RandomAffine +
     RandomResizedCrop, same regime as RLBenchDataPreprocessor in
     3d_flowmatch_actor).
  3. Optional bilinear image resize on GPU (antialias=True).

Usage
-----
    from agents.maniflow_bc.gpu_preprocessor import ManiFlowGPUPreprocessor

    preprocessor = ManiFlowGPUPreprocessor(
        image_size=(128, 128),
        augment_prob=0.8,
        custom_imsize=None,   # None = keep original size
    ).cuda()

    # Inside the training loop (already on GPU tensors from DataLoader):
    rgb_proc, pcd_proc = preprocessor(
        rgb,         # (B, ncam, 3, H, W)  float32 [0,1]
        depth,       # (B, ncam, 1, H, W)  float32
        extrinsics,  # (B, ncam, 4, 4)     float32
        intrinsics,  # (B, ncam, 3, 3)     float32
        augment=True,
    )
    # rgb_proc  : (B, ncam, 3, H', W')  float32
    # pcd_proc  : (B, ncam, 3, H', W')  float32  world-frame point cloud
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import augmentation as K


# ---------------------------------------------------------------------------
# GPU depth → world-frame point cloud
# ---------------------------------------------------------------------------

class DepthToPointCloud(nn.Module):
    """
    Un-project depth maps to world-frame point clouds on the GPU.

    Adapted from 3d_flowmatch_actor/utils/depth2cloud/rlbench.py but
    generalised to work with any (H, W) without pre-allocated pixel grids
    (the grids are created once and cached on the device).
    """

    def __init__(self, image_size: tuple[int, int] = (128, 128)):
        super().__init__()
        H, W = image_size
        u = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
        v = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
        ones = torch.ones(H, W, dtype=torch.float32)
        # pixel_grid: (1, 3, H, W) — broadcast over batch
        pixel_grid = torch.stack([u, v, ones], dim=0).unsqueeze(0)
        self.register_buffer("pixel_grid", pixel_grid)   # moves with .to(device)

    def forward(
        self,
        depth: torch.Tensor,       # (B*ncam, 1, H, W) float  (depth in metres)
        extrinsics: torch.Tensor,  # (B*ncam, 4, 4)    float  cam-to-world
        intrinsics: torch.Tensor,  # (B*ncam, 3, 3)    float
    ) -> torch.Tensor:             # (B*ncam, 3, H, W) float  world-frame xyz
        BN, _, H, W = depth.shape

        # ---------- inverse camera-projection matrix -----------------------
        # P = K @ [R|t]  →  P^{-1} used to un-project pixel rays
        C  = extrinsics[:, :3, 3:]                    # (BN, 3, 1)
        R  = extrinsics[:, :3, :3]                    # (BN, 3, 3)
        R_inv = R.transpose(1, 2)                     # (BN, 3, 3)
        t_inv = -torch.bmm(R_inv, C)                  # (BN, 3, 1)
        ext_inv = torch.cat([R_inv, t_inv], dim=-1)   # (BN, 3, 4)
        cam_proj = torch.bmm(intrinsics, ext_inv)     # (BN, 3, 4)
        cam_proj_homo = torch.cat([
            cam_proj,
            torch.tensor([[0, 0, 0, 1]], dtype=cam_proj.dtype,
                         device=cam_proj.device).expand(BN, 1, 4),
        ], dim=1)                                     # (BN, 4, 4)
        proj_inv = torch.linalg.inv(cam_proj_homo.float())[:, :3]  # (BN,3,4)
        proj_inv = proj_inv.to(depth.dtype)

        # ---------- un-project ---------------------------------------------
        # scaled pixel: u*d, v*d, d
        scaled = self.pixel_grid * depth              # (BN, 3, H, W)
        scaled_homo = torch.cat(
            [scaled, torch.ones(BN, 1, H, W, dtype=depth.dtype,
                                device=depth.device)], dim=1
        )                                             # (BN, 4, H, W)
        flat = scaled_homo.view(BN, 4, -1)            # (BN, 4, HW)
        world = torch.bmm(proj_inv, flat)             # (BN, 3, HW)
        return world.view(BN, 3, H, W)


# ---------------------------------------------------------------------------
# GPU augmentation + preprocessing module
# ---------------------------------------------------------------------------

class ManiFlowGPUPreprocessor(nn.Module):
    """
    Batched GPU pre-processor for ManiFlow BC observations.

    Steps
    -----
    1. Depth → Point Cloud on GPU (bfloat16 intermediate for speed).
    2. Kornia RandomAffine + RandomResizedCrop augmentation (applied jointly
       to RGB and PCD so spatial consistency is preserved).
    3. Optional bilinear resize to ``custom_imsize``.

    All operations run in float32 (with bfloat16 intermediates for the
    augmentation step to save bandwidth) and return float32 tensors.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (128, 128),
        augment_prob: float = 0.8,
        crop_prob: float = 0.1,
        custom_imsize: int | None = None,
    ):
        super().__init__()
        self._image_size   = image_size
        self._custom_imsize = custom_imsize

        # Depth → PCD module (registered buffers move with .to(device))
        self.depth2pcd = DepthToPointCloud(image_size)

        # Kornia augmentation pipeline — applied to (B*ncam, 6, H, W) where
        # the first 3 channels are RGB and the next 3 are XYZ point cloud.
        # Using reflection padding to avoid black borders from affine shifts.
        self.aug = K.AugmentationSequential(
            K.RandomAffine(
                degrees=0,
                translate=0.0,
                scale=(0.75, 1.25),
                padding_mode="reflection",
                p=augment_prob,
            ),
            K.RandomResizedCrop(
                size=image_size,
                scale=(0.95, 1.05),
                p=crop_prob,
            ),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        rgb: torch.Tensor,         # (B, ncam, 3, H, W)  float32 [0,1]
        depth: torch.Tensor,       # (B, ncam, 1, H, W)  float32
        extrinsics: torch.Tensor,  # (B, ncam, 4, 4)     float32
        intrinsics: torch.Tensor,  # (B, ncam, 3, 3)     float32
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        rgb_out : (B, ncam, 3, H', W')  float32
        pcd_out : (B, ncam, 3, H', W')  float32  world-frame XYZ
        """
        B, ncam, _, H, W = rgb.shape
        BN = B * ncam

        # 1. Flatten batch × camera dims
        rgb_flat   = rgb.view(BN, 3, H, W)
        depth_flat = depth.view(BN, 1, H, W)
        ext_flat   = extrinsics.view(BN, 4, 4)
        int_flat   = intrinsics.view(BN, 3, 3)

        # 2. Depth → PCD in bfloat16 for memory efficiency
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16):
            pcd_flat = self.depth2pcd(
                depth_flat.to(torch.bfloat16),
                ext_flat.to(torch.bfloat16),
                int_flat.to(torch.bfloat16),
            ).float()                                    # (BN, 3, H, W)

        # 3. Augmentation: apply the SAME transform to RGB and PCD channels
        if augment:
            obs = torch.cat([rgb_flat.half(), pcd_flat.half()], dim=1)  # (BN,6,H,W)
            obs = self.aug(obs)
            rgb_out = obs[:, :3].float()                 # (BN, 3, H, W)
            pcd_out = obs[:, 3:].float()                 # (BN, 3, H, W)
        else:
            rgb_out = rgb_flat                           # already float32
            pcd_out = pcd_flat

        # 4. Optional resize
        if self._custom_imsize is not None and self._custom_imsize != H:
            sz = self._custom_imsize
            rgb_out = F.interpolate(
                rgb_out, (sz, sz), mode="bilinear", antialias=True
            )
            pcd_out = F.interpolate(
                pcd_out, (sz, sz), mode="bilinear", antialias=True
            )

        # 5. Restore (B, ncam, 3, H', W')
        _, _, H2, W2 = rgb_out.shape
        rgb_out = rgb_out.view(B, ncam, 3, H2, W2)
        pcd_out = pcd_out.view(B, ncam, 3, H2, W2)

        return rgb_out, pcd_out

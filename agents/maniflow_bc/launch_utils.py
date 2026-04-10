"""
ManiFlow launch utilities
=========================
Pure-Zarr data pipeline — no YARR replay buffer.

Data flow
---------
  convert_replay_to_zarr.py  →  {split}.zarr + instructions.json
  create_zarr_loader()        →  PyTorch DataLoader (ManiFlowZarrDataset)
  OfflineTrainRunner          →  calls agent.update(batch) each step

Batch keys produced by maniflow_collate_fn (mirrors base_collate_fn):
  task          list[str]           length B
  instr         list[str]           length B
  rgb           (B, ncam, 3, H, W)   float32
  pcd           (B, ncam, 3, H, W)   float32
  proprioception (B, nhist, nhand, 8) float32
  action        (B, 8)               float32  [xyz qxyzw grip]  (nhand+T squeezed)
  extrinsics    (B, ncam, 4, 4)      float32
  intrinsics    (B, ncam, 3, 3)      float32
  nerf_multi_view_rgb/depth/camera      (B, nview)  object (optional)
  nerf_next_multi_view_rgb/depth/camera (B, nview)  object (optional)
"""

import json
import logging
import random
from typing import List, Optional

import numpy as np

from agents.maniflow_bc.preprocess_agent import ManiFlowPreprocessAgent
from agents.maniflow_bc.voxel_flow_encoder import VoxelFlowEncoder
from agents.maniflow_bc.qattention_maniflow_agent import ManiFlowBCAgent
from agents.maniflow_bc.qattention_stack_agent import ManiFlowStackAgent
from agents.maniflow_bc.zarr_dataset import ManiFlowZarrDataset, maniflow_collate_fn
from agents.maniflow_bc.gpu_preprocessor import ManiFlowGPUPreprocessor

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig
from termcolor import cprint
from lightning_fabric import Fabric


# ---------------------------------------------------------------------------
# Module-level worker init — must be at module scope so it is picklable
# by the DataLoader multiprocessing backend.
# ---------------------------------------------------------------------------

def _seed_worker(worker_id: int):
    """Seed numpy/random per worker for reproducible augmentation."""
    import random as _random
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Fast Zarr-based DataLoader
# ---------------------------------------------------------------------------

def create_zarr_loader(
        zarr_path: str,
        instructions_path: str,
        batch_size: int,
        cameras: List[str] = None,
        camera_inds: Optional[List[int]] = None,
        chunk_size: int = 1,
        num_workers: int = 4,
        mem_gb: float = 8.0,
        relative_action: bool = False,
        copies: int = 10,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
) -> tuple:
    """
    Build a DataLoader backed by ``ManiFlowZarrDataset``.

    Mirrors 3D FlowMatch Actor's ``get_loaders`` (utils/trainers/base.py):
      - ``prefetch_factor=4``    → keep 4 batches pre-loaded in worker RAM
      - ``persistent_workers``   → workers stay alive across epochs
      - ``pin_memory=True``      → faster H→D transfer for CUDA tensors
      - Seeded ``worker_init_fn``→ reproducible augmentation per worker
      - ``DistributedSampler``   → proper data sharding in DDP runs

    Parameters
    ----------
    zarr_path          : Path to ``{split}.zarr``.
    instructions_path  : Path to ``instructions.json`` (task→variation→list[str]).
    batch_size         : Batch size per GPU.
    cameras            : Camera names (must match converter order).
    chunk_size         : Action chunk size T (must match method.action_chunk_size).
    num_workers        : DataLoader worker processes (default 4).
    mem_gb             : LRU cache per worker process in GiB.
    relative_action    : Return delta actions (SO(3) relative).
    copies             : Virtual dataset length multiplier.
    distributed        : Use DistributedSampler (for DDP).
    rank / world_size  : DDP rank / world-size.
    seed               : Global RNG seed for reproducibility.

    Returns
    -------
    (loader, dataset, sampler)
    """
    with open(instructions_path, 'r') as f:
        instructions = json.load(f)

    dataset = ManiFlowZarrDataset(
        zarr_path=zarr_path,
        instructions=instructions,
        cameras=cameras or ["front"],
        camera_inds=camera_inds,
        chunk_size=chunk_size,
        mem_gb=mem_gb,
        relative_action=relative_action,
        copies=copies,
    )

    def _seed_worker_local(worker_id: int):  # kept as local for backward-compat; unused
        pass

    g = torch.Generator()
    g.manual_seed(seed)

    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        collate_fn=maniflow_collate_fn,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
        generator=g,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    return loader, dataset, sampler


# ---------------------------------------------------------------------------
# GPU Pre-processor factory
# ---------------------------------------------------------------------------

def create_gpu_preprocessor(
        image_size: tuple = (128, 128),
        augment_prob: float = 0.8,
        crop_prob: float = 0.1,
        custom_imsize: Optional[int] = None,
) -> ManiFlowGPUPreprocessor:
    """
    Instantiate a ``ManiFlowGPUPreprocessor`` (GPU-side depth→PCD + kornia
    augmentation).

    The returned module should be moved to the training device:
        preprocessor = create_gpu_preprocessor(...).to(device)

    During training call:
        rgb_proc, pcd_proc = preprocessor(rgb, depth, extrinsics, intrinsics,
                                          augment=True)
    During evaluation:
        rgb_proc, pcd_proc = preprocessor(rgb, depth, extrinsics, intrinsics,
                                          augment=False)
    """
    return ManiFlowGPUPreprocessor(
        image_size=image_size,
        augment_prob=augment_prob,
        crop_prob=crop_prob,
        custom_imsize=custom_imsize,
    )


# ---------------------------------------------------------------------------
# create_agent
# ---------------------------------------------------------------------------

def create_agent(cfg: DictConfig) -> ManiFlowPreprocessAgent:
    """
    Build the ManiFlow agent stack from config.

    Key config keys (under cfg.method):
      voxel_sizes, final_dim, language_model_dim,
      # Transformer denoising head:
      embedding_dim            (default: 120)
      num_attn_heads           (default: 8)
      num_shared_attn_layers   (default: 4)
      voxel_token_downsample   (default: 5)
      denoise_timesteps        (default: 100)
      action_dim               (default: 8)
      action_chunk_size        (default: 1  — T steps per prediction)
      lv2_batch_size           (default: 1  — inner loss re-sampling loop)
      workspace_bounds         (default: scene_bounds from rlbench config)
      pos_loss_weight          (default: 30.0)
      rot_loss_weight          (default: 10.0)
      grip_loss_weight         (default: 1.0)
      # legacy MLP stubs (kept for YAML back-compat, not used functionally):
      flow_context_dim, flow_hidden_dim, flow_num_layers
    """
    depth_0bounds = cfg.rlbench.scene_bounds
    cam_resolution = cfg.rlbench.camera_resolution

    # Transformer head hyperparameters (read from cfg, fall back to defaults)
    embedding_dim = getattr(cfg.method, 'embedding_dim',          120)
    num_attn_heads = getattr(cfg.method, 'num_attn_heads',         8)
    num_shared_attn_layers = getattr(cfg.method, 'num_shared_attn_layers', 4)
    voxel_token_downsample = getattr(cfg.method, 'voxel_token_downsample', 5)
    num_fps_tokens = getattr(cfg.method, 'num_fps_tokens',         512)
    denoise_timesteps = getattr(cfg.method, 'denoise_timesteps',      100)
    action_dim = getattr(cfg.method, 'action_dim',             8)
    action_chunk_size = getattr(cfg.method, 'action_chunk_size', 1)
    lv2_batch_size = getattr(cfg.method, 'lv2_batch_size', 1)
    workspace_bounds = list(getattr(cfg.method, 'workspace_bounds',
                                    list(depth_0bounds)))
    # Legacy stubs (ignored by new VoxelFlowEncoder but kept for YAML compat)
    flow_context_dim = getattr(cfg.method, 'flow_context_dim',  256)
    flow_hidden_dim = getattr(cfg.method, 'flow_hidden_dim',   512)
    flow_num_layers = getattr(cfg.method, 'flow_num_layers',   4)

    qattention_agents = []
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):

        flow_encoder = VoxelFlowEncoder(
            voxel_size=vox_size,
            initial_dim=3 + 3 + 1 + 3,   # rgb + xyz + density + normal
            low_dim_size=4,
            im_channels=cfg.method.final_dim,
            lang_feat_dim=1024,             # CLIP sentence embedding
            lang_emb_dim=cfg.method.language_model_dim,
            rot_dim=6,                       # always ortho6D
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            voxel_token_downsample=voxel_token_downsample,
            num_fps_tokens=num_fps_tokens,
            coordinate_bounds=list(depth_0bounds),
            denoise_timesteps=denoise_timesteps,
            activation=cfg.method.activation,
            lang_fusion_type=cfg.method.lang_fusion_type,
            # legacy stubs
            context_dim=flow_context_dim,
            flow_hidden_dim=flow_hidden_dim,
            flow_num_layers=flow_num_layers,
            voxel_patch_size=cfg.method.voxel_patch_size,
            voxel_patch_stride=cfg.method.voxel_patch_stride,
            cfg=cfg,
        )

        qattention_agent = ManiFlowBCAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            flow_encoder=flow_encoder,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth -
                                                   1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            lr=cfg.method.lr,
            training_iterations=cfg.framework.training_iterations,
            lr_scheduler=cfg.method.lr_scheduler,
            num_warmup_steps=cfg.method.num_warmup_steps,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size,
            voxel_feature_size=3,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            transform_augmentation=cfg.method.transform_augmentation.apply_se3,
            transform_augmentation_xyz=cfg.method.transform_augmentation.aug_xyz,
            transform_augmentation_rpy=cfg.method.transform_augmentation.aug_rpy,
            transform_augmentation_rot_resolution=cfg.method.transform_augmentation.aug_rot_resolution,
            optimizer_type=cfg.method.optimizer,
            num_devices=cfg.ddp.num_devices,
            denoise_timesteps=denoise_timesteps,
            action_dim=action_dim,
            action_chunk_size=action_chunk_size,
            lv2_batch_size=lv2_batch_size,
            workspace_bounds=workspace_bounds,
            pos_loss_weight=getattr(cfg.method, 'pos_loss_weight',  30.0),
            rot_loss_weight=getattr(cfg.method, 'rot_loss_weight',  10.0),
            grip_loss_weight=getattr(cfg.method, 'grip_loss_weight',  1.0),
            cfg=cfg.method,
        )
        qattention_agents.append(qattention_agent)

    flow_stack_agent = ManiFlowStackAgent(
        qattention_agents=qattention_agents,
        camera_names=cfg.rlbench.cameras,
    )
    # ManiFlowPreprocessAgent normalises uint8 RGB to [-1,1] and does NOT
    # access YARR-specific keys (low_dim_state, demo, nerf_multi_view_rgb,
    # lang_goal), so it is safe to use with the Zarr DataLoader.
    preprocess_agent = ManiFlowPreprocessAgent(pose_agent=flow_stack_agent, norm_rgb=True)
    return preprocess_agent

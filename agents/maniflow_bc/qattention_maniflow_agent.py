"""
ManiFlow BC Agent
=================
Drop-in replacement for QAttentionPerActBCAgent that:
  - Keeps the VoxelGrid voxelisation pipeline unchanged.
  - Keeps the NeuralRenderer (Gaussian Splatting) auxiliary loss unchanged.
  - Replaces the Perceiver IO + discrete classification head with a
    continuous flow-matching action head (VoxelFlowEncoder).

Action space: [x, y, z, qx, qy, qz, qw, gripper_open]  (8-DoF, continuous)
Loss: flow-matching velocity prediction (L1) on position/rotation +
      optional neural rendering auxiliary loss.
"""

import logging
import os
import contextlib
import warnings
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# matplotlib internally uses an old pyparsing API that emits
# PyparsingDeprecationWarning on every import.  Suppress once at module load.
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*parseString.*')
    warnings.filterwarnings('ignore', message='.*resetCache.*')
    warnings.filterwarnings('ignore', message='.*enablePackrat.*')
    import matplotlib
    matplotlib.use('Agg')   # non-interactive backend — safe in Docker / headless
    import matplotlib.pyplot as plt

from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, ImageSummary, Summary
from termcolor import colored, cprint
import io

from helpers.utils import visualise_voxel
from voxel.voxel_grid import VoxelGrid
from voxel.augmentation import apply_se3_augmentation_continuous
from helpers.clip.core.clip import build_model, load_clip
import PIL.Image as Image
import transformers
from helpers.optim.lamb import Lamb
from torch.nn.parallel import DistributedDataParallel as DDP
from agents.maniflow_bc.neural_rendering import NeuralRenderer
from agents.maniflow_bc.utils import (
    visualize_pcd,
    quat_xyzw_to_ortho6d,
    ortho6d_to_quat_xyzw,
)
from helpers.language_model import create_language_model

from agents.maniflow_bc.voxel_flow_encoder import VoxelFlowEncoder
from agents.maniflow_bc.rf_scheduler import RFScheduler

from lightning_fabric import Fabric

NAME = 'ManiFlowAgent'

# ---------------------------------------------------------------------------
# Utility helpers (shared with manigaussian_bc)
# ---------------------------------------------------------------------------

def _PSNR_torch(img1, img2, max_val=1):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(torch.tensor(float(max_val)) / torch.sqrt(mse))


def _parse_camera_file(file_path):
    """
    Parses camera extrinsics and intrinsics from a text file.
    Expected format: 
    - Lines 0-3: 4x4 extrinsic matrix (row-major)
    - Line 4: empty or comment
    - Lines 5-7: 3x3 intrinsic matrix (row-major)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ext = []
    for x in lines[0:4]:
        ext += [float(y) for y in x.split()]
    ext = np.array(ext).reshape(4, 4)
    intr = []
    for x in lines[5:8]:
        intr += [float(y) for y in x.split()]
    intr = np.array(intr).reshape(3, 3)
    focal = intr[0, 0]
    return ext, intr, focal


def _parse_img_file(file_path, mask_gt_rgb=False):
    """
    Parses an RGB image from a file path and normalizes it to [0, 1].
    """
    img = Image.open(file_path).convert('RGB')
    return np.asarray(img).astype(np.float32) / 255.0


def _parse_depth_file(file_path):
    """ Parses a depth image from a file path and converts it to a float32 numpy array.
    The depth image is expected to be in a single-channel format (e.g., 16-bit PNG),"""
    d = Image.open(file_path).convert('L')
    return np.asarray(d).astype(np.float32)


def visualize_feature_map_by_normalization(features):
    """
    Normalize a feature map to [0, 1] range for display.
    features: (B, C, H, W)  — takes first batch element
    Returns:  (H, W, C) numpy array in [0, 1]
    """
    MIN_DENOMINATOR = 1e-12
    features = features[0].cpu().detach().numpy()   # (C, H, W)
    features = features.transpose(1, 2, 0)           # (H, W, C)
    features = features / (
        np.linalg.norm(features, axis=-1, keepdims=True) + MIN_DENOMINATOR
    )
    # shift from [-1, 1] → [0, 1] for imshow
    features = (features - features.min()) / (features.max() - features.min() + MIN_DENOMINATOR)
    return features


# ---------------------------------------------------------------------------
# QFunction wrapper (keeps NeuralRenderer, replaces _qnet with VoxelFlowEncoder)
# ---------------------------------------------------------------------------

class QFunctionFlow(nn.Module):
    """
    Thin wrapper around VoxelFlowEncoder + NeuralRenderer.
    Mirrors the interface of the original QFunction in manigaussian_bc.
    """

    def __init__(self,
                 flow_encoder: VoxelFlowEncoder,
                 voxelizer: VoxelGrid,
                 bounds_offset: float,
                 device,
                 training: bool,
                 use_ddp: bool = True,
                 cfg=None,
                 fabric: Fabric = None):
        super().__init__()
        self._flow_encoder = flow_encoder.to(device)
        self._voxelizer    = voxelizer
        self._bounds_offset = bounds_offset
        self.cfg = cfg
        self.device = device

        self._coord_trans = torch.diag(
            torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        ).to(device)

        if cfg.use_neural_rendering:
            self._neural_renderer = NeuralRenderer(cfg.neural_renderer).to(device)
            if training and use_ddp:
                self._neural_renderer = fabric.setup(self._neural_renderer)
        else:
            self._neural_renderer = None
        cprint(f"[NeuralRenderer]: {cfg.use_neural_rendering}", "cyan")

        if training and use_ddp:
            cprint("[QFunctionFlow] use DDP: True", "cyan")
            self._flow_encoder = fabric.setup(self._flow_encoder)

    # ------------------------------------------------------------------
    # Scene encoding (returns voxel features + lang embed + context)
    # ------------------------------------------------------------------
    def encode_scene(self, rgb_pcd, depth, proprio, pcd,
                     camera_extrinsics, camera_intrinsics,
                     lang_goal_emb, lang_token_embs,
                     bounds=None, prev_bounds=None,
                     prev_layer_voxel_grid=None):
        """
        Encodes the input scene into a voxel grid and extracts features using the flow encoder.
        Args:
            rgb_pcd: list of tuples (rgb, pcd) for each camera view
            depth: list of depth maps for each camera view
            proprio: proprioceptive state
            pcd: list of point clouds for each camera view
            camera_extrinsics: list of extrinsic matrices for each camera view
            camera_intrinsics: list of intrinsic matrices for each camera view
            lang_goal_emb: language goal embedding
            lang_token_embs: language token embeddings
            bounds: optional pre-computed bounds for voxelization
            prev_bounds: optional bounds from previous layer (for multi-layer setups)
            prev_layer_voxel_grid: optional voxel grid from previous layer (for multi-layer setups)
        Returns:
            voxel_grid: (B, C, D, H, W) voxel grid of the scene
            voxel_feat: (B, F) features extracted from the voxel grid by the flow encoder
            lang_embedd: (B, L) language embedding output by the flow encoder
            context: additional context output by the flow encoder (e.g., for velocity prediction)
            bounds: (B, 6) bounds used for voxelization
        """

        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1
        )   # (B, N_points, 3)
        # Flatten RGB features in the same order as the flattened point cloud
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1] # C from (B, C, H, W) = 3 for RGB
        flat_feats = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1
        )

        voxel_grid, _ = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_feats,
            coord_bounds=bounds, return_density=True
        )
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach() # (B, C, D, H, W) = (B, 10, 100, 100, 100)

        # Pad bounds if necessary to ensure consistent shape for flow encoder
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        voxel_feat, lang_embedd, context = self._flow_encoder(
            voxel_grid, proprio, lang_goal_emb, lang_token_embs,
            prev_layer_voxel_grid, bounds, prev_bounds
        )
        return voxel_grid, voxel_feat, lang_embedd, context, bounds

    # ------------------------------------------------------------------
    # Full forward (training + neural rendering loss)
    # ------------------------------------------------------------------
    def forward(self, rgb_pcd, depth, proprio, pcd,
                camera_extrinsics, camera_intrinsics,
                lang_goal_emb, lang_token_embs,
                bounds=None, prev_bounds=None,
                prev_layer_voxel_grid=None,
                # neural rendering args
                use_neural_rendering=False,
                nerf_target_rgb=None, nerf_target_depth=None,
                nerf_target_pose=None, nerf_target_camera_intrinsic=None,
                lang_goal=None,
                nerf_next_target_rgb=None, nerf_next_target_pose=None,
                nerf_next_target_depth=None,
                nerf_next_target_camera_intrinsic=None,
                gt_embed=None, step=None, action=None):
        """
        Full forward pass that encodes the scene, predicts velocity (if noisy_action and timestep are provided),
        and computes the neural rendering loss (if use_neural_rendering is True).
        
        Args:
            rgb_pcd: list of tuples (rgb, pcd) for each camera view
            depth: list of depth maps for each camera view
            proprio: proprioceptive state
            pcd: list of point clouds for each camera view
            camera_extrinsics: list of extrinsic matrices for each camera view
            camera_intrinsics: list of intrinsic matrices for each camera view
            lang_goal_emb: language goal embedding
            lang_token_embs: language token embeddings
            bounds: optional pre-computed bounds for voxelization
            prev_bounds: optional bounds from previous layer (for multi-layer setups)
            prev_layer_voxel_grid: optional voxel grid from previous layer (for multi-layer setups)
            use_neural_rendering: whether to compute the neural rendering loss
            nerf_target_rgb: target RGB image for neural rendering loss
            nerf_target_depth: target depth image for neural rendering loss
            nerf_target_pose: target camera pose for neural rendering loss
            nerf_target_camera_intrinsic: target camera intrinsic for neural rendering loss
            lang_goal: language goal (for neural rendering loss)
            nerf_next_target_rgb: next target RGB image for neural rendering loss
            nerf_next_target_pose: next target camera pose for neural rendering loss
            nerf_next_target_depth: next target depth image for neural rendering loss
            nerf_next_target_camera_intrinsic: next target camera intrinsic for neural rendering loss
            gt_embed: ground truth embedding (for neural rendering loss)
            step: current training step (for neural rendering loss)
            action: current action (for neural rendering loss)
        
        Returns:
            context: dict output by the flow encoder used for velocity prediction
            voxel_grid: (B, C, D, H, W) voxel grid of the scene
            voxel_feat: (B, F) features extracted from the voxel grid by the flow encoder
            lang_embedd: (B, L) language embedding output by the flow encoder
            rendering_loss_dict: dictionary of losses from the neural renderer (if use_neural_rendering is True)
        """

        voxel_grid, voxel_feat, lang_embedd, context, bounds = self.encode_scene(
            rgb_pcd, depth, proprio, pcd,
            camera_extrinsics, camera_intrinsics,
            lang_goal_emb, lang_token_embs,
            bounds, prev_bounds, prev_layer_voxel_grid
        )

        # Neural rendering auxiliary loss (unchanged from manigaussian_bc)
        rendering_loss_dict = {}
        if use_neural_rendering and self._neural_renderer is not None:
            b = rgb_pcd[0][0].shape[0]
            rgb   = [rp[0] for rp in rgb_pcd]
            depth_list = depth

            focal = camera_intrinsics[0][:, 0, 0]
            cx, cy = 128 / 2, 128 / 2
            device = rgb_pcd[0][0].device
            c = torch.tensor([cx, cy], dtype=torch.float32).unsqueeze(0).to(device)

            if nerf_target_rgb is not None:
                gt_pose  = nerf_target_pose @ self._coord_trans
                rendering_loss_dict, _ = self._neural_renderer(
                    rgb=rgb[0], pcd=pcd[0], depth=depth_list[0],
                    language=lang_embedd,
                    dec_fts=voxel_feat,
                    gt_rgb=nerf_target_rgb,
                    gt_depth=nerf_target_depth,
                    focal=focal, c=c,
                    gt_pose=gt_pose,
                    gt_intrinsic=nerf_target_camera_intrinsic,
                    lang_goal=lang_goal,
                    next_gt_pose=nerf_next_target_pose,
                    next_gt_intrinsic=nerf_next_target_camera_intrinsic,
                    next_gt_rgb=nerf_next_target_rgb,
                    step=step, action=action,
                    training=True,
                )
            else:
                rendering_loss_dict = {
                    'loss': 0., 'loss_rgb': 0., 'loss_embed': 0.,
                    'l1': 0., 'psnr': 0., 'loss_dyna': 0., 'loss_reg': 0.,
                }

        return context, voxel_grid, voxel_feat, lang_embedd, rendering_loss_dict

    # ------------------------------------------------------------------
    # Inference-only rendering (mirrors manigaussian_bc QFunction.render)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def render(self, rgb_pcd, proprio, pcd,
               lang_goal_emb, lang_token_embs,
               tgt_pose, tgt_intrinsic,
               camera_extrinsics=None, camera_intrinsics=None,
               depth=None, bounds=None,
               prev_bounds=None, prev_layer_voxel_grid=None,
               nerf_target_rgb=None, lang_goal=None,
               nerf_next_target_rgb=None, nerf_next_target_depth=None,
               nerf_next_target_pose=None,
               nerf_next_target_camera_intrinsic=None,
               action=None, step=None):

        voxel_grid, voxel_feat, lang_embedd, context, bounds = self.encode_scene(
            rgb_pcd, depth, proprio, pcd,
            camera_extrinsics, camera_intrinsics,
            lang_goal_emb, lang_token_embs,
            bounds, prev_bounds, prev_layer_voxel_grid
        )

        _, ret_dict = self._neural_renderer(
            pcd=pcd[0], rgb=rgb_pcd[0][0],
            dec_fts=voxel_feat,
            language=lang_embedd,
            gt_pose=tgt_pose, gt_intrinsic=tgt_intrinsic,
            gt_rgb=nerf_target_rgb,
            lang_goal=lang_goal,
            next_gt_rgb=nerf_next_target_rgb,
            next_gt_pose=nerf_next_target_pose,
            next_gt_intrinsic=nerf_next_target_camera_intrinsic,
            step=step, action=action,
            training=False,
        )
        return (
            ret_dict.render_novel,
            ret_dict.next_render_novel,
            ret_dict.render_embed,
            ret_dict.gt_embed,
        )


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class ManiFlowBCAgent(Agent):
    """
    Flow-matching BC agent for ManiGaussian.

    - Uses continuous 8-DoF action: [x, y, z, qx, qy, qz, qw, gripper]
    - Trains a rectified flow / velocity predictor.
    - Keeps the Gaussian Splatting auxiliary rendering loss intact.
    """

    def __init__(self,
                 layer: int,
                 coordinate_bounds: list,
                 flow_encoder: VoxelFlowEncoder,
                 camera_names: list,
                 batch_size: int,
                 voxel_size: int,
                 bounds_offset: float,
                 voxel_feature_size: int,
                 image_crop_size: int,
                 lr: float = 5e-4,
                 lr_scheduler: bool = False,
                 training_iterations: int = 100_000,
                 num_warmup_steps: int = 3_000,
                 include_low_dim_state: bool = True,
                 image_resolution: list = None,
                 lambda_weight_l2: float = 1e-6,
                 transform_augmentation: bool = True,
                 transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                 transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                 transform_augmentation_rot_resolution: int = 5,
                 optimizer_type: str = 'lamb',
                 num_devices: int = 1,
                 # flow-matching hyperparameters
                 denoise_timesteps: int = 100,
                 action_dim: int = 8,
                 action_chunk_size: int = 1,
                 lv2_batch_size: int = 1,
                 workspace_bounds: list = None,
                 # loss weights (matching base_denoise_actor defaults)
                 pos_loss_weight: float = 30.0,
                 rot_loss_weight: float = 10.0,
                 grip_loss_weight: float = 1.0,
                 # FIX #2: configurable gripper threshold so inference matches
                 # the BCE training prior (0.5).  Expose via cfg for tuning.
                 grip_threshold: float = 0.5,
                 cfg=None):

        self._layer           = layer
        self._coordinate_bounds = coordinate_bounds
        self._flow_encoder    = flow_encoder
        self._camera_names    = camera_names
        self._batch_size      = batch_size
        self._voxel_size      = voxel_size
        self._bounds_offset   = bounds_offset
        self._voxel_feature_size = voxel_feature_size
        self._image_crop_size = image_crop_size
        self._lr              = lr
        self._lr_scheduler    = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._num_cameras     = len(camera_names)
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz or [0., 0., 0.])
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy or [0., 0., 45.]
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type  = optimizer_type
        self._num_devices     = num_devices
        self._denoise_timesteps = denoise_timesteps
        self._action_dim       = action_dim        # GT replay dim: 8 (3+4quat+1)
        self._action_chunk_size = action_chunk_size  # T: 1=single step, >1=chunk
        self._lv2_batch_size   = lv2_batch_size      # inner loss re-sampling iterations
        self._pos_loss_weight  = pos_loss_weight
        self._rot_loss_weight  = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._grip_threshold   = grip_threshold

        # Workspace normalisation buffers (mirrors base_denoise_actor)
        _wb = workspace_bounds or [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        self._ws_min = torch.tensor(_wb[:3], dtype=torch.float32)
        self._ws_max = torch.tensor(_wb[3:], dtype=torch.float32)

        self.cfg = cfg
        self.use_neural_rendering = cfg.use_neural_rendering
        cprint(f"[ManiFlowBCAgent] use_neural_rendering: {self.use_neural_rendering}", "cyan")

        # Flow-matching schedulers
        self._pos_scheduler = RFScheduler(noise_sampler="logit_normal",
                                          noise_sampler_config={"mean": 0.0, "std": 1.5})
        self._rot_scheduler = RFScheduler(noise_sampler="logit_normal",
                                          noise_sampler_config={"mean": 0.0, "std": 1.5})

        self._name = NAME + '_layer' + str(self._layer)

        if self.use_neural_rendering:
            cprint(f"[ManiFlowBCAgent] nerf weight: {cfg.neural_renderer.lambda_nerf}", "red")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, training: bool, device: torch.device = None,
              use_ddp: bool = True, fabric: Fabric = None):
        self._training = training
        # `device or torch.device('cpu')` is WRONG when device=0 (int from
        # fabric.global_rank): bool(0)==False → evaluates to cpu.
        # Explicitly map: None→cpu, int→cuda:N, else pass through.
        if device is None:
            self._device = torch.device('cpu')
        elif isinstance(device, int):
            self._device = torch.device(f'cuda:{device}')
        else:
            self._device = torch.device(device)
        # Pin CUDA context to the correct GPU (per-process), matching
        # the GitHub forum recommendation for diff_gaussian_rasterization.
        if self._device.type == 'cuda':
            torch.cuda.set_device(self._device)
        device = self._device

        self._voxelizer = VoxelGrid(
            coord_bounds=self._coordinate_bounds.cpu()
            if isinstance(self._coordinate_bounds, torch.Tensor)
            else self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )

        self._q = QFunctionFlow(
            self._flow_encoder,
            self._voxelizer,
            self._bounds_offset,
            device, training, use_ddp, self.cfg, fabric=fabric,
        ).to(device).train(training)

        grid_for_crop = torch.arange(
            0, self._image_crop_size, device=device
        ).unsqueeze(0).repeat(self._image_crop_size, 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat(
            [grid_for_crop.transpose(1, 0), grid_for_crop], dim=2
        ).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(
            self._coordinate_bounds, device=device
        ).unsqueeze(0)

        if training:
            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999), adam=False,
                )
            else:
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            self._optimizer = fabric.setup_optimizers(self._optimizer)

            if self._lr_scheduler:
                self._scheduler = (
                    transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                        self._optimizer,
                        num_warmup_steps=self._num_warmup_steps,
                        num_training_steps=self._training_iterations,
                        # FIX #14: a single cosine decay (num_cycles=1) prevents
                        # the 20+ hard restarts that reset Adam momentum and cause
                        # repeated loss spikes throughout training.
                        num_cycles=1,
                    )
                )
            logging.info('# ManiFlow Params: %d M' % (
                sum(p.numel() for n, p in self._q.named_parameters()
                    if p.requires_grad and 'clip' not in n) / 1e6
            ))
        else:
            for param in self._q.parameters():
                param.requires_grad = False
            self.language_model = create_language_model(self.cfg.language_model,
                                                        device=self._device)
            self._voxelizer.to(device)
            self._q.to(device)

        # Initialise summary dicts so update_summaries / update_wandb_summaries
        # are safe to call before the first update() (e.g. at step 0).
        self._summaries        = {}
        self._wandb_summaries  = {}
        self._crop_summary     = []

        # FIX #15: EMA of model weights — significantly improves eval success
        # rate with zero additional training cost.  EMA decay=0.999 matches
        # the 3D FlowMatch Actor reference.  Only active during training; the
        # EMA shadow copy is swapped in for inference via _use_ema_weights().
        if training:
            self._ema_decay = getattr(self.cfg, 'ema_decay', 0.999)
            self._ema_weights: dict = {}   # name → EMA parameter tensor (CPU)
            self._ema_step = 0
            cprint(f"[ManiFlowBCAgent] EMA enabled (decay={self._ema_decay})", "cyan")
        else:
            self._ema_decay   = None
            self._ema_weights = {}
            self._ema_step    = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def _encoder(self) -> VoxelFlowEncoder:
        """Unwrapped VoxelFlowEncoder (strips Fabric/DDP .module if present)."""
        return getattr(self._q._flow_encoder, 'module', self._q._flow_encoder)

    # ------------------------------------------------------------------
    # EMA helpers (FIX #15)
    # ------------------------------------------------------------------
    def _update_ema(self):
        """Update the EMA shadow weights from the current live parameters."""
        decay = self._ema_decay
        for name, param in self._q.named_parameters():
            if not param.requires_grad:
                continue
            data = param.data.detach().cpu()
            if name not in self._ema_weights:
                self._ema_weights[name] = data.clone()
            else:
                self._ema_weights[name].mul_(decay).add_(data, alpha=1.0 - decay)
        self._ema_step += 1

    def _use_ema_weights(self):
        """
        Context manager: swap EMA weights into the live model for inference,
        then restore the original weights on exit.

        Usage::
            with self._use_ema_weights():
                actions = self._denoise_action(context)
        """
        @contextlib.contextmanager
        def _ctx():
            if not self._ema_weights:
                yield
                return
            # Save originals and swap in EMA
            originals = {}
            for name, param in self._q.named_parameters():
                if name in self._ema_weights:
                    originals[name] = param.data.clone()
                    param.data.copy_(
                        self._ema_weights[name].to(param.device)
                    )
            try:
                yield
            finally:
                # Restore originals
                for name, param in self._q.named_parameters():
                    if name in originals:
                        param.data.copy_(originals[name])

        return _ctx()

    # ------------------------------------------------------------------
    def _preprocess_inputs(self, replay_sample, sample_id=None):
        obs, depths, pcds, exs, ins = [], [], [], [], []
        self._crop_summary = []
        for n in self._camera_names:    # default: [front, left_shoulder, right_shoulder, wrist] or [front]
            if sample_id is not None:   # default: None (training: full batch), int (inference: single sample)
                sl = slice(sample_id, sample_id + 1)
                rgb   = replay_sample['%s_rgb' % n][sl]
                depth = replay_sample['%s_depth' % n][sl]
                pcd   = replay_sample['%s_point_cloud' % n][sl]
                extin = replay_sample['%s_camera_extrinsics' % n][sl]
                intin = replay_sample['%s_camera_intrinsics' % n][sl]
            else:
                rgb   = replay_sample['%s_rgb' % n]
                depth = replay_sample['%s_depth' % n]
                pcd   = replay_sample['%s_point_cloud' % n]
                extin = replay_sample['%s_camera_extrinsics' % n]
                intin = replay_sample['%s_camera_intrinsics' % n]
            obs.append([rgb, pcd])
            depths.append(depth)
            pcds.append(pcd)
            exs.append(extin)
            ins.append(intin)
        return obs, depths, pcds, exs, ins

    def _act_preprocess_inputs(self, observation):
        obs, depths, pcds, exs, ins = [], [], [], [], []
        for n in self._camera_names:
            obs.append([observation['%s_rgb' % n],
                        observation['%s_point_cloud' % n]])
            depths.append(observation['%s_depth' % n])
            pcds.append(observation['%s_point_cloud' % n])
            exs.append(observation['%s_camera_extrinsics' % n].squeeze(0))
            ins.append(observation['%s_camera_intrinsics' % n].squeeze(0))
        return obs, depths, pcds, exs, ins

    # ------------------------------------------------------------------
    # Workspace normalisation helpers (mirrors base_denoise_actor exactly)
    # ------------------------------------------------------------------
    def _normalize_pos(self, signal: torch.Tensor) -> torch.Tensor:
        """Normalise first 3 dims of signal to [-1, 1] using workspace bounds."""
        _min = self._ws_min.to(signal.device)
        _max = self._ws_max.to(signal.device)
        out = signal.clone()
        out[..., :3] = (signal[..., :3] - _min) / (_max - _min) * 2.0 - 1.0
        return out

    def _unnormalize_pos(self, signal: torch.Tensor) -> torch.Tensor:
        """Inverse of _normalize_pos: maps first 3 dims from [-1,1] to world coords."""
        _min = self._ws_min.to(signal.device)
        _max = self._ws_max.to(signal.device)
        out = signal.clone()
        out[..., :3] = (signal[..., :3] + 1.0) / 2.0 * (_max - _min) + _min
        return out

    def _convert_rot(self, signal: torch.Tensor) -> torch.Tensor:
        """Convert quaternion xyzw at [..., 3:7] to ortho6D, return (..., 9)."""
        rot = quat_xyzw_to_ortho6d(
            signal[..., 3:7].reshape(-1, 4)
        ).reshape(*signal.shape[:-1], 6)
        return torch.cat([signal[..., :3], rot], dim=-1)   # (..., 9)

    def _unconvert_rot(self, signal: torch.Tensor) -> torch.Tensor:
        """Convert ortho6D at [..., 3:9] back to quaternion xyzw, return (..., 7)."""
        quat = ortho6d_to_quat_xyzw(
            signal[..., 3:9].reshape(-1, 6)
        ).reshape(*signal.shape[:-1], 4)
        return torch.cat([signal[..., :3], quat], dim=-1)  # (..., 7)

    # ------------------------------------------------------------------
    # Flow-matching loss computation
    # ------------------------------------------------------------------
    def _compute_flow_loss(self, gt_trajectory: torch.Tensor,
                           context) -> torch.Tensor:
        """
        Rectified-flow loss — mirrors ``base_denoise_actor.compute_loss`` exactly.

        Input convention (matches 3D FlowMatch Actor)
        ----------------------------------------------
        gt_trajectory : (B, T, nhand, 8)
            Last dim = [x, y, z, qx, qy, qz, qw, gripper_open]  (xyzw format)
            T    = action_chunk_size  (1 = single-step, >1 = chunk)
            nhand = 1 for single-arm tasks

        Steps
        -----
        1. Separate gripper openness: gt_openess = gt_trajectory[..., -1:]
        2. Strip gripper:             gt_trajectory = gt_trajectory[..., :-1]   (B,T,nhand,7)
        3. Normalise positions to [-1,1]
        4. Convert rotation: quat_xyzw → ortho6D  → (B, T*nhand, 9) → (B,T,nhand,9)
        5. lv2_batch_size inner loop: sample noise + timestep, add noise, predict, loss
        """
        device = gt_trajectory.device
        B, T, nhand, _ = gt_trajectory.shape

        # 1. Separate gripper
        gt_openess    = gt_trajectory[..., -1:]          # (B, T, nhand, 1)
        gt_trajectory = gt_trajectory[..., :-1]          # (B, T, nhand, 7)

        # 2. Normalise positions
        gt_trajectory = self._normalize_pos(gt_trajectory)   # (B, T, nhand, 7)

        # 3. Convert rotation: flatten nhand into batch for the rotation converter,
        #    then unflatten — matches original's flatten(1,2)/unflatten(1,(traj_len,nhand))
        gt_trajectory = self._convert_rot(
            gt_trajectory.flatten(1, 2)          # (B, T*nhand, 7)
        ).unflatten(1, (T, nhand))               # (B, T, nhand, 9)

        total_loss = 0.
        # Keep track of last-iteration quantities for logging.
        denoise_target = None
        pred_list      = None
        for _ in range(self._lv2_batch_size):
            # FIX #3: draw *independent* noise tensors for position and rotation.
            # Using the same noise array creates a correlated gradient signal
            # because the pos/rot subspaces have different geometric scales.
            noise_pos = torch.randn(gt_trajectory[..., :3].shape, device=device)  # (B, T, nhand, 3)
            noise_rot = torch.randn(gt_trajectory[..., 3:].shape, device=device)  # (B, T, nhand, 6)

            # Use a SINGLE shared timestep for both pos and rot.
            # Rationale: predict_velocity receives ONE scalar t as AdaLN
            # conditioning.  If t_pos ≠ t_rot, the model is told t_pos while
            # seeing rotation tokens at a different noise level t_rot — it
            # cannot correctly attribute how much denoising work remains for
            # rotation.  At inference both modalities always share the same t
            # (both start from randn at t=1 and are stepped with the same idx),
            # so independent training timesteps create a train/inference
            # distribution mismatch that prevents the rotation loss from
            # converging and degrades rotation accuracy.  A single shared t
            # eliminates the mismatch.
            t = self._pos_scheduler.sample_noise_step(B, device)  # (B,) shared

            noisy_pos = self._pos_scheduler.add_noise(
                gt_trajectory[..., :3], noise_pos, t)
            noisy_rot = self._rot_scheduler.add_noise(
                gt_trajectory[..., 3:], noise_rot, t)
            # Flatten nhand into T for the transformer head: (B, T*nhand, 9)
            noisy_traj = torch.cat([noisy_pos, noisy_rot], dim=-1).flatten(1, 2)

            pred_list = self._encoder.predict_velocity(noisy_traj, t, context)

            # FIX #4: compute denoise_target *inside* the loop from the
            # current iteration's noise draw.  Computing it once outside means
            # iterations 1..N-1 receive the target from the Nth noise draw,
            # giving them a wrong supervision signal.
            # RF target: v* = noise − x_clean  (separate targets per modality)
            target_pos = self._pos_scheduler.prepare_target(
                noise_pos, gt_trajectory[..., :3])   # (B, T, nhand, 3)
            target_rot = self._rot_scheduler.prepare_target(
                noise_rot, gt_trajectory[..., 3:])   # (B, T, nhand, 6)
            denoise_target = torch.cat([target_pos, target_rot], dim=-1)  # (B, T, nhand, 9)

            for layer_pred in pred_list:
                # Unflatten back to (B, T, nhand, 10)
                lp = layer_pred.unflatten(1, (T, nhand))
                pos     = lp[..., :3]    # (B, T, nhand, 3)
                rot     = lp[..., 3:9]   # (B, T, nhand, 6)
                openess = lp[..., 9:10]  # (B, T, nhand, 1)

                iter_loss = (
                    self._pos_loss_weight  * F.l1_loss(pos, denoise_target[..., :3])
                    + self._rot_loss_weight  * F.l1_loss(rot, denoise_target[..., 3:])
                    + self._grip_loss_weight * F.binary_cross_entropy_with_logits(
                        openess, gt_openess)
                )
                total_loss = total_loss + iter_loss

        total_loss = total_loss / self._lv2_batch_size

        # Individual losses from last inner iteration / last layer for logging
        lp_last = pred_list[-1].unflatten(1, (T, nhand))
        pos_loss  = F.l1_loss(lp_last[..., :3],   denoise_target[..., :3])
        rot_loss  = F.l1_loss(lp_last[..., 3:9],  denoise_target[..., 3:])
        grip_loss = F.binary_cross_entropy_with_logits(lp_last[..., 9:10], gt_openess)
        return total_loss, pos_loss, rot_loss, grip_loss

    # ------------------------------------------------------------------
    # Denoising loop (inference) — matches base_denoise_actor.conditional_sample
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _denoise_action(self, context,
                        num_steps: int = None) -> torch.Tensor:
        """
        Run the full denoising loop to obtain a clean action chunk.

        Mirrors ``base_denoise_actor.conditional_sample``.

        Parameters
        ----------
        context : dict with keys 'voxel_tokens', 'voxel_pos', 'lang_tokens',
                  'proprio_feats' (all tensors).

        Returns
        -------
        actions : (B, T, nhand, 8)  [x,y,z, qx,qy,qz,qw, gripper]
                  T     = self._action_chunk_size
                  nhand = 1 (single-arm)
        """
        _ref   = context['voxel_tokens']
        B      = _ref.shape[0]
        device = _ref.device
        T      = self._action_chunk_size
        nhand  = 1
        steps  = num_steps or self._denoise_timesteps

        self._pos_scheduler.set_timesteps(steps, device=device)
        self._rot_scheduler.set_timesteps(steps, device=device)

        # Start from pure noise (B, T*nhand, 9): normalised pos + rot6d
        # The transformer head works on the flat (T*nhand) token sequence.
        trajectory = torch.randn(B, T * nhand, 9, device=device)

        out = None
        # Single shared timestep — matches training where both pos and rot
        # use the same t.  Both schedulers have identical parameters so
        # iterating over pos_scheduler's timesteps is sufficient.
        for idx, t_val in enumerate(self._pos_scheduler.timesteps):
            t = t_val.expand(B)
            out_list = self._encoder.predict_velocity(trajectory, t, context)
            out = out_list[-1]   # (B, T*nhand, 10)

            pos = self._pos_scheduler.step(
                out[..., :3], idx, trajectory[..., :3]).prev_sample
            rot = self._rot_scheduler.step(
                out[..., 3:9], idx, trajectory[..., 3:9]).prev_sample
            trajectory = torch.cat([pos, rot], dim=-1)   # (B, T*nhand, 9)

        # Unflatten nhand dimension, then post-process
        trajectory = trajectory.unflatten(1, (T, nhand))  # (B, T, nhand, 9)
        out = out.unflatten(1, (T, nhand))                 # (B, T, nhand, 10)

        # Denormalise position then unconvert rotation back to quaternion
        trajectory = self._unnormalize_pos(trajectory)    # pos → world coords
        trajectory = self._unconvert_rot(                 # (B, T, nhand, 7)
            trajectory.flatten(1, 2)
        ).unflatten(1, (T, nhand))
        grip_prob  = out[..., 9:10].sigmoid()             # (B, T, nhand, 1)
        # FIX #2: use the configurable threshold (default 0.5, matching
        # the BCE training prior) instead of the hard-coded 0.6.
        # Tune via cfg.method.grip_threshold if tasks need more confidence.
        grip_binary = (grip_prob > self._grip_threshold).float()
        actions = torch.cat([trajectory, grip_binary], dim=-1)  # (B, T, nhand, 8)
        return actions

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------
    def update(self, step: int, replay_sample: dict,
               fabric: Fabric) -> dict:
        """
        Update the agent's parameters based on the replay sample.
        """
        
        device = self._device
        rank   = fabric.global_rank   # int (0 on main process)

        action_gt           = replay_sample['action'].float().to(device)   # (B, 8)
        action_gripper_pose = replay_sample['gripper_pose'].float().to(device)   # (B, 7)
        lang_goal_emb       = replay_sample['lang_goal_emb'].float().to(device)
        lang_token_embs     = replay_sample['lang_token_embs'].float().to(device)
        prev_layer_voxel_grid = replay_sample.get('prev_layer_voxel_grid', None)
        prev_layer_bounds     = replay_sample.get('prev_layer_bounds', None)
        lang_goal             = replay_sample['lang_goal']

        obs, depth, pcd, extrinsics, intrinsics = self._preprocess_inputs(replay_sample)
        # Move all per-camera tensors to the training device so that SE3
        # augmentation and the forward pass see a consistent device.
        pcd        = [p.to(device) for p in pcd]
        extrinsics = [e.to(device) for e in extrinsics]
        intrinsics = [i.to(device) for i in intrinsics]
        depth      = [d.to(device) for d in depth]
        obs        = [[rgb.to(device), pc.to(device)] for rgb, pc in obs]
        bs = pcd[0].shape[0]

        # ---- Load NeRF multi-view data (same as manigaussian_bc) ----------
        # Skip entirely when use_neural_rendering=False to avoid unnecessary
        # file I/O and potential crashes if NeRF paths were not stored.
        if self.use_neural_rendering:
            nerf_multi_view_rgb_path   = replay_sample['nerf_multi_view_rgb']
            nerf_multi_view_depth_path = replay_sample['nerf_multi_view_depth']
            nerf_multi_view_camera_path = replay_sample['nerf_multi_view_camera']
            nerf_next_multi_view_rgb_path   = replay_sample['nerf_next_multi_view_rgb']
            nerf_next_multi_view_depth_path = replay_sample['nerf_next_multi_view_depth']
            nerf_next_multi_view_camera_path = replay_sample['nerf_next_multi_view_camera']

            if (nerf_multi_view_rgb_path is None
                    or nerf_multi_view_rgb_path[0, 0] is None):
                cprint(nerf_multi_view_rgb_path, 'red')
                cprint(replay_sample['indices'], 'red')
                nerf_target_rgb = None
                nerf_target_camera_extrinsic = None
                cprint('Warning: NeRF multi-view RGB paths are None. Skipping neural rendering loss for this batch.', 'yellow')
                raise ValueError('nerf_multi_view_rgb_path is None')

            num_view         = nerf_multi_view_rgb_path.shape[-1]
            num_view_by_user = self.cfg.num_view_for_nerf
            assert num_view_by_user <= num_view, f'num_view_by_user {num_view_by_user} should be less than num_view {num_view}'
            interval = num_view // num_view_by_user # Sample views at a fixed interval to ensure coverage of all views across batches
            # Subsample all three path arrays consistently (fixes rgb/depth/camera misalignment)
            nerf_multi_view_rgb_path    = nerf_multi_view_rgb_path[:, ::interval]
            nerf_multi_view_depth_path  = nerf_multi_view_depth_path[:, ::interval]
            nerf_multi_view_camera_path = nerf_multi_view_camera_path[:, ::interval]

            # Sample a random view index for each user in the batch, and index into the subsampled paths
            view_idx = np.random.randint(0, num_view_by_user)
            nerf_multi_view_rgb_path    = nerf_multi_view_rgb_path[:, view_idx]
            nerf_multi_view_depth_path  = nerf_multi_view_depth_path[:, view_idx]
            nerf_multi_view_camera_path = nerf_multi_view_camera_path[:, view_idx]

            next_view_idx = np.random.randint(0, num_view_by_user)
            # Also subsample next-view paths consistently before indexing
            nerf_next_multi_view_rgb_path    = nerf_next_multi_view_rgb_path[:, ::interval]
            nerf_next_multi_view_depth_path  = nerf_next_multi_view_depth_path[:, ::interval]
            nerf_next_multi_view_camera_path = nerf_next_multi_view_camera_path[:, ::interval]
            nerf_next_multi_view_rgb_path    = nerf_next_multi_view_rgb_path[:, next_view_idx]
            nerf_next_multi_view_depth_path  = nerf_next_multi_view_depth_path[:, next_view_idx]
            nerf_next_multi_view_camera_path = nerf_next_multi_view_camera_path[:, next_view_idx]

            nerf_target_rgbs, nerf_target_depths = [], []
            nerf_target_camera_extrinsics, nerf_target_camera_intrinsics = [], []
            nerf_next_target_rgbs, nerf_next_target_depths = [], []
            nerf_next_target_camera_extrinsics, nerf_next_target_camera_intrinsics = [], []

            mask_gt_rgb = self.cfg.neural_renderer.dataset.mask_gt_rgb
            for i in range(bs):
                nerf_target_rgbs.append(_parse_img_file(nerf_multi_view_rgb_path[i],
                                                        mask_gt_rgb=mask_gt_rgb))
                nerf_target_depths.append(_parse_depth_file(nerf_multi_view_depth_path[i]))
                ext, intr, _ = _parse_camera_file(nerf_multi_view_camera_path[i])
                nerf_target_camera_extrinsics.append(ext)
                nerf_target_camera_intrinsics.append(intr)

                nerf_next_target_rgbs.append(_parse_img_file(nerf_next_multi_view_rgb_path[i],
                                                             mask_gt_rgb=mask_gt_rgb))
                nerf_next_target_depths.append(_parse_depth_file(nerf_next_multi_view_depth_path[i]))
                next_ext, next_intr, _ = _parse_camera_file(nerf_next_multi_view_camera_path[i])
                nerf_next_target_camera_extrinsics.append(next_ext)
                nerf_next_target_camera_intrinsics.append(next_intr)

            def _to_t(arr): return torch.from_numpy(np.stack(arr)).float().to(device)
            nerf_target_rgb      = _to_t(nerf_target_rgbs)
            nerf_target_depth    = _to_t(nerf_target_depths)
            nerf_target_cam_ext  = _to_t(nerf_target_camera_extrinsics)
            nerf_target_cam_intr = _to_t(nerf_target_camera_intrinsics)
            nerf_next_target_rgb      = _to_t(nerf_next_target_rgbs)
            nerf_next_target_depth    = _to_t(nerf_next_target_depths)
            nerf_next_target_cam_ext  = _to_t(nerf_next_target_camera_extrinsics)
            nerf_next_target_cam_intr = _to_t(nerf_next_target_camera_intrinsics)
        else:
            # No NeRF: pass None for all nerf args
            nerf_target_rgb = nerf_target_depth = nerf_target_cam_ext = nerf_target_cam_intr = None
            nerf_next_target_rgb = nerf_next_target_depth = nerf_next_target_cam_ext = nerf_next_target_cam_intr = None

        # ---- Scene bounds and optional SE3 augmentation ------------------
        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            cp     = replay_sample['attention_coordinate_layer_%d' % (self._layer - 1)].to(device)
            bounds = torch.cat([cp - self._bounds_offset,
                                cp + self._bounds_offset], dim=1)

        proprio = replay_sample['low_dim_state'].float().to(device) if self._include_low_dim_state else None

        if self._transform_augmentation:
            action_gt, pcd, extrinsics, _ = apply_se3_augmentation_continuous(
                pcd, extrinsics,
                action_gripper_pose,
                action_gt,
                bounds, self._layer,
                self._transform_augmentation_xyz,
                self._transform_augmentation_rpy,
                self._device,
            )

        # ---- Forward pass -------------------------------------------------
        # First, encode scene to get context (no denoising step needed for loss)
        (context, voxel_grid, voxel_feat,
         lang_embedd, rendering_loss_dict) = self._q(
            obs, depth, proprio, pcd,
            extrinsics, intrinsics,
            lang_goal_emb, lang_token_embs,
            bounds, prev_layer_bounds, prev_layer_voxel_grid,
            use_neural_rendering=self.use_neural_rendering,
            nerf_target_rgb=nerf_target_rgb,
            nerf_target_depth=nerf_target_depth,
            nerf_target_pose=nerf_target_cam_ext,
            nerf_target_camera_intrinsic=nerf_target_cam_intr,
            lang_goal=lang_goal,
            nerf_next_target_rgb=nerf_next_target_rgb,
            nerf_next_target_depth=nerf_next_target_depth,
            nerf_next_target_pose=nerf_next_target_cam_ext,
            nerf_next_target_camera_intrinsic=nerf_next_target_cam_intr,
            step=step, action=action_gt,
        )

        # ---- Flow-matching loss ------------------------------------------
        # Reshape (B, 8) → (B, 1, 1, 8) to match the 3D FlowMatch Actor's
        # (B, T, nhand, 8) convention.  T=1 and nhand=1 for keypoint-based
        # single-arm BC; action_chunk_size > 1 would expand T further.
        T = self._action_chunk_size
        nhand = 1
        # Expand single keypoint action to T steps (all identical — the RF
        # loss learns to denoise any position within the trajectory)
        gt_traj = action_gt.unsqueeze(1).unsqueeze(2).expand(
            -1, T, nhand, -1)   # (B, T, nhand, 8)
        flow_loss, pos_loss, rot_loss, grip_loss = self._compute_flow_loss(
            gt_traj, context)

        # ---- Total loss --------------------------------------------------
        if self.use_neural_rendering:
            lambda_nerf = self.cfg.neural_renderer.lambda_nerf
            lambda_bc   = self.cfg.lambda_bc
            total_loss  = lambda_bc * flow_loss + lambda_nerf * rendering_loss_dict['loss']
        else:
            lambda_bc  = self.cfg.lambda_bc
            total_loss = lambda_bc * flow_loss

        if step % 10 == 0 and rank == 0:
            if self.use_neural_rendering:
                psnr          = rendering_loss_dict['psnr']
                loss_rgb      = rendering_loss_dict['loss_rgb']
                loss_embed    = rendering_loss_dict['loss_embed']
                loss_dyna     = rendering_loss_dict.get('loss_dyna', 0.)
                loss_reg      = rendering_loss_dict.get('loss_reg', 0.)
                lambda_nerf   = self.cfg.neural_renderer.lambda_nerf
                cprint(
                    f'total L: {total_loss.item():.4f} | '
                    f'L_flow: {flow_loss.item():.4f} x {lambda_bc:.3f} | '
                    f'L_pos: {pos_loss.item():.4f} | '
                    f'L_rot: {rot_loss.item():.4f} | '
                    f'L_grip: {grip_loss.item():.4f} | '
                    f'L_rgb: {loss_rgb:.4f} | '
                    f'L_embed: {loss_embed:.4f} | '
                    f'L_dyna: {loss_dyna:.4f} | '
                    f'L_reg: {loss_reg:.4f} | '
                    f'psnr: {psnr:.2f}',
                    'green'
                )
                if self.cfg.use_wandb:
                    import wandb
                    wandb.log({
                        'train/total_loss': total_loss.item(),
                        'train/flow_loss': flow_loss.item(),
                        'train/pos_loss': pos_loss.item(),
                        'train/rot_loss': rot_loss.item(),
                        'train/grip_loss': grip_loss.item(),
                        'train/psnr': psnr,
                        'train/rgb_loss': loss_rgb,
                        'train/embed_loss': loss_embed,
                        'train/dyna_loss': loss_dyna,
                        'train/reg_loss': loss_reg,
                    }, step=step)
            else:
                cprint(
                    f'total L: {total_loss.item():.4f} | '
                    f'L_flow: {flow_loss.item():.4f} x {lambda_bc:.3f} | '
                    f'L_pos: {pos_loss.item():.4f} | '
                    f'L_rot: {rot_loss.item():.4f} | '
                    f'L_grip: {grip_loss.item():.4f}',
                    'green'
                )
                if self.cfg.use_wandb:
                    import wandb
                    wandb.log({
                        'train/total_loss': total_loss.item(),
                        'train/flow_loss': flow_loss.item(),
                        'train/pos_loss': pos_loss.item(),
                        'train/rot_loss': rot_loss.item(),
                        'train/grip_loss': grip_loss.item(),
                    }, step=step)

        self._optimizer.zero_grad()
        fabric.backward(total_loss)
        # FIX #6: tighter gradient clipping for the Transformer denoising head.
        # max_norm=10.0 was too permissive — spiky GS-rasterizer backward passes
        # occasionally corrupt the Transformer weights.  1.0 matches the
        # 3D FlowMatch Actor reference and is standard for diffusion Transformers.
        torch.nn.utils.clip_grad_norm_(self._q.parameters(), max_norm=1.0)
        self._optimizer.step()
        # FIX #15: update EMA shadow weights every step (negligible overhead).
        self._update_ema()

        # ---- Optional render preview (same cadence as manigaussian_bc) ---
        render_freq = (self.cfg.neural_renderer.render_freq
                       if self.use_neural_rendering else None)
        to_render   = (
            render_freq is not None
            and step % render_freq == 0
            and nerf_target_cam_ext is not None
        )
        if to_render:
            rgb_render, next_rgb_render, embed_render, gt_embed_render = self._q.render(
                rgb_pcd=obs, proprio=proprio, pcd=pcd,
                lang_goal_emb=lang_goal_emb,
                lang_token_embs=lang_token_embs,
                bounds=bounds, prev_bounds=prev_layer_bounds,
                prev_layer_voxel_grid=prev_layer_voxel_grid,
                tgt_pose=nerf_target_cam_ext,
                tgt_intrinsic=nerf_target_cam_intr,
                nerf_target_rgb=nerf_target_rgb,
                lang_goal=lang_goal,
                nerf_next_target_rgb=nerf_next_target_rgb,
                nerf_next_target_depth=nerf_next_target_depth,
                nerf_next_target_pose=nerf_next_target_cam_ext,
                nerf_next_target_camera_intrinsic=nerf_next_target_cam_intr,
                step=step, action=action_gt,
            )

            # NOTE: [h, w, 3] — take first batch element
            rgb_gt       = nerf_target_rgb[0]
            rgb_render   = rgb_render[0]
            psnr_val     = _PSNR_torch(rgb_render, rgb_gt)
            psnr_dyna    = None
            if next_rgb_render is not None:
                next_rgb_gt     = nerf_next_target_rgb[0]
                next_rgb_render = next_rgb_render[0]
                psnr_dyna       = _PSNR_torch(next_rgb_render, next_rgb_gt)

            if rank == 0:
                os.makedirs('recon', exist_ok=True)
                rgb_src = obs[0][0][0].permute(1, 2, 0).cpu() / 2 + 0.5

                fig, axs = plt.subplots(1, 7, figsize=(21, 3))
                # 0: input view
                axs[0].imshow(rgb_src.numpy())
                axs[0].set_title('src')
                # 1: target GT
                axs[1].imshow(rgb_gt.cpu().numpy())
                axs[1].set_title('tgt')
                # 2: predicted RGB
                axs[2].imshow(rgb_render.cpu().numpy())
                axs[2].set_title(f'psnr={psnr_val:.2f}')
                # 3: predicted feature/embed (normalised for display)
                if embed_render is not None:
                    embed_vis = visualize_feature_map_by_normalization(
                        embed_render.permute(0, 3, 1, 2))
                    axs[3].imshow(embed_vis)
                axs[3].set_title('embed pred')
                # 4: GT embed (only available when a foundation model is configured)
                if gt_embed_render is not None:
                    gt_embed_vis = visualize_feature_map_by_normalization(gt_embed_render)
                    axs[4].imshow(gt_embed_vis)
                axs[4].set_title('embed gt')
                # 5: next-frame prediction (dynamic field)
                if next_rgb_render is not None:
                    axs[5].imshow(next_rgb_render.cpu().numpy())
                    axs[5].set_title(f'next psnr={psnr_dyna:.2f}')
                # 6: next-frame GT
                if next_rgb_render is not None:
                    axs[6].imshow(next_rgb_gt.cpu().numpy())
                axs[6].set_title('next tgt')
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()

                if self.cfg.use_wandb:
                    import wandb
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    image = Image.open(buf)
                    image.load()  # force PIL to read before buf closes
                    wandb.log({'eval/recon_img': wandb.Image(image)}, step=step)
                    buf.close()
                    workdir = os.getcwd()
                    cprint(f'Saved {workdir}/recon/{step}_rgb.png to wandb', 'cyan')
                else:
                    plt.savefig(f'recon/{step}_rgb.png')
                    workdir = os.getcwd()
                    cprint(f'Saved {workdir}/recon/{step}_rgb.png locally', 'cyan')
                plt.close()

        if self._lr_scheduler:
            self._scheduler.step()

        self._summaries = {
            'losses/total_loss': total_loss.item(),
            'losses/flow_loss':  flow_loss.item(),
            'losses/pos_loss':   pos_loss.item(),
            'losses/rot_loss':   rot_loss.item(),
            'losses/grip_loss':  grip_loss.item(),
        }
        if self.use_neural_rendering and rendering_loss_dict:
            self._summaries['losses/rgb_loss']   = rendering_loss_dict['loss_rgb']
            self._summaries['losses/embed_loss'] = rendering_loss_dict['loss_embed']
            self._summaries['losses/dyna_loss']  = rendering_loss_dict.get('loss_dyna', 0.)
            self._summaries['losses/reg_loss']   = rendering_loss_dict.get('loss_reg',  0.)
            self._summaries['psnr']              = rendering_loss_dict['psnr']

        self._wandb_summaries = dict(self._summaries)

        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]
        if prev_layer_bounds is None:
            prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        return {
            'total_loss':             total_loss,
            'prev_layer_voxel_grid':  prev_layer_voxel_grid,
            'prev_layer_bounds':      prev_layer_bounds,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def act(self, step: int, observation: dict,
            deterministic: bool = False) -> ActResult:
        bounds            = self._coordinate_bounds
        prev_layer_voxel_grid = observation.get('prev_layer_voxel_grid', None)
        prev_layer_bounds     = observation.get('prev_layer_bounds', None)
        lang_goal             = observation['lang_goal']

        with torch.no_grad():
            lang_goal_emb, lang_token_embs = self.language_model.extract(lang_goal)

        proprio = observation['low_dim_state'] if self._include_low_dim_state else None

        obs, depth, pcd, extrinsics, intrinsics = self._act_preprocess_inputs(observation)

        # Move to device
        # From YARR, each observation tensor arrives as (1, timesteps, C, H, W).
        # We take [:, 0] to strip the timestep dim (dim-1), giving (1, C, H, W) —
        # the same (B, C, H, W) shape that encode_scene expects.
        # NOTE: all four per-camera tensors (rgb, pcd, depth) must use [:, 0]
        # to strip the *timestep* dimension.  Using [0] instead would strip the
        # *batch* dimension, giving (T, C, H, W) — wrong when T > 1.
        obs    = [[o[0][:, 0].to(self._device), o[1][:, 0].to(self._device)] for o in obs]
        depth  = [d[:, 0].to(self._device) for d in depth]
        # observation['low_dim_state'] arrives from YARR as (1, timesteps, 4).
        # Strip the timestep dimension (dim-1) to match the training shape (B, 4),
        # mirroring PreprocessAgent._strip_time's `v[:, 0]` in the replay path.
        proprio = proprio[:, 0, :].to(self._device) if proprio is not None else None
        # pcd arrives as (1, timesteps, 3, H, W) — use [:, 0] like obs/depth.
        pcd    = [p[:, 0].to(self._device) for p in pcd]
        extrinsics = [e.to(self._device) for e in extrinsics]
        intrinsics = [i.to(self._device) for i in intrinsics]
        lang_goal_emb   = lang_goal_emb.to(self._device)
        lang_token_embs = lang_token_embs.to(self._device)
        bounds          = torch.as_tensor(bounds, device=self._device)

        # prev_layer_voxel_grid and prev_layer_bounds are lists of tensors
        # (appended by act() across layers), not bare tensors — use list comp.
        prev_layer_voxel_grid = (
            [v.to(self._device) for v in prev_layer_voxel_grid]
            if prev_layer_voxel_grid is not None else None
        )
        prev_layer_bounds = (
            [b.to(self._device) for b in prev_layer_bounds]
            if prev_layer_bounds is not None else None
        )

        # Run the entire inference forward pass — scene encoding AND denoising —
        # under the same EMA weight snapshot so that the 3D-CNN / VL-attention
        # context and the Transformer head are always evaluated with the same
        # parameters they were trained together on.
        # _use_ema_weights() is a no-op when _ema_weights is empty (eval-only
        # load without a training run) or when training hasn't started yet.
        with torch.no_grad(), self._use_ema_weights():
            voxel_grid, voxel_feat, lang_embedd, context, bounds = self._q.encode_scene(
                obs, depth, proprio, pcd,
                extrinsics, intrinsics,
                lang_goal_emb, lang_token_embs,
                bounds, prev_layer_bounds, prev_layer_voxel_grid,
            )
            actions = self._denoise_action(context)   # (1, T, nhand=1, 8)

        # Take first step, first hand for execution
        action_np = actions[0, 0, 0].cpu().numpy()   # [x, y, z, qx, qy, qz, qw, gripper]

        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]
        if prev_layer_bounds is None:
            prev_layer_bounds = [bounds]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        # rollout_generator.py calls np.array(v) on EVERY value in
        # observation_elements.  Only scalar/1-D numpy-friendly values belong
        # here.  Large tensors (voxel_grid, bounds) and tensor lists go into
        # `info`, which YARR never tries to numpy-ify.
        observation_elements = {
            'attention_coordinate': actions[0:1, 0, 0, :3].cpu(),
        }
        info = {
            'voxel_grid_depth%d' % self._layer: voxel_grid.cpu(),
            # Pass-through for the stack agent's next-layer call.
            # Kept as CPU tensors; act() moves them back with .to(device).
            'prev_layer_voxel_grid': [v.cpu() for v in prev_layer_voxel_grid],
            'prev_layer_bounds':     [b.cpu() for b in prev_layer_bounds],
        }
        self._act_voxel_grid = voxel_grid[0]
        return ActResult(
            action_np,
            observation_elements=observation_elements,
            info=info,
        )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def update_summaries(self) -> List[Summary]:
        summaries = []
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (self._name, n), v))
        for (name, crop) in self._crop_summary:
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0
            summaries.append(ImageSummary('%s/crops/%s' % (self._name, name), crops))
        for tag, param in self._q.named_parameters():
            if param.grad is None:
                continue
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (self._name, tag), param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (self._name, tag), param.data))
        return summaries

    def update_wandb_summaries(self):
        return dict(self._wandb_summaries)

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        device = self._device
        if isinstance(device, int):
            device = torch.device('cuda:%d' % device)
        weight_file = os.path.join(savedir, '%s.pt' % self._name)
        state_dict  = torch.load(weight_file, map_location=device)
        merged      = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_flow_encoder.module', '_flow_encoder')
                k = k.replace('_neural_renderer.module', '_neural_renderer')
            # Always skip voxelizer buffers: they are batch-size-dependent
            # (e.g. _ones_max_coords, _flat_output, _tiled_batch_indices,
            # _index_grid) and will mismatch when loading a training
            # checkpoint (batch_size=8) into an eval model (batch_size=1).
            if '_voxelizer' in k:
                continue
            if k in merged:
                merged[k] = v
            else:
                logging.warning(f"key {k} in checkpoint but not in model.")
        msg = self._q.load_state_dict(merged, strict=False)
        if msg.missing_keys:
            cprint(f"missing keys: {msg.missing_keys}", 'yellow')
        cprint(f"Loaded weights from {weight_file}", 'cyan')

        # FIX #15: load EMA shadow weights if available (written by save_weights).
        ema_file = os.path.join(savedir, '%s_ema.pt' % self._name)
        if os.path.isfile(ema_file):
            self._ema_weights = torch.load(ema_file, map_location='cpu')
            cprint(f"Loaded EMA weights from {ema_file}", 'cyan')

    def save_weights(self, savedir: str):
        # Exclude batch-size-dependent voxelizer buffers so checkpoints are
        # portable across different batch sizes (train vs eval).
        state_dict = {k: v for k, v in self._q.state_dict().items()
                      if '_voxelizer' not in k}
        torch.save(
            state_dict,
            os.path.join(savedir, '%s.pt' % self._name)
        )
        # FIX #15: persist EMA shadow weights alongside the live checkpoint.
        if self._ema_weights:
            torch.save(
                self._ema_weights,
                os.path.join(savedir, '%s_ema.pt' % self._name)
            )

    def load_clip(self): pass
    def unload_clip(self): pass

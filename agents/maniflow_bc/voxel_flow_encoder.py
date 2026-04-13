"""
VoxelFlowEncoder
================
Keeps the 3D-CNN voxel encoding pipeline from PerceiverVoxelLangEncoder
(needed to supply `voxel_grid_feature` and `lang_embedd` to the Gaussian
Splatting renderer) but replaces the Perceiver Transformer and discrete
classification heads with a **Transformer denoising head** that exactly
mirrors the architecture of 3D FlowMatch Actor (`base_denoise_actor.py`)
while adapting the scene representation to use the 3D-CNN voxel feature map.

Voxel-to-token adaptation
--------------------------
3D FlowMatch Actor conditions the Transformer on a FPS-subsampled point cloud
of 3D tokens.  ManiFlow has no point cloud at the Transformer level; instead
it has a 3D-CNN voxel feature volume (B, C, V, V, V).  The adaptation is:

  1.  The voxel volume is spatially downsampled via AdaptiveAvgPool3d and then
      flattened into N = ds^3 tokens of size C (ds = V // voxel_token_downsample).
  2.  Each token is assigned a 3D world coordinate computed from its voxel
      index and the workspace bounding box — used for 3D RoPE exactly as
      point-cloud positions are in the original.
  3.  Language tokens are a separate sequence, cross-attended to by the action
      query, exactly as in the original TransformerHead.
  4.  The denoising head is the same TransformerHead structure:
        trajectory query -> lang cross-attn -> scene cross-attn (AdaLN)
        -> joint self-attn (AdaLN) -> pos / rot / openness output heads.

The encoder still returns (voxel_grid_feature, lang_embedd) so that
NeuralRenderer can be used unchanged.

Trajectory output
-----------------
The TransformerHead operates on a trajectory of T action tokens:
    noisy_trajectory : (B, T, 9)   — [x,y,z, r1..r6]  (normalized pos + 6D rot)
    timestep         : (B,)
    → predicted velocity : list of [(B, T, 10)]  — [pos_vel, rot_vel, grip_logit]

Setting T=1 recovers single-step prediction.
T is controlled by `action_chunk_size` in the config.

Action space (GT/output)
------------------------
8-DoF: [x, y, z, qx, qy, qz, qw, gripper_open]
Internal (flow space): 9-DoF trajectory token [x_norm,y_norm,z_norm, r1..r6]
  + 1 grip logit predicted separately via BCE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from helpers.network_utils import (
    DenseBlock,
    SpatialSoftmax3D,
    MultiLayer3DEncoderShallow,
)
from agents.maniflow_bc.position_encodings import (
    SinusoidalPosEmb,
    RotaryPositionEncoding3D,
    PositionEmbeddingLearnedMLP,
)
from agents.maniflow_bc.transformer_layers import AttentionModule


# ---------------------------------------------------------------------------
# TransformerHead  (exact port of base_denoise_actor.TransformerHead + 3D RoPE)
# ---------------------------------------------------------------------------

class TransformerHead(nn.Module):
    """
    Denoising head matching 3D FlowMatch Actor's TransformerHead exactly,
    adapted to use voxel tokens as scene context instead of a point cloud.

    Input trajectory shape:  (B, T, 9)  — [x_norm, y_norm, z_norm, r1..r6]
                                           (normalised pos + 6D ortho rotation)
    Output (per forward call): list of length 1 containing (B, T, 10)
                                — [pos_vel(3), rot_vel(6), grip_logit(1)]

    T = action_chunk_size (1 = single-step, >1 = action chunk / trajectory).

    Parameters
    ----------
    embedding_dim : int
        Feature dimension throughout the Transformer (default 120).
    num_attn_heads : int
        Number of attention heads.  Must divide embedding_dim evenly.
    num_shared_attn_layers : int
        Number of AdaLN self-attention layers in the shared trunk (default 4).
    rot_dim : int
        Rotation representation dimension inside the head (default 6, ortho6D).
    """

    def __init__(
        self,
        embedding_dim: int = 120,
        num_attn_heads: int = 8,
        num_shared_attn_layers: int = 4,
        rot_dim: int = 6,           # 6D ortho rotation (Zhou et al. CVPR 2019)
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        # traj token dim = 3 pos + rot_dim  (grip is NOT part of noisy traj)
        traj_input_dim = 3 + rot_dim

        # ---- Time embedding (mirrors base_denoise_actor exactly) ---------- #
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # ---- Proprioception encoder --------------------------------------- #
        # In base_denoise_actor: Linear(embedding_dim * nhist) -> ReLU -> Linear
        # We keep nhist=1 so it's just embedding_dim -> embedding_dim.
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # ---- Trajectory encoder ------------------------------------------ #
        # Matches base_denoise_actor.traj_encoder: Linear(9 or 6) -> embedding_dim
        self.traj_encoder = nn.Linear(traj_input_dim, embedding_dim)

        # Step positional embedding for trajectory tokens
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # ---- 3D RoPE for voxel positions --------------------------------- #
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # ---- Cross-attention: trajectory query -> language --------------- #
        self.traj_lang_attention = AttentionModule(
            num_layers=1,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=False,
            is_self=False,
        )

        # ---- Cross-attention: trajectory query -> voxel scene tokens ----- #
        # dim_fw=4*embedding_dim matches the standard Transformer FFW ratio
        # and aligns with vl_attention / traj_lang_attention in the reference.
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=True,
            use_adaln=True,
            is_self=False,
        )

        # ---- Shared self-attention trunk --------------------------------- #
        self.self_attn = AttentionModule(
            num_layers=num_shared_attn_layers,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=True,
            use_adaln=True,
            is_self=True,
        )

        # ---- Position head (xyz velocity → 3) ---------------------------- #
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=True,
            use_adaln=True,
            is_self=True,
        )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3),
        )

        # ---- Rotation head (6D ortho velocity → 6) ----------------------- #
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=True,
            use_adaln=True,
            is_self=True,
        )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rot_dim),
        )

        # ---- Gripper head (logit → 1) ------------------------------------ #
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    # ---------------------------------------------------------------------- #

    def encode_denoising_timestep(
        self,
        timestep:      torch.Tensor,   # (B,)
        proprio_feats: torch.Tensor,   # (B, embedding_dim) or None
    ) -> torch.Tensor:
        """Compute AdaLN conditioning signal = time_emb + proprio_emb."""
        time_feats = self.time_emb(timestep)
        if proprio_feats is not None:
            time_feats = time_feats + self.curr_gripper_emb(proprio_feats)
        return time_feats   # (B, embedding_dim)

    def forward(
        self,
        noisy_trajectory: torch.Tensor,            # (B, T, 9)  pos_norm+rot6d
        timestep:         torch.Tensor,            # (B,)  float in [0,1]
        voxel_tokens:     torch.Tensor,            # (B, M, embedding_dim)  FPS subset — self-attn
        voxel_pos:        torch.Tensor,            # (B, M, 3)              FPS positions
        lang_tokens:      torch.Tensor,            # (B, L, embedding_dim)
        proprio_feats:    torch.Tensor = None,     # (B, embedding_dim)  pre-projected
        traj_xyz_world:   torch.Tensor = None,     # (B, T, 3)  unnormalised world coords
        full_vox_tokens:  torch.Tensor = None,     # (B, N, embedding_dim)  full set — cross-attn
        full_vox_pos:     torch.Tensor = None,     # (B, N, 3)              full positions
    ):
        """
        Returns a list of length 1 containing (B, T, 10):
            [pos_velocity(3), rot_velocity(6), grip_logit(1)]
        The list wrapper exactly matches base_denoise_actor's return convention.

        voxel_tokens / voxel_pos   : FPS-subsampled tokens — used in the joint
                                     self-attention trunk for efficiency.
        full_vox_tokens / full_vox_pos : full (non-subsampled) token set — used
                                     in cross-attention so the trajectory can
                                     attend to fine-grained scene detail.
                                     Falls back to voxel_tokens if None.
        """
        B, T, _ = noisy_trajectory.shape

        # Fall back gracefully when full tokens are not provided
        if full_vox_tokens is None:
            full_vox_tokens = voxel_tokens
        if full_vox_pos is None:
            full_vox_pos = voxel_pos

        # ---- AdaLN conditioning signal ------------------------------------ #
        time_embs = self.encode_denoising_timestep(timestep, proprio_feats)  # (B, emb)

        # ---- Encode trajectory tokens ------------------------------------ #
        # Mirrors base_denoise_actor: traj_feats = traj_encoder(trajectory)
        traj_feats = self.traj_encoder(noisy_trajectory)   # (B, T, emb)

        # Step-index positional embedding (one per timestep in [0, T))
        # Mirrors base_denoise_actor.traj_time_emb applied per step
        step_indices = torch.arange(T, device=noisy_trajectory.device)  # (T,)
        traj_time_pos = self.traj_time_emb(step_indices)                 # (T, emb)
        traj_time_pos = traj_time_pos.unsqueeze(0).expand(B, -1, -1)    # (B, T, emb)

        # ---- 3D RoPE positions ------------------------------------------- #
        # Full voxel token positions for cross-attention
        rel_full_vox_pos = self.relative_pe_layer(full_vox_pos)    # (B, N, emb, 2)
        # FPS voxel token positions for joint self-attention
        rel_vox_pos      = self.relative_pe_layer(voxel_pos)       # (B, M, emb, 2)
        # Use world-space trajectory coordinates for RoPE so that the relative
        # positional encoding is computed in the same coordinate space as the
        # voxel tokens.  This exactly mirrors base_denoise_actor.policy_forward_pass
        # which computes traj_xyz = self.unnormalize_pos(trajectory)[..., :3].
        # traj_xyz_world is supplied by predict_velocity (unnormalized via coord_bounds).
        # Fall back to normalized coords only if not provided (e.g. standalone use).
        if traj_xyz_world is not None:
            traj_xyz = traj_xyz_world                             # (B, T, 3) world
        else:
            traj_xyz = noisy_trajectory[:, :, :3]                 # (B, T, 3) fallback
        rel_traj_pos = self.relative_pe_layer(traj_xyz)           # (B, T, emb, 2)

        # ---- Cross-attention: trajectory -> language --------------------- #
        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=lang_tokens,
            seq1_sem_pos=traj_time_pos,
            seq2_sem_pos=None,
        )[-1]   # (B, T, emb)
        # Re-add traj_time_pos (mirrors base_denoise_actor pattern)
        traj_feats = traj_feats + traj_time_pos

        # ---- Cross-attention: trajectory -> FULL voxel scene ------------- #
        # Uses the full (non-subsampled) token set so the trajectory can
        # attend to fine-grained scene detail.  Mirrors base_denoise_actor
        # which cross-attends to full rgb3d_feats before the self-attn trunk.
        traj_feats = self.cross_attn(
            seq1=traj_feats,
            seq2=full_vox_tokens,
            seq1_pos=rel_traj_pos,
            seq2_pos=rel_full_vox_pos,
            ada_sgnl=time_embs,
        )[-1]   # (B, T, emb)

        # ---- Joint self-attention (trajectory + FPS-subsampled tokens) --- #
        # Uses the smaller FPS subset for computational efficiency.
        # Mirrors base_denoise_actor: features = cat([traj, fps_scene_feats])
        joint_feats = torch.cat([traj_feats, voxel_tokens], dim=1)  # (B, T+M, emb)
        joint_pos   = torch.cat([rel_traj_pos, rel_vox_pos], dim=1)

        joint_feats = self.self_attn(
            seq1=joint_feats,
            seq2=joint_feats,
            seq1_pos=joint_pos,
            seq2_pos=joint_pos,
            ada_sgnl=time_embs,
        )[-1]   # (B, T+M, emb)

        # ---- Position head ----------------------------------------------- #
        # Mirrors base_denoise_actor.predict_pos exactly:
        #   position_features = position_self_attn(features, features, ...)[-1]
        #   position_features = position_features[:, :traj_len]
        #   position_features = position_proj(position_features)
        #   position = position_predictor(position_features)
        pos_feat = self.position_self_attn(
            seq1=joint_feats,
            seq2=joint_feats,
            seq1_pos=joint_pos,
            seq2_pos=joint_pos,
            ada_sgnl=time_embs,
        )[-1][:, :T, :]                          # slice traj tokens → (B, T, emb)
        pos_feat  = self.position_proj(pos_feat)  # (B, T, emb)
        position  = self.position_predictor(pos_feat)   # (B, T, 3)

        # ---- Rotation head ----------------------------------------------- #
        rot_feat = self.rotation_self_attn(
            seq1=joint_feats,
            seq2=joint_feats,
            seq1_pos=joint_pos,
            seq2_pos=joint_pos,
            ada_sgnl=time_embs,
        )[-1][:, :T, :]                          # (B, T, emb)
        rot_feat  = self.rotation_proj(rot_feat)  # (B, T, emb)
        rotation  = self.rotation_predictor(rot_feat)   # (B, T, 6)

        # ---- Gripper head ------------------------------------------------ #
        # Uses joint traj tokens from the shared trunk (matches base_denoise_actor)
        openess = self.openess_predictor(joint_feats[:, :T, :])   # (B, T, 1)
        # openess = self.openess_predictor(pos_feat)   # (B, T, 1)  # ← alternative: use position head features instead of joint_feats for grip prediction.  Both work but joint_feats is slightly better.

        out = torch.cat([position, rotation, openess], dim=-1)    # (B, T, 10)
        return [out]    # list wrapper mirrors base_denoise_actor return convention


# ---------------------------------------------------------------------------
# VoxelFlowEncoder
# ---------------------------------------------------------------------------

class VoxelFlowEncoder(nn.Module):
    """
    3D-CNN voxel encoder + Transformer flow-matching action head.

    The 3D-CNN pipeline is identical to PerceiverVoxelLangEncoder (keeping
    GS renderer compatibility).  The denoising head is a TransformerHead
    that mirrors 3D FlowMatch Actor, with voxel tokens as the scene context.

    Parameters
    ----------
    voxel_size : int
        Number of voxels per side (e.g. 100 -> 100^3 grid).
    initial_dim : int
        Input channels of the raw voxel grid (default 10).
    low_dim_size : int
        Proprioception dimension (default 4).
    im_channels : int
        3D-CNN output channels (default 64).
    lang_feat_dim : int
        Sentence-level language embedding size (default 1024, CLIP).
    lang_emb_dim : int
        Token-level language embedding size (default 512, CLIP tokens).
    embedding_dim : int
        Transformer feature dimension.  Must be divisible by num_attn_heads.
    num_attn_heads : int
        Number of Transformer attention heads (default 8).
    num_shared_attn_layers : int
        Shared trunk depth (default 4).
    rot_dim : int
        Rotation representation dimension inside the Transformer head.
        Default 6 (ortho6D).  Traj token dim = 3 + rot_dim.
    voxel_token_downsample : int
        Spatial stride for downsampling voxel tokens before Transformer.
        E.g. 5 gives (V/5)^3 tokens from a V^3 volume.
    num_fps_tokens : int
        Number of tokens to keep after farthest-point sampling (FPS) of the
        downsampled voxel token set before the joint self-attention.
        Mirrors the FPS subsampling done by 3D FlowMatch Actor on its point
        cloud.  Set to 0 to disable (use all tokens).  Default: 512.
    coordinate_bounds : list[float]
        Workspace [x_min, y_min, z_min, x_max, y_max, z_max] used to assign
        world-space 3D coordinates to voxel tokens for 3D RoPE.
    lang_fusion_type : str
        'seq' or 'concat' (default 'seq').
    activation : str
        Activation for DenseBlock (default 'lrelu').
    cfg : DictConfig
        Full config passed through for NeuralRenderer compatibility.
    """

    def __init__(
        self,
        voxel_size: int = 100,
        initial_dim: int = 10,
        low_dim_size: int = 4,
        im_channels: int = 64,
        lang_feat_dim: int = 1024,
        lang_emb_dim: int = 512,
        embedding_dim: int = 120,
        num_attn_heads: int = 8,
        num_shared_attn_layers: int = 4,
        rot_dim: int = 6,           # 6D ortho rotation representation
        num_fps_tokens: int = 512,
        coordinate_bounds=None,
        lang_fusion_type: str = 'seq',
        activation: str = 'lrelu',
        # Legacy API keys kept for backward-compatibility (ignored internally)
        context_dim: int = 256,
        flow_hidden_dim: int = 512,
        flow_num_layers: int = 4,
        denoise_timesteps: int = 100,
        voxel_patch_size: int = 5,
        voxel_patch_stride: int = 5,
        voxel_token_downsample: int = 5,
        cfg=None,
    ):
        super().__init__()
        assert embedding_dim % num_attn_heads == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by "
            f"num_attn_heads ({num_attn_heads})"
        )
        self.cfg                    = cfg
        self.voxel_size             = voxel_size
        self.im_channels            = im_channels
        self.rot_dim                = rot_dim
        self.embedding_dim          = embedding_dim
        self.low_dim_size           = low_dim_size
        self.lang_fusion_type       = lang_fusion_type
        self.voxel_token_downsample = voxel_token_downsample
        self.num_fps_tokens         = num_fps_tokens
        self.denoise_timesteps      = denoise_timesteps

        if coordinate_bounds is None:
            coordinate_bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        self.register_buffer(
            'coord_bounds',
            torch.tensor(coordinate_bounds, dtype=torch.float32)
        )

        # ------------------------------------------------------------------ #
        # 1.  3D-CNN voxel encoder  (unchanged from PerceiverVoxelLangEncoder)#
        # ------------------------------------------------------------------ #
        self.encoder_3d = MultiLayer3DEncoderShallow(
            in_channels=initial_dim,
            out_channels=im_channels,
        )

        # ------------------------------------------------------------------ #
        # 2.  Language pre-processing                                         #
        # ------------------------------------------------------------------ #
        if lang_fusion_type == 'concat':
            self.lang_preprocess = nn.Linear(lang_feat_dim, im_channels)
            self._lang_out_dim   = im_channels
        else:  # 'seq'
            self.lang_preprocess = nn.Linear(lang_emb_dim, im_channels * 2)
            self._lang_out_dim   = im_channels * 2

        # ------------------------------------------------------------------ #
        # 4.  Project voxel tokens and lang tokens into embedding_dim         #
        # ------------------------------------------------------------------ #
        self._vox_ds = voxel_size // voxel_token_downsample   # downsampled side

        self.voxel_token_proj = nn.Sequential(
            nn.Linear(im_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.lang_token_proj = nn.Sequential(
            nn.Linear(self._lang_out_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Learned absolute positional encoding for voxel tokens (supplements RoPE)
        self.voxel_pos_emb = PositionEmbeddingLearnedMLP(
            dim=3, num_pos_feats=embedding_dim
        )

        # ------------------------------------------------------------------ #
        # 4b. Vision-Language cross-attention                                 #
        #     Mirrors encoder3d.vl_attention: scene tokens attend to language #
        #     tokens so that the visual features become task-conditioned       #
        #     BEFORE entering the denoising Transformer head.                 #
        #                                                                      #
        # FIX F: add rotary_pe=True so that the subsampled voxel tokens       #
        # retain their relative spatial positions during VL cross-attention.  #
        # Without RoPE, "pick the red cube on the LEFT" and "pick the red     #
        # cube on the RIGHT" produce identical spatial attention patterns.     #
        # Language tokens use seq2_pos=None (no spatial position needed).     #
        # ------------------------------------------------------------------ #
        self.vl_attention = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=True,    # FIX F: spatially-aware VL attention
            use_adaln=False,
            is_self=False,
        )
        # Shared 3D RoPE encoder for voxel positions used in vl_attention.
        # Reusing TransformerHead.relative_pe_layer would create a cross-module
        # dependency; instead we register a standalone instance here so that
        # vl_attention can encode positions during encode_scene (before the
        # transformer head is called).
        self.vl_relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # ------------------------------------------------------------------ #
        # 5.  Proprio projection: low_dim_size -> embedding_dim              #
        #     Projects raw proprio before passing to TransformerHead's        #
        #     curr_gripper_emb (which operates in embedding space).           #
        #                                                                      #
        # FIX D: low_dim_state is 4-dimensional                               #
        #   [gripper_open, joint_vel_x, joint_vel_y, joint_vel_z]             #
        # Only gripper_open (dim 0) is causally relevant for *what action to  #
        # take* — the xyz dims encode spatial state that belongs in the scene  #
        # token stream, not in the AdaLN conditioning signal.  Mixing spatial  #
        # coordinates into the time-conditioning pathway forces the model to   #
        # disentangle them from the timestep signal, making denoising harder.  #
        # We therefore project only gripper_open (1-D) into embedding_dim.    #
        # ------------------------------------------------------------------ #
        if low_dim_size > 0:
            self.proprio_proj = nn.Sequential(
                nn.Linear(1, embedding_dim),   # gripper_open only (dim 0)
                nn.ReLU(),
            )
        else:
            self.proprio_proj = None

        # ------------------------------------------------------------------ #
        # 6.  Transformer denoising head                                      #
        # ------------------------------------------------------------------ #
        self.transformer_head = TransformerHead(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=rot_dim,
        )

        # ------------------------------------------------------------------ #
        # FIX B: Pre-compute the voxel world-coordinate grid once and         #
        # register it as a buffer so it moves with the module to any device.  #
        # Previously _voxel_world_coords() rebuilt an (N, 3) tensor on every  #
        # forward pass via linspace + meshgrid + stack — ~8000 FP32 allocs    #
        # per step.  The grid is fully determined by coord_bounds (a buffer)  #
        # and _vox_ds (fixed at init) so it never changes.                    #
        # ------------------------------------------------------------------ #
        ds = self._vox_ds
        # coord_bounds may not be registered yet — use the raw list
        cb = torch.tensor(coordinate_bounds, dtype=torch.float32)
        bb_min, bb_max = cb[:3], cb[3:]
        lin = torch.linspace(0.5 / ds, 1.0 - 0.5 / ds, ds)
        xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing='ij')
        _grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)   # (N, 3)
        _grid = _grid * (bb_max - bb_min) + bb_min                   # world coords
        self.register_buffer('_vox_world_grid', _grid)               # (N, 3)

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    def _voxel_world_coords(self, B: int, device) -> torch.Tensor:
        """
        Return (B, N_vox, 3) world-coordinate grid for downsampled voxel tokens.

        FIX B: The grid is pre-computed once in __init__ and stored as a
        registered buffer (_vox_world_grid).  This call simply broadcasts it
        to the requested batch size — zero extra allocation every forward pass.
        """
        # _vox_world_grid is already on the right device (moved by .to() / fabric)
        return self._vox_world_grid.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)

    @staticmethod
    @torch.no_grad()
    def _spatial_grid_sample(vox_ds: int, n_samples: int,
                              device) -> torch.Tensor:
        """
        FIX G: O(N) spatial stratified subsampling for the coarse pre-VL pass.

        The full voxel token grid has N = vox_ds^3 tokens (e.g. 20^3 = 8000).
        Running _density_based_sample (O(N²) cdist) on 8000 tokens allocates a
        ~2 GB (B×8000×8000) distance matrix — the dominant memory cost per step.

        For the coarse pre-VL subsampling (8000 → ~1024) spatial uniformity is
        sufficient: every cell in a coarser G×G×G grid selects one representative
        token, giving O(N) cost and uniform workspace coverage.

        Parameters
        ----------
        vox_ds   : int    side length of the full downsampled grid (e.g. 20)
        n_samples: int    approximate number of tokens to keep
        device   : torch.device

        Returns
        -------
        idx : (1, M) long indices, broadcastable to (B, M)
        """
        # Largest integer G such that G^3 <= n_samples
        G = max(1, int(n_samples ** (1.0 / 3.0)))
        # stride in vox_ds space to cover it with G cells per axis
        stride = max(1, vox_ds // G)

        # Build indices of one anchor token per stride-cube
        i_coords = torch.arange(0, vox_ds, stride, device=device)
        j_coords = torch.arange(0, vox_ds, stride, device=device)
        k_coords = torch.arange(0, vox_ds, stride, device=device)
        ii, jj, kk = torch.meshgrid(i_coords, j_coords, k_coords, indexing='ij')
        # Flat index in the (vox_ds, vox_ds, vox_ds) grid
        flat_idx = (ii * vox_ds * vox_ds + jj * vox_ds + kk).reshape(-1)   # (M,)
        return flat_idx.unsqueeze(0)   # (1, M)

    @staticmethod
    @torch.no_grad()
    def _density_based_sample(features: torch.Tensor, n_samples: int,
                               k: int = 8) -> torch.Tensor:
        """
        Density-based feature-space subsampling.

        Mirrors ``base_encoder.density_based_sampler`` from 3D FlowMatch Actor
        exactly.  Selects the M tokens that are *most isolated* in feature
        space — i.e. those with the largest average distance to their k nearest
        neighbours.  This is superior to spatial FPS for voxel grids because:
          - Spatial FPS wastes samples on empty / uniform background voxels.
          - Feature-space diversity keeps semantically distinct tokens
            (object surfaces, edges, contact regions) regardless of spatial
            layout.

        Parameters
        ----------
        features : (B, N, C)  token feature matrix
        n_samples: int         number of tokens to keep (M)
        k        : int         neighbourhood size for density estimate (default 8)

        Returns
        -------
        idx : (B, M)  long indices into the N dimension
        """
        B, N, C = features.shape
        device  = features.device

        if n_samples >= N:
            return torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

        # (B, N, N) pairwise L2 distances in feature space
        dists = torch.cdist(features, features, p=2)

        # FIX C: torch.cdist includes dist(i,i)=0, which topk(largest=False)
        # always selects as one of the k nearest neighbours — reducing the
        # effective neighbourhood from k to k-1 and dampening every token's
        # sparsity score by 1/k.  Using k+1 and discarding slot 0 (the self-
        # distance) gives a clean k-neighbour estimate.
        knn_dists, _ = dists.topk(k=k + 1, dim=-1, largest=False)   # (B, N, k+1)
        sparsity = knn_dists[..., 1:].mean(dim=-1)                    # (B, N) — skip self

        # Keep the M most isolated (most sparse / least dense) tokens
        idx = sparsity.topk(n_samples, dim=-1, largest=True).indices   # (B, M)
        return idx

    # ---------------------------------------------------------------------- #
    # encode_scene                                                              #
    # ---------------------------------------------------------------------- #

    def encode_scene(
        self,
        voxel_grid:      torch.Tensor,   # (B, C_init, V, V, V)
        proprio:         torch.Tensor,   # (B, low_dim_size)
        lang_goal_emb:   torch.Tensor,   # (B, lang_feat_dim)
        lang_token_embs: torch.Tensor,   # (B, 77, lang_emb_dim)
    ):
        """
        This function encodes the raw inputs into the voxel and language features needed for the Gaussian Splatting renderer and the Transformer action head.
        
        Args:
        voxel_grid       : (B, C_init, V, V, V)  raw voxel grid input
        proprio          : (B, low_dim_size)     raw proprio vector (e.g. gripper state)
        lang_goal_emb    : (B, lang_feat_dim)   sentence-level language embedding (e.g. CLIP text encoder output)
        lang_token_embs  : (B, 77, lang_emb_dim) token-level language embeddings (e.g. CLIP token embeddings)
        
        Returns
        -------
        voxel_grid_feature : (B, im_channels, V, V, V)  — for GS renderer
        lang_embedd        : (B, seq, lang_out_dim)      — for GS renderer
        context            : dict
            'voxel_tokens'  (B, M, embedding_dim)   M = num_fps_tokens (or N)
            'voxel_pos'     (B, M, 3)
            'lang_tokens'   (B, L, embedding_dim)
            'proprio_feats' (B, embedding_dim)  or None
        """
        B      = voxel_grid.shape[0]
        device = voxel_grid.device

        # 3D-CNN
        d0, _ = self.encoder_3d(voxel_grid)   # (B, im_C, V, V, V)

        # Language
        if self.lang_fusion_type == 'seq':
            l = self.lang_preprocess(lang_token_embs.float())      # (B, 77, im_C*2)
        else:
            l = self.lang_preprocess(lang_goal_emb.float()).unsqueeze(1)  # (B,1,im_C)

        # Downsample voxel volume -> token sequence
        ds = self._vox_ds
        vox_ds     = F.adaptive_avg_pool3d(d0, (ds, ds, ds))        # (B, im_C, ds,ds,ds)
        vox_tokens = rearrange(vox_ds, 'b c x y z -> b (x y z) c')  # (B, N, im_C)

        # Project to embedding_dim
        vox_tokens  = self.voxel_token_proj(vox_tokens)    # (B, N, emb)
        lang_tokens = self.lang_token_proj(l)               # (B, L, emb)

        # World-space positions for 3D RoPE
        voxel_pos = self._voxel_world_coords(B, device)    # (B, N, 3)

        # FIX H: normalise voxel world coords to [-1, 1] before the learned
        # absolute pos MLP so its input scale is consistent and well-conditioned.
        # The 3D RoPE handles raw world coords internally (correct), but the
        # learned MLP has no built-in normalisation — raw coords (e.g. z ∈
        # [0.6, 1.6]) can land in a regime where Xavier-initialised weights
        # produce wildly varying output magnitudes.
        bb_min = self.coord_bounds[:3]   # (3,)
        bb_max = self.coord_bounds[3:]   # (3,)
        voxel_pos_norm = (voxel_pos - bb_min) / (bb_max - bb_min) * 2.0 - 1.0  # [-1, 1]

        # Add learned absolute positional encoding (on normalised coords)
        vox_tokens = vox_tokens + self.voxel_pos_emb(voxel_pos_norm)

        # FIX #7: subsample BEFORE VL-attention to cut the O(N × L) cost.
        # With voxel_size=100 and downsample=5, N=8000 tokens.  Running
        # vl_attention on 8000 tokens × 77 lang tokens is the dominant
        # memory/compute bottleneck.  We keep a larger intermediate set
        # (vl_fps_size = 2 × num_fps_tokens, capped at N) for VL-attention
        # so that scene semantics are well-covered before the final FPS.
        # The full 8000-token set is still used in cross-attention inside the
        # Transformer head (fine-grained spatial detail).
        #
        # FIX G: The original code used _density_based_sample (O(N² × C) cdist)
        # for this coarse 8000→1024 pass, which peaks at ~2 GB per batch.  For
        # the pre-VL pass spatial uniformity is sufficient — we use O(N)
        # _spatial_grid_sample instead.  _density_based_sample is still used for
        # the second (fine) 1024→512 pass where feature diversity matters.
        N_full = vox_tokens.shape[1]
        vl_fps_size = min(2 * max(self.num_fps_tokens, 1), N_full)
        if self.num_fps_tokens > 0 and vl_fps_size < N_full:
            # Coarse spatial grid sample — O(N), uniform coverage
            vl_idx     = self._spatial_grid_sample(
                self._vox_ds, vl_fps_size, device=device
            ).expand(B, -1).contiguous()                                         # (B, vl_fps_size)
            # All derived index expansions must be contiguous for gather / scatter_.
            vl_idx_exp    = vl_idx.unsqueeze(-1).expand(-1, -1, vox_tokens.shape[-1]).contiguous()
            vl_tokens_sub = torch.gather(vox_tokens, 1, vl_idx_exp)              # (B, vl_fps_size, emb)
            vl_pos_sub    = torch.gather(voxel_pos,  1,
                                         vl_idx.unsqueeze(-1).expand(-1, -1, 3).contiguous())  # (B, vl_fps_size, 3)
        else:
            vl_tokens_sub = vox_tokens
            vl_pos_sub    = voxel_pos

        # Vision-Language cross-attention: subsampled voxel tokens attend to
        # language tokens so that the scene representation becomes
        # task-conditioned before entering the denoising head.
        # FIX F: pass 3D RoPE positions for voxel tokens so that VL attention
        # is spatially aware (e.g. "pick left" vs "pick right" differentially
        # conditions left- vs right-side voxels).  Language tokens have no
        # spatial position (seq2_pos=None).
        vl_rope_pos = self.vl_relative_pe_layer(vl_pos_sub)   # (B, vl_fps_size, emb, 2)
        vl_tokens_sub = self.vl_attention(
            seq1=vl_tokens_sub,
            seq2=lang_tokens,
            seq1_pos=vl_rope_pos,
            seq2_pos=None,
        )[-1]   # (B, vl_fps_size, emb)

        # Write task-conditioned features back into the full token set so that
        # the cross-attention in the Transformer head still has access to all N
        # world positions with updated (language-conditioned) features.
        if self.num_fps_tokens > 0 and vl_fps_size < N_full:
            vox_tokens = vox_tokens.clone()
            vox_tokens.scatter_(
                1,
                vl_idx_exp,
                vl_tokens_sub,
            )

        # Density-based feature-space subsampling: keep M semantically
        # diverse tokens for the joint self-attention trunk.
        # Mirrors base_encoder.run_dps / density_based_sampler from 3D
        # FlowMatch Actor.  We pass the FULL vox_tokens to cross_attn
        # (fine-grained detail) and only the subsampled fps_tokens to
        # the joint self-attn (efficiency).
        if self.num_fps_tokens > 0:
            dps_idx    = self._density_based_sample(vl_tokens_sub, self.num_fps_tokens)  # (B, M)
            # Map local vl_fps indices back to the full token index space
            if vl_fps_size < N_full:
                global_dps_idx = torch.gather(vl_idx, 1, dps_idx)        # (B, M) in [0, N)
            else:
                global_dps_idx = dps_idx
            idx_exp    = global_dps_idx.unsqueeze(-1).expand(-1, -1, vox_tokens.shape[-1]).contiguous()
            fps_tokens = torch.gather(vox_tokens, 1, idx_exp)             # (B, M, emb)
            fps_pos    = torch.gather(voxel_pos, 1,
                                      global_dps_idx.unsqueeze(-1).expand(-1, -1, 3).contiguous())  # (B,M,3)
        else:
            fps_tokens, fps_pos = vox_tokens, voxel_pos

        # Proprio embedding: project gripper_open only -> embedding_dim
        # FIX D: slice dim 0 (gripper_open) so that spatial joint-velocity
        # dims don't contaminate the AdaLN time-conditioning signal.
        proprio_feats = None
        if self.proprio_proj is not None and proprio is not None:
            gripper_open  = proprio[:, :1].float()               # (B, 1)
            proprio_feats = self.proprio_proj(gripper_open)       # (B, emb)

        context = {
            'voxel_tokens':      fps_tokens,    # (B, M, emb) — FPS subset, for self-attn
            'voxel_pos':         fps_pos,        # (B, M, 3)
            'full_vox_tokens':   vox_tokens,     # (B, N, emb) — full set, for cross-attn
            'full_vox_pos':      voxel_pos,      # (B, N, 3)
            'lang_tokens':       lang_tokens,
            'proprio_feats':     proprio_feats,
        }
        return d0, l, context

    # ---------------------------------------------------------------------- #
    # predict_velocity                                                          #
    # ---------------------------------------------------------------------- #

    def predict_velocity(
        self,
        noisy_trajectory: torch.Tensor,  # (B, T, 9)  pos_norm + rot6d
        t:                torch.Tensor,  # (B,)  denoising timestep index
        context,                         # dict from encode_scene
    ) -> list:
        """
        Predict flow velocity v = noise - x_clean.

        Returns a list of length 1 containing a tensor of shape (B, T, 10):
            [..., :3]  position velocity
            [..., 3:9] rotation velocity (ortho6D)
            [..., 9:]  gripper logit
        """
        if isinstance(context, dict):
            voxel_tokens      = context['voxel_tokens']       # (B, M, emb)  FPS — self-attn
            voxel_pos         = context['voxel_pos']           # (B, M, 3)
            # Full (non-subsampled) tokens for cross-attention.  Falls back to
            # the FPS subset if the key is absent (e.g. in standalone tests).
            full_vox_tokens   = context.get('full_vox_tokens', voxel_tokens)
            full_vox_pos      = context.get('full_vox_pos',    voxel_pos)
            lang_tokens       = context['lang_tokens']
            proprio_feats     = context.get('proprio_feats', None)
        else:
            B = noisy_trajectory.shape[0]
            device = noisy_trajectory.device
            voxel_tokens    = torch.zeros(B, 1, self.embedding_dim, device=device)
            voxel_pos       = torch.zeros(B, 1, 3, device=device)
            full_vox_tokens = voxel_tokens
            full_vox_pos    = voxel_pos
            lang_tokens     = torch.zeros(B, 1, self.embedding_dim, device=device)
            proprio_feats   = None

        # Unnormalize trajectory xyz ([-1,1] → world coords) for 3D RoPE.
        # This mirrors base_denoise_actor.policy_forward_pass which computes
        #   traj_xyz = self.unnormalize_pos(trajectory)[..., :3]
        # before passing world-space positions to the prediction head.
        bb_min = self.coord_bounds[:3].to(noisy_trajectory.device)
        bb_max = self.coord_bounds[3:].to(noisy_trajectory.device)
        traj_xyz_norm  = noisy_trajectory[:, :, :3]                        # (B, T, 3) in [-1,1]
        traj_xyz_world = (traj_xyz_norm + 1.0) / 2.0 * (bb_max - bb_min) + bb_min  # world coords

        return self.transformer_head(
            noisy_trajectory=noisy_trajectory.float(),
            timestep=t.float(),
            voxel_tokens=voxel_tokens.float(),         # FPS subset → self-attn
            voxel_pos=voxel_pos.float(),
            full_vox_tokens=full_vox_tokens.float(),   # full set → cross-attn
            full_vox_pos=full_vox_pos.float(),
            lang_tokens=lang_tokens.float(),
            proprio_feats=proprio_feats.float() if proprio_feats is not None else None,
            traj_xyz_world=traj_xyz_world.float(),
        )

    # ---------------------------------------------------------------------- #
    # forward                                                                   #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        voxel_grid:      torch.Tensor,
        proprio:         torch.Tensor,
        lang_goal_emb:   torch.Tensor,
        lang_token_embs: torch.Tensor,
        prev_layer_voxel_grid=None,
        bounds=None,
        prev_layer_bounds=None,
        mask=None,
    ):
        """
        Agruments
        ---------
        voxel_grid       : (B, C_init, V, V, V)  raw voxel grid input
        proprio          : (B, low_dim_size)     raw proprio vector (e.g. gripper open/close state)
        lang_goal_emb    : (B, lang_feat_dim)     sentence-level language embedding (e.g. CLIP)
        lang_token_embs   : (B, seq, lang_emb_dim)  token-level language embeddings (e.g. CLIP tokens)

        prev_layer_voxel_grid: (B, C_prev, V, V, V)  from previous layer (ignored)
        bounds: None (ignored)
        prev_layer_bounds: None (ignored)
        mask: None (ignored)
        
        Returns
        -------
        voxel_grid_feature : (B, im_channels, V, V, V)
        lang_embedd        : (B, seq, lang_out_dim)
        context            : dict  {'voxel_tokens', 'voxel_pos', 'lang_tokens'}
        """
        voxel_grid_feature, lang_embedd, context = self.encode_scene(
            voxel_grid, proprio, lang_goal_emb, lang_token_embs
        )
        return voxel_grid_feature, lang_embedd, context

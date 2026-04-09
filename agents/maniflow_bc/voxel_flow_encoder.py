"""
VoxelFlowEncoder
================
Keeps the 3D-CNN voxel encoding pipeline from PerceiverVoxelLangEncoder
(needed to supply `voxel_grid_feature` and `lang_embedd` to the Gaussian
Splatting renderer) but replaces the Perceiver Transformer and discrete
classification heads with a **Transformer denoising head** that mirrors the
architecture of 3D FlowMatch Actor (`denoise_actor_3d.py`) while adapting
the scene representation to use the 3D-CNN voxel feature map.

Voxel-to-token adaptation
--------------------------
3D FlowMatch Actor conditions the Transformer on a FPS-subsampled point cloud
of 3D tokens.  maniflow has no point cloud at the Transformer level; instead
it has a 3D-CNN voxel feature volume (B, C, V, V, V).  The adaptation is:

  1.  The voxel volume is spatially downsampled via AdaptiveAvgPool3d and then
      flattened into N = ds^3 tokens of size C (ds = V // voxel_token_downsample).
  2.  Each token is assigned a 3D world coordinate computed from its voxel
      index and the workspace bounding box — used for 3D RoPE exactly as
      point-cloud positions are in the original.
  3.  Language tokens are a separate sequence, cross-attended to by the action
      query, exactly as in the original TransformerHead.
  4.  The denoising head is the same TransformerHead structure:
        action -> lang cross-attn -> scene cross-attn (AdaLN)
        -> joint self-attn (AdaLN) -> pos / rot / openness output heads.

The encoder still returns (voxel_grid_feature, lang_embedd) so that
NeuralRenderer can be used unchanged.

The action head forward signature is:
    predict_velocity(noisy_action, t, context) -> (B, action_dim)
where context is a dict {'voxel_tokens', 'voxel_pos', 'lang_tokens'}.

Action space
------------
Single-step 8-DoF: [x, y, z, qx, qy, qz, qw, gripper_open]
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
# TransformerHead  (mirrors base_denoise_actor.TransformerHead + 3D RoPE)
# ---------------------------------------------------------------------------

class TransformerHead(nn.Module):
    """
    Denoising head matching 3D FlowMatch Actor's TransformerHead, adapted for
    a single-step 8-DoF action instead of a multi-frame trajectory.

    Scene context: flat sequence of voxel tokens with 3D RoPE positions.
    Language context: flat sequence of language tokens.

    Parameters
    ----------
    embedding_dim : int
        Feature dimension throughout the Transformer (default 120).
    num_attn_heads : int
        Number of attention heads.  Must divide embedding_dim evenly.
    num_shared_attn_layers : int
        Number of AdaLN self-attention layers in the shared trunk (default 4).
    action_dim : int
        Raw action dimension (default 8 = xyz+quat+gripper).
    """

    def __init__(
        self,
        embedding_dim: int = 120,
        num_attn_heads: int = 8,
        num_shared_attn_layers: int = 4,
        action_dim: int = 8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim

        # ---- Time embedding ---------------------------------------------- #
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # ---- Proprioception encoder (fused into AdaLN, mirrors 3D FlowMatch) #
        # curr_gripper_emb: Linear(proprio_dim * embedding_dim) -> ReLU -> Linear
        # This is added to time_emb to form the AdaLN conditioning signal.
        # low_dim_size is stored at VoxelFlowEncoder level; TransformerHead
        # receives the already-embedded proprio vector of size embedding_dim.
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        # ---- Action query encoder ---------------------------------------- #
        # Encode the 8-DoF noisy action into a single (B, 1, embedding_dim) token
        self.action_encoder = nn.Linear(action_dim, embedding_dim)

        # Step positional embedding (we have only 1 step / token)
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # ---- 3D RoPE for voxel positions --------------------------------- #
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # ---- Cross-attention: action query -> language ------------------- #
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

        # ---- Cross-attention: action query -> voxel scene tokens --------- #
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=True,   # 3D RoPE on voxel positions
            use_adaln=True,
            is_self=False,
        )

        # ---- Shared self-attention trunk --------------------------------- #
        self.self_attn = AttentionModule(
            num_layers=num_shared_attn_layers,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=True,
            use_adaln=True,
            is_self=True,
        )

        # ---- Position head (xyz, -> 3) ----------------------------------- #
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
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

        # ---- Rotation head (quaternion velocity, -> 4) ------------------- #
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
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
            nn.Linear(embedding_dim, 4),
        )

        # ---- Gripper head (logit -> 1) ----------------------------------- #
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    # ---------------------------------------------------------------------- #

    def forward(
        self,
        noisy_action:  torch.Tensor,            # (B, action_dim)
        timestep:      torch.Tensor,            # (B,)  float in [0,1]
        voxel_tokens:  torch.Tensor,            # (B, N_vox, embedding_dim)
        voxel_pos:     torch.Tensor,            # (B, N_vox, 3)  world coords
        lang_tokens:   torch.Tensor,            # (B, L, embedding_dim)
        proprio_feats: torch.Tensor = None,     # (B, embedding_dim)  pre-projected
    ) -> torch.Tensor:
        """Returns predicted velocity: (B, action_dim)."""
        B = noisy_action.shape[0]

        # ---- AdaLN signal (time + proprio) -------------------------------- #
        # Mirrors base_denoise_actor.encode_denoising_timestep:
        #   time_embs = time_emb(t) + curr_gripper_emb(proprio)
        time_embs = self.time_emb(timestep)   # (B, emb)
        if proprio_feats is not None:
            time_embs = time_embs + self.curr_gripper_emb(proprio_feats)

        # ---- Encode action as a single-token query ----------------------- #
        act_feat = self.action_encoder(noisy_action).unsqueeze(1)  # (B, 1, emb)
        step_pos = self.traj_time_emb(
            torch.zeros(1, device=noisy_action.device)
        ).unsqueeze(0).expand(B, 1, -1)        # (B, 1, emb)
        # NOTE: do NOT add step_pos here; it is applied inside traj_lang_attention
        # via seq1_sem_pos and then re-added AFTER the output (matching original).

        # ---- 3D RoPE positions ------------------------------------------- #
        # Voxel tokens: (B, N, emb, 2) from 3D world coords
        rel_vox_pos = self.relative_pe_layer(voxel_pos)
        # Action token position = xyz from the noisy action, matching the
        # original base_denoise_actor which uses traj_xyz = trajectory[..., :3].
        # This spatially grounds the action token in the RoPE distance metric.
        act_xyz = noisy_action[:, :3].unsqueeze(1)          # (B, 1, 3)
        rel_act_pos = self.relative_pe_layer(act_xyz)        # (B, 1, emb, 2)

        # ---- Cross-attention: action -> language ------------------------- #
        act_feat = self.traj_lang_attention(
            seq1=act_feat,
            seq2=lang_tokens,
            seq1_sem_pos=step_pos,
            seq2_sem_pos=None,
        )[-1]   # (B, 1, emb)
        # Re-add step_pos after attention output (mirrors base_denoise_actor pattern)
        act_feat = act_feat + step_pos

        # ---- Cross-attention: action -> voxel scene ---------------------- #
        act_feat = self.cross_attn(
            seq1=act_feat,
            seq2=voxel_tokens,
            seq1_pos=rel_act_pos,
            seq2_pos=rel_vox_pos,
            ada_sgnl=time_embs,
        )[-1]   # (B, 1, emb)

        # ---- Joint self-attention (action token + voxel tokens) ---------- #
        joint_feats = torch.cat([act_feat, voxel_tokens], dim=1)  # (B, 1+N, emb)
        joint_pos   = torch.cat([rel_act_pos, rel_vox_pos], dim=1)

        joint_feats = self.self_attn(
            seq1=joint_feats,
            seq2=joint_feats,
            seq1_pos=joint_pos,
            seq2_pos=joint_pos,
            ada_sgnl=time_embs,
        )[-1]   # (B, 1+N, emb)

        # ---- Position head ----------------------------------------------- #
        # Project FIRST, then attend: mirrors base_denoise_actor.predict_pos:
        #   position_features = self.position_self_attn(features, features, ...)
        #   position_features = position_features[:, :traj_len]  # slice action tokens
        # The projection acts as an input gate before the dedicated self-attn block.
        pos_feat = self.position_proj(joint_feats)   # (B, 1+N, emb)
        pos_feat = self.position_self_attn(
            seq1=pos_feat,
            seq2=pos_feat,
            seq1_pos=joint_pos,
            seq2_pos=joint_pos,
            ada_sgnl=time_embs,
        )[-1][:, :1, :]   # (B, 1+N, emb) -> slice action token -> (B, 1, emb)
        position  = self.position_predictor(pos_feat).squeeze(1)   # (B, 3)

        # ---- Rotation head ----------------------------------------------- #
        rot_feat  = self.rotation_proj(joint_feats)  # (B, 1+N, emb)
        rot_feat  = self.rotation_self_attn(
            seq1=rot_feat,
            seq2=rot_feat,
            seq1_pos=joint_pos,
            seq2_pos=joint_pos,
            ada_sgnl=time_embs,
        )[-1][:, :1, :]   # (B, 1+N, emb) -> slice action token -> (B, 1, emb)
        rotation  = self.rotation_predictor(rot_feat).squeeze(1)   # (B, 4)

        # ---- Gripper head ------------------------------------------------ #
        # Use the action token from joint_feats (not pos_feat which is
        # already projected through the position head's linear layer).
        openess = self.openess_predictor(joint_feats[:, :1, :]).squeeze(1)  # (B, 1)

        return torch.cat([position, rotation, openess], dim=-1)     # (B, 8)


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
    action_dim : int
        Continuous action dimension (default 8).
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
        action_dim: int = 8,
        voxel_token_downsample: int = 5,
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
        self.action_dim             = action_dim
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
        # 5.  Proprio projection: low_dim_size -> embedding_dim              #
        #     Projects raw proprio before passing to TransformerHead's        #
        #     curr_gripper_emb (which operates in embedding space).           #
        # ------------------------------------------------------------------ #
        if low_dim_size > 0:
            self.proprio_proj = nn.Sequential(
                nn.Linear(low_dim_size, embedding_dim),
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
            action_dim=action_dim,
        )

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    def _voxel_world_coords(self, B: int, device) -> torch.Tensor:
        """
        Build (B, N_vox, 3) world-coordinate grid for downsampled voxel tokens.
        Each coordinate is the centre of the voxel cell in world space.
        """
        ds = self._vox_ds
        bb_min = self.coord_bounds[:3]
        bb_max = self.coord_bounds[3:]

        lin = torch.linspace(0.5 / ds, 1.0 - 0.5 / ds, ds, device=device)
        # indexing='ij': xx[i,j,k]=lin[i], yy[i,j,k]=lin[j], zz[i,j,k]=lin[k]
        # rearrange 'b c x y z -> b (x y z) c' flattens so token n corresponds
        # to voxel (x=n//ds^2, y=(n//ds)%ds, z=n%ds), matching lin[i,j,k] order.
        xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing='ij')
        grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # (N, 3)
        grid = grid * (bb_max - bb_min).to(device) + bb_min.to(device)
        return grid.unsqueeze(0).expand(B, -1, -1)   # (B, N, 3)

    @staticmethod
    @torch.no_grad()
    def _fps(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Farthest Point Sampling on a set of 3D points.

        Parameters
        ----------
        xyz      : (B, N, 3)
        n_samples: int, number of points to keep (M <= N)

        Returns
        -------
        idx : (B, M)  indices into the N dimension
        """
        B, N, _ = xyz.shape
        device  = xyz.device

        if n_samples >= N:
            return torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

        # Pairwise squared distances are too large; iterate greedily per batch
        idx     = torch.zeros(B, n_samples, dtype=torch.long, device=device)
        # Start from a random seed to avoid always picking corner tokens
        farthest = torch.randint(0, N, (B,), device=device)
        distance = torch.full((B, N), float('inf'), device=device)

        for i in range(n_samples):
            idx[:, i] = farthest
            # Gather current farthest point coords: (B, 1, 3)
            centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
            # Squared distance from centroid to all points
            dist = ((xyz - centroid) ** 2).sum(dim=-1)   # (B, N)
            distance = torch.minimum(distance, dist)
            farthest = distance.argmax(dim=-1)            # (B,)

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

        # Add learned absolute positional encoding
        vox_tokens = vox_tokens + self.voxel_pos_emb(voxel_pos)

        # FPS subsampling: reduce N -> M for joint self-attention
        # (mirrors 3D FlowMatch Actor's fps_subsampling_factor on point cloud)
        if self.num_fps_tokens > 0:
            fps_idx    = self._fps(voxel_pos, self.num_fps_tokens)   # (B, M)
            idx_exp    = fps_idx.unsqueeze(-1).expand(-1, -1, vox_tokens.shape[-1])
            fps_tokens = torch.gather(vox_tokens, 1, idx_exp)        # (B, M, emb)
            fps_pos    = torch.gather(voxel_pos, 1,
                                      fps_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B,M,3)
        else:
            fps_tokens, fps_pos = vox_tokens, voxel_pos

        # Proprio embedding (low_dim_size -> embedding_dim)
        proprio_feats = None
        if self.proprio_proj is not None and proprio is not None:
            proprio_feats = self.proprio_proj(proprio.float())   # (B, emb)

        context = {
            'voxel_tokens':  fps_tokens,
            'voxel_pos':     fps_pos,
            'lang_tokens':   lang_tokens,
            'proprio_feats': proprio_feats,
        }
        return d0, l, context

    # ---------------------------------------------------------------------- #
    # predict_velocity                                                          #
    # ---------------------------------------------------------------------- #

    def predict_velocity(
        self,
        noisy_action: torch.Tensor,   # (B, action_dim)
        t:            torch.Tensor,   # (B,)  float in [0,1]
        context,                      # dict from encode_scene
    ) -> torch.Tensor:
        """Predict flow velocity v = noise - x_clean."""
        if isinstance(context, dict):
            voxel_tokens  = context['voxel_tokens']
            voxel_pos     = context['voxel_pos']
            lang_tokens   = context['lang_tokens']
            proprio_feats = context.get('proprio_feats', None)
        else:
            # Legacy plain-tensor fallback: build trivial single-token contexts
            B = noisy_action.shape[0]
            voxel_tokens  = torch.zeros(B, 1, self.embedding_dim,
                                        device=noisy_action.device)
            voxel_pos     = torch.zeros(B, 1, 3, device=noisy_action.device)
            lang_tokens   = torch.zeros(B, 1, self.embedding_dim,
                                        device=noisy_action.device)
            proprio_feats = None

        return self.transformer_head(
            noisy_action=noisy_action.float(),
            timestep=t.float(),
            voxel_tokens=voxel_tokens.float(),
            voxel_pos=voxel_pos.float(),
            lang_tokens=lang_tokens.float(),
            proprio_feats=proprio_feats.float() if proprio_feats is not None else None,
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

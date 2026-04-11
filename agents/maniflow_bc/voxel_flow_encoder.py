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
from collections import OrderedDict
from einops import rearrange
from torchvision.ops import FeaturePyramidNetwork

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
# _EfficientFPN — multi-scale feature pyramid (mirrors encoder3d exactly)
# ---------------------------------------------------------------------------

class _EfficientFPN(FeaturePyramidNetwork):
    """
    Feature Pyramid Network that fuses all CLIP RN50 levels top-down and
    returns a single output at `output_level` resolution.

    Matches encoder3d.feature_pyramid([64,256,512,1024,2048], emb, "res3")
    from the reference 3D FlowMatch Actor exactly:
      • subclasses torchvision FeaturePyramidNetwork
      • adds early-stop: only processes down to `output_level` (saves compute)
      • ensures contiguous memory for Conv2d weights (required for backward hooks)
    """

    def __init__(
        self,
        in_channels_list,
        out_channels,
        extra_blocks=None,
        norm_layer=None,
        output_level: str = 'res3',
    ):
        super().__init__(in_channels_list, out_channels, extra_blocks, norm_layer)
        self.output_level = output_level

        # Ensure Conv2d weight tensors are contiguous (required for backward
        # hooks added below; torchvision FPN may create non-contiguous ones).
        for idx in range(len(self.inner_blocks)):
            block = self.inner_blocks[idx]
            if isinstance(block, nn.Conv2d):
                new_block = nn.Conv2d(
                    block.in_channels, block.out_channels, block.kernel_size,
                    stride=block.stride, padding=block.padding,
                    dilation=block.dilation,
                    bias=(block.bias is not None),
                    padding_mode=block.padding_mode,
                ).to(memory_format=torch.contiguous_format)
                new_block.weight.data.copy_(block.weight.data)
                if block.bias is not None:
                    new_block.bias.data.copy_(block.bias.data)
                self.inner_blocks[idx] = new_block

        for block in self.inner_blocks:
            if isinstance(block, nn.Conv2d):
                block.weight.register_hook(lambda grad: grad.contiguous())

    def forward(self, x: OrderedDict) -> OrderedDict:
        names  = list(x.keys())
        values = [v.contiguous(memory_format=torch.contiguous_format)
                  for v in x.values()]

        # Top-most level (res5)
        last_inner = self.get_result_from_inner_blocks(values[-1], -1)
        results    = [self.get_result_from_layer_blocks(last_inner, -1)]
        result_names = [names[-1]]

        if names[-1] != self.output_level:
            for idx in range(len(values) - 2, -1, -1):
                inner_lateral = self.get_result_from_inner_blocks(values[idx], idx)
                feat_shape    = inner_lateral.shape[-2:]
                inner_top_down = F.interpolate(last_inner, size=feat_shape, mode='nearest')
                last_inner = (inner_lateral + inner_top_down).contiguous()
                results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
                result_names.insert(0, names[idx])
                if names[idx] == self.output_level:
                    break  # early stop — don't compute levels below output_level

        if self.extra_blocks is not None:
            results, result_names = self.extra_blocks(results, values, result_names)

        return OrderedDict({
            k: v.contiguous(memory_format=torch.contiguous_format)
            for k, v in zip(result_names, results)
        })


# ---------------------------------------------------------------------------
# CLIPVoxelLifter
# ---------------------------------------------------------------------------

class CLIPVoxelLifter(nn.Module):
    """
    Replaces MultiLayer3DEncoderShallow with a CLIP ResNet50 visual backbone
    whose 2D spatial features are lifted ("splatted") into the 3D voxel volume
    using the per-pixel point cloud.

    Pipeline  (now matches 3D FlowMatch Actor encoder3d exactly for steps 1-4)
    --------
    1.  Normalize the RGB image with CLIP's own mean/std.
    2.  Run CLIP RN50 stem + all 4 residual layers, collecting 5 feature maps:
            res1 : (B, 64,   H/4,  W/4)   — stem output (after avgpool)
            res2 : (B, 256,  H/4,  W/4)   — layer1
            res3 : (B, 512,  H/8,  W/8)   — layer2  ← FPN output level
            res4 : (B, 1024, H/16, W/16)  — layer3
            res5 : (B, 2048, H/32, W/32)  — layer4
        Mirrors ModifiedResNetFeatures.forward() from the reference exactly.
    3.  [Gap #1 FIX] Pass all 5 maps through an EfficientFeaturePyramidNetwork
        (top-down multi-scale fusion) that outputs a single feature map at
        res3 resolution (H/8, W/8) with `embedding_dim` channels.
        Mirrors encoder3d: feature_pyramid([64,256,512,1024,2048], emb, "res3").
    4.  [Gap #2] Vision→Language cross-attention (2 layers, no RoPE):
        every spatial pixel token attends to all language tokens, grounding
        visual features to the task instruction — mirrors encoder3d.vl_attention.
    5.  Scatter-add per-pixel features into the 3D voxel grid using the point
        cloud, then average per occupied voxel cell.
    6.  A final LayerNorm produces (B, embedding_dim, V, V, V).

    This output is drop-in compatible with the 3D-CNN output so the rest of
    VoxelFlowEncoder (TransformerHead, GS renderer) is completely unchanged.

    Parameters
    ----------
    clip_model : nn.Module
        The CLIP visual encoder (ModifiedResNet), e.g. `clip_full.visual`.
    embedding_dim : int
        Output feature dimension — equals VoxelFlowEncoder.embedding_dim.
        The FPN projects all feature levels to this width directly.
    voxel_size : int
        Number of voxel cells per side (default 100).
    coordinate_bounds : torch.Tensor, shape (6,)
        [x_min, y_min, z_min, x_max, y_max, z_max] workspace bounds.
    num_attn_heads : int
        Number of heads for vl_attention (must divide embedding_dim).
    num_vl_attn_layers : int
        Cross-attention layers for vision→language grounding (default 2).
    finetune_backbone : bool
        Whether to allow gradients through the CLIP backbone (default False).
    """

    # CLIP RN50 image normalisation constants
    _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

    # RN50 channel widths at each residual level (width=64, Bottleneck ×4)
    # res1: stem post-avgpool  → 64   (= width)
    # res2: layer1             → 256  (= width * 4)
    # res3: layer2             → 512  (= width * 8)
    # res4: layer3             → 1024 (= width * 16)
    # res5: layer4             → 2048 (= width * 32)
    _RN50_CHANNELS = [64, 256, 512, 1024, 2048]

    def __init__(
        self,
        clip_model: nn.Module,
        embedding_dim: int = 120,
        voxel_size: int = 100,
        coordinate_bounds: torch.Tensor = None,
        num_attn_heads: int = 8,
        num_vl_attn_layers: int = 2,
        finetune_backbone: bool = False,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.embedding_dim = embedding_dim
        self.clip_visual = clip_model   # ModifiedResNet (CLIP RN50 visual encoder)

        if not finetune_backbone:
            for p in self.clip_visual.parameters():
                p.requires_grad = False

        # Register CLIP normalisation as buffers (moves with .to(device))
        mean = torch.tensor(self._CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(self._CLIP_STD,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('clip_mean', mean)
        self.register_buffer('clip_std',  std)

        if coordinate_bounds is not None:
            self.register_buffer('coord_bounds', coordinate_bounds.float())
        else:
            self.register_buffer(
                'coord_bounds',
                torch.tensor([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6], dtype=torch.float32)
            )

        # ---- [Gap #1 FIX] Feature Pyramid Network ----------------------- #
        # Mirrors encoder3d:
        #   feature_pyramid = EfficientFeaturePyramidNetwork(
        #       [64, 256, 512, 1024, 2048], embedding_dim, output_level="res3")
        # Fuses all 5 CLIP feature levels top-down and outputs a single map at
        # res3 resolution (H/8, W/8) with `embedding_dim` channels.
        self.feature_pyramid = _EfficientFPN(
            in_channels_list=self._RN50_CHANNELS,
            out_channels=embedding_dim,
            output_level='res3',
        )

        # ---- [Gap #2] Vision→Language cross-attention -------------------- #
        # Mirrors encoder3d.vl_attention exactly:
        #   AttentionModule(num_layers=2, d_model=emb, dim_fw=4*emb,
        #                   n_heads=..., pre_norm=False, rotary_pe=False,
        #                   use_adaln=False, is_self=False)
        self.vl_attention = AttentionModule(
            num_layers=num_vl_attn_layers,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=False,
            is_self=False,
        )

        # Final LayerNorm on the lifted volume.
        # voxel_token_proj in VoxelFlowEncoder handles embedding_dim→embedding_dim.
        self.out_norm = nn.LayerNorm(embedding_dim)

    # ---------------------------------------------------------------------- #

    def _extract_multi_level(self, x: torch.Tensor) -> OrderedDict:
        """
        Run CLIP RN50 stem + all 4 residual layers and return all 5 feature maps.

        Replicates ModifiedResNetFeatures.forward() from the reference:
            res1 : stem output  (B, 64,   H/4,  W/4)
            res2 : layer1       (B, 256,  H/4,  W/4)
            res3 : layer2       (B, 512,  H/8,  W/8)
            res4 : layer3       (B, 1024, H/16, W/16)
            res5 : layer4       (B, 2048, H/32, W/32)
        """
        v = self.clip_visual
        # Stem: three 3×3 convs → avgpool  (total stride 4)
        x = x.type(v.conv1.weight.dtype)
        x = v.relu(v.bn1(v.conv1(x)))
        x = v.relu(v.bn2(v.conv2(x)))
        x0 = v.relu(v.bn3(v.conv3(x)))   # (B, 64, H/2, W/2)
        x0 = v.avgpool(x0)               # (B, 64, H/4, W/4)  ← res1
        x1 = v.layer1(x0)               # (B, 256,  H/4, W/4) ← res2
        x2 = v.layer2(x1)               # (B, 512,  H/8, W/8) ← res3
        x3 = v.layer3(x2)               # (B, 1024, H/16,W/16)← res4
        x4 = v.layer4(x3)               # (B, 2048, H/32,W/32)← res5
        return OrderedDict([
            ('res1', x0.float()),
            ('res2', x1.float()),
            ('res3', x2.float()),
            ('res4', x3.float()),
            ('res5', x4.float()),
        ])

    def _normalize_clip(self, rgb: torch.Tensor) -> torch.Tensor:
        """[-1,1] → CLIP mean/std normalisation."""
        rgb_01 = (rgb * 0.5 + 0.5).clamp(0.0, 1.0)
        return (rgb_01 - self.clip_mean) / self.clip_std

    def _lift_to_voxel(
        self,
        feats_2d: torch.Tensor,   # (B, C, H_f, W_f)  features at FPN output resolution
        pcd:      torch.Tensor,   # (B, 3, H_img, W_img) world XYZ per pixel
    ) -> torch.Tensor:
        """
        Lift 2D per-pixel features into a 3D voxel volume via scatter_add.

        Returns
        -------
        voxel_vol : (B, C, V, V, V)  averaged features per voxel cell
        """
        B, C, H_f, W_f = feats_2d.shape
        V = self.voxel_size
        device = feats_2d.device

        bb_min = self.coord_bounds[:3].to(device)
        bb_max = self.coord_bounds[3:].to(device)

        # Bilinearly downsample pcd to feature map resolution
        pcd_ds = F.interpolate(pcd.float(), size=(H_f, W_f),
                               mode='bilinear', align_corners=False)  # (B,3,H_f,W_f)

        pcd_flat  = pcd_ds.permute(0, 2, 3, 1).reshape(B, -1, 3)    # (B, N, 3)
        feat_flat = feats_2d.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, N, C)
        N = pcd_flat.shape[1]

        # Bin world coords into voxel grid [0, V-1]^3
        pcd_clamped = pcd_flat.clamp(
            min=bb_min.unsqueeze(0).unsqueeze(0),
            max=bb_max.unsqueeze(0).unsqueeze(0),
        )
        vox_float = (pcd_clamped - bb_min) / (bb_max - bb_min + 1e-8) * V
        vox_idx   = vox_float.long().clamp(0, V - 1)  # (B, N, 3)
        ix, iy, iz = vox_idx[..., 0], vox_idx[..., 1], vox_idx[..., 2]
        lin_idx = ix * (V * V) + iy * V + iz           # (B, N)

        voxel_sum   = torch.zeros(B, V*V*V, C, device=device, dtype=feat_flat.dtype)
        voxel_count = torch.zeros(B, V*V*V, 1, device=device, dtype=feat_flat.dtype)
        lin_idx_exp = lin_idx.unsqueeze(-1)
        voxel_sum.scatter_add_(1, lin_idx_exp.expand(-1, -1, C), feat_flat)
        voxel_count.scatter_add_(1, lin_idx_exp,
                                 torch.ones(B, N, 1, device=device, dtype=feat_flat.dtype))

        voxel_avg = voxel_sum / (voxel_count + 1e-8)               # (B, V^3, C)
        voxel_vol = voxel_avg.reshape(B, V, V, V, C).permute(0, 4, 1, 2, 3)  # (B,C,V,V,V)
        return voxel_vol

    def forward(
        self,
        rgb:         torch.Tensor,   # (B, 3, H, W)  ManiFlow RGB in [-1,1]
        pcd:         torch.Tensor,   # (B, 3, H, W)  world XYZ per pixel
        lang_tokens: torch.Tensor,   # (B, L, embedding_dim)  projected lang tokens
    ):
        """
        Returns
        -------
        (voxel_feat, [])
            voxel_feat : (B, embedding_dim, V, V, V)
        """
        # 1. CLIP normalise
        rgb_clip = self._normalize_clip(rgb)

        # 2. Extract all 5 CLIP feature levels
        _frozen = not any(p.requires_grad for p in self.clip_visual.parameters())
        if _frozen:
            with torch.no_grad():
                feat_dict = self._extract_multi_level(rgb_clip)
        else:
            feat_dict = self._extract_multi_level(rgb_clip)

        # 3. [Gap #1] FPN: multi-scale fusion → res3 at (B, emb, H/8, W/8)
        fpn_out  = self.feature_pyramid(feat_dict)   # OrderedDict {'res3': ...}
        feat2d   = fpn_out['res3']                   # (B, embedding_dim, H/8, W/8)
        B, C_emb, H_f, W_f = feat2d.shape

        # 4. [Gap #2] Vision→Language cross-attention
        N_pix = H_f * W_f
        vis_tokens = feat2d.permute(0, 2, 3, 1).reshape(B, N_pix, C_emb)  # (B, N, emb)
        vis_tokens = self.vl_attention(
            seq1=vis_tokens,
            seq2=lang_tokens,
        )[-1]                                                                # (B, N, emb)
        feat2d = vis_tokens.reshape(B, H_f, W_f, C_emb).permute(0, 3, 1, 2)  # (B,emb,H,W)

        # 5. Lift into voxel volume
        voxel_vol = self._lift_to_voxel(feat2d, pcd)   # (B, emb, V, V, V)

        # 6. Final LayerNorm
        B2, C2, V1, V2, V3 = voxel_vol.shape
        out = self.out_norm(
            voxel_vol.permute(0, 2, 3, 4, 1).reshape(-1, C2)
        ).reshape(B2, V1, V2, V3, C2).permute(0, 4, 1, 2, 3)   # (B, emb, V, V, V)
        return out, []


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
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
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
            dim_fw=embedding_dim,
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

        # ---- Rotation head (6D ortho velocity → 6) ----------------------- #
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
        voxel_tokens:     torch.Tensor,            # (B, N_vox, embedding_dim)
        voxel_pos:        torch.Tensor,            # (B, N_vox, 3)  world coords
        lang_tokens:      torch.Tensor,            # (B, L, embedding_dim)
        proprio_feats:    torch.Tensor = None,     # (B, embedding_dim)  pre-projected
        coord_bounds:     torch.Tensor = None,     # (6,) [x_min,y_min,z_min,x_max,y_max,z_max]
    ):
        """
        Returns a list of length 1 containing (B, T, 10):
            [pos_velocity(3), rot_velocity(6), grip_logit(1)]
        The list wrapper exactly matches base_denoise_actor's return convention
        (which may return multiple intermediate predictions for deep supervision;
        here we always return a single prediction).
        """
        B, T, _ = noisy_trajectory.shape

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
        rel_vox_pos  = self.relative_pe_layer(voxel_pos)          # (B, N, emb, 2)
        # Trajectory xyz positions: unnormalize from [-1,1] to world coords
        # so that RoPE encodes the same spatial scale as voxel_pos.
        # Mirrors base_denoise_actor where traj_xyz = unnormalize_pos(traj)[..., :3].
        # The noisy trajectory itself stays normalised for the flow process;
        # unnormalization is only for RoPE to match voxel_pos coordinate space.
        traj_xyz_norm = noisy_trajectory[:, :, :3]                # (B, T, 3) in [-1,1]
        if coord_bounds is not None:
            ws_min = coord_bounds[:3].to(traj_xyz_norm.device)    # (3,)
            ws_max = coord_bounds[3:].to(traj_xyz_norm.device)    # (3,)
            # [-1,1] → world coords: x_world = (x_norm + 1) / 2 * (max - min) + min
            traj_xyz = (traj_xyz_norm + 1.0) / 2.0 * (ws_max - ws_min) + ws_min
        else:
            # Fallback: use normalised coords if bounds not supplied
            traj_xyz = traj_xyz_norm
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

        # ---- Cross-attention: trajectory -> voxel scene ------------------ #
        traj_feats = self.cross_attn(
            seq1=traj_feats,
            seq2=voxel_tokens,
            seq1_pos=rel_traj_pos,
            seq2_pos=rel_vox_pos,
            ada_sgnl=time_embs,
        )[-1]   # (B, T, emb)

        # ---- Joint self-attention (trajectory tokens + FPS voxel tokens) - #
        # Exactly mirrors base_denoise_actor: features = cat([traj, fps_scene])
        joint_feats = torch.cat([traj_feats, voxel_tokens], dim=1)  # (B, T+N, emb)
        joint_pos   = torch.cat([rel_traj_pos, rel_vox_pos], dim=1)

        joint_feats = self.self_attn(
            seq1=joint_feats,
            seq2=joint_feats,
            seq1_pos=joint_pos,
            seq2_pos=joint_pos,
            ada_sgnl=time_embs,
        )[-1]   # (B, T+N, emb)

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
        # CLIP backbone option
        use_clip_backbone: bool = False,   # True → replace 3D-CNN with CLIPVoxelLifter
        finetune_clip_backbone: bool = False,
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
        self.use_clip_backbone      = use_clip_backbone

        if coordinate_bounds is None:
            coordinate_bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        self.register_buffer(
            'coord_bounds',
            torch.tensor(coordinate_bounds, dtype=torch.float32)
        )

        # ------------------------------------------------------------------ #
        # 1.  Scene encoder                                                    #
        #     Option A (default): 3D-CNN on voxelised RGB                     #
        #     Option B:           CLIP RN50 + feature-lifting connector        #
        # ------------------------------------------------------------------ #
        if use_clip_backbone:
            from helpers.clip.core.clip import build_model, load_clip
            clip_raw, _ = load_clip('RN50', jit=False, device='cpu')
            clip_full   = build_model(clip_raw.state_dict())
            del clip_raw
            self.encoder_3d = None   # not used
            self.clip_lifter = CLIPVoxelLifter(
                clip_model=clip_full.visual,
                embedding_dim=embedding_dim,
                voxel_size=voxel_size,
                coordinate_bounds=torch.tensor(coordinate_bounds, dtype=torch.float32),
                num_attn_heads=num_attn_heads,
                num_vl_attn_layers=2,
                finetune_backbone=finetune_clip_backbone,
            )
            # Project the lifted voxel volume from embedding_dim → im_channels so
            # the GS renderer (GeneralizableGSEmbedNet) always receives the
            # expected d_latent=im_channels channels via F.grid_sample.
            # A 1×1×1 conv preserves spatial dims and adds no receptive-field bias.
            if embedding_dim != im_channels:
                self.clip_renderer_proj = nn.Conv3d(
                    embedding_dim, im_channels, kernel_size=1, bias=False
                )
            else:
                self.clip_renderer_proj = nn.Identity()
        else:
            self.clip_lifter = None
            self.clip_renderer_proj = None
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

        # When use_clip_backbone=True, CLIPVoxelLifter outputs embedding_dim
        # channels directly; when False, 3D-CNN outputs im_channels.
        _vox_in_channels = embedding_dim if use_clip_backbone else im_channels
        self.voxel_token_proj = nn.Sequential(
            nn.Linear(_vox_in_channels, embedding_dim),
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
        # 5.  Proprio encoding                                                #
        #     [Gap #4] Mirrors encoder3d.gripper_context_head:               #
        #       a) Learnable base token + linear projection of raw proprio    #
        #          scalars (gripper_open, finger_pos×2, time) onto it        #
        #       b) 3-layer cross-attention: gripper token → voxel tokens     #
        #          with 3D RoPE on both sides                                 #
        #     The output (B, 1, embedding_dim) is squeezed to (B, embedding  #
        #     _dim) for the curr_gripper_emb inside TransformerHead.         #
        # ------------------------------------------------------------------ #
        # Shared 3D RoPE layer used by both gripper_context_head and
        # TransformerHead (both need to encode world-space positions).
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Always-active: gripper_context_head runs unconditionally.
        # proprio_proj injects the raw scalar values (gripper_open, finger
        # positions, time) into the learnable base token so the cross-attention
        # token is genuinely informative, not a constant embedding.
        # When low_dim_size == 0, proprio_proj is a no-op (nn.Identity on a
        # zero tensor), and the base embedding alone is used.
        self.curr_gripper_embed = nn.Embedding(1, embedding_dim)
        if low_dim_size > 0:
            # Linear: low_dim_size → embedding_dim  (no bias; base embed provides bias)
            self.proprio_proj = nn.Linear(low_dim_size, embedding_dim, bias=False)
        else:
            self.proprio_proj = None
        # 3-layer cross-attention: gripper token → scene (voxel) tokens
        # Mirrors encoder3d.gripper_context_head exactly:
        #   AttentionModule(num_layers=3, d_model=emb, dim_fw=emb,
        #                   n_heads=..., rotary_pe=True, use_adaln=False,
        #                   pre_norm=False, is_self=False)
        self.gripper_context_head = AttentionModule(
            num_layers=3,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=True,
            use_adaln=False,
            is_self=False,
        )

        # ------------------------------------------------------------------ #
        # 6.  Transformer denoising head                                      #
        # ------------------------------------------------------------------ #
        self.transformer_head = TransformerHead(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=rot_dim,
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
        xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing='ij')
        grid = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # (N, 3)
        grid = grid * (bb_max - bb_min).to(device) + bb_min.to(device)
        return grid.unsqueeze(0).expand(B, -1, -1)   # (B, N, 3)

    @staticmethod
    @torch.no_grad()
    def _dps(features: torch.Tensor, n_samples: int, k: int = 8) -> torch.Tensor:
        """
        Density-Based Sampling in **feature space** — mirrors the reference
        ``base_encoder.py::density_based_sampler`` exactly.

        Keeps the M tokens that are *most isolated* in embedding space (highest
        average distance to their k nearest neighbours).  For a voxel grid this
        discards the large sea of near-identical empty/background voxels and
        retains the informative object-surface tokens where features are
        distinctive — the opposite of what FPS on spatial coordinates would do.

        Parameters
        ----------
        features  : (B, N, C)  voxel token embeddings
        n_samples : int         number of tokens to keep (M <= N)
        k         : int         number of nearest neighbours for density estimate

        Returns
        -------
        idx : (B, M)  indices into the N dimension

        Implementation note
        -------------------
        Naively computing the full (B, N, N) pairwise distance matrix via
        torch.cdist would require ~1 GB of GPU memory at B=4, N=8000 and
        cause an OOM.  Instead we compute approximate kNN densities by
        chunking the query dimension so that only a (B, chunk, N) slice
        is live at any one time.  The chunk size is chosen so the slice
        fits comfortably in ~256 MB regardless of B, N, or C.
        """
        B, N, C = features.shape
        device  = features.device

        if n_samples >= N:
            return torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

        k_eff = min(k + 1, N)
        M = min(n_samples, N)

        # Chunk size: keep peak memory below ~256 MB.
        # Each (B, chunk, N) float32 slice = B * chunk * N * 4 bytes.
        # With B≤8 and N≤8000: chunk = 256 MB / (B * N * 4).
        max_bytes  = 256 * 1024 * 1024   # 256 MB
        chunk_size = max(1, max_bytes // (B * N * 4))
        chunk_size = min(chunk_size, N)

        density = torch.zeros(B, N, device=device)

        f_norm = features  # (B, N, C)
        for start in range(0, N, chunk_size):
            end   = min(start + chunk_size, N)
            q     = f_norm[:, start:end, :]          # (B, chunk, C)
            # (B, chunk, N) — pairwise L2 between the chunk queries and all keys
            d_chunk = torch.cdist(q, f_norm, p=2)    # (B, chunk, N)
            knn_d, _ = d_chunk.topk(k=k_eff, dim=-1, largest=False)
            density[:, start:end] = knn_d[:, :, 1:].mean(dim=-1)

        # Keep the M most sparse / most distinctive tokens
        idx = density.topk(M, dim=-1, largest=True).indices   # (B, M)
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
        # Optional inputs used by CLIPVoxelLifter (ignored when use_clip_backbone=False)
        rgb:             torch.Tensor = None,   # (B, 3, H, W)  raw RGB in [-1,1]
        pcd:             torch.Tensor = None,   # (B, 3, H, W)  world XYZ per pixel
        gripper_pose:    torch.Tensor = None,   # (B, 7) [x,y,z,qx,qy,qz,qw] current gripper pose
    ):
        """
        Encodes the raw inputs into voxel and language features.

        Args
        ----
        voxel_grid       : (B, C_init, V, V, V)  raw voxel grid input
        proprio          : (B, low_dim_size)
        lang_goal_emb    : (B, lang_feat_dim)
        lang_token_embs  : (B, 77, lang_emb_dim)
        rgb              : (B, 3, H, W)  — only used when use_clip_backbone=True
        pcd              : (B, 3, H, W)  — only used when use_clip_backbone=True

        Returns
        -------
        voxel_grid_feature : (B, im_channels, V, V, V)  — for GS renderer
        lang_embedd        : (B, seq, lang_out_dim)      — for GS renderer
        context            : dict with keys:
            'voxel_tokens'  (B, M, embedding_dim)
            'voxel_pos'     (B, M, 3)
            'lang_tokens'   (B, L, embedding_dim)
            'proprio_feats' (B, embedding_dim) or None
        """
        B      = voxel_grid.shape[0]
        device = voxel_grid.device

        # ---- Language tokens (must be available before scene encoding) -- #
        # Project raw lang embeddings to embedding_dim first so they can be
        # passed to CLIPVoxelLifter.vl_attention (Gap #2).
        if self.lang_fusion_type == 'seq':
            l = self.lang_preprocess(lang_token_embs.float())      # (B, 77, im_C*2)
        else:
            l = self.lang_preprocess(lang_goal_emb.float()).unsqueeze(1)  # (B,1,im_C)
        # Project to embedding_dim for attention use
        lang_tokens_emb = self.lang_token_proj(l)   # (B, L, embedding_dim)

        # ---- Scene encoding --------------------------------------------- #
        if self.use_clip_backbone:
            # CLIP ResNet50 backbone + vl_attention + feature-lifting connector
            assert rgb is not None and pcd is not None, (
                "rgb and pcd must be provided to encode_scene when use_clip_backbone=True"
            )
            # Pass lang_tokens_emb so vl_attention can ground visual features.
            # Returns (B, embedding_dim, V, V, V).
            d0_emb, _ = self.clip_lifter(rgb, pcd, lang_tokens=lang_tokens_emb)
            # Project to im_channels for the GS renderer (d_latent=im_channels).
            # Transformer tokens are built from d0_emb (embedding_dim) below.
            d0 = self.clip_renderer_proj(d0_emb)  # (B, im_channels, V, V, V)
        else:
            # 3D-CNN on voxelised RGB (default)
            d0, _ = self.encoder_3d(voxel_grid)   # (B, im_channels, V, V, V)
            d0_emb = d0                            # same tensor; same channels

        # Downsample voxel volume -> token sequence
        ds = self._vox_ds
        # Use d0_emb (embedding_dim channels) for Transformer tokens so the
        # richer CLIP features are used for policy; d0 (im_channels) is only
        # needed for the GS renderer.
        vox_ds     = F.adaptive_avg_pool3d(d0_emb, (ds, ds, ds))    # (B, _vox_in_C, ds,ds,ds)
        vox_tokens = rearrange(vox_ds, 'b c x y z -> b (x y z) c')  # (B, N, _vox_in_C)

        # Project to embedding_dim
        vox_tokens  = self.voxel_token_proj(vox_tokens)    # (B, N, emb)
        # lang_tokens already projected above; keep alias for context dict
        lang_tokens = lang_tokens_emb                       # (B, L, emb)

        # World-space positions for 3D RoPE
        voxel_pos = self._voxel_world_coords(B, device)    # (B, N, 3)

        # Add learned absolute positional encoding
        vox_tokens = vox_tokens + self.voxel_pos_emb(voxel_pos)

        # DPS subsampling: reduce N -> M for joint self-attention.
        # DPS works in feature space (not spatial coords), so it naturally
        # discards the many near-identical empty/background voxels and keeps
        # the informative object-surface tokens — matches base_encoder.run_dps.
        if self.num_fps_tokens > 0:
            fps_idx    = self._dps(vox_tokens, self.num_fps_tokens)  # (B, M) — feature-space DPS
            idx_exp    = fps_idx.unsqueeze(-1).expand(-1, -1, vox_tokens.shape[-1])
            fps_tokens = torch.gather(vox_tokens, 1, idx_exp)        # (B, M, emb)
            fps_pos    = torch.gather(voxel_pos, 1,
                                      fps_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B,M,3)
        else:
            fps_tokens, fps_pos = vox_tokens, voxel_pos

        # ---- [Gap #4] Proprio encoding via gripper_context_head ---------- #
        # Learnable base token + raw proprio scalars injected via proprio_proj.
        # gripper_context_head always runs (unconditional — no longer gated on
        # `proprio is not None`).  When proprio is unavailable, the base embed
        # alone is used (proprio_proj contribution is zero).
        #
        # Token construction (mirrors encoder3d.encode_proprio):
        #   gripper_feats = curr_gripper_embed + proprio_proj(proprio)
        #   → 3-layer cross-attn: gripper token → voxel scene tokens (3D RoPE)
        gripper_feats = self.curr_gripper_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        # (B, 1, emb) — contiguous for rotary PE bmm

        if self.proprio_proj is not None and proprio is not None:
            # proprio: (B, low_dim_size) — inject scalars into the token
            proprio_offset = self.proprio_proj(proprio.float().to(device))  # (B, emb)
            gripper_feats = gripper_feats + proprio_offset.unsqueeze(1)     # (B, 1, emb)

        # Gripper world-xyz for 3D RoPE.
        # `gripper_pose` (B,7) = [x,y,z,qx,qy,qz,qw] is forwarded from the
        # replay sample during training and from the observation dict at eval.
        # If unavailable, fall back to the spatial mean of the point-cloud so
        # the token still carries a plausible scene-centre position.
        if gripper_pose is not None:
            gripper_xyz = gripper_pose[:, :3].float().to(device)    # (B, 3)
        elif pcd is not None:
            # pcd: (B, 3, H, W) — mean over spatial dims → (B, 3) approx centre
            gripper_xyz = pcd.float().to(device).flatten(2).mean(dim=2)  # (B, 3)
        else:
            gripper_xyz = torch.zeros(B, 3, device=device, dtype=torch.float32)
        gripper_pos = self.relative_pe_layer(
            gripper_xyz.unsqueeze(1)                            # (B, 1, 3)
        )                                                       # (B, 1, emb, 2) RoPE

        # Scene token 3D RoPE positions
        scene_pos = self.relative_pe_layer(fps_pos)             # (B, M, emb, 2)

        # 3-layer cross-attention: gripper → scene
        gripper_feats = self.gripper_context_head(
            seq1=gripper_feats,
            seq2=fps_tokens,
            seq1_pos=gripper_pos,
            seq2_pos=scene_pos,
        )[-1]                                                    # (B, 1, emb)

        proprio_feats = gripper_feats.squeeze(1)                 # (B, emb)

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
            voxel_tokens  = context['voxel_tokens']
            voxel_pos     = context['voxel_pos']
            lang_tokens   = context['lang_tokens']
            proprio_feats = context.get('proprio_feats', None)
        else:
            B = noisy_trajectory.shape[0]
            device = noisy_trajectory.device
            voxel_tokens  = torch.zeros(B, 1, self.embedding_dim, device=device)
            voxel_pos     = torch.zeros(B, 1, 3, device=device)
            lang_tokens   = torch.zeros(B, 1, self.embedding_dim, device=device)
            proprio_feats = None

        return self.transformer_head(
            noisy_trajectory=noisy_trajectory.float(),
            timestep=t.float(),
            voxel_tokens=voxel_tokens.float(),
            voxel_pos=voxel_pos.float(),
            lang_tokens=lang_tokens.float(),
            proprio_feats=proprio_feats.float() if proprio_feats is not None else None,
            coord_bounds=self.coord_bounds,   # pass workspace bounds for RoPE unnormalization
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
        rgb:          torch.Tensor = None,   # (B, 3, H, W) — used when use_clip_backbone=True
        pcd:          torch.Tensor = None,   # (B, 3, H, W) — used when use_clip_backbone=True
        gripper_pose: torch.Tensor = None,   # (B, 7) [x,y,z,qx,qy,qz,qw] current gripper pose
    ):
        """
        Arguments
        ---------
        voxel_grid       : (B, C_init, V, V, V)  raw voxel grid input
        proprio          : (B, low_dim_size)
        lang_goal_emb    : (B, lang_feat_dim)
        lang_token_embs  : (B, seq, lang_emb_dim)
        prev_layer_voxel_grid, bounds, prev_layer_bounds, mask: ignored (legacy)
        rgb              : (B, 3, H, W)  — required when use_clip_backbone=True
        pcd              : (B, 3, H, W)  — required when use_clip_backbone=True
        gripper_pose     : (B, 7)        — current gripper world pose [x,y,z,qx,qy,qz,qw]

        Returns
        -------
        voxel_grid_feature : (B, im_channels, V, V, V)
        lang_embedd        : (B, seq, lang_out_dim)
        context            : dict  {'voxel_tokens', 'voxel_pos', 'lang_tokens', 'proprio_feats'}
        """
        voxel_grid_feature, lang_embedd, context = self.encode_scene(
            voxel_grid, proprio, lang_goal_emb, lang_token_embs,
            rgb=rgb, pcd=pcd, gripper_pose=gripper_pose,
        )
        return voxel_grid_feature, lang_embedd, context

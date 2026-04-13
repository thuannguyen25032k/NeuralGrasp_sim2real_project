# ManiFlow BC — Complete Workflow

> **One-sentence summary:** ManiFlow BC is a **rectified-flow robot manipulation policy** that encodes a 3D voxel scene with a CNN+Transformer denoising head to predict continuous 8-DoF end-effector actions, with an **optional** Gaussian Splatting neural-rendering auxiliary loss for richer scene supervision.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Shared Backbone (Both Variants)](#2-shared-backbone-both-variants)
3. [Variant A — ManiFlow Without Neural Rendering](#3-variant-a--maniflow-without-neural-rendering)
4. [Variant B — ManiFlow With Neural Rendering](#4-variant-b--maniflow-with-neural-rendering)
5. [Training Loop Comparison](#5-training-loop-comparison)
6. [Inference Step-by-Step](#6-inference-step-by-step)
7. [Module Inventory](#7-module-inventory)
8. [Key Design Decisions](#8-key-design-decisions)
9. [Configuration & Scripts](#9-configuration--scripts)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          MANIFLOW BC — OVERVIEW                          │
│                                                                          │
│  Inputs                                                                  │
│  ──────                                                                  │
│   RGB-D cameras ──► Point Cloud ──► VoxelGrid                           │
│   Language goal ──► CLIP encoder ──► (sentence emb, token embs)         │
│   Proprioception ──► gripper state (4-DoF)                              │
│                                                                          │
│                     ┌─────────────────────────────────────────┐         │
│                     │      VoxelFlowEncoder                   │         │
│                     │  3D-CNN ──► voxel tokens                │         │
│                     │  Language proj ──► lang tokens          │         │
│                     │  FPS subsampling (N → 512 tokens)       │         │
│                     │  3D RoPE position encoding              │         │
│                     │  Proprio projection ──► AdaLN signal    │         │
│                     └────────────────┬────────────────────────┘         │
│                                      │  context dict                    │
│                    ┌─────────────────▼───────────────────────┐          │
│                    │       TransformerHead (denoising)        │          │
│                    │   z_T ~ N(0,I)  [noisy action]          │          │
│                    │   for t = T … 1 (Euler):                │          │
│                    │     v = predict_velocity(z_t, t, ctx)   │          │
│                    │     z_{t-1} = z_t − Δt · v             │          │
│                    │   z_0 = clean action (B, 8)             │          │
│                    └─────────────────────────────────────────┘          │
│                                                                          │
│  ┌─ WITH neural rendering ────────────────────────────────────────────┐ │
│  │  voxel_grid_feature + lang_embedd ──► NeuralRenderer               │ │
│  │  (GeneralizableGSEmbedNet) ──► Gaussian parameters                 │ │
│  │  ──► diff. rasterizer ──► rendered_rgb, rendered_embed             │ │
│  │  ──► L_rgb + L_embed (auxiliary supervision)                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  Output action: [x, y, z, qx, qy, qz, qw, gripper_open]  (8-DoF)       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Shared Backbone (Both Variants)

Both variants share the same input processing, voxelisation, scene encoding, and denoising loop.  The only difference is whether `NeuralRenderer` is instantiated and its auxiliary loss is added to the total.

### 2.1 Input Processing

```
RGB-D frames (N cameras)                Language goal string
┌──────────────────────┐                ┌─────────────────────────────┐
│  (B, 3, H, W) × N   │                │  "pick up the red cup"      │
└──────────┬───────────┘                └──────────┬──────────────────┘
           │                                       │
           ▼                                       ▼
┌──────────────────────┐                ┌─────────────────────────────┐
│  Point Cloud PCDs    │                │  CLIP encoder (frozen)      │
│  (B, N·H·W, 3)       │                │  lang_goal_emb   (B, 1024)  │
└──────────┬───────────┘                │  lang_token_embs (B, 77,512)│
           │                            └──────────┬──────────────────┘
           ▼                                       │
┌──────────────────────────────────────────────────┼───────────┐
│  VoxelGrid.coords_to_bounding_voxel_grid(...)    │           │
│  scene_bounds = [-0.3,-0.5,0.6, 0.7,0.5,1.6]    │           │
│  voxel_grid (B, 10, V, V, V)   V=100             │ Proprio   │
└──────────────────────────────────────────────────┴───────────┘
```

- **`gpu_preprocessor.py`** handles per-camera image cropping and point-cloud projection on the GPU.
- **`VoxelGrid`** (`voxel/voxel_grid.py`) bins the point cloud into a 100³ voxel volume with 10 channels (RGB + occupancy + extras).
- **CLIP** is loaded frozen; both the 1024-D sentence embedding and 77-token × 512-D token embeddings are kept.

---

### 2.2 VoxelFlowEncoder — Scene Encoding

**File:** `agents/maniflow_bc/voxel_flow_encoder.py`  
**Class:** `VoxelFlowEncoder`

```
voxel_grid (B, 10, V, V, V)
        │
        ▼ Step A — 3D-CNN
MultiLayer3DEncoderShallow (helpers/network_utils.py)
        │
        ▼
d0  (B, im_channels=64, V, V, V)   ──────────────────► voxel_grid_feature
                                                         (for GS renderer)

lang_token_embs (B, 77, 512)
        │
        ▼ Step B — Language pre-processing
nn.Linear(512 → im_channels×2 = 128)
        │
        ▼
l   (B, 77, 128)  ────────────────────────────────────► lang_embedd
                                                         (for GS renderer)

d0 (B, 64, V, V, V)
        │
        ▼ Step C — Voxel token extraction
AdaptiveAvgPool3d → (B, 64, ds, ds, ds)    ds = V / 5 = 20
rearrange → (B, ds³, 64)  =  (B, 8000, 64)
nn.Linear(64 → embedding_dim=192) + LayerNorm
        │
        ▼
vox_tokens (B, 8000, 192)

        │
        ▼ Step D — World-space 3D positions + learned pos embedding
_voxel_world_coords()  →  voxel_pos  (B, 8000, 3)
PositionEmbeddingLearnedMLP(voxel_pos)  →  pos_emb  (B, 8000, 192)
vox_tokens += pos_emb

        │
        ▼ Step E — Farthest Point Sampling (FPS)
_fps(voxel_pos, num_fps_tokens=512)
fps_tokens (B, 512, 192),  fps_pos (B, 512, 3)

proprio (B, 4)
        │
        ▼ Step F — Proprio projection
nn.Linear(4 → 192) + ReLU
        │
        ▼
proprio_feats (B, 192)

─────────────────────────────────────────────
context = {
    'voxel_tokens':  fps_tokens   (B, 512, 192)
    'voxel_pos':     fps_pos      (B, 512, 3)
    'lang_tokens':   lang_tokens  (B, 77, 192)
    'proprio_feats': proprio_feats (B, 192)
}
─────────────────────────────────────────────
```

---

### 2.3 TransformerHead — Velocity Prediction

**File:** `agents/maniflow_bc/voxel_flow_encoder.py`  
**Class:** `TransformerHead`

At each denoising step the head receives `(noisy_trajectory, t, context)` and outputs the predicted rectified-flow velocity.

```
noisy_trajectory (B, T, 9)   ← [pos_norm(3) + rot6D(6)]
timestep         (B,)        ← t ∈ [0, 1]
─────────────────────────────────────────────────────────────────────

┌─ 1. AdaLN conditioning signal ────────────────────────────────────┐
│  SinusoidalPosEmb(t) → time_emb   (B, 192)                        │
│  proprio_feats ───────► curr_gripper_emb  (B, 192)                │
│  ada_sgnl = time_emb + curr_gripper_emb   (B, 192)                │
└───────────────────────────────────────────────────────────────────┘

┌─ 2. Encode trajectory tokens ─────────────────────────────────────┐
│  noisy_trajectory ──► nn.Linear(9→192) ──► traj_feats  (B, T, 192)│
│  SinusoidalPosEmb([0…T]) ──────────────► traj_time_pos (B, T, 192)│
└───────────────────────────────────────────────────────────────────┘

┌─ 3. 3D RoPE positions ────────────────────────────────────────────┐
│  RotaryPositionEncoding3D(voxel_pos)  → rel_vox_pos  (B,512,192,2)│
│  RotaryPositionEncoding3D(traj_xyz)   → rel_traj_pos (B, T,192,2)│
└───────────────────────────────────────────────────────────────────┘

┌─ 4. Cross-attention: trajectory → language  (1 layer, no RoPE) ──┐
│  Q = traj_feats  (B, T, 192)                                      │
│  K,V = lang_tokens (B, 77, 192)                                   │
│  → traj_feats  (B, T, 192)  +  traj_time_pos                     │
└───────────────────────────────────────────────────────────────────┘

┌─ 5. Cross-attention: trajectory → voxel scene  (2 layers, AdaLN) ┐
│  Q = traj_feats  (B, T, 192)  + rel_traj_pos  [3D RoPE]          │
│  K,V = fps_tokens (B,512,192) + rel_vox_pos   [3D RoPE]          │
│  AdaLN conditioned on ada_sgnl                                    │
│  → traj_feats  (B, T, 192)                                        │
└───────────────────────────────────────────────────────────────────┘

┌─ 6. Joint self-attention  (4 layers, AdaLN, 3D RoPE) ────────────┐
│  joint = cat([traj_feats, fps_tokens], dim=1)  (B, T+512, 192)   │
│  joint_pos = cat([rel_traj_pos, rel_vox_pos])                     │
│  Tokens attend to each other across action + scene                │
│  → joint_feats  (B, T+512, 192)                                   │
└───────────────────────────────────────────────────────────────────┘

┌─ 7. Specialised output heads ─────────────────────────────────────┐
│                                                                   │
│  ┌─ Position head ──────────────────────────────────────────────┐ │
│  │  pos_self_attn (2 layers, AdaLN, RoPE)                       │ │
│  │  slice traj tokens [:, :T, :]                                │ │
│  │  position_proj → position_predictor → pos_vel  (B, T, 3)    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─ Rotation head ──────────────────────────────────────────────┐ │
│  │  rot_self_attn (2 layers, AdaLN, RoPE)                       │ │
│  │  rotation_proj → rotation_predictor → rot_vel  (B, T, 6)    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─ Gripper head ───────────────────────────────────────────────┐ │
│  │  openess_predictor (joint traj tokens) → grip_logit (B,T,1) │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  output = cat([pos_vel, rot_vel, grip_logit], dim=-1) (B, T, 10) │
└───────────────────────────────────────────────────────────────────┘
```

---

### 2.4 Rectified Flow Scheduler

**File:** `agents/maniflow_bc/rf_scheduler.py`  
**Class:** `RFScheduler`

**Training** — add noise and compute target velocity:
```
x_clean ∈ ℝ⁹          (normalised pos + rot6D per action step)
noise   ~ N(0, I)      (same shape)
t       ~ logit-Normal(μ=0, σ=1.5)   (biased toward mid-noise)

z_t     = (1 − t) · x_clean + t · noise      [linear interpolation]
target  = noise − x_clean                      [rectified-flow velocity]
```

**Inference** — Euler integration from pure noise:
```
z_T ~ N(0, I)
for idx, t in enumerate(timesteps):           # T=100 steps, descending
    v  = predict_velocity(z_t, t, context)
    z_{t-1} = z_t − (t − t_prev) · v         [Euler step]
z_0 = clean action trajectory
```

---

## 3. Variant A — ManiFlow Without Neural Rendering

**Config key:** `method.use_neural_rendering=False`  
**Launch script:** `scripts/train_maniflow_no_nerf.sh`

### 3.1 Forward Pass

```
observation (RGB-D, language, proprio)
        │
        ▼
QFunctionFlow.encode_scene()
        │  returns: (voxel_grid, voxel_feat, lang_embedd, context, bounds)
        ▼
NeuralRenderer  ─── SKIPPED (self._neural_renderer = None)
rendering_loss_dict = {}
        │
        ▼
context  ──────────────────────────────────────────────────────────────►
```

### 3.2 Training Loss (No Neural Rendering)

```
gt_action (B, 8)  ──► reshape to (B, T=1, nhand=1, 8)
                │
                ▼ _compute_flow_loss()

For i in range(lv2_batch_size):      # default 1; recommended 4
    noise    ~ N(0, I)               # (B, T, nhand, 9)
    t        ~ logit-Normal(0, 1.5)  # (B,)
    noisy_z  = RF_interpolate(x_clean, noise, t)
    pred_vel = predict_velocity(noisy_z, t, context)
    target   = noise − x_clean
    iter_loss = pos_weight * L1(pred_pos, target_pos)
              + rot_weight * L1(pred_rot, target_rot)
              + grip_weight * BCE(pred_grip, gt_grip)

L_flow = mean(iter_loss over lv2_batch_size)

─────────────────────────────────────────────────────────────────────────
L_total = λ_bc · L_flow               (λ_bc = 1.0 by default)
─────────────────────────────────────────────────────────────────────────
```

**Loss weights (from `ManiFlow_BC.yaml`):**

| Loss term     | Weight | Description |
|---------------|--------|-------------|
| `L_pos`       | 30.0   | L1 on xyz velocity |
| `L_rot`       | 10.0   | L1 on ortho6D rotation velocity |
| `L_grip`      | 1.0    | Binary cross-entropy on gripper logit |
| `λ_bc`        | 1.0    | Overall BC loss scale |

### 3.3 Data Flow Diagram (No Neural Rendering)

```
┌───────────────────────────────────────────────────────────────────────┐
│                   TRAINING STEP (no neural rendering)                 │
│                                                                       │
│  replay_sample                                                        │
│      │                                                                │
│      ├── RGB-D, pcd, proprio ──► encode_scene() ──► context          │
│      ├── lang_goal_emb, lang_token_embs ────────────────┘            │
│      │                                                                │
│      └── action_gt (B, 8)                                            │
│                │                                                      │
│                ▼                                                      │
│         reshape → (B, 1, 1, 8)                                       │
│                │                                                      │
│                ▼                                                      │
│         _compute_flow_loss(gt_traj, context)                         │
│                │                                                      │
│                ▼                                                      │
│         L_total = λ_bc · L_flow                                      │
│                │                                                      │
│                ▼                                                      │
│         backward() → clip_grad_norm(max=10) → optimizer.step()       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 4. Variant B — ManiFlow With Neural Rendering

**Config key:** `method.use_neural_rendering=True`  
**Launch script:** `scripts/train_maniflow_nerf.sh`

This variant adds a **Gaussian Splatting auxiliary loss** that provides dense scene geometry and semantic supervision.

### 4.1 Additional Data Requirements

Each RLBench episode must include **multi-view NeRF images** stored alongside the demo:

```
data/train_data/<task>/
    episode0/
        low_dim_obs.pkl
        variation0/
            ...
            nerf_multi_view_rgb/     ← N novel-view RGB images (e.g. 20)
            nerf_multi_view_depth/   ← corresponding depth maps
            nerf_multi_view_camera/  ← camera extrinsics + intrinsics
            nerf_next_multi_view_*/  ← same for next keyframe
```

### 4.2 NeuralRenderer Pipeline

**Files:** `agents/maniflow_bc/neural_rendering.py`, `agents/maniflow_bc/models_embed.py`

```
voxel_grid_feature (B, 64, V, V, V)   ← from 3D-CNN encoder
lang_embedd        (B, 77, 128)        ← from language pre-processor
rgb                (B, 3, H, W)        ← input camera image (first camera)
pcd                (B, 3, H, W)        ← input point cloud (first camera)
depth              (B, 1, H, W)        ← input depth (first camera)
focal, c           (B,), (B, 2)        ← camera intrinsics
gt_pose            (B, 4, 4)           ← novel-view camera extrinsic
        │
        ▼
GeneralizableGSEmbedNet.forward()
────────────────────────────────────────────────────────────────────────
  Step 1 — Lift points to 3D
    pcd  ──► world_to_canonical() ──► xyz in [0,1]³
    Optional positional encoding (PositionalEncoding)
    d_in: xyz(3) + code(39 if use_code else 0)

  Step 2 — Sample voxel features
    sample_in_canonical_voxel(xyz, voxel_feat)
    ──► point_feature  (B, N, 128)

  Step 3 — Regress Gaussian parameters (ResnetFC MLP)
    inputs: [xyz_encoded, point_feature, lang_embedd]
    ──► GS raw params (B, N, Σ_split_dims)
    split_dims = [xyz(3), opacity(1), scale(3), rotation(4),
                  feature_dc(3), feature(3), sh_higher(9)]

  Step 4 — Apply activations
    xyz      : raw offset  (clamp to workspace)
    opacity  : sigmoid
    scale    : exp
    rotation : normalize  (unit quaternion)

  (Optional) Step 5 — Dynamic Gaussian deformation
    gs_deformation_field(ResnetFC):
      inputs: [all_raw_params, xyz_encoded, lang, action]
      ──► Δxyz (3) + Δrot (4)   (per-Gaussian deformation)
    Next-frame Gaussians:
      xyz_next = xyz + Δxyz
      rot_next = normalize(rot + Δrot)

        │
        ▼
Differential Gaussian Rasterizer  (third_party/gaussian-splatting)
    Gaussians → rendered_rgb   (B, H, W, 3)
    Gaussians → rendered_embed (B, H, W, d_embed)
```

### 4.3 Foundation Model Feature Extraction

The semantic embedding target `gt_embed` is extracted from the **ground-truth novel-view RGB** using a frozen foundation model:

| Foundation model | Feature dims | Preprocessing |
|---|---|---|
| `diffusion` | 512 (LDM UNet blocks 5,7 + decoder 2,5) | Resize to 512×512 |
| `dinov2` | 1024 (ViT-L/14 patch tokens) | Resize to 224×8 px |

Both are then **dimensionality-reduced to `d_embed`** via batched PCA:

```python
# Vectorised batched PCA (avoids Python loop over batch)
A = feat.reshape(B, C, -1).permute(0, 2, 1)   # (B, H*W, C)
A -= A.mean(dim=1, keepdim=True)               # mean-centre
_, _, Vh = torch.linalg.svd(A, full_matrices=False)
gt_embed = torch.bmm(A, Vh[:, :d_embed].permute(0,2,1))
           .reshape(B, d_embed, H, W)          # (B, d_embed, H, W)
```

### 4.4 Rendering Losses

```
rendered_rgb   (B, H, W, 3)   ←─┐
gt_rgb         (B, H, W, 3)   ←─┤
                                 │
L_rgb   = λ_rgb · [ 0.8 · L1(rendered, gt)
                  + 0.2 · (1 − D-SSIM)(rendered, gt) ]

rendered_embed (B, H, W, d_embed)  ←─┐
gt_embed       (B, H, W, d_embed)  ←─┤
  (both mean-centred + L2-normalised  │
   for stable cosine matching)        │
L_embed = λ_embed · loss_embed_fn(rendered_embed, gt_embed)
          where loss_embed_fn ∈ {l2, cosine}

(Optional dynamic field)
rendered_next  (B, H, W, 3)  ← next-frame Gaussians rendered at next-pose
gt_next_rgb    (B, H, W, 3)
L_dyna  = λ_dyna · [ L1 + (1-D-SSIM) ](rendered_next, gt_next_rgb)

L_render = L_rgb + L_embed + L_dyna + L_reg
```

### 4.5 Training Loss (With Neural Rendering)

```
──────────────────────────────────────────────────────────────────────
L_total = λ_bc · L_flow  +  λ_nerf · L_render
──────────────────────────────────────────────────────────────────────

where:
  L_flow  = pos_weight · L1_pos + rot_weight · L1_rot + grip_weight · BCE_grip
  L_render = L_rgb + L_embed [+ L_dyna + L_reg if use_dynamic_field]
```

**Loss weights (from `train_maniflow_nerf.sh` defaults):**

| Loss term      | Weight       | Description |
|----------------|--------------|-------------|
| `λ_bc`         | 1.0          | Flow-matching BC loss scale |
| `λ_nerf`       | 0.02         | Outer NeRF loss scale |
| `λ_embed`      | 0.5          | Semantic embedding loss weight |
| `λ_dyna`       | 0.1          | Dynamic field loss weight |
| `λ_reg`        | 0.0          | Gaussian regularisation weight |
| `λ_rgb`        | 1.0          | RGB photometric loss weight |

### 4.6 Data Flow Diagram (With Neural Rendering)

```
┌────────────────────────────────────────────────────────────────────────┐
│                   TRAINING STEP (with neural rendering)                │
│                                                                        │
│  replay_sample                                                         │
│      │                                                                 │
│      ├── RGB-D, pcd, proprio ──► encode_scene()  ──► context          │
│      │                              ├─ voxel_feat (B,64,V,V,V)        │
│      │                              └─ lang_embedd (B,77,128)         │
│      ├── lang_goal_emb, lang_token_embs ──────────────────┘           │
│      │                                                                 │
│      ├── nerf_multi_view_rgb/depth/camera  ──► nerf_target_*          │
│      ├── nerf_next_multi_view_*            ──► nerf_next_target_*     │
│      │                                                                 │
│      │  Foundation model (frozen)                                     │
│      │  nerf_target_rgb ──► DINO/Diffusion ──► gt_embed               │
│      │                                                                 │
│      │  NeuralRenderer.forward()                                      │
│      │  ┌────────────────────────────────────────────────────────┐    │
│      │  │ voxel_feat + lang_embedd + pcd + rgb                   │    │
│      │  │ ──► GeneralizableGSEmbedNet                            │    │
│      │  │ ──► Gaussian params (xyz, opacity, scale, rot, SH)     │    │
│      │  │ ──► Gaussian rasterizer  ──► rendered_rgb, embed       │    │
│      │  │ (optional) deformation ──► next-frame render           │    │
│      │  └────────────────────────────────────────────────────────┘    │
│      │        │                                                        │
│      │        ▼                                                        │
│      │   rendering_loss_dict = { L_rgb, L_embed, L_dyna, psnr, … }    │
│      │                                                                 │
│      └── action_gt (B, 8) ──► _compute_flow_loss() ──► L_flow        │
│                                                                        │
│  L_total = λ_bc · L_flow  +  λ_nerf · L_render                       │
│                                                                        │
│  backward() → clip_grad_norm(max=10) → optimizer.step()               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Training Loop Comparison

| Aspect | Without Neural Rendering | With Neural Rendering |
|---|---|---|
| **`use_neural_rendering`** | `False` | `True` |
| **`NeuralRenderer`** | Not instantiated | `GeneralizableGSEmbedNet` + rasterizer |
| **Multi-view NeRF data** | Not required | Required (RGB + depth + camera per keyframe) |
| **Foundation model** | Not used | DINOv2 or Diffusion (frozen) |
| **Loss** | `λ_bc · L_flow` | `λ_bc · L_flow + λ_nerf · L_render` |
| **Batch size** | 8 (recommended) | 1 (GS rasterizer per scene) |
| **Context dict** | Same | Same |
| **Inference** | Identical | Identical |
| **GPU memory** | Lower | Higher (GS backward is expensive) |
| **Recommended for** | Fast sanity check, ablation | Full training with scene geometry |

---

## 6. Inference Step-by-Step

Both variants share the same inference path — `NeuralRenderer` is never called at test time.

```
Given: a new robot observation (RGB-D, language, proprio)

1.  Language encoding  (once per episode)
    ├── CLIP → lang_goal_emb  (B, 1024)
    └── CLIP → lang_token_embs (B, 77, 512)

2.  Input preprocessing
    └── GPU preprocessor → per-camera RGB + pcd pairs, depth, extrinsics

3.  Voxelise
    └── VoxelGrid.coords_to_bounding_voxel_grid(pcd, feats)
        → voxel_grid (B, 10, V, V, V)

4.  Encode scene
    └── QFunctionFlow.encode_scene(...)
        → context  { voxel_tokens(B,512,192), voxel_pos(B,512,3),
                     lang_tokens(B,77,192), proprio_feats(B,192) }

5.  Initialise noisy trajectory
    └── z_T ~ N(0, I)   shape (B, T·nhand, 9)
        T = action_chunk_size = 1

6.  Denoise  (Euler integration, T_steps=100)
    └── for idx, t in enumerate(pos_scheduler.timesteps):
           v_out = encoder.predict_velocity(z_t, t, context)
                   → (B, T, 10) = [pos_vel(3), rot_vel(6), grip_logit(1)]
           pos = pos_scheduler.step(v_out[:,:,:3], idx, z_t[:,:,:3])
           rot = rot_scheduler.step(v_out[:,:,3:9], idx, z_t[:,:,3:9])
           z_{idx+1} = cat([pos, rot])

7.  Post-process clean trajectory z_0
    ├── _unnormalize_pos(z_0[:,:,:3])   → xyz in world coords
    ├── _unconvert_rot(z_0[:,:,3:9])    → quaternion xyzw (normalised)
    └── sigmoid(v_out[:,:,9])           → gripper open probability

8.  Return action[0, 0, 0]  → [x, y, z, qx, qy, qz, qw, gripper]
    └── Execute on robot end-effector
```

---

## 7. Module Inventory

| File | Class / Function | Role |
|---|---|---|
| `voxel_flow_encoder.py` | `VoxelFlowEncoder` | 3D-CNN encoder + `TransformerHead`; outputs `(voxel_grid_feature, lang_embedd, context)` |
| `voxel_flow_encoder.py` | `TransformerHead` | Denoising head: cross-attn → self-attn → pos/rot/grip heads |
| `qattention_maniflow_agent.py` | `QFunctionFlow` | Wraps `VoxelFlowEncoder` + `NeuralRenderer`; `encode_scene()`, `forward()`, `render()` |
| `qattention_maniflow_agent.py` | `ManiFlowBCAgent` | Main agent: `build()`, `update()`, `act()`, `_compute_flow_loss()`, `_denoise_action()` |
| `rf_scheduler.py` | `RFScheduler` | Rectified-flow noise schedule (logit-Normal sampler) & Euler inference steps |
| `neural_rendering.py` | `NeuralRenderer` | Gaussian Splatting auxiliary loss: encode → rasterize → L_rgb + L_embed |
| `models_embed.py` | `GeneralizableGSEmbedNet` | Per-point Gaussian parameter regression from voxel features + language |
| `transformer_layers.py` | `AttentionModule`, `AdaLN`, `FFWLayer` | Transformer building blocks (cross-attn, self-attn, AdaLN conditioning) |
| `position_encodings.py` | `SinusoidalPosEmb`, `RotaryPositionEncoding3D`, `PositionEmbeddingLearnedMLP` | Time & 3D spatial position encodings |
| `loss.py` | `l1_loss`, `l2_loss`, `ssim`, `cosine_loss` | Photometric + semantic embedding losses |
| `launch_utils.py` | `create_replay`, `fill_replay` | Replay buffer construction; demo → keypoint extraction |
| `gpu_preprocessor.py` | `GPUPreprocessor` | GPU-side image crop + point-cloud projection |
| `voxel/voxel_grid.py` | `VoxelGrid` | 3D binning of point clouds into a dense voxel volume |
| `voxel/augmentation.py` | `apply_se3_augmentation_continuous` | SE3 data augmentation on actions + point clouds |
| `helpers/language_model.py` | `create_language_model` | Builds frozen CLIP (or T5) language encoder |
| `helpers/preprocess_agent.py` | `PreprocessAgent` | Stacks preprocessing in the YARR agent pipeline |

---

## 8. Key Design Decisions

| Design Choice | Rationale |
|---|---|
| **Rectified Flow (vs. DDPM)** | Linear trajectories in action space → fewer denoising steps (100 Euler vs. 1000 DDPM), faster inference |
| **3D-CNN voxel encoder (shared)** | Compact spatial representation; preserves scene geometry; reusable by both the Transformer head and the GS renderer |
| **FPS subsampling (8000 → 512 tokens)** | Keeps joint self-attention $O(N^2)$ tractable; FPS preserves spatial coverage better than random sampling |
| **3D RoPE on voxel positions** | Relative spatial encoding → generalises to novel workspace extents without re-learning position biases |
| **AdaLN (time + proprio)** | Injects denoising timestep and robot state into every attention layer as a modulation signal, without hard token concatenation |
| **Separate pos / rot / grip heads** | Decouples geometrically distinct DoFs; allows independent loss weighting (30/10/1) |
| **Ortho6D rotation representation** | Avoids quaternion antipodal ambiguity and gimbal lock; 6D → continuous 6D space → cleaner gradient flow |
| **logit-Normal timestep sampler** | Concentrates training on mid-noise steps (σ=1.5) where the velocity field is most informative |
| **GS auxiliary loss (optional)** | Provides dense geometry + semantic scene supervision at training time without extra robot roll-outs; disabled at inference |
| **DINOv2 / Diffusion feature matching** | Distils rich pre-trained visual representations into the Gaussians' feature spherical harmonics, improving scene understanding |
| **lv2_batch_size inner loop** | Multiplies gradient diversity $N×$ by re-sampling noise per step, reusing the expensive 3D-CNN context encoding |

---

## 9. Configuration & Scripts

### 9.1 Key Config File

**`conf/method/ManiFlow_BC.yaml`** — primary hyperparameter file:

```
use_neural_rendering: True   ← flip to False to disable GS renderer

# Flow-matching head
embedding_dim: 192           # Transformer feature dim (must divide by num_attn_heads)
num_attn_heads: 8
num_shared_attn_layers: 4
voxel_token_downsample: 5    # spatial stride: 100/5 = 20 → 8000 tokens
num_fps_tokens: 512          # FPS subsampling target
denoise_timesteps: 100       # Euler steps at inference
action_chunk_size: 1         # T > 1 predicts action chunk
lv2_batch_size: 1            # inner noise re-sampling (4 recommended)

# Optimizer
lr: 0.0001                   # Adam (NOT LAMB — LAMB overshoots)
lr_scheduler: False

# Loss weights
lambda_bc: 1.0
pos_loss_weight: 30.0
rot_loss_weight: 10.0
grip_loss_weight: 1.0
```

### 9.2 Training Scripts

| Script | Description |
|---|---|
| `scripts/train_maniflow_nerf.sh` | Full training **with** GS neural rendering (`use_neural_rendering=True`, `batch_size=1`) |
| `scripts/train_maniflow_no_nerf.sh` | Faster training **without** neural rendering (`use_neural_rendering=False`, `batch_size=8`) |

#### Usage

```bash
# Without neural rendering (fast baseline)
bash scripts/train_maniflow_no_nerf.sh  0  12346  my_exp_name

# With neural rendering (full model)
bash scripts/train_maniflow_nerf.sh     0  12345  my_exp_name

# Arguments: [GPU_IDs]  [PORT]  [EXP_NAME]
# GPU_IDs can be comma-separated for multi-GPU: "0,1"
```

### 9.3 Evaluation

```bash
bash scripts/eval.sh  <exp_name>  <checkpoint_step>
```

Evaluation runs `eval.py` which calls `ManiFlowBCAgent.act()` in a live RLBench environment.  The GS renderer is **never used** during evaluation.

---

## Appendix: Action Space Convention

| Dimension | Meaning | Training representation | Output |
|---|---|---|---|
| `[0:3]` | End-effector xyz | Normalised to `[-1, 1]` via workspace bounds | World-frame metres |
| `[3:7]` | Rotation (xyzw quaternion) | Converted to ortho6D `[r1..r6]` for flow | Unit quaternion |
| `[7]`   | Gripper open/close | Binary target; predicted as logit | Sigmoid probability |

**Internal flow trajectory token:** `(B, T, 9)` = `[x_norm, y_norm, z_norm, r1, r2, r3, r4, r5, r6]`  
**Gripper** is predicted separately via BCE and appended at post-processing.

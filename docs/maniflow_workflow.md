# ManiFlow BC — Architecture & Workflow

> **One-sentence summary:** ManiFlow BC is a **flow-matching robot manipulation policy** that lifts a 3D voxel scene representation through a Transformer denoising head to predict continuous end-effector actions, with an optional Gaussian-Splatting neural rendering auxiliary loss.

---

## 1. High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            ONLINE INFERENCE LOOP                             │
│                                                                              │
│  RGB-D cameras ──►  Point Cloud  ──►  VoxelGrid  ──►  VoxelFlowEncoder     │
│  Language goal  ──►  CLIP tokens  ──────────────────►  (scene context)      │
│  Proprioception ──────────────────────────────────►                         │
│                                                                              │
│          ┌─────────────────────────────────────────────────────┐            │
│          │       Rectified Flow Denoising  (T steps)           │            │
│          │  z_T ~ N(0,I)  ──► z_{T-1} ──► … ──► z_0 = action  │            │
│          └─────────────────────────────────────────────────────┘            │
│                                                                              │
│  Predicted action: [x, y, z, qx, qy, qz, qw, gripper_open]  (8-DoF)        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Data-Flow Diagram

```
══════════════════════════════════════════════════════════════════════
 INPUTS
══════════════════════════════════════════════════════════════════════

 RGB-D frames (N cameras)          Language goal string
 ┌──────────────────────┐          ┌──────────────────────┐
 │  (B, 3, H, W) × N   │          │  "pick up the cup"   │
 └──────────┬───────────┘          └──────────┬───────────┘
            │                                  │
            ▼                                  ▼
 ┌──────────────────────┐          ┌──────────────────────────────┐
 │   Point Cloud PCDs   │          │   CLIP  (frozen)             │
 │  (B, N·H·W, 3)       │          │  lang_goal_emb  (B, 1024)    │
 └──────────┬───────────┘          │  lang_token_embs(B, 77, 512) │
            │                      └──────────┬───────────────────┘
            ▼                                  │
 ┌──────────────────────┐                      │
 │   VoxelGrid          │                      │
 │  coords_to_bounding_ │                      │
 │  voxel_grid(...)     │    Proprioception     │
 │  (B, C_init, V,V,V)  │◄── (B, low_dim_size) │
 └──────────┬───────────┘                      │
            │                                  │
            └──────────────┬───────────────────┘
                           │
                           ▼
══════════════════════════════════════════════════════════════════════
 VoxelFlowEncoder.encode_scene()
══════════════════════════════════════════════════════════════════════

  Step A — 3D-CNN Voxel Encoding
  ────────────────────────────────────────────────────
  voxel_grid (B, C_init, V,V,V)
         │
         ▼
  MultiLayer3DEncoderShallow
         │
         ▼
  d0  (B, im_channels=64, V,V,V)   ─────────────────► voxel_grid_feature
                                                        (kept for GS renderer)

  Step B — Language Pre-processing
  ────────────────────────────────────────────────────
  lang_token_embs (B, 77, 512)
         │
         ▼
  nn.Linear(512 → im_channels×2)
         │
         ▼
  l   (B, 77, 128)                  ─────────────────► lang_embedd
                                                        (kept for GS renderer)

  Step C — Voxel Token Extraction
  ────────────────────────────────────────────────────
  d0 (B, 64, V,V,V)
         │
         ▼
  AdaptiveAvgPool3d → (B, 64, ds,ds,ds)     ds = V / voxel_token_downsample
         │
         ▼
  rearrange → (B, ds³, 64)
         │
         ▼
  nn.Linear(64 → embedding_dim=120)  +  LayerNorm
         │
         ▼
  vox_tokens (B, N, 120)             N = ds³ ≈ 8000 for V=100, ds=20

  Step D — World-Space Positions & Learned Pos Emb
  ────────────────────────────────────────────────────
  coord_bounds [x_min,y_min,z_min, x_max,y_max,z_max]
         │
         ▼
  _voxel_world_coords() → voxel_pos (B, N, 3)
         │
         ▼
  PositionEmbeddingLearnedMLP  →  vox_tokens += pos_emb

  Step E — Farthest Point Sampling (FPS)
  ────────────────────────────────────────────────────
  vox_tokens (B, N, 120),  voxel_pos (B, N, 3)
         │
         ▼
  _fps(voxel_pos, num_fps_tokens=512)
         │
         ▼
  fps_tokens (B, M, 120),  fps_pos (B, M, 3)    M = 512

  Step F — Proprio Projection
  ────────────────────────────────────────────────────
  proprio (B, low_dim_size=4)
         │
         ▼
  nn.Linear(4 → 120) + ReLU
         │
         ▼
  proprio_feats (B, 120)

  ──────────────────────────────────────────────────────────────
  context = {
      'voxel_tokens':  fps_tokens  (B, 512, 120)
      'voxel_pos':     fps_pos     (B, 512, 3)
      'lang_tokens':   lang_tokens (B, 77, 120)
      'proprio_feats': proprio_feats (B, 120)
  }
  ──────────────────────────────────────────────────────────────

══════════════════════════════════════════════════════════════════════
 RFScheduler  (Rectified Flow)
══════════════════════════════════════════════════════════════════════

 ┌─ TRAINING ──────────────────────────────────────────────────┐
 │                                                              │
 │  x_clean (B, 8)  +  noise ~ N(0,I)                          │
 │       │                  │                                   │
 │       └──── t ~ logit-Normal(0, 1.5) ────┐                  │
 │                                          ▼                   │
 │            z_t = (1-t)·x_clean + t·noise   [linear interp]  │
 │            target = noise − x_clean         [velocity]       │
 │                                                              │
 └──────────────────────────────────────────────────────────────┘

 ┌─ INFERENCE ─────────────────────────────────────────────────┐
 │                                                              │
 │  z_T ~ N(0,I)                                               │
 │    │                                                         │
 │    ▼  for t = T, T-1, …, 1  (Euler steps)                   │
 │  z_{t-1} = z_t − (t − t') · v_θ(z_t, t, context)           │
 │    │                                                         │
 │    ▼                                                         │
 │  z_0  =  predicted clean action  (B, 8)                     │
 │                                                              │
 └──────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════
 TransformerHead.forward()  ←  VoxelFlowEncoder.predict_velocity()
══════════════════════════════════════════════════════════════════════

 Inputs:
   noisy_action  (B, 8)       ← z_t from RFScheduler
   timestep      (B,)         ← t  in [0, 1]
   voxel_tokens  (B, 512, 120)
   voxel_pos     (B, 512, 3)
   lang_tokens   (B, 77, 120)
   proprio_feats (B, 120)

  ┌─ 1. AdaLN conditioning signal ───────────────────────────┐
  │                                                           │
  │  SinusoidalPosEmb(t) ──► time_emb   (B, 120)             │
  │  proprio_feats ──────────► curr_gripper_emb (B, 120)      │
  │                                                           │
  │  ada_sgnl = time_emb + curr_gripper_emb    (B, 120)       │
  └───────────────────────────────────────────────────────────┘

  ┌─ 2. Action token encoding ────────────────────────────────┐
  │                                                           │
  │  noisy_action ──► nn.Linear(8 → 120) ──► act_feat (B,1,120) │
  │  step_pos = SinusoidalPosEmb([0]) ──► (B, 1, 120)        │
  └───────────────────────────────────────────────────────────┘

  ┌─ 3. 3D RoPE positions ────────────────────────────────────┐
  │                                                           │
  │  RotaryPositionEncoding3D(voxel_pos)  ──► rel_vox_pos     │
  │                     (B, 512, 3) ──────── (B, 512, 120, 2) │
  │  RotaryPositionEncoding3D(act_xyz)    ──► rel_act_pos     │
  │                     (B, 1, 3)   ──────── (B, 1,   120, 2) │
  └───────────────────────────────────────────────────────────┘

  ┌─ 4. Cross-attention: action → language ───────────────────┐
  │                                                           │
  │  Q = act_feat  (B, 1, 120)                                │
  │  K,V = lang_tokens (B, 77, 120)   [no RoPE, standard]    │
  │  AttentionModule(layers=1, heads=8)                       │
  │  act_feat = output + step_pos                             │
  └───────────────────────────────────────────────────────────┘

  ┌─ 5. Cross-attention: action → voxel scene  (AdaLN) ───────┐
  │                                                           │
  │  Q = act_feat  (B, 1, 120)  + rel_act_pos  [3D RoPE]     │
  │  K,V = fps_tokens (B,512,120)+ rel_vox_pos [3D RoPE]     │
  │  AttentionModule(layers=2, heads=8, AdaLN, RoPE)          │
  │  Conditioning: ada_sgnl modulates via Adaptive LayerNorm  │
  └───────────────────────────────────────────────────────────┘

  ┌─ 6. Joint self-attention  (AdaLN) ────────────────────────┐
  │                                                           │
  │  joint = cat([act_feat, fps_tokens], dim=1)  (B,513,120)  │
  │  joint_pos = cat([rel_act_pos, rel_vox_pos]) (B,513,120,2)│
  │  AttentionModule(layers=4, heads=8, AdaLN, RoPE)          │
  │  Tokens attend to each other across action + scene        │
  └───────────────────────────────────────────────────────────┘

  ┌─ 7. Specialised output heads ─────────────────────────────┐
  │                                                           │
  │  ┌─ Position head ─────────────────────────────────────┐  │
  │  │  pos_self_attn (layers=2, AdaLN, RoPE)              │  │
  │  │  slice action token [:, :1, :]                      │  │
  │  │  position_proj → position_predictor → (B, 3)  xyz   │  │
  │  └─────────────────────────────────────────────────────┘  │
  │                                                           │
  │  ┌─ Rotation head ─────────────────────────────────────┐  │
  │  │  rot_self_attn (layers=2, AdaLN, RoPE)              │  │
  │  │  slice action token [:, :1, :]                      │  │
  │  │  rotation_proj → rotation_predictor → (B, 4)  quat  │  │
  │  └─────────────────────────────────────────────────────┘  │
  │                                                           │
  │  ┌─ Gripper head ──────────────────────────────────────┐  │
  │  │  openess_predictor (reuses pos_feat) → (B, 1) logit │  │
  │  └─────────────────────────────────────────────────────┘  │
  │                                                           │
  │  output = cat([xyz, quat, gripper], dim=-1)  (B, 8)      │
  └───────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════
 TRAINING LOSS  (QAttentionManiFlowBCAgent.update())
══════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────┐
  │  pred_velocity  (B, 8)  ← TransformerHead output             │
  │  target_velocity= noise − x_clean  (B, 8)  ← RFScheduler    │
  │                                                              │
  │  L_pos = L1(pred[:,:3], target[:,:3])   ← position          │
  │  L_rot = L1(pred[:,3:7], target[:,3:7]) ← rotation          │
  │  L_grip= BCE(pred[:,7], target[:,7])    ← gripper open/close │
  │                                                              │
  │  L_bc  = L_pos + λ_rot·L_rot + λ_grip·L_grip                │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  (Optional) Neural Rendering Auxiliary Loss                  │
  │                                                              │
  │  voxel_grid_feature + lang_embedd                            │
  │       │                                                      │
  │       ▼                                                      │
  │  NeuralRenderer (GeneralizableGSEmbedNet)                    │
  │       │                                                      │
  │       ▼                                                      │
  │  Gaussian parameters (means, scales, rotations, opacities,  │
  │                        RGB/feature spherical harmonics)      │
  │       │                                                      │
  │       ▼                                                      │
  │  Differential rasterizer  ──► rendered_rgb, rendered_depth   │
  │                                                              │
  │  L_rgb   = λ_rgb  · (L1 + D-SSIM)(rendered, gt_rgb)         │
  │  L_embed = λ_embed· L_embed_fn(rendered_feat, gt_feat)       │
  │                 [L2 / cosine vs. DINOv2 / Diffusion feats]   │
  │                                                              │
  │  L_total = L_bc + L_rgb + L_embed                            │
  └──────────────────────────────────────────────────────────────┘
```

---

## 3. Module Inventory

| File | Role |
|---|---|
| `voxel_flow_encoder.py` | `VoxelFlowEncoder` — 3D-CNN encoder + `TransformerHead` |
| `qattention_maniflow_agent.py` | `QFunctionFlow` + `QAttentionManiFlowBCAgent` — training/inference loop |
| `rf_scheduler.py` | `RFScheduler` — rectified flow noise schedule & Euler sampler |
| `transformer_layers.py` | `AttentionModule`, `AdaLN`, `FFWLayer` — Transformer primitives |
| `position_encodings.py` | `SinusoidalPosEmb`, `RotaryPositionEncoding3D`, `PositionEmbeddingLearnedMLP` |
| `neural_rendering.py` | `NeuralRenderer` — GS auxiliary loss |
| `models_embed.py` | `GeneralizableGSEmbedNet` — Gaussian parameter regressor |
| `loss.py` | `l1_loss`, `l2_loss`, `ssim`, `cosine_loss` |
| `rf_scheduler.py` | Rectified flow scheduler (training noise & inference Euler steps) |
| `zarr_dataset.py` | Zarr-backed dataset for offline BC training |
| `launch_utils.py` | Agent + model construction helpers |
| `gpu_preprocessor.py` | GPU-side image / point cloud preprocessing |

---

## 4. Inference Step-by-Step

```
Given: a new observation (RGB-D, language, proprio)

1.  Pre-process
    ├── Project RGB-D → point cloud (GPU preprocessor)
    └── Encode language → (lang_goal_emb, lang_token_embs) via CLIP

2.  Voxelise
    └── VoxelGrid.coords_to_bounding_voxel_grid(pcd, feats)
        → voxel_grid (B, C_init, V, V, V)

3.  Encode scene
    └── VoxelFlowEncoder.encode_scene(voxel_grid, proprio,
                                       lang_goal_emb, lang_token_embs)
        → context  { voxel_tokens, voxel_pos, lang_tokens, proprio_feats }

4.  Initialise noisy action
    └── z_T ~ N(0, I)   shape (B, 8)

5.  Denoise  (Euler integration, T=10 steps)
    └── for t_idx in range(T):
           v = VoxelFlowEncoder.predict_velocity(z_t, t, context)
           z_{t-1} = RFScheduler.step(v, t_idx, z_t).prev_sample

6.  Post-process clean action z_0
    ├── position  = z_0[:3]            (world-frame xyz)
    ├── rotation  = z_0[3:7]           (unit quaternion, normalise)
    └── gripper   = sigmoid(z_0[7])    (open probability)

7.  Execute on robot
```

---

## 5. Key Design Decisions

| Design Choice | Rationale |
|---|---|
| **Rectified Flow (vs. DDPM)** | Straight-line trajectories in action space → fewer denoising steps, faster inference |
| **3D-CNN voxel encoder** | Compact scene representation; preserves spatial structure; reuses GS renderer pipeline |
| **FPS subsampling (N→512)** | Keeps joint self-attention O(N²) tractable while preserving scene coverage |
| **3D RoPE on voxel positions** | Relative spatial encoding — generalises to novel workspace extents |
| **AdaLN conditioning** | Time + proprio signal modulates every attention layer without hard injection |
| **Separate position/rotation/gripper heads** | Decouples geometrically distinct DoFs; allows different loss weighting |
| **GS auxiliary loss** | Provides dense geometry supervision signal without extra robot roll-outs |
| **logit-Normal timestep sampler** | Biases training towards medium-noise timesteps where learning signal is richest |

# ManiFlow BC — Workflow & Architecture

> **Branch**: `main` | **Method name**: `ManiFlow_BC`  
> **Key files**: `agents/maniflow_bc/`, `voxel/augmentation.py`, `helpers/`, `conf/method/ManiFlow_BC.yaml`

---

## 1. Overview

ManiFlow BC is a **continuous-action, flow-matching behaviour-cloning** agent for robot manipulation. It is a drop-in replacement for `ManiGaussian_BC` that:

- Keeps the **3D-CNN voxel encoder** and **Gaussian Splatting (GS) auxiliary renderer** from ManiGaussian unchanged.
- Replaces the Perceiver IO + discrete classification head with a **Transformer denoising head** (rectified flow / flow-matching) that predicts continuous 8-DoF actions.
- Shares the voxel feature map between the BC head and the GS renderer, so both losses are trained jointly.

**Action space** (8-DoF):
```
[x, y, z, qx, qy, qz, qw, gripper_open]
```
The quaternion is stored in **xyzw** format throughout. Gripper open/close is a binary float `{0.0, 1.0}`.

---

## 2. Data Pipeline

### 2.1 Demo Storage → Replay Buffer

```
RLBench demo (.pkl)
        │
        ▼
fill_replay()                     [launch_utils.py]
        │
        ├─ keypoint_discovery()   [heuristic: gripper state change]
        │
        └─ _add_keypoints_to_replay()   per keypoint k:
                │
                ├─ obs         = demo[prev_keypoint]    (current observation)
                ├─ obs_tp1     = demo[keypoint]         (target keypoint)
                ├─ obs_tm1     = demo[keypoint - 1]     (for terminal next_obs)
                │
                ├─ action      = _get_action(obs_tp1)   → (8,) float32
                │     [x,y,z, qx,qy,qz,qw, gripper_open]
                │
                ├─ obs_dict    = extract_obs(obs, t=frame_idx, next_obs=obs_tp1 | obs_tm1)
                │     → low_dim_state (4,): [gripper_open, finger0, finger1, time_token]
                │     → {cam}_rgb (3,128,128), {cam}_depth (1,128,128)
                │     → {cam}_point_cloud (3,128,128)
                │     → {cam}_camera_extrinsics (4,4), {cam}_camera_intrinsics (3,3)
                │     → nerf_multi_view_rgb/depth/camera  (object arrays, paths)
                │     → nerf_next_multi_view_rgb/depth/camera  (for GS dynamics)
                │
                ├─ lang embedding  = language_model.extract(description)
                │     → lang_goal_emb (1024,) CLIP sentence emb
                │     → lang_token_embs (77, 512) CLIP token embs
                │
                ├─ gripper_pose     = obs_tp1.gripper_pose   (7,)  ← GT action pose
                ├─ obs_gripper_pose = obs.gripper_pose        (7,)  ← current obs pose
                │
                └─ replay.add(action, reward, terminal, ...)
```

**Time token** in `low_dim_state[3]`: `time = (1 - t/(L-1)) * 2 - 1 ∈ [-1, 1]`  
where `t` = absolute frame index, `L` = `episode_length`.

**`next_obs` for GS dynamics**:
- Non-terminal transitions: `next_obs = obs_tp1` (next keypoint — real motion signal)
- Terminal transition: `next_obs = obs_tm1` (one raw frame before keypoint — near-zero motion, consistent with original ManiGaussian)

**Replay buffer schema** (`create_replay`):

| Key | Shape | Type | Notes |
|---|---|---|---|
| `action` | `(8,)` | float32 | GT continuous action |
| `low_dim_state` | `(4,)` | float32 | ObservationElement → auto-gets `_tp1` |
| `{cam}_rgb` | `(3,128,128)` | float32 | ObservationElement |
| `{cam}_depth` | `(1,128,128)` | float32 | ObservationElement |
| `{cam}_point_cloud` | `(3,128,128)` | float32 | ObservationElement |
| `{cam}_camera_extrinsics` | `(4,4)` | float32 | ObservationElement |
| `{cam}_camera_intrinsics` | `(3,3)` | float32 | ObservationElement |
| `nerf_multi_view_rgb` | `(N_views,)` | object | file paths |
| `nerf_next_multi_view_rgb` | `(N_views,)` | object | file paths |
| `gripper_pose` | `(7,)` | float32 | **target** pose (GT) |
| `obs_gripper_pose` | `(7,)` | float32 | **current** pose (for encoder) |
| `lang_goal_emb` | `(1024,)` | float32 | CLIP sentence embedding |
| `lang_token_embs` | `(77, 512)` | float32 | CLIP token embeddings |
| `ignore_collisions` | `(1,)` | float32 | From current obs |
| `demo` | `()` | bool | Extra replay element |

---

## 3. Training Loop

```
OfflineTrainRunner.start()
        │
        ▼  sample batch from replay buffer (B samples)
PreprocessAgent.update()              [helpers/preprocess_agent.py]
        │
        ├─ _strip_time(v): strip YARR timestep dim → v[:, 0]
        ├─ normalize RGB:   (x / 255) * 2 - 1  (for non-nerf rgb keys)
        ├─ save nerf path arrays before .float() (they are object arrays)
        │
        ▼
ManiFlowStackAgent.update()           [qattention_stack_agent.py]
        │
        └─ for each layer agent (depth 0, 1, ...):
                ManiFlowBCAgent.update()    [qattention_maniflow_agent.py]
```

### 3.1 `ManiFlowBCAgent.update()` — Step by Step

```
1. Unpack replay sample
   ├─ action_gt           (B, 8)    ← target continuous action
   ├─ obs_gripper_pose    (B, 7)    ← current obs gripper pose
   ├─ gripper_pose        (B, 7)    ← same as action_gt[:, :7], kept for aug pivot
   ├─ lang_goal_emb       (B, 1024)
   ├─ lang_token_embs     (B, 77, 512)
   ├─ proprio             (B, 4)    ← low_dim_state
   └─ obs, depth, pcd, extrinsics, intrinsics  (per-camera tensors)

2. Load NeRF multi-view data  [only if use_neural_rendering=True]
   ├─ Parse RGB, depth, camera files from paths
   ├─ Subsample views: interval = N_views // num_view_for_nerf
   └─ Pick random view index per batch

3. Compute bounds
   ├─ Layer 0:  bounds = coordinate_bounds  (global workspace)
   └─ Layer k:  bounds = [attention_coord_{k-1} ± bounds_offset]  (local crop)

4. SE3 Augmentation  [if transform_augmentation.apply_se3]
   ├─ Pivot: obs_gripper_pose  (current obs pose, NOT the GT target)
   ├─ apply_se3_augmentation_continuous(pcd, extrinsics, pivot, action_gt, ...)
   │     → Random rotation (max ±45° yaw by default) + bounded translation
   │     → Transforms: action_gt position+rotation, all camera PCDs,
   │                   camera extrinsics, obs_gripper_pose
   └─ obs_gripper_pose ← augmented version  (keeps encoder consistent with scene)

5. Forward pass: QFunctionFlow(...)
   │
   ├─ VoxelFlowEncoder.forward()   [voxel_flow_encoder.py]
   │     → encode_scene():
   │           ① Voxelize PCD → voxel_grid  (B, 10, 100, 100, 100)
   │           ② 3D-CNN encoder → voxel_feat  (B, 128, V', V', V')
   │           ③ AdaptiveAvgPool3d → flatten → N tokens  (B, N, 128)
   │           ④ FPS subsample → M tokens  (B, 512, 128)  [M = num_fps_tokens]
   │           ⑤ Language preprocessing:
   │                 lang_goal_emb  → CLIP sentence emb proj
   │                 lang_token_embs → token sequence proj
   │           ⑥ Gripper context head:
   │                 curr_gripper_embed + proprio_proj(proprio)
   │                 3-layer cross-attn: gripper token ← scene tokens  (3D RoPE)
   │                 → proprio_feats  (B, emb)
   │           ⑦ Build context dict:
   │                 { voxel_tokens (B,M,emb), voxel_pos (B,M,3),
   │                   lang_tokens (B,L,emb), proprio_feats (B,emb) }
   │
   └─ NeuralRenderer(...)  [if use_neural_rendering]
         → GS rendering loss dict  {loss, loss_rgb, loss_embed, loss_dyna, psnr}

6. Flow-matching loss: _compute_flow_loss(action_gt, context)
   │
   ├─ Reshape action_gt  (B,8) → (B, T, nhand=1, 8)  [T = action_chunk_size]
   ├─ Separate:  gt_openess (B,T,1,1) | gt_traj (B,T,1,7)
   ├─ Normalize positions:  xyz → [-1, 1]  using workspace_bounds
   ├─ Convert rotation:  quat_xyzw → ortho6D  → gt_traj (B,T,1,9)
   │
   └─ Inner loop × lv2_batch_size:
         ① Sample noise    ~ N(0,I)          shape (B,T,1,9)
         ② Sample timestep t ~ logit-normal  shape (B,)
         ③ Noisy trajectory = (1-t)*x_clean + t*noise   [RF forward process]
         ④ Flatten: (B,T*nhand,9) → TransformerHead.predict_velocity()
         ⑤ RF target (velocity): v* = noise - x_clean
         ⑥ Loss per layer prediction:
               L = pos_weight * L1(pred_pos,   v*_pos)
                 + rot_weight * L1(pred_rot,   v*_rot)
                 + grip_weight * BCE(pred_grip, gt_openess)

7. Total loss
   ├─ use_neural_rendering=True:
   │     total = lambda_bc * flow_loss + lambda_nerf * rendering_loss
   └─ use_neural_rendering=False:
         total = lambda_bc * flow_loss

8. Backward + gradient clip (max_norm=10.0) + optimizer step
9. Optional LR scheduler step
```

---

## 4. Model Architecture

### 4.1 VoxelFlowEncoder

```
Input: voxel_grid (B, 10, 100, 100, 100)
       proprio     (B, 4)
       lang_goal_emb  (B, 1024)
       lang_token_embs (B, 77, 512)
       gripper_pose   (B, 7)  optional

                    ┌─────────────────────────────────────┐
                    │  3D-CNN Encoder                      │
                    │  MultiLayer3DEncoderShallow          │
                    │  → voxel feature map  (B, 128, V,V,V)│
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                                         ▼
   ┌──────────────────────┐             ┌───────────────────────────┐
   │  NeuralRenderer (GS) │             │  Flow-matching head        │
   │  (uses voxel_feat    │             │                            │
   │   and lang_embedd)   │             │  AdaptiveAvgPool3d(stride) │
   └──────────────────────┘             │  → flatten → N tokens      │
                                        │  → FPS → M=512 tokens      │
                                        │  → 3D RoPE on world coords │
                                        │                            │
                                        │  Language pre-processing:  │
                                        │  lang_goal_emb → Linear    │
                                        │  lang_token_embs → Linear  │
                                        │                            │
                                        │  Gripper context head:     │
                                        │  curr_gripper_embed (1,emb)│
                                        │  + proprio_proj(proprio)   │
                                        │  → 3-layer cross-attn      │
                                        │    gripper ← scene tokens  │
                                        │  → proprio_feats (B, emb)  │
                                        └────────────┬───────────────┘
                                                     ▼
                                              context dict
                                  { voxel_tokens, voxel_pos,
                                    lang_tokens, proprio_feats }
```

### 4.2 TransformerHead (Denoising Head)

```
Input:  noisy_trajectory (B, T, 9)   — [x_norm, y_norm, z_norm, r1..r6]
        timestep t        (B,)        — float in [0, 1]
        context dict

AdaLN conditioning signal:
    time_embs = time_emb(t) + curr_gripper_emb(proprio_feats)
              = SinusoidalPosEmb(t) + 2-layer MLP(proprio_feats)

Trajectory tokens:
    traj_feats = traj_encoder(noisy_traj)   + step_pos_emb   (B, T, emb)

─── Stage 1: Cross-attention → Language ──────────────────────────────
    traj_feats = cross_attn(traj → lang_tokens, no RoPE, no AdaLN)

─── Stage 2: Cross-attention → Voxel Scene ───────────────────────────
    traj_feats = cross_attn(traj → voxel_tokens, 3D RoPE, AdaLN)

─── Stage 3: Joint Self-attention ────────────────────────────────────
    joint = cat([traj_feats, voxel_tokens], dim=1)   (B, T+M, emb)
    joint = self_attn(joint, AdaLN, 3D RoPE)  × num_shared_attn_layers

─── Output Heads (slice traj tokens [:, :T]) ─────────────────────────
    pos head:  2-layer self-attn → Linear → position  (B, T, 3)
    rot head:  2-layer self-attn → Linear → rotation  (B, T, 6)  ortho6D
    grip head: Linear → openness (B, T, 1)  logit

Output: [(B, T, 10)]  [pos_vel(3), rot_vel(6), grip_logit(1)]
```

### 4.3 CLIP Backbone Path (optional, `use_clip_backbone=True`)

When enabled, the 3D-CNN encoder is replaced by:
```
RGB (B, 3, 128, 128)
    → CLIP RN50 (frozen when finetune_clip_backbone=False)
    → {res2, res3, res4, res5} feature pyramid
    → _EfficientFPN → 2D feature map (B, final_dim, H', W')
    → CLIPVoxelLifter: back-project via PCD into 3D voxel volume
    → voxel_feat (B, final_dim, V, V, V)
```
The rest of the pipeline (FPS, Transformer head, GS renderer) is unchanged.

---

## 5. SE3 Augmentation Detail

```python
apply_se3_augmentation_continuous(pcd, camera_extrinsics, pivot_pose,
                                   action_gt, bounds, ...)
```

The scene is rotated around `obs_gripper_pose` (the **current** observation pose):

```
Random perturbation:
    t_shift ~ U(-aug_xyz * bounds_extent, +aug_xyz * bounds_extent)
    R_delta = euler_to_matrix(roll, pitch, yaw)   yaw ~ U(-45°, +45°)

For every world-frame point p  (PCD, camera origins, action target):
    p' = R_delta^T * (p - t_pivot) + clamp(t_pivot + t_shift)

Applied to:
    ✓ Point clouds   (perturb_se3)
    ✓ Camera extrinsics  (perturb_se3_camera_pose, with .clone() to avoid in-place mutation)
    ✓ action_gt position + rotation  (direct SE3 math, no voxel discretization)
    ✓ obs_gripper_pose  (so encoder gripper token is in the augmented frame)
    ✗ RGB images  (cameras are fixed; augmentation is a world-frame rotation)
```

---

## 6. Inference (Evaluation)

```
CustomRLBenchEnv.step()
        │  obs dict including obs_gripper_pose (7,)
        ▼
YARR RolloutGenerator.generator()
        │  wraps obs → (1, timesteps=1, ...) tensors
        ▼
PreprocessAgent.act()
        │  normalize RGB: (x/255)*2-1
        ▼
ManiFlowStackAgent.act()
        │
        └─ ManiFlowBCAgent.act()
                │
                ├─ Language model: extract(lang_goal) → lang_goal_emb, lang_token_embs
                │
                ├─ Shape-fix obs_gripper_pose:
                │     YARR path:  (1, 1, 7) → while dim>2: [:,0] → (1, 7)
                │     Direct path: (7,) → unsqueeze(0) → (1, 7)
                │
                ├─ encode_scene() → context dict
                │
                └─ _denoise_action(context, num_steps=denoise_timesteps=10)
                        │
                        │  Start: trajectory ~ N(0,I)  (B, T, 9)
                        │  Euler steps:
                        │    for t in [1.0, 0.9, ..., 0.1]:
                        │        v = TransformerHead(trajectory, t, context)
                        │        pos = pos_scheduler.step(v_pos, t, traj_pos)
                        │        rot = rot_scheduler.step(v_rot, t, traj_rot)
                        │        trajectory = cat([pos, rot])
                        │
                        │  Post-process:
                        │    unnormalize_pos: [-1,1] → world coords
                        │    unconvert_rot: ortho6D → quat_xyzw
                        │    grip: sigmoid(logit) > 0.5 → binary
                        │
                        └─ actions (B=1, T=1, nhand=1, 8)  → action_np (8,)

ManiFlowStackAgent appends ignore_collisions → 9-DoF action (9,)
RLBench executes:  MoveArmThenGripper [x,y,z,qx,qy,qz,qw, gripper, ignore_col]
```

---

## 7. Loss Formulation

### 7.1 Rectified Flow (BC loss)

$$
\mathcal{L}_{\text{flow}} = \frac{1}{L} \sum_{l=1}^{L} \left[
  w_{\text{pos}} \cdot \|v_\theta^{\text{pos}} - v^*_{\text{pos}}\|_1
  + w_{\text{rot}} \cdot \|v_\theta^{\text{rot}} - v^*_{\text{rot}}\|_1
  + w_{\text{grip}} \cdot \text{BCE}(v_\theta^{\text{grip}},\, y_{\text{grip}})
\right]
$$

where:
- $L$ = `lv2_batch_size` (inner re-sampling iterations, default 4)
- $v^* = \epsilon - x_{\text{clean}}$ is the velocity target (noise minus clean action)
- $z_t = (1-t) \cdot x_{\text{clean}} + t \cdot \epsilon$, $t \sim$ logit-normal($\mu$=0, $\sigma$=1.5)
- Weights: $w_{\text{pos}}=30$, $w_{\text{rot}}=10$, $w_{\text{grip}}=1$

### 7.2 Total Loss

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{bc}} \cdot \mathcal{L}_{\text{flow}}
  + \lambda_{\text{nerf}} \cdot \mathcal{L}_{\text{GS}}
$$

| Hyper-param | Value | Notes |
|---|---|---|
| `lambda_bc` | 1.0 | BC loss weight |
| `lambda_nerf` | 0.02 | GS auxiliary weight |
| `lambda_rgb` | 1.0 | inside GS loss |
| `lambda_embed` | 0.5 | inside GS loss |
| `lambda_dyna` | 0.01 | inside GS dynamics loss |

---

## 8. Key Configuration Reference

| Config key | Default | Meaning |
|---|---|---|
| `action_dim` | 8 | `[x,y,z,qx,qy,qz,qw,grip]` |
| `action_chunk_size` | 1 | T steps per denoising call |
| `lv2_batch_size` | 4 | Inner loss re-sampling |
| `denoise_timesteps` | 10 | Euler steps at inference |
| `embedding_dim` | 120 | Transformer feature dim |
| `num_attn_heads` | 8 | Attention heads |
| `num_shared_attn_layers` | 4 | Joint self-attn depth |
| `voxel_token_downsample` | 5 | Pool stride; N = (100/5)³ = 8000 tokens |
| `num_fps_tokens` | 512 | Tokens after FPS subsampling |
| `workspace_bounds` | `[-0.3,-0.5,0.6, 0.7,0.5,1.6]` | For pos normalization |
| `use_clip_backbone` | True | CLIP RN50 instead of 3D-CNN |
| `optimizer` | `adam` | lr=1e-4, constant schedule |
| `transform_augmentation.aug_rpy` | `[0,0,45]` | ±45° yaw SE3 aug |

---

## 9. Bug Fixes Applied (vs. Original Codebase)

| # | File | Description |
|---|---|---|
| 1 | `qattention_maniflow_agent.py` `act()` | Missing `gripper_pose=` arg in `encode_scene` call |
| 2 | `qattention_maniflow_agent.py` `render()` | Missing `gripper_pose=None` keyword |
| 3 | `custom_rlbench_env.py` both `extract_obs` | `obs_gripper_pose` never written to `obs_dict` |
| 4 | `launch_utils.py` `_add_keypoints_to_replay` | Terminal `add_final` stored stale (second-to-last) gripper pose |
| 5 | `qattention_maniflow_agent.py` `update()` | SE3 aug pivot used GT target pose instead of current obs pose |
| 6 | `augmentation.py` + `update()` | `obs_gripper_pose` not transformed after SE3 aug |
| 7 | `augmentation.py` `perturb_se3_camera_pose` | In-place mutation of replay-buffer extrinsic tensors |
| 8 | `qattention_maniflow_agent.py` `act()` | `obs_gripper_pose` missing `.unsqueeze(0)` for bare `(7,)` tensor |

# ManiFlow BC — Workflow & Architecture

> **Branch**: `main` | **Method name**: `ManiFlow_BC`  
> **Key files**: `agents/maniflow_bc/`, `voxel/augmentation.py`, `helpers/`, `conf/method/ManiFlow_BC.yaml`

---

## 1. Overview

ManiFlow BC is a **continuous-action, flow-matching behaviour-cloning** agent for robot manipulation. It is a drop-in replacement for `ManiGaussian_BC` that:

- Keeps the **3D-CNN voxel encoder** and **Gaussian Splatting (GS) auxiliary renderer** from ManiGaussian unchanged.
- Replaces the Perceiver IO + discrete classification head with a **Transformer denoising head** (rectified flow) that predicts continuous 8-DoF actions.
- Shares the voxel feature map between the BC head and the GS renderer so both losses train jointly.
- Maintains an **EMA shadow copy** of all model weights (decay=0.999) and swaps it in at inference time.

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
        ├─ language_model created INSIDE worker process
        │     (avoids fp16/CPU cross-process crash)
        │
        ├─ keypoint_discovery()   [heuristic: gripper state change]
        │
        └─ _add_keypoints_to_replay()   per keypoint k (0-indexed):
                │
                ├─ obs         = demo[i]           (augmentation start frame)
                ├─ obs_tp1     = demo[keypoint]    (target keypoint)
                ├─ obs_tm1     = demo[keypoint-1]  (for terminal next_obs)
                │
                ├─ action      = _get_action(obs_tp1)   → (8,) float32
                │     [x,y,z, qx,qy,qz,qw, gripper_open]
                │     quat is L2-normalised only (no sign flip)
                │
                ├─ obs_dict    = extract_obs(obs, t=k, next_obs=obs_tp1 | obs_tm1)
                │     t = k  (0-based keypoint counter, treating each augmented
                │              start as a fresh episode — matches eval self._i=0)
                │     → low_dim_state (4,): [gripper_open, finger0, finger1, time_token]
                │     → {cam}_rgb (3,128,128), {cam}_depth (1,128,128)
                │     → {cam}_point_cloud (3,128,128)
                │     → {cam}_camera_extrinsics (4,4), {cam}_camera_intrinsics (3,3)
                │     → nerf_multi_view_rgb/depth/camera  (object arrays, file paths)
                │     → nerf_next_multi_view_rgb/depth/camera
                │
                ├─ lang embedding  = language_model.extract(description)
                │     → lang_goal_emb (1024,)  CLIP sentence emb
                │     → lang_token_embs (77, 512)  CLIP token embs
                │
                ├─ gripper_pose  = obs_tp1.gripper_pose  (7,)  ← target pose for SE3 aug
                │
                └─ replay.add(action, reward, terminal, ...)

        Terminal obs: extract_obs(obs_tp1, t=k+1, ...)  → replay.add_final(...)
```

**Time token** in `low_dim_state[3]`:  
`time = (1 − t / (L−1)) × 2 − 1 ∈ [−1, 1]`  
where `t` = keypoint counter `k`, `L` = `episode_length`.

**`next_obs` for GS dynamics**:
- Non-terminal: `next_obs = obs_tp1` (next keypoint — real motion)
- Terminal: `next_obs = obs_tm1` (frame before keypoint — near-zero motion)

**Demo augmentation**: `demo_augmentation_every_n=10` causes `fill_replay` to loop over raw frames at stride 10 and create sub-episodes starting from each frame. Each sub-episode's keypoint counter resets to `k=0`, matching eval semantics (agent always starts with `self._i=0`).

**Replay buffer schema** (`create_replay`):

| Key | Shape | Type | Notes |
|---|---|---|---|
| `action` | `(8,)` | float32 | GT continuous action |
| `low_dim_state` | `(4,)` | float32 | ObservationElement (YARR auto-adds `_tp1`) |
| `{cam}_rgb` | `(3,128,128)` | float32 | ObservationElement |
| `{cam}_depth` | `(1,128,128)` | float32 | ObservationElement |
| `{cam}_point_cloud` | `(3,128,128)` | float32 | ObservationElement |
| `{cam}_camera_extrinsics` | `(4,4)` | float32 | ObservationElement |
| `{cam}_camera_intrinsics` | `(3,3)` | float32 | ObservationElement |
| `nerf_multi_view_rgb` | `(N_views,)` | object | file paths |
| `nerf_next_multi_view_rgb` | `(N_views,)` | object | file paths |
| `gripper_pose` | `(7,)` | float32 | target pose (used as SE3 aug pivot) |
| `lang_goal_emb` | `(1024,)` | float32 | CLIP sentence embedding |
| `lang_token_embs` | `(77, 512)` | float32 | CLIP token embeddings |
| `ignore_collisions` | `(1,)` | float32 | From current obs |
| `demo` | `()` | bool | Extra replay element |

---

## 3. Training Loop

```
OfflineTrainRunner.start()
        │
        ▼  sample batch (B samples) from TaskUniformReplayBuffer
PreprocessAgent.update()              [helpers/preprocess_agent.py]
        │
        ├─ _strip_time(v): strip YARR timestep dim → v[:, 0]  (tensors with ndim>2 only)
        ├─ normalize RGB:  (x / 255) * 2 − 1   (for non-nerf keys)
        ├─ nerf path arrays kept as object arrays (not cast to float)
        │
        ▼
ManiFlowStackAgent.update()           [qattention_stack_agent.py]
        │
        └─ for each layer (currently only layer 0):
                ManiFlowBCAgent.update()    [qattention_maniflow_agent.py]
```

### 3.1 `ManiFlowBCAgent.update()` — Step by Step

```
1. Unpack replay sample
   ├─ action_gt        (B, 8)   ← target continuous action
   ├─ gripper_pose     (B, 7)   ← target gripper pose (SE3 aug pivot)
   ├─ lang_goal_emb    (B, 1024)
   ├─ lang_token_embs  (B, 77, 512)
   ├─ proprio          (B, 4)   ← low_dim_state
   └─ obs, depth, pcd, extrinsics, intrinsics  (per-camera tensors)

2. Load NeRF multi-view data  [only if use_neural_rendering=True]
   ├─ Subsample views consistently:  interval = N_views // num_view_for_nerf
   └─ Pick one random view index per update step

3. Compute bounds
   ├─ Layer 0:  bounds = coordinate_bounds  (global workspace [-0.3,-0.5,0.6,0.7,0.5,1.6])
   └─ Layer k:  bounds = [attention_coord_{k-1} ± bounds_offset]

4. SE3 Augmentation  [if transform_augmentation.apply_se3=True]
   ├─ Pivot: gripper_pose (target pose from replay — NOT current obs pose)
   ├─ apply_se3_augmentation_continuous(pcd, extrinsics, pivot, action_gt, ...)
   │     → yaw ~ U(−45°, +45°), translation ~ U(±5% workspace extent)
   │     → Transforms: action_gt xyz+quat, all camera PCDs, camera extrinsics
   └─ Returns augmented (action_gt, pcd, extrinsics, _)

5. Forward pass: QFunctionFlow(...)   [qattention_maniflow_agent.py]
   │
   ├─ encode_scene()   [voxel_flow_encoder.py → VoxelFlowEncoder]
   │     ① Voxelize PCD → voxel_grid  (B, 10, 100, 100, 100)
   │     ② 3D-CNN  → d0  (B, final_dim=192, V', V', V')
   │     ③ AdaptiveAvgPool3d(5) → flatten → N=8000 tokens  (B, 8000, 192)
   │     ④ Project → embedding_dim=256:  voxel_token_proj
   │     ⑤ Voxel world-coord positions → normalize to [−1,1]:  voxel_pos_norm
   │     ⑥ Add learned abs pos emb:  voxel_tokens += voxel_pos_emb(voxel_pos_norm)
   │     ⑦ Spatial grid subsample 8000 → 2×512=1024 tokens  (O(N), coarse)
   │     ⑧ VL cross-attention:  voxel_tokens ← lang_tokens  (3D RoPE on voxel side)
   │     ⑨ Density-based FPS:  1024 → num_fps_tokens=512  (O(N²), fine diversity)
   │     ⑩ proprio_proj(proprio) → proprio_feats  (B, emb)
   │     ⑪ Build context dict:
   │           { voxel_tokens (B,512,256), voxel_pos (B,512,3),
   │             lang_tokens (B,L,256), proprio_feats (B,256) }
   │
   └─ NeuralRenderer(...)  [if use_neural_rendering]
         → GS rendering loss dict {loss, loss_rgb, loss_embed, loss_dyna, psnr}

6. Flow-matching loss: _compute_flow_loss(action_gt, context)
   │
   ├─ Reshape: action_gt (B,8) → gt_traj (B,T=1,nhand=1,8)
   ├─ Separate: gt_openess (B,T,1,1) | gt_traj (B,T,1,7)
   ├─ _normalize_pos: xyz → [−1,1]  using workspace_bounds
   ├─ _convert_rot: quat_xyzw → ortho6D (Gram-Schmidt) → (B,T,1,9)
   │
   └─ Inner loop × lv2_batch_size=3:
         ① noise_pos ~ N(0,I)  (B,T,1,3)   ← INDEPENDENT from noise_rot
            noise_rot ~ N(0,I)  (B,T,1,6)
         ② t ~ logit-normal(μ=0, σ=1.5)  shape (B,)  ← SINGLE shared timestep
         ③ noisy_pos = (1−t)*x_pos + t*noise_pos
            noisy_rot = (1−t)*x_rot + t*noise_rot
         ④ noisy_traj = cat([noisy_pos, noisy_rot]).flatten(1,2)  (B, T, 9)
         ⑤ pred_list = encoder.predict_velocity(noisy_traj, t, context)
         ⑥ target_pos = noise_pos − x_pos  (computed INSIDE loop from current draw)
            target_rot = noise_rot − x_rot
         ⑦ Loss (summed over deep-supervision layers):
               L = pos_weight=30 × L1(pred_pos, target_pos)
                 + rot_weight=20 × L1(pred_rot, target_rot)
                 + grip_weight=1  × BCE(pred_grip, gt_openess)
         total_loss += L

   total_loss /= lv2_batch_size=3

7. Full training loss
   ├─ use_neural_rendering=True:
   │     total = lambda_bc=1.0 × flow_loss + lambda_nerf=0.1 × rendering_loss
   └─ use_neural_rendering=False:
         total = lambda_bc=1.0 × flow_loss

8. optimizer.zero_grad()
   fabric.backward(total_loss)
   clip_grad_norm_(q.parameters(), max_norm=1.0)
   optimizer.step()
   _update_ema()   ← EMA shadow updated AFTER every optimizer step (decay=0.999)

9. Optional LR scheduler step (cosine with warmup when lr_scheduler=True)
```

---

## 4. Model Architecture

### 4.1 VoxelFlowEncoder  (`voxel_flow_encoder.py`)

```
Input: voxel_grid (B, 10, 100, 100, 100)
       proprio     (B, 4)
       lang_goal_emb  (B, 1024)    ← CLIP sentence emb
       lang_token_embs (B, 77, 512) ← CLIP token embs

  ┌─────────────────────────────────────────────────────────────────┐
  │  3D-CNN Encoder  (MultiLayer3DEncoderShallow)                   │
  │  → d0  (B, final_dim=192, V', V', V')                          │
  └──────────────────────────┬──────────────────────────────────────┘
                             │ shared voxel feature map
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
  ┌──────────────────┐           ┌──────────────────────────────────┐
  │ NeuralRenderer   │           │ Flow-matching head               │
  │ (GS renderer,    │           │                                  │
  │  aux loss only)  │           │  AdaptiveAvgPool3d(stride=5)     │
  └──────────────────┘           │  → flatten → 8000 tokens         │
                                 │  → voxel_token_proj → emb=256    │
                                 │  + learned abs pos emb           │
                                 │                                  │
                                 │  ① Spatial grid subsample→1024   │
                                 │  ② VL cross-attn (3D RoPE)       │
                                 │     voxel tokens ← lang tokens   │
                                 │  ③ Density FPS → 512 tokens      │
                                 │                                  │
                                 │  lang_token_proj: (B,77,256)     │
                                 │  proprio_proj:    (B,256)        │
                                 └──────────────┬───────────────────┘
                                                ▼
                                         context dict
                             { voxel_tokens (B, 512, 256),
                               voxel_pos    (B, 512, 3),
                               lang_tokens  (B, 77, 256),
                               proprio_feats (B, 256) }
```

### 4.2 TransformerHead (Denoising Head)

```
Input:  noisy_traj  (B, T*nhand, 9)   [x_norm, y_norm, z_norm, r1…r6]
        timestep t  (B,)              float in [0, 1]
        context dict

AdaLN conditioning:
    ada_cond = SinusoidalPosEmb(t)          (B, emb)
             + 2-layer MLP(proprio_feats)   (B, emb)

Trajectory tokens:
    traj_tokens = traj_encoder(noisy_traj) + step_pos_emb   (B, T, 256)

─── Stage 1: Cross-attn → Language ─────────────────────────────────────
    traj_tokens = cross_attn(Q=traj, KV=lang_tokens)   [no RoPE, no AdaLN]

─── Stage 2: Cross-attn → Voxel scene ──────────────────────────────────
    traj_tokens = cross_attn(Q=traj, KV=voxel_tokens, 3D RoPE, AdaLN)

─── Stage 3: Joint self-attn ────────────────────────────────────────────
    joint = cat([traj_tokens, voxel_tokens], dim=1)   (B, T+512, 256)
    joint = self_attn(joint, AdaLN, 3D RoPE)  ×  num_shared_attn_layers=6

─── Output heads  (slice first T tokens) ────────────────────────────────
    pos head:  2-layer self-attn → Linear → velocity_pos  (B, T, 3)
    rot head:  2-layer self-attn → Linear → velocity_rot  (B, T, 6)  ortho6D
    grip head: Linear           → openness logit          (B, T, 1)

Output: list of (B, T*nhand, 10)  per deep-supervision layer
        last dim = [vel_pos(3), vel_rot(6), grip_logit(1)]
```

---

## 5. SE3 Augmentation Detail

```python
apply_se3_augmentation_continuous(
    pcd, camera_extrinsics, action_gripper_pose,   # pivot = gripper_pose from replay
    action_gt, bounds, ...)
```

The scene is perturbed around the **target gripper pose** (`gripper_pose`):

```
Random perturbation:
    t_shift ~ U(−aug_xyz × bounds_extent, +aug_xyz × bounds_extent)
              aug_xyz = [0.05, 0.05, 0.05]
    R_delta = euler_to_matrix(roll=0, pitch=0, yaw ~ U(−45°, +45°))

For every world-frame point p:
    p' = R_delta^T × (p − t_pivot) + clamp(t_pivot + t_shift)

Applied to:
    ✓ Point clouds         (perturb_se3)
    ✓ Camera extrinsics    (perturb_se3_camera_pose, cloned to avoid in-place mutation)
    ✓ action_gt xyz+quat   (direct SE3 math, no voxel discretization)
    ✗ RGB images           (cameras are fixed; augmentation is world-frame only)
    ✗ obs_gripper_pose     (not used in current ManiFlow encoder forward)
```

---

## 6. Inference (Evaluation)

```
CustomRLBenchEnv.step(action)
        │  self._i += 1  ← incremented BEFORE extract_obs
        │                   so time token t=self._i matches training (t=k, after k actions)
        ▼
YARR RolloutGenerator
        │  wraps obs → (1, timesteps=1, ...) tensors
        ▼
PreprocessAgent.act()
        │  normalize RGB: (x/255)*2−1
        ▼
ManiFlowStackAgent.act()
        │
        └─ ManiFlowBCAgent.act()
                │
                ├─ language_model.extract(lang_goal)
                │     → lang_goal_emb (1, 1024), lang_token_embs (1, 77, 512)
                │
                ├─ Strip YARR timestep dim from all obs tensors:
                │     rgb/pcd/depth: obs[:, 0]   (1,T,C,H,W) → (1,C,H,W)
                │     proprio:       proprio[:, 0, :]          (1,T,4) → (1,4)
                │     extr/intr:     .squeeze(0)  already (4,4)/(3,3) from _act_preprocess
                │
                ├─ with torch.no_grad(), _use_ema_weights():
                │     encode_scene() → context dict
                │     _denoise_action(context, num_steps=denoise_timesteps=20)
                │
                │     _denoise_action:
                │       trajectory ~ N(0,I)  (B=1, T*nhand=1, 9)
                │       pos/rot schedulers set_timesteps(20)
                │       logit-normal warped inference grid  (FIX #5)
                │       Euler loop (20 steps, shared t for pos+rot):
                │           v = TransformerHead(traj, t, context)
                │           pos = pos_scheduler.step(v[..,:3],  idx, traj[..,:3])
                │           rot = rot_scheduler.step(v[..3:9], idx, traj[..3:9])
                │           traj = cat([pos, rot])
                │       _unnormalize_pos: [−1,1] → world coords
                │       _unconvert_rot: ortho6D → quat_xyzw  (Shepperd)
                │       grip: sigmoid(v[...,9]) > grip_threshold=0.5 → binary
                │
                └─ action_np = actions[0, 0, 0].cpu().numpy()   (8,)

ManiFlowStackAgent appends ignore_collisions (from obs) → (9,)
RLBench: MoveArmThenGripper [x,y,z,qx,qy,qz,qw, grip, ignore_col]
```

---

## 7. EMA (Exponential Moving Average)

```
Training:
    After every optimizer.step():
        _update_ema():
            for each trainable param:
                ema[name] = 0.999 × ema[name] + 0.001 × param.data   (on CPU)

Save (every save_freq=10000 steps):
    logs/<exp>/seed0/weights/<step>/
        ManiFlowAgent_layer0.pt        ← raw weights
        ManiFlowAgent_layer0_ema.pt    ← EMA shadow weights

Eval / act():
    load_weights() automatically loads _ema.pt if it exists.
    _use_ema_weights() context manager:
        → swaps EMA params into live model
        → runs encode_scene + _denoise_action
        → restores original params on exit
    No-ops gracefully if _ema.pt is missing (prints red WARNING).
```

---

## 8. Loss Formulation

### 8.1 Rectified Flow (BC loss)

$$
\mathcal{L}_{\text{flow}} = \frac{1}{N_{\text{lv2}}} \sum_{i=1}^{N_{\text{lv2}}} \sum_{l=1}^{L_{\text{depth}}} \left[
  w_{\text{pos}} \cdot \|v_\theta^{\text{pos}} - v^*_{\text{pos}}\|_1
  + w_{\text{rot}} \cdot \|v_\theta^{\text{rot}} - v^*_{\text{rot}}\|_1
  + w_{\text{grip}} \cdot \text{BCE}(v_\theta^{\text{grip}},\, y_{\text{grip}})
\right]
$$

where:
- $N_{\text{lv2}} = 3$ (`lv2_batch_size`, inner re-sampling iterations — normalised)
- $L_{\text{depth}}$ = number of deep-supervision Transformer layers (summed, not normalised)
- $v^* = \varepsilon - x_{\text{clean}}$ is the RF velocity target
- $z_t = (1-t)\,x_{\text{clean}} + t\,\varepsilon$,  $t \sim \text{logit-normal}(\mu{=}0,\, \sigma{=}1.5)$
- Independent $\varepsilon_{\text{pos}},\,\varepsilon_{\text{rot}}$;  single shared $t$ for both
- $w_{\text{pos}}=30$,  $w_{\text{rot}}=20$,  $w_{\text{grip}}=1$

### 8.2 Total Loss

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{bc}} \cdot \mathcal{L}_{\text{flow}}
  + \lambda_{\text{nerf}} \cdot \mathcal{L}_{\text{GS}}
$$

| Hyper-param | Value | Notes |
|---|---|---|
| `lambda_bc` | 1.0 | BC loss multiplier |
| `lambda_nerf` | 0.1 | GS auxiliary multiplier |
| `lambda_rgb` | 1.0 | inside GS loss |
| `lambda_embed` | 0.5 | inside GS loss |
| `lambda_dyna` | 0.01 | inside GS dynamics loss |

---

## 9. Key Configuration Reference (`conf/method/ManiFlow_BC.yaml`)

| Config key | Current value | Meaning |
|---|---|---|
| `action_dim` | 8 | `[x,y,z,qx,qy,qz,qw,grip]` |
| `action_chunk_size` | 1 | T steps per denoising call |
| `lv2_batch_size` | 3 | Inner RF loss re-sampling iterations |
| `denoise_timesteps` | 20 | Euler steps at inference |
| `embedding_dim` | 256 | Transformer feature dim |
| `num_attn_heads` | 8 | Attention heads |
| `num_shared_attn_layers` | 6 | Joint self-attn depth |
| `final_dim` | 192 | 3D-CNN output channels |
| `voxel_token_downsample` | 5 | Pool stride → N=(100/5)³=8000 tokens |
| `num_fps_tokens` | 512 | Tokens after density-FPS subsampling |
| `workspace_bounds` | `[-0.3,-0.5,0.6, 0.7,0.5,1.6]` | xyz normalisation range |
| `pos_loss_weight` | 30.0 | L1 position loss weight |
| `rot_loss_weight` | 20.0 | L1 rotation loss weight |
| `grip_loss_weight` | 1.0 | BCE grip loss weight |
| `grip_threshold` | 0.5 | Sigmoid threshold at inference |
| `ema_decay` | 0.999 | EMA decay (0 = disabled) |
| `optimizer` | `adam` | lr=1e-4 |
| `lr_scheduler` | False | cosine+warmup when True |
| `transform_augmentation.aug_rpy` | `[0,0,45]` | ±45° yaw SE3 augmentation |
| `use_neural_rendering` | True | Enable GS auxiliary loss |

---

## 10. Bug Fixes Applied (vs. Original Codebase)

| # | File | Description |
|---|---|---|
| 1 | `custom_rlbench_env.py` — both `step()` | `self._i` now incremented **before** `extract_obs`, so time token at step k is `k`, matching training |
| 2 | `launch_utils.py` — `_add_keypoints_to_replay` | `t=k` (not `obs_frame_idx`); terminal obs uses `t=k+1`; `k_offset` removed — each augmented start treated as fresh episode |
| 3 | `qattention_maniflow_agent.py` — `_compute_flow_loss` | Independent `noise_pos`/`noise_rot` (FIX #3); target recomputed **inside** `lv2_batch_size` loop (FIX #4) |
| 4 | `qattention_maniflow_agent.py` — `_compute_flow_loss` | Single shared timestep `t` for pos+rot (eliminates train/eval distribution mismatch in AdaLN) |
| 5 | `qattention_maniflow_agent.py` — `_denoise_action` | `grip_threshold` now configurable (default 0.5, was hardcoded 0.6) |
| 6 | `rf_scheduler.py` | Logit-normal warped inference grid (FIX #5) — inference timestep distribution matches training sampler |
| 7 | `qattention_maniflow_agent.py` — `update()` | Gradient clip `max_norm=1.0` (was 10.0); single cosine LR cycle (was 20+ hard restarts) |
| 8 | `qattention_maniflow_agent.py` — `build()` | EMA shadow weights enabled during training (decay=0.999), swapped in for `act()` |
| 9 | `qattention_maniflow_agent.py` — `load_weights()` | Fabric `_forward_module` / DDP `module` unwrapped before `load_state_dict`; `weights_only=True` |
| 10 | `augmentation.py` — `perturb_se3_camera_pose` | `.clone()` before in-place modification — prevents replay-buffer corruption |

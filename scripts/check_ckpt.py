"""
Diagnostic script: verify that every checkpoint in logs/maniflow_run1/seed0/weights
loads cleanly into the eval model built from the current config.

Simulates the exact load_weights() remapping logic so any future
architecture/config drift is caught before silently breaking eval.

Reports per checkpoint:
  - shape mismatches   → would silently stay random via strict=False
  - not_in_model       → checkpoint key not recognised by current model
  - missing_from_ckpt  → learned model key absent from checkpoint (random init!)

Usage (inside the neuralgrasp Docker container):
    cd /app && uv run python scripts/check_ckpt.py [weights_root]

Default weights_root: /app/logs/maniflow_run1/seed0/weights
"""
import sys, os
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/third_party/YARR')
sys.path.insert(0, '/app/third_party/RLBench')
sys.path.insert(0, '/app/third_party/PyRep')

import torch
from omegaconf import OmegaConf

# ── Build eval model from current config ─────────────────────────────────────
base_cfg = OmegaConf.load('/app/conf/config.yaml')
method_cfg = OmegaConf.load('/app/conf/method/ManiFlow_BC.yaml')
OmegaConf.set_struct(base_cfg, False)
base_cfg.method = method_cfg
base_cfg.method.use_neural_rendering = False   # eval never uses renderer

from agents.maniflow_bc.launch_utils import create_agent
agent = create_agent(base_cfg)
qa = agent._pose_agent._qattention_agents[0]
qa.build(training=False, device=torch.device('cpu'), use_ddp=False)
model_sd = qa._q.state_dict()

print(f"Model embedding_dim       : {base_cfg.method.embedding_dim}")
print(f"Model num_attn_heads      : {base_cfg.method.num_attn_heads}")
print(f"Model num_shared_attn_layers: {base_cfg.method.num_shared_attn_layers}")
print(f"Model total state_dict keys: {len(model_sd)}")
print()

# ── Replicate load_weights() remapping exactly ───────────────────────────────
def remap_key(k):
    """Mirror the key-remapping logic in ManiFlowBCAgent.load_weights()."""
    if '_voxelizer' in k:
        return None                                      # always skipped
    k = k.replace('_flow_encoder.module',    '_flow_encoder')
    k = k.replace('_neural_renderer.module', '_neural_renderer')
    # Backward-compat: old Sequential proprio_proj → current nn.Linear
    k = k.replace('proprio_proj.0.weight',   'proprio_proj.weight')
    k = k.replace('proprio_proj.0.bias',     'proprio_proj.bias')
    return k

# Learned model keys: exclude voxelizer (non-learned) and neural_renderer
# (absent at eval because use_neural_rendering=False).
learned_model_keys = {k for k in model_sd
                      if '_voxelizer' not in k and '_neural_renderer' not in k}

# ── Iterate every checkpoint ─────────────────────────────────────────────────
weights_root = sys.argv[1] if len(sys.argv) > 1 \
    else '/app/logs/maniflow_run1/seed0/weights'
steps = sorted(os.listdir(weights_root), key=int)
print(f"Checking {len(steps)} checkpoints under: {weights_root}")
print(f"Steps: {steps}\n")

all_ok = True
for step in steps:
    ckpt_path = os.path.join(weights_root, step, 'ManiFlowAgent_layer0.pt')
    sd = torch.load(ckpt_path, map_location='cpu')

    shape_mismatches, not_in_model, ckpt_mapped = [], [], set()

    for k_orig, v in sd.items():
        k = remap_key(k_orig)
        if k is None:
            continue                                     # voxelizer — skip
        ckpt_mapped.add(k)
        if k in model_sd:
            if model_sd[k].shape != v.shape:
                shape_mismatches.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            not_in_model.append(k)

    missing_from_ckpt = learned_model_keys - ckpt_mapped

    ok = not shape_mismatches and not not_in_model and not missing_from_ckpt
    all_ok = all_ok and ok
    status = 'OK  ✓' if ok else 'FAIL ✗'
    print(f"  step={step:>6}  [{status}]"
          f"  shape_mm={len(shape_mismatches)}"
          f"  not_in_model={len(not_in_model)}"
          f"  missing_from_ckpt={len(missing_from_ckpt)}")

    for k, cs, ms in shape_mismatches:
        print(f"      SHAPE MISMATCH : {k}  ckpt={cs}  model={ms}")
    for k in not_in_model:
        print(f"      NOT IN MODEL   : {k}")
    for k in sorted(missing_from_ckpt):
        print(f"      MISSING IN CKPT (random init!): {k}  model_shape={tuple(model_sd[k].shape)}")

print()
print("=" * 60)
print(f"Overall: {'ALL CHECKPOINTS OK ✓' if all_ok else 'FAILURES DETECTED ✗'}")

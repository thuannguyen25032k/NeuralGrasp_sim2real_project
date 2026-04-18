'''
Modified from: https://github.com/Kunhao-Liu/3D-OVS/blob/main/models/DINO_extractor.py
'''
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def _patch_pep604_in_dir(root_dir: str) -> int:
    """
    Walk `root_dir` and inject `from __future__ import annotations` at the
    top of every .py file that contains PEP 604 union syntax (`X | Y`).
    This makes all annotations lazy-evaluated strings, which keeps the
    DINOv2 `main` branch importable under Python 3.9 (the default evaluation
    model of function-signature annotations would otherwise raise
    `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`).

    Returns the number of files patched.
    """
    # Match `float | None`, `int | str`, `X | None = ...`, etc.
    # The pattern requires an identifier/qualname on each side of the pipe
    # so we don't false-match `a | b` in ordinary expressions.
    pat = re.compile(r'\b[A-Za-z_][A-Za-z0-9_\.\[\]]*\s*\|\s*[A-Za-z_][A-Za-z0-9_\.\[\]]*')
    future = 'from __future__ import annotations\n'
    patched = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    src = f.read()
            except Exception:
                continue
            if future in src:
                continue
            if not pat.search(src):
                continue
            # Preserve shebang and module docstring position: insert after
            # any leading comment / docstring block.
            lines = src.splitlines(keepends=True)
            insert_at = 0
            # Skip shebang
            if lines and lines[0].startswith('#!'):
                insert_at = 1
            # Skip encoding / copyright comments and blank lines
            while insert_at < len(lines) and (
                lines[insert_at].lstrip().startswith('#') or lines[insert_at].strip() == ''
            ):
                insert_at += 1
            # If a module docstring starts here, skip to after it
            if insert_at < len(lines) and lines[insert_at].lstrip().startswith(('"""', "'''")):
                quote = '"""' if '"""' in lines[insert_at] else "'''"
                # single-line docstring
                if lines[insert_at].count(quote) >= 2:
                    insert_at += 1
                else:
                    insert_at += 1
                    while insert_at < len(lines) and quote not in lines[insert_at]:
                        insert_at += 1
                    insert_at += 1  # past the closing line
            lines.insert(insert_at, future)
            try:
                with open(fp, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                patched += 1
            except Exception:
                pass
    return patched


class VitExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vitl14'):
        super().__init__()
        # FIX (Python 3.9 ↔ DINOv2 `main`): the current DINOv2 codebase uses
        # PEP 604 union syntax (`float | None`) in function signatures, which
        # Python 3.9 evaluates eagerly and rejects with
        #   TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
        # We fix it by (a) letting torch.hub download the repo as normal into
        # its cache, and (b) injecting `from __future__ import annotations`
        # at the top of every .py file in the cache that uses PEP 604 — this
        # makes annotations lazy-evaluated strings, which Python 3.9 accepts.
        # This is a one-off patch on the cached source tree; no network calls.
        hub_dir = torch.hub.get_dir()
        cache_dir = os.path.join(hub_dir, 'facebookresearch_dinov2_main')

        # First, ensure the repo is cached (downloads if missing, using main).
        # We call with skip_validation to avoid re-downloading if already cached.
        if not os.path.isdir(cache_dir):
            # Force download of 'main' (this will succeed — the failure was
            # at the subsequent *import* step, not the download).
            try:
                torch.hub.set_dir(hub_dir)
                torch.hub._get_cache_or_reload(
                    'facebookresearch/dinov2', force_reload=False,
                    trust_repo=True, calling_fn='load',
                    verbose=True, skip_validation=True,
                )
            except Exception:
                # If the private API signature differs in this torch version,
                # fall back to a best-effort load that will populate the cache
                # before failing — the subsequent patched import will succeed.
                try:
                    torch.hub.load(
                        'facebookresearch/dinov2', model_name,
                        trust_repo=True,
                    )
                except Exception:
                    pass  # expected to fail; cache is now populated

        # Patch the cached source tree so PEP 604 is lazy-evaluated.
        if os.path.isdir(cache_dir):
            n = _patch_pep604_in_dir(cache_dir)
            if n > 0:
                print(f"[VitExtractor] Patched {n} DINOv2 file(s) for Python 3.9 compatibility.")

        # Now the import path is safe: load from the already-cached + patched tree.
        self.model = torch.hub.load(
            'facebookresearch/dinov2', model_name,
            source='github', trust_repo=True,
        )
        self.model.eval()
        self.patch_size = 14
        self.feature_dims = 1024
        self.preprocess = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet
        ])

        self._freeze()

    def _freeze(self):
        super().train(mode=False)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, input_img):
        B, C, H, W = input_img.shape
        input_img = self.preprocess(input_img)
        dino_ret = self.model.forward_features(input_img)['x_norm_patchtokens']
        dino_ret = dino_ret.transpose(1, 2).reshape([B, -1, H//self.patch_size, W//self.patch_size])    # [B, 1024, 128, 128]
        return dino_ret

import torch
import numpy as np
import torch.autograd.profiler as profiler
import visdom
import time

# ---------------------------------------------------------------------------
# 6D rotation representation utilities
# (Zhou et al., "On the Continuity of Rotation Representations in Neural
#  Networks", CVPR 2019)
# ---------------------------------------------------------------------------

def normalise_quat(x: torch.Tensor) -> torch.Tensor:
    """Normalise a quaternion tensor along its last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def quat_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert xyzw quaternion (..., 4) to rotation matrix (..., 3, 3)."""
    quat = normalise_quat(quat)
    x, y, z, w = quat.unbind(-1)
    B = quat.shape[:-1]
    mat = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*B, 3, 3)
    return mat


def matrix_to_ortho6d(matrix: torch.Tensor) -> torch.Tensor:
    """Extract the first two columns of a rotation matrix as 6D rep (..., 6)."""
    # matrix: (..., 3, 3)
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)   # (..., 6)


def ortho6d_to_matrix(ortho6d: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt: 6D rep (..., 6) → rotation matrix (..., 3, 3)."""
    a1 = ortho6d[..., :3]
    a2 = ortho6d[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack([b1, b2, b3], dim=-1)           # (..., 3, 3)


def matrix_to_quat_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) → xyzw quaternion (..., 4)."""
    # Shepperd method
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    quats = torch.zeros(*matrix.shape[:-2], 4, device=matrix.device,
                        dtype=matrix.dtype)
    # Case: trace > 0
    s     = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2   # 4w
    quats[..., 0] = (m[..., 2, 1] - m[..., 1, 2]) / s   # x
    quats[..., 1] = (m[..., 0, 2] - m[..., 2, 0]) / s   # y
    quats[..., 2] = (m[..., 1, 0] - m[..., 0, 1]) / s   # z
    quats[..., 3] = 0.25 * s                              # w

    # Case: m00 largest diagonal — use a more numerically stable path
    cond1 = (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2]) & (trace <= 0)
    s1    = torch.sqrt(torch.clamp(1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], min=1e-10)) * 2
    q1    = torch.stack([0.25 * s1,
                         (m[..., 0, 1] + m[..., 1, 0]) / s1,
                         (m[..., 0, 2] + m[..., 2, 0]) / s1,
                         (m[..., 2, 1] - m[..., 1, 2]) / s1], dim=-1)
    quats = torch.where(cond1.unsqueeze(-1), q1, quats)

    cond2 = (m[..., 1, 1] > m[..., 2, 2]) & ~cond1 & (trace <= 0)
    s2    = torch.sqrt(torch.clamp(1.0 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2], min=1e-10)) * 2
    q2    = torch.stack([(m[..., 0, 1] + m[..., 1, 0]) / s2,
                         0.25 * s2,
                         (m[..., 1, 2] + m[..., 2, 1]) / s2,
                         (m[..., 0, 2] - m[..., 2, 0]) / s2], dim=-1)
    quats = torch.where(cond2.unsqueeze(-1), q2, quats)

    cond3 = ~cond1 & ~cond2 & (trace <= 0)
    s3    = torch.sqrt(torch.clamp(1.0 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1], min=1e-10)) * 2
    q3    = torch.stack([(m[..., 0, 2] + m[..., 2, 0]) / s3,
                         (m[..., 1, 2] + m[..., 2, 1]) / s3,
                         0.25 * s3,
                         (m[..., 1, 0] - m[..., 0, 1]) / s3], dim=-1)
    quats = torch.where(cond3.unsqueeze(-1), q3, quats)

    return normalise_quat(quats)    # (..., 4)  xyzw


def quat_xyzw_to_ortho6d(quat: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion (..., 4) → 6D rotation rep (..., 6). Used in training."""
    return matrix_to_ortho6d(quat_xyzw_to_matrix(quat))


def ortho6d_to_quat_xyzw(ortho6d: torch.Tensor) -> torch.Tensor:
    """6D rotation rep (..., 6) → xyzw quaternion (..., 4). Used at inference."""
    return matrix_to_quat_xyzw(ortho6d_to_matrix(ortho6d))


@torch.no_grad()
def visualize_pcd(xyz, attention_coordinate=None, rgb=None, name='xyz', sleep=0):
    '''
    use visdom to visualize point cloud in training process
    xyz: (B, N, 3)
    rgb: (B, 3, H, W)
    '''
    vis = visdom.Visdom()
    if rgb is not None:
        rgb_vis = rgb[0].detach().cpu().numpy()
        vis.image(rgb_vis, win='rgb', opts=dict(title='rgb'))

    # point cloud
    pc_vis = xyz[0].detach().cpu().numpy()  # (128*128, 3)

    # visualize ground-truth action_trans (B,3) in point cloud (blue)
    if attention_coordinate is not None:
        action = attention_coordinate[0].unsqueeze(0).detach().cpu().numpy()
        pc_vis_aug = np.concatenate([pc_vis, action], axis=0)
        label_vis = np.concatenate([np.zeros((pc_vis.shape[0], 1))+1, np.zeros((1,1))+2], axis=0)
    else:
        pc_vis_aug = pc_vis
        label_vis = np.zeros((pc_vis.shape[0], 1))+1
    label_vis = label_vis.astype(int)

    vis.scatter(
        X=pc_vis_aug, Y=label_vis, win=name, 
        opts=dict(
            title=name,
            markersize=1,
            markercolor=np.array([[0,0,255], [255,0,0]]) if attention_coordinate is not None else np.array([[0,0,255]]),
            # blue and red
        )
    )
    if sleep > 0:
        time.sleep(sleep)


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
    )
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj


def gen_rays(poses, width, height, focal, z_near, z_far, c=None):
    """
    Generate camera rays
    :return (B, H, W, 8)
    """
    num_images = poses.shape[0]
    device = poses.device
    cam_unproj_map = (
        unproj_map(width, height, focal.squeeze(), c=c, device=device)
        .unsqueeze(0)
        .repeat(num_images, 1, 1, 1)
    )
    cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
    cam_raydir = torch.matmul(
        poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)
    )[:, :, :, :, 0]

    cam_nears = (
        torch.tensor(z_near, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    cam_fars = (
        torch.tensor(z_far, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    return torch.cat(
        (cam_centers, cam_raydir, cam_nears, cam_fars), dim=-1
    )  # (B, H, W, 8)


def combine_interleaved(t, inner_dims=(1,), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])    # [1, 1, 16384, 512]
    if agg_type == "average":   # default
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t    # [1, 16384, 512]

class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = embed.view(x.shape[0], -1)
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.num_freqs,
            d_in,
            conf.freq_factor,
            conf.include_input,
        )

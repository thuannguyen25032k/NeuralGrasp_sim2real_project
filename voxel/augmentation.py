import numpy as np
import torch
from helpers import utils
from pytorch3d import transforms as torch3d_tf
from termcolor import cprint

import einops
from scipy.spatial.transform import Rotation as R


def perturb_se3(pcd,
                trans_shift_4x4,
                rot_shift_4x4,
                action_gripper_4x4,
                bounds):
    """ Perturb point clouds with given transformation.
    :param pcd: list of point clouds [[bs, 3, N], ...] for N cameras
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds
    """
    # baatch bounds if necessary
    bs = pcd[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_pcd = []
    for p in pcd:
        p_shape = p.shape
        num_points = p_shape[-1] * p_shape[-2]

        action_trans_3x1 = action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        trans_shift_3x1 = trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)

        # flatten point cloud
        p_flat = p.reshape(bs, 3, -1)
        p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)

        # shift points to have action_gripper pose as the origin
        p_flat_4x1_action_origin[:, :3, :] = p_flat - action_trans_3x1

        # apply rotation
        perturbed_p_flat_4x1_action_origin = torch.bmm(p_flat_4x1_action_origin.transpose(2, 1),
                                                       rot_shift_4x4).transpose(2, 1)

        # apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(action_then_trans_3x1[:, 0],
                                              min=bounds_x_min, max=bounds_x_max)
        action_then_trans_3x1_y = torch.clamp(action_then_trans_3x1[:, 1],
                                              min=bounds_y_min, max=bounds_y_max)
        action_then_trans_3x1_z = torch.clamp(action_then_trans_3x1[:, 2],
                                              min=bounds_z_min, max=bounds_z_max)
        action_then_trans_3x1 = torch.stack([action_then_trans_3x1_x,
                                             action_then_trans_3x1_y,
                                             action_then_trans_3x1_z], dim=1)

        # shift back the origin
        perturbed_p_flat_3x1 = perturbed_p_flat_4x1_action_origin[:, :3, :] + action_then_trans_3x1

        perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
        perturbed_pcd.append(perturbed_p)
    return perturbed_pcd


def perturb_se3_camera_pose(camera_pose,
                trans_shift_4x4,
                rot_shift_4x4,
                action_gripper_4x4,
                bounds):
    """ Perturb point clouds with given transformation.
    :param pcd: list of point clouds [[bs, 3, N], ...] for N cameras
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds
    """
    # batch bounds if necessary
    bs = camera_pose[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_camera_pose = []
    for cam_pose in camera_pose:
        cam_pose = cam_pose.clone()   # avoid mutating the replay-buffer tensor in-place
        cam_R, cam_T = cam_pose[:, :3, :3], cam_pose[:, :3, 3:]

        # action_trans_3x1 = action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(bs, 1, 1)
        # trans_shift_3x1 = trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(bs, 1, 1)
        action_trans_3x1 = action_gripper_4x4[:, 0:3, 3].unsqueeze(-1)
        trans_shift_3x1 = trans_shift_4x4[:, 0:3, 3].unsqueeze(-1)

        cam_T = cam_T - action_trans_3x1    # [bs, 3, 1]
        cam_T_4x1 = torch.ones(bs, 4, 1).to(cam_T.device)
        cam_T_4x1[:, :3, :] = cam_T
        cam_T_4x1 = torch.bmm(cam_T_4x1.transpose(2, 1), rot_shift_4x4).transpose(2, 1)

        cam_R = torch.bmm(cam_R.transpose(2, 1), rot_shift_4x4[:, :3, :3]).transpose(2, 1)

        # apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(action_then_trans_3x1[:, 0],
                                              min=bounds_x_min, max=bounds_x_max)
        action_then_trans_3x1_y = torch.clamp(action_then_trans_3x1[:, 1],
                                              min=bounds_y_min, max=bounds_y_max)
        action_then_trans_3x1_z = torch.clamp(action_then_trans_3x1[:, 2],
                                              min=bounds_z_min, max=bounds_z_max)
        action_then_trans_3x1 = torch.stack([action_then_trans_3x1_x,
                                             action_then_trans_3x1_y,
                                             action_then_trans_3x1_z], dim=1)

        # shift back the origin
        cam_T_4x1[:, :3]  = cam_T_4x1[:, :3] + action_then_trans_3x1

        # get new camera pose
        cam_T = cam_T_4x1[:, :3, :]
        cam_pose[:, :3, :3], cam_pose[:, :3, 3:] = cam_R, cam_T
        perturbed_camera_pose.append(cam_pose)

    return perturbed_camera_pose


def apply_se3_augmentation(pcd,
                           action_gripper_pose,
                           action_trans,
                           action_rot_grip,
                           bounds,
                           layer,
                           trans_aug_range,
                           rot_aug_range,
                           rot_aug_resolution,
                           voxel_size,
                           rot_resolution,
                           device):
    """ Apply SE3 augmentation to a point clouds and actions.
    :param pcd: list of point clouds [[bs, 3, H, W], ...] for N cameras
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param action_trans: discretized translation action [bs, 3]
    :param action_rot_grip: discretized rotation and gripper action [bs, 4]
    :param bounds: metric scene bounds of voxel grid [bs, 6]
    :param layer: voxelization layer (always 1 for PerAct)
    :param trans_aug_range: range of translation augmentation [x_range, y_range, z_range]
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :param rot_aug_resolution: degree increments for discretized augmentation rotations
    :param voxel_size: voxelization resoltion
    :param rot_resolution: degree increments for discretized rotations
    :param device: torch device
    :return: perturbed action_trans, action_rot_grip, pcd
    """

    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = action_gripper_pose[:, :3]
    action_gripper_quat_wxyz = torch.cat((action_gripper_pose[:, 6].unsqueeze(1),
                                          action_gripper_pose[:, 3:6]), dim=1)
    action_gripper_rot = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz)
    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    perturbed_trans = torch.full_like(action_trans, -1.)
    perturbed_rot_grip = torch.full_like(action_rot_grip, -1.)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    while torch.any(perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 10:
            # raise Exception('Failing to perturb action and keep it within bounds.')
            cprint('Failing to perturb action and keep it within bounds. use non-perturbed value.', 'red')
            # return original action
            return action_trans, action_rot_grip, pcd

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)
        trans_shift = trans_range * utils.rand_dist((bs, 3)).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = utils.rand_discrete((bs, 1),
                                   min=-roll_aug_steps,
                                   max=roll_aug_steps) * np.deg2rad(rot_aug_resolution)
        pitch = utils.rand_discrete((bs, 1),
                                    min=-pitch_aug_steps,
                                    max=pitch_aug_steps) * np.deg2rad(rot_aug_resolution)
        yaw = utils.rand_discrete((bs, 1),
                                  min=-yaw_aug_steps,
                                  max=yaw_aug_steps) * np.deg2rad(rot_aug_resolution)
        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(torch.cat((roll, pitch, yaw), dim=1), "XYZ")
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        # rotate then translate the 4x4 keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
        perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(perturbed_action_gripper_4x4[:, :3, :3])
        perturbed_action_quat_xyzw = torch.cat([perturbed_action_quat_wxyz[:, 1:],
                                                perturbed_action_quat_wxyz[:, 0].unsqueeze(1)],
                                               dim=1).cpu().numpy()

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        trans_indicies, rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            trans_idx = utils.point_to_voxel_index(perturbed_action_trans[b], voxel_size, bounds_np)
            trans_indicies.append(trans_idx.tolist())

            quat = perturbed_action_quat_xyzw[b]
            quat = utils.normalize_quaternion(perturbed_action_quat_xyzw[b])
            if quat[-1] < 0:
                quat = -quat
            disc_rot = utils.quaternion_to_discrete_euler(quat, rot_resolution)
            rot_grip_indicies.append(disc_rot.tolist() + [int(action_rot_grip[b, 3].cpu().numpy())])

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        perturbed_trans = torch.from_numpy(np.array(trans_indicies)).to(device=device)
        perturbed_rot_grip = torch.from_numpy(np.array(rot_grip_indicies)).to(device=device)

    action_trans = perturbed_trans
    action_rot_grip = perturbed_rot_grip

    # apply perturbation to pointclouds
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

    return action_trans, action_rot_grip, pcd

def apply_se3_augmentation_with_camera_pose(pcd,
                            camera_pose,
                           action_gripper_pose,
                           action_trans,
                           action_rot_grip,
                           bounds,
                           layer,
                           trans_aug_range,
                           rot_aug_range,
                           rot_aug_resolution,
                           voxel_size,
                           rot_resolution,
                           device):
    """ Apply SE3 augmentation to a point clouds and actions.
    :param pcd: list of point clouds [[bs, 3, H, W], ...] for N cameras
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param action_trans: discretized translation action [bs, 3]
    :param action_rot_grip: discretized rotation and gripper action [bs, 4]
    :param bounds: metric scene bounds of voxel grid [bs, 6]
    :param layer: voxelization layer (always 1 for PerAct)
    :param trans_aug_range: range of translation augmentation [x_range, y_range, z_range]
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :param rot_aug_resolution: degree increments for discretized augmentation rotations
    :param voxel_size: voxelization resoltion
    :param rot_resolution: degree increments for discretized rotations
    :param device: torch device
    :return: perturbed action_trans, action_rot_grip, pcd
    """

    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = action_gripper_pose[:, :3]
    action_gripper_quat_wxyz = torch.cat((action_gripper_pose[:, 6].unsqueeze(1),
                                          action_gripper_pose[:, 3:6]), dim=1)
    action_gripper_rot = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz)
    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    perturbed_trans = torch.full_like(action_trans, -1.)
    perturbed_rot_grip = torch.full_like(action_rot_grip, -1.)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    while torch.any(perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 10:
            # raise Exception('Failing to perturb action and keep it within bounds.')
            cprint('Failing to perturb action and keep it within bounds. use non-perturbed value.', 'red')
            # return original action
            return action_trans, action_rot_grip, pcd, camera_pose

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)
        trans_shift = trans_range * utils.rand_dist((bs, 3)).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = utils.rand_discrete((bs, 1),
                                   min=-roll_aug_steps,
                                   max=roll_aug_steps) * np.deg2rad(rot_aug_resolution)
        pitch = utils.rand_discrete((bs, 1),
                                    min=-pitch_aug_steps,
                                    max=pitch_aug_steps) * np.deg2rad(rot_aug_resolution)
        yaw = utils.rand_discrete((bs, 1),
                                  min=-yaw_aug_steps,
                                  max=yaw_aug_steps) * np.deg2rad(rot_aug_resolution)
        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(torch.cat((roll, pitch, yaw), dim=1), "XYZ")
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        # rotate then translate the 4x4 keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
        perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(perturbed_action_gripper_4x4[:, :3, :3])
        perturbed_action_quat_xyzw = torch.cat([perturbed_action_quat_wxyz[:, 1:],
                                                perturbed_action_quat_wxyz[:, 0].unsqueeze(1)],
                                               dim=1).cpu().numpy()

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        trans_indicies, rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            trans_idx = utils.point_to_voxel_index(perturbed_action_trans[b], voxel_size, bounds_np)
            trans_indicies.append(trans_idx.tolist())

            quat = perturbed_action_quat_xyzw[b]
            quat = utils.normalize_quaternion(perturbed_action_quat_xyzw[b])
            if quat[-1] < 0:
                quat = -quat
            disc_rot = utils.quaternion_to_discrete_euler(quat, rot_resolution)
            rot_grip_indicies.append(disc_rot.tolist() + [int(action_rot_grip[b, 3].cpu().numpy())])

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        perturbed_trans = torch.from_numpy(np.array(trans_indicies)).to(device=device)
        perturbed_rot_grip = torch.from_numpy(np.array(rot_grip_indicies)).to(device=device)

    action_trans = perturbed_trans
    action_rot_grip = perturbed_rot_grip

    # apply perturbation to pointclouds
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)
    camera_pose = perturb_se3_camera_pose(camera_pose, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)
    return action_trans, action_rot_grip, pcd, camera_pose


def apply_se3_augmentation_continuous(
        pcd,
        camera_pose,
        action_gripper_pose,
        action_gt,
        bounds,
        layer,
        trans_aug_range,
        rot_aug_range,
        device,
        obs_gripper_pose=None):
    """
    SE3 augmentation for continuous-action agents (e.g. ManiFlow BC).

    Unlike `apply_se3_augmentation_with_camera_pose`, this function:
      - Transforms the continuous 8-DoF action directly in SE3, avoiding
        any voxel-index discretization and the associated retry loop.
      - Returns the augmented action as a float tensor (B, 8)
        [x, y, z, qx, qy, qz, qw, gripper_open].
      - Keeps the point-cloud and camera-pose perturbation identical to
        the original function (needed by the Gaussian-Splatting renderer).
      - Optionally transforms obs_gripper_pose (current obs gripper pose,
        which is passed to encode_scene for gripper_context_head) into the
        same augmented world frame as pcd, so scene tokens and the gripper
        context are expressed in a consistent coordinate system.

    Parameters
    ----------
    pcd               : list of (B, 3, H, W) point clouds per camera
    camera_pose       : list of (B, 4, 4) camera extrinsic matrices
    action_gripper_pose : (B, 7) current gripper pose [xyz, qx,qy,qz,qw]
                          used as the rotation pivot for SE3 perturbation
    action_gt         : (B, 8) continuous ground-truth action
                        [x, y, z, qx, qy, qz, qw, gripper_open]
    bounds            : (B, 6) or (1, 6) workspace bounds [xmin..zmax]
    layer             : voxelization layer index (0 for single-layer PerAct)
    trans_aug_range   : (3,) tensor, fraction of workspace to use as shift range
    rot_aug_range     : list [roll_deg, pitch_deg, yaw_deg] max rotation
                        Sampled continuously (uniform) — no discretization.
    device            : torch device
    obs_gripper_pose  : (B, 7) or None — current observation gripper pose
                        [xyz, qx, qy, qz, qw].  When provided it is transformed
                        by the same SE3 as pcd and returned as the 4th element.
                        This keeps the gripper_context_head's position token
                        consistent with the augmented scene tokens.

    Returns
    -------
    action_gt_aug       : (B, 8) augmented continuous action (same dtype as input)
    pcd_aug             : list of perturbed point clouds
    camera_pose_aug     : list of perturbed camera extrinsics
    obs_gripper_pose_aug: (B, 7) augmented obs gripper pose, or None if not provided
    """
    bs = pcd[0].shape[0]
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # ---- Build 4x4 matrix from current gripper pose (rotation pivot) ----
    action_gripper_trans = action_gripper_pose[:, :3]
    action_gripper_quat_wxyz = torch.cat(
        (action_gripper_pose[:, 6].unsqueeze(1), action_gripper_pose[:, 3:6]), dim=1
    )
    action_gripper_rot   = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz)
    action_gripper_4x4   = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    # ---- Sample random SE3 perturbation ----------------------------------
    trans_range  = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)
    trans_shift  = trans_range * utils.rand_dist((bs, 3)).to(device=device)
    trans_shift_4x4 = identity_4x4.detach().clone()
    trans_shift_4x4[:, 0:3, 3] = trans_shift

    # Continuous uniform sampling — no discretization needed for a continuous-action agent.
    # rot_aug_range gives the max absolute angle in degrees per axis.
    # Tensors are created directly on `device` to avoid a CPU→GPU copy error.
    roll  = ((torch.rand(bs, 1, device=device) * 2 - 1) * np.deg2rad(rot_aug_range[0]))
    pitch = ((torch.rand(bs, 1, device=device) * 2 - 1) * np.deg2rad(rot_aug_range[1]))
    yaw   = ((torch.rand(bs, 1, device=device) * 2 - 1) * np.deg2rad(rot_aug_range[2]))

    rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
        torch.cat((roll, pitch, yaw), dim=1), "XYZ"
    )
    rot_shift_4x4 = identity_4x4.detach().clone()
    rot_shift_4x4[:, :3, :3] = rot_shift_3x3

    # ---- Apply the same SE3 perturbation to the TARGET action (action_gt) --------
    # The scene perturbation in perturb_se3 applies this to every world-frame point p:
    #
    #   p' = bmm(p^T, rot_shift_4x4)^T + clamp(t_pivot + t_shift)
    #      = rot_shift_4x4^T * (p - t_pivot) + clamp(t_pivot + t_shift)
    #      = R_delta^T * (p - t_pivot) + clamp(t_pivot + t_shift)
    #
    # where R_delta = rot_shift_4x4[:, :3, :3]  (pytorch3d euler_angles_to_matrix output,
    # a standard column-vector rotation matrix), and t_pivot = action_gripper xyz.
    #
    # The same transformation must be applied to action_gt:
    #   aug_pos = R_delta^T * (t_action_gt - t_pivot) + clamp(t_pivot + t_shift)
    #   aug_rot = R_delta^T * R_action_gt    (left-multiply = same world-frame rotation)

    # Build action_gt rotation matrix from quaternion (xyzw → wxyz for pytorch3d)
    action_gt_quat_xyzw = action_gt[:, 3:7]                                   # (B, 4) xyzw
    action_gt_quat_wxyz = torch.cat(
        [action_gt_quat_xyzw[:, 3:4], action_gt_quat_xyzw[:, :3]], dim=1
    )                                                                           # (B, 4) wxyz
    action_gt_rot = torch3d_tf.quaternion_to_matrix(action_gt_quat_wxyz)       # (B, 3, 3)

    # R_delta^T — matches the row-vector bmm transpose used in perturb_se3.
    # Use rot_shift_3x3 directly (already computed above); do NOT re-read from rot_shift_4x4
    # to avoid accidental aliasing if rot_shift_4x4 were ever modified.
    rot_shift_3x3_T = rot_shift_3x3.transpose(1, 2)                           # (B, 3, 3) = R_delta^T

    # Rotation: aug_R = R_delta^T * R_action_gt
    aug_rot = torch.bmm(rot_shift_3x3_T, action_gt_rot)                        # (B, 3, 3)

    # Position: R_delta^T * (t_action_gt - t_pivot) + clamp(t_pivot + t_shift)
    t_pivot  = action_gripper_4x4[:, 0:3, 3]                                  # (B, 3)
    t_action = action_gt[:, :3]                                                # (B, 3)
    offset   = t_action - t_pivot                                              # (B, 3)
    aug_offset = torch.einsum('bij,bj->bi', rot_shift_3x3_T, offset)          # (B, 3)

    # Clamp the new pivot (t_pivot + t_shift) exactly as perturb_se3 does:
    # it uses global min/max across the batch, so we must match that here.
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)
    bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
    bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
    bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()
    new_pivot = t_pivot + trans_shift                                          # (B, 3)
    new_pivot = torch.stack([
        new_pivot[:, 0].clamp(bounds_x_min, bounds_x_max),
        new_pivot[:, 1].clamp(bounds_y_min, bounds_y_max),
        new_pivot[:, 2].clamp(bounds_z_min, bounds_z_max),
    ], dim=1)
    aug_pos = aug_offset + new_pivot                                           # (B, 3)

    # Convert aug_rot back to quaternion: pytorch3d returns wxyz, action layout needs xyzw
    aug_quat_wxyz = torch3d_tf.matrix_to_quaternion(aug_rot)                   # (B, 4) wxyz
    # Re-normalise to guard against numerical drift in matrix_to_quaternion
    aug_quat_wxyz = aug_quat_wxyz / (aug_quat_wxyz.norm(dim=-1, keepdim=True) + 1e-8)
    aug_quat_xyzw = torch.cat(
        [aug_quat_wxyz[:, 1:], aug_quat_wxyz[:, 0:1]], dim=1
    )                                                                           # (B, 4) xyzw

    # Gripper open/close is scene-invariant — keep unchanged
    aug_grip = action_gt[:, 7:8]

    action_gt_aug = torch.cat([aug_pos, aug_quat_xyzw, aug_grip], dim=-1)

    # ---- Apply same perturbation to point clouds and camera poses --------
    pcd_aug         = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)
    camera_pose_aug = perturb_se3_camera_pose(camera_pose, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

    # ---- Optionally transform obs_gripper_pose into the augmented frame ----
    # After SE3 augmentation the pcd (and thus the scene tokens produced by
    # encode_scene) are in the *augmented* world frame.  obs_gripper_pose is
    # passed to encode_scene → gripper_context_head, which uses its xyz as a
    # world-space 3D position.  If we left obs_gripper_pose in the original
    # frame, the gripper position token would be inconsistent with the scene
    # tokens — the same mismatch that motivated fixing action_gt above.
    #
    # The transformation is identical to what we did for action_gt:
    #   aug_pos = R_delta^T * (t_obs_grip - t_pivot) + clamp(t_pivot + t_shift)
    #   aug_rot = R_delta^T * R_obs_grip
    obs_gripper_pose_aug = None
    if obs_gripper_pose is not None:
        obs_grip_quat_xyzw = obs_gripper_pose[:, 3:7]                         # (B, 4) xyzw
        obs_grip_quat_wxyz = torch.cat(
            [obs_grip_quat_xyzw[:, 3:4], obs_grip_quat_xyzw[:, :3]], dim=1
        )                                                                       # (B, 4) wxyz
        obs_grip_rot = torch3d_tf.quaternion_to_matrix(obs_grip_quat_wxyz)     # (B, 3, 3)

        t_obs_grip = obs_gripper_pose[:, :3]                                   # (B, 3)
        obs_offset = t_obs_grip - t_pivot                                      # (B, 3)
        aug_obs_offset = torch.einsum('bij,bj->bi', rot_shift_3x3_T, obs_offset)  # (B, 3)
        aug_obs_pos = aug_obs_offset + new_pivot                               # (B, 3)

        aug_obs_rot = torch.bmm(rot_shift_3x3_T, obs_grip_rot)                 # (B, 3, 3)
        aug_obs_quat_wxyz = torch3d_tf.matrix_to_quaternion(aug_obs_rot)       # (B, 4) wxyz
        # Re-normalise to guard against numerical drift
        aug_obs_quat_wxyz = aug_obs_quat_wxyz / (
            aug_obs_quat_wxyz.norm(dim=-1, keepdim=True) + 1e-8
        )
        aug_obs_quat_xyzw = torch.cat(
            [aug_obs_quat_wxyz[:, 1:], aug_obs_quat_wxyz[:, 0:1]], dim=1
        )                                                                       # (B, 4) xyzw

        obs_gripper_pose_aug = torch.cat([aug_obs_pos, aug_obs_quat_xyzw], dim=-1)  # (B, 7)

    return action_gt_aug, pcd_aug, camera_pose_aug, obs_gripper_pose_aug


### ref: https://github.com/vlc-robot/polarnet
# NOT USED
def random_rotate_pcd_and_action(pcd, action, rot_range, rot=None):
    '''
    pcd: (B, 3, npoints)
    action: (B, 8)
    shift_range: float
    '''

    if rot is None:
        rot = np.random.uniform(-rot_range, rot_range)
    r = R.from_euler('z', rot, degrees=True)

    pos_ori = einops.rearrange(pcd, 'b c n -> (b n) c')
    pos_new = r.apply(pos_ori)
    pcd = einops.rearrange(pos_new, '(b n) c -> b c n', b=pcd.shape[0], n=pcd.shape[2])
    action[..., :3] = r.apply(action[..., :3])
    
    a_ori = R.from_quat(action[..., 3:7])
    a_new = r * a_ori
    action[..., 3:7] = a_new.as_quat()

    return pcd, action

def random_shift_pcd_and_action(pcd, action, shift_range, shift=None):
    '''
    pcd: (B, 3, npoints)
    action: (B, 8)
    shift_range: float
    '''
    if shift is None:
        shift = np.random.uniform(-shift_range, shift_range, size=(3, ))

    pcd = pcd + shift[None, :, None]
    action[..., :3] += shift[None, :]

    return pcd, action

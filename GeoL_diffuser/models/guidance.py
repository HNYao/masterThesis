import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import open3d as o3d


class Guidance:
    def __init__(self):
        pass

    def scale_xyR_pose(self, pose_xyR, xyR_min_bound, xyR_max_bound):
        """
        scale the pose_xyR to [-1, 1]
        pose_xyR: B * H * 3
        """
        if len(pose_xyR.shape) == 3:
            min_bound_batch = xyR_min_bound.unsqueeze(1)
            max_bound_batch = xyR_max_bound.unsqueeze(1)
        elif len(pose_xyR.shape) == 4:
            min_bound_batch = xyR_min_bound.unsqueeze(1).unsqueeze(1)
            max_bound_batch = xyR_max_bound.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError("Invalid shape of the input pose_xyR")

        scale = max_bound_batch - min_bound_batch
        pose_xyR = (pose_xyR - min_bound_batch) / (scale + 1e-6)
        pose_xyR = 2 * pose_xyR - 1
        pose_xyR = pose_xyR.clamp(-1, 1)

        return pose_xyR

    def descale_xyR_pose(self, pose_xyR, xyR_min_bound, xyR_max_bound):
        """
        descale the pose_4d to the original range
        pose_xyR: B * N * H * 3
        """
        if len(pose_xyR.shape) == 3:
            min_bound_batch = xyR_min_bound.unsqueeze(1)
            max_bound_batch = xyR_max_bound.unsqueeze(1)
        elif len(pose_xyR.shape) == 4:
            min_bound_batch = xyR_min_bound.unsqueeze(1).unsqueeze(1)
            max_bound_batch = xyR_max_bound.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError("Invalid shape of the input pose_xyR")

        scale = max_bound_batch - min_bound_batch
        pose_xyR = (pose_xyR + 1) / 2
        pose_xyR = pose_xyR * scale + min_bound_batch
        return pose_xyR

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the trajectory to use to compute losses and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        return loss_tot, guide_losses


class OneGoalGuidance(Guidance):
    def __init__(self):
        super(OneGoalGuidance, self).__init__()

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the trajectory to use to compute losses
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        bsize, num_samp, num_hypo, _ = x.size()
        # goal_pose_xyR = data_batch["goal_pose_xyR"]  # [B, 3]
        # goal_pose_xyR = goal_pose_xyR[:, None]  # [B, 1, 3]
        # gt_pose_xyR_min_bound = data_batch["gt_pose_xyR_min_bound"]
        # gt_pose_xyR_max_bound = data_batch["gt_pose_xyR_max_bound"]
        # x_goal = self.scale_xyR_pose(
        #     goal_pose_xyR, gt_pose_xyR_min_bound, gt_pose_xyR_max_bound
        # )  # [B, 1, 3]
        # x_goal = x_goal[:, None].expand(-1, num_samp, num_hypo, -1)  # [B, N, H, 3]
        # Debug only ...
        x_goal = torch.Tensor([0.0317, -0.0118, -0.0099])[None, None, None].cuda()
        x_goal = x_goal.expand(bsize, num_samp, num_hypo, -1)
        goal_loss = F.mse_loss(
            x_goal[..., :2], x[..., :2], reduction="none"
        )  # [B, N, H, 3]
        goal_loss = goal_loss.mean(dim=-1)  # [B, N, H]
        guide_losses["goal_loss"] = goal_loss

        # if t[0] == 0:
        #     import pdb

        #     pdb.set_trace()

        # print()
        goal_loss = goal_loss.mean(dim=-1)  # [B, N]
        print("goal_loss: ", goal_loss[0, 0], x_goal[0, 0, 0, :2], x[0, 0, 0, :2])
        goal_loss = goal_loss.mean() * 2000
        loss_tot += goal_loss
        return loss_tot, guide_losses

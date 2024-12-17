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

class DiffuserGuidance:
    def __init__(self, **kwargs):
        self.goal_weight = kwargs.get("goal_weight", 0)
        self.affordance_weight = kwargs.get("affordance_weight", 0)
    
        if self.goal_weight != 0:
            self.goal_guidance = OneGoalGuidance()
        if self.affordance_weight != 0:
            self.affordance_guidance = AffordanceGuidance()
    
    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the trajectory to use to compute losses
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        if self.goal_weight != 0:
            goal_loss, goal_guide_losses = self.goal_guidance.compute_guidance_loss(x, t, data_batch)
            guide_losses.update(goal_guide_losses)
            loss_tot += goal_loss * self.goal_weight
        
        if self.affordance_weight != 0:
            affordance_loss, affordance_guide_losses = self.affordance_guidance.compute_guidance_loss(x, t, data_batch)
            guide_losses.update(affordance_guide_losses)
            loss_tot += affordance_loss * self.affordance_weight
        
        loss_per_traj = 0 # NOTE: not trajectory actually, more like points sampling
        for k, v in guide_losses.items():
            loss_per_traj += v
        guide_losses["loss_per_traj"] = loss_per_traj
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

        # FIXME: hard-coded goal, need to think about proper way to pass goal and scale it
        x_goal = torch.Tensor([0.6477, 0.2714, 1.040])[None, None, None].cuda()
        #x_goal = torch.Tensor([100, 100, 100])[None, None, None].cuda()
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
        goal_loss = goal_loss.mean() * 500
        loss_tot += goal_loss
        return loss_tot, guide_losses

class AffordanceGuidance(Guidance):
    def __init__(self):
        super(AffordanceGuidance, self).__init__()

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evalueates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the trajectory to use to compute losses
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """

        guide_losses = dict()
        loss_tot = 0.0

        bsize, num_samp, num_hypo, _ = x.size() 
        affordance = data_batch["affordance"] # [B, 2048, 1]
        #position = self.scale_xyz_pose(data_batch["pc_position"], data_batch["gt_pose_xyz_max_bound"], data_batch["gt_pose_xyz_min_bound"]) # [B, 2048, 3]
        position = data_batch["pc_position"] # [B, 2048, 3]
        # TODO: find the top k affordance values and the corresponding positions
        affordance = affordance.squeeze(-1) # [B, 2048]
        k = 10 # top k affordance values
        topk_affordance, topk_idx = torch.topk(affordance, k, dim=-1)
        topk_positions = torch.gather(position, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, position.size(-1)))
        avg_topk_positions = topk_positions.mean(dim=1)  # (B, 3)
        avg_topk_positions = avg_topk_positions[:, None, None].expand(-1, num_samp, num_hypo, -1)  # (B, N, H, 3)
        affordance_loss = F.mse_loss(
            avg_topk_positions[..., :2], x[..., :2], reduction="none"
        ) # (B, N, H, 3)
        affordance_loss = affordance_loss.mean(dim=-1) # (B, N, H)
        guide_losses["affordance_loss"] = affordance_loss
        affordance_loss = affordance_loss.mean(dim=-1)  # (B, N)
        affordance_loss = affordance_loss.mean() * 500 # (B,)
        loss_tot += affordance_loss

        return loss_tot, guide_losses
    

    def scale_xyz_pose(self, pose_xyz, xyz_min_bound, xyz_max_bound):
        """
        scale the pose_xyz to [-1, 1]
        pose_xyR: B * H * 3
        """
        if len(pose_xyz.shape) == 3:
            min_bound_batch = xyz_min_bound.unsqueeze(1)
            max_bound_batch = xyz_max_bound.unsqueeze(1)
        elif len(pose_xyz.shape) == 4:
            min_bound_batch = xyz_min_bound.unsqueeze(1).unsqueeze(1)
            max_bound_batch = xyz_max_bound.unsqueeze(1).unsqueeze(1)
        elif len(pose_xyz.shape) == 2:
            min_bound_batch = xyz_min_bound
            max_bound_batch = xyz_max_bound

        scale = max_bound_batch - min_bound_batch
        pose_xyz = (pose_xyz - min_bound_batch) / (scale + 1e-6)
        pose_xyz = 2 * pose_xyz - 1
        pose_xyz = pose_xyz.clamp(-1, 1)

        return pose_xyz

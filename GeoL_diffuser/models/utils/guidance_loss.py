import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class Guidance:
    def __init__(self, scale=1.0):
        self.scale = scale
        pass

    def scale_pose(self, pose_4d, min_bound, max_bound):
        """
        scale the pose_4d to [-1, 1]
        pose_4d: B * H * 4
        """
        if len(pose_4d.shape) == 3:
            min_bound_batch = min_bound.unsqueeze(1)
            max_bound_batch = max_bound.unsqueeze(1)
        elif len(pose_4d.shape) == 4:
            min_bound_batch = min_bound.unsqueeze(1).unsqueeze(1)
            max_bound_batch = max_bound.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError("Invalid shape of the input pose_4d")

        scale = max_bound_batch - min_bound_batch
        pose_4d = (pose_4d - min_bound_batch) / (scale+1e-6)
        pose_4d = 2 * pose_4d - 1
        pose_4d = pose_4d.clamp(-1, 1)

        return pose_4d
    
    def descale_pose(self, pose_4d, min_bound, max_bound):
        """
        descale the pose_4d to the original range
        pose_4d: B * N * H * 4
        """
        if len(pose_4d.shape)==3:
            min_bound_batch = min_bound.unsqueeze(1)
            max_bound_batch = max_bound.unsqueeze(1)
        elif len(pose_4d.shape)==4:
            min_bound_batch = min_bound.unsqueeze(1).unsqueeze(1)
            max_bound_batch = max_bound.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError("Invalid shape of the input pose_4d")
        
        scale = max_bound_batch - min_bound_batch
        pose_4d = (pose_4d + 1) / 2
        pose_4d = pose_4d * scale + min_bound_batch
        return pose_4d

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
    def __init__(
        self,
        non_collide_weight=0,
        affordance_weight=0, # the final position should have high affordance score

    ):
        self.non_collide_weight = non_collide_weight
        self.affordance_weight = affordance_weight
        
        self.noncollide_guidance = NonCollideGuidance() # TODO: implement NonCollideGuidance
        self.affordance_guidance = AffordanceGuidance() # TODO: implement AffordanceGuidance
    
    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the pose to use to compute losses and
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        if self.non_collide_weight > 0:
            goal_loss, losses_dict = self.noncollide_guidance.compute_guidance_loss(x, t, data_batch)
            loss_tot += goal_loss * self.non_collide_weight
            guide_losses.update(losses_dict)
        
        if self.affordance_weight > 0:
            affordance_loss, losses_dict = self.affordance_guidance.compute_guidance_loss(x, t, data_batch)
            loss_tot += affordance_loss * self.affordance_weight
            guide_losses.update(losses_dict)
        
        loss_per_pose = 0
        for k, v in guide_losses.items():
            loss_per_pose += v
        
        guide_losses["total_loss"] = loss_per_pose

        return loss_tot, guide_losses

class NonCollideGuidance(Guidance):
    def __init__(self, scale=1.0):
        super().__init__(scale)
        # NOTEL: scale may be innecessary
    
    def compute_guidance_loss(self, x, t, data_batch, num_noncollide=60):
        '''
        Evalute all guidance losses, ttoal and individual values.
        - x: (B, N, H, 3) diffusion state
        - data_batch: various tensors of size (B, ...) that may be needed for loss calculations
        '''
        guide_losses = dict()
        loss_tot = 0.0
        batch_size, num_samp, horizon, _ = x.size()

        voxel_bounds = data_batch["voxel_bounds"] # TODO: add voxel bounds to data_batch
        object_points = data_batch["object_points"] # TODO: add object points to data_batch
        num_pcdobj = object_points.size(1)

        gt_pose_4d_min_bound = data_batch["gt_pose_4d_min_bound"]
        gt_pose_4d_max_bound = data_batch["gt_pose_4d_max_bound"]
        tsdf = data_batch["tsdf_grid_fine"][:, None] # B * 1 * D * H * W # TODO: add tsdf_grid_fine to data_batch

        # Acquire the tsdf of the trajectory points
        waypoints = self.descale_trajectory(
            x, gt_pose_4d_min_bound, gt_pose_4d_max_bound
        )  # [B, N, H, 3]

        ### with obj points
        waypoints = waypoints.unsqueeze(2).expand(
            -1, -1, object_points.size(1), -1, -1
        )[
            ..., -num_noncollide:, :
        ]  # [B, N, O, H, 3]
        waypoints_init = waypoints[:, :, :, :1]  # [B, N, O, 1, 3]
        travel_dist = waypoints - waypoints_init  # [B, N, O, H, 3]

        object_points = (
            object_points.unsqueeze(1)
            .unsqueeze(3)
            .expand(-1, num_samp, -1, num_noncollide, -1)
        )  # [B, N, O, H, 3]
        query_points = object_points + travel_dist  # [B, N, O, H, 3]
        query_points = query_points.view(batch_size, num_samp, -1, 3)  # [B, N, O*H, 3]

        voxel_bounds = voxel_bounds.unsqueeze(-1).repeat(1, 1, 3)  # [B, 2, 3]
        voxel_bounds_min = voxel_bounds[:, 0][:, None, None].repeat(
            1, num_samp, num_noncollide * num_pcdobj, 1
        )  # [B, N, O*H, 3]
        voxel_bounds_max = voxel_bounds[:, 1][:, None, None].repeat(
            1, num_samp, num_noncollide * num_pcdobj, 1
        )  # [B, N, O*H, 3]
        query_points = (query_points - voxel_bounds_min) / (
            voxel_bounds_max - voxel_bounds_min
        )  # [B, N, O*H, 3]
        query_grids = query_points * 2 - 1  # [B, N, H, 3]

        query_grids = query_grids[..., [2, 1, 0]]  # [B, N, H, 3]
        query_grids = query_grids.unsqueeze(-2)  # [B, N, H, 1, 3]

        query_tsdf = F.grid_sample(
            tsdf, query_grids, align_corners=True
        )  # [B, 1, N, H, 1]
        query_tsdf = query_tsdf.squeeze(-1).squeeze(1)  # [B, N, H]
        map_loss = F.relu(-(query_tsdf - 0.1)).mean(dim=-1)  # [B, N]

        guide_losses = {"collision_loss": map_loss}
        map_loss = map_loss.mean()
        loss_tot += map_loss
        return loss_tot, guide_losses
    

class AffordanceGuidance(Guidance):
    def __init__(self, scale):
        super().__init__(scale)
        # NOTEL: scale may be innecessary
    
    def compute_guidance_loss(self, x, t, data_batch):
        '''
        Evaluate the closeness of the final position to the object affordance
        - x: (B, N, H, 3) diffusion state
        - data_batch: various tensors of size (B, ...) that may be needed for loss calculations
        '''
        guide_losses = dict()
        loss_tot = 0.0
        batch_size, num_samp, horizon, _ = x.size()

        pass


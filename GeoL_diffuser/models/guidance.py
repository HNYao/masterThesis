import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import open3d as o3d
from matplotlib import cm


class Guidance:
    def __init__(self):
        pass

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

    def descale_xyz_pose(self, pose_xyz, xyz_min_bound, xyz_max_bound):
        """
        descale the pose_xyz to the original range
        pose_xyz: B * N * H * 3
        """
        if len(pose_xyz.shape) == 3:
            min_bound_batch = xyz_min_bound.unsqueeze(1)
            max_bound_batch = xyz_max_bound.unsqueeze(1)
        elif len(pose_xyz.shape) == 4:
            min_bound_batch = xyz_min_bound.unsqueeze(1).unsqueeze(1)
            max_bound_batch = xyz_max_bound.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError("Invalid shape of the input pose_xyz")

        scale = max_bound_batch - min_bound_batch
        pose_xyz = (pose_xyz + 1) / 2
        pose_xyz = pose_xyz * scale + min_bound_batch
        return pose_xyz

    def get_heatmap(self, values, cmap_name="turbo", invert=False):
        if invert:
            values = -values
        values = (values - values.min()) / (values.max() - values.min())
        colormaps = cm.get_cmap(cmap_name)
        rgb = colormaps(values)[..., :3]  # don't need alpha channel
        return rgb

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

        # find the top k affordance
        affordance = data_batch["affordance"] # [B, 2048, 1]
        position = data_batch["pc_position"] # [B, 2048, 3]
        affordance = affordance.squeeze(-1) # [B, 2048]
        k = 10 # top k affordance values
        topk_affordance, topk_idx = torch.topk(affordance, k, dim=-1)
        topk_positions = torch.gather(position, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, position.size(-1)))
        avg_topk_positions = topk_positions.mean(dim=1)  # (B, 3) unscaled
        avg_topk_positions_debug = avg_topk_positions.clone() # (B, 3) unscaled
        avg_topk_positions = avg_topk_positions[:, None, None].expand(-1, num_samp, num_hypo, -1)  # (B, N, H, 3)

        # scale the topk affordance
        avg_topk_positions = self.scale_xyz_pose(avg_topk_positions, data_batch["gt_pose_xyz_min_bound"], data_batch["gt_pose_xyz_max_bound"]) # (B, N, H, 3)

        affordance_loss = F.mse_loss(
            avg_topk_positions[..., :2], x[..., :2], reduction="none"
        ) # (B, N, H, 3) scaled avg_topk_positions
        affordance_loss = affordance_loss.mean(dim=-1) # (B, N, H)
        affordance_loss_debug = affordance_loss.clone()
        guide_losses["affordance_loss"] = affordance_loss # (B, N, H)
        affordance_loss = affordance_loss.mean(dim=-1)  # (B, N)
        
        affordance_loss = affordance_loss.mean() * 500 # (B,)
        loss_tot += affordance_loss

        ####### DEBUG visualize avg_topk_positions
        #avg_topk_positions_debug = self.scale_xyz_pose(avg_topk_positions_debug, data_batch["gt_pose_xyz_max_bound"], data_batch["gt_pose_xyz_min_bound"]) # (B, 3)
        avg_topk_positions_debug = avg_topk_positions_debug[0].cpu().detach().numpy() # (3,) unsacled
        position_debug = position[0].cpu().detach().numpy() # (2048, 3)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(position_debug)
        pc.paint_uniform_color([0.5, 0.5, 0.5])
        avg_topk_sphere = self.draw_sphere_at_point(avg_topk_positions_debug)
        descaled_x = self.descale_xyz_pose(x, data_batch["gt_pose_xyz_min_bound"], data_batch["gt_pose_xyz_max_bound"])
        pred_points = descaled_x[0].view(-1, 3).cpu().detach().numpy()
        distances = np.sqrt(
            ((position_debug[:, :2][:, None, :] - pred_points[:, :2]) ** 2).sum(axis=2)
        ) # x, y distance 

        scenepts_to_anchor_dist = np.min(distances, axis=1)  # [num_points]
        scenepts_to_anchor_id = np.argmin(distances, axis=1)  # [num_points]
        topk_points_id = np.argsort(scenepts_to_anchor_dist, axis=0)[: pred_points.shape[0]]
        tokk_points_id_corr_anchor = scenepts_to_anchor_id[topk_points_id]

        guide_cost = affordance_loss_debug[0].flatten().cpu().detach().numpy() # [N*H]
        guide_cost = guide_cost[tokk_points_id_corr_anchor]
        guide_cost_color =  self.get_heatmap(guide_cost[None], invert=False)[0]

        points_for_place= position_debug[topk_points_id]
        vis = [pc, avg_topk_sphere]

        for ii, pos in enumerate(points_for_place):
            pos_vis = o3d.geometry.TriangleMesh.create_sphere()
            pos_vis.compute_vertex_normals()
            pos_vis.scale(0.03, center=(0, 0, 0))
            pos_vis.translate(pos[:3])
            vis_color = guide_cost_color[ii]
            pos_vis.paint_uniform_color(vis_color)
            vis.append(pos_vis)


        #o3d.visualization.draw_geometries(vis)
        #o3d.io.write_point_cloud("outputs/debug/AffordanceGuide_pc.ply", vis)

        ##############################################

        return loss_tot, guide_losses
    
    def draw_sphere_at_point(self, center, radius=0.06, color=[0.1, 0.1, 0.7]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        sphere.translate(center)
        return sphere



import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import open3d as o3d
from matplotlib import cm
from GeoL_diffuser.models.helpers import TSDFVolume, get_view_frustum
from GeoL_diffuser.models.utils.fit_plane import *
from sklearn.cluster import DBSCAN


class Guidance:
    def __init__(self):
        pass

    def scale_xy_pose(self, pose_xy, xy_min_bound, xy_max_bound):
        """
        scale the pose_xyz to [-1, 1]
        pose_xyR: B * H * 3
        """
        if len(pose_xy.shape) == 3:
            min_bound_batch = xy_min_bound.unsqueeze(1)
            max_bound_batch = xy_max_bound.unsqueeze(1)
        elif len(pose_xy.shape) == 4:
            min_bound_batch = xy_min_bound.unsqueeze(1).unsqueeze(1)
            max_bound_batch = xy_max_bound.unsqueeze(1).unsqueeze(1)
        elif len(pose_xy.shape) == 2:
            min_bound_batch = xy_min_bound
            max_bound_batch = xy_max_bound

        scale = max_bound_batch - min_bound_batch
        pose_xy = (pose_xy - min_bound_batch) / (scale + 1e-6)
        pose_xy = 2 * pose_xy - 1
        pose_xy = pose_xy.clamp(-1, 1)

        return pose_xy

    def descale_xy_pose(self, pose_xy, xy_min_bound, xy_max_bound):
        """
        descale the pose_xyz to the original range
        pose_xyz: B * N * H * 3
        """
        if len(pose_xy.shape) == 3:
            min_bound_batch = xy_min_bound.unsqueeze(1)
            max_bound_batch = xy_max_bound.unsqueeze(1)
        elif len(pose_xy.shape) == 4:
            min_bound_batch = xy_min_bound.unsqueeze(1).unsqueeze(1)
            max_bound_batch = xy_max_bound.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError("Invalid shape of the input pose_xy")

        scale = max_bound_batch - min_bound_batch
        pose_xy = (pose_xy + 1) / 2
        pose_xy = pose_xy * scale + min_bound_batch
        return pose_xy


    def scale_xyR_pose(self, pose_xyR, xyR_min_bound, xyR_max_bound):
        """
        scale the pose_xy to [-1, 1]
        pose_xy: B * H * 3
        """
        if len(pose_xyR.shape) == 3:
            min_bound_batch = xyR_min_bound.unsqueeze(1)
            max_bound_batch = xyR_max_bound.unsqueeze(1)
        elif len(pose_xyR.shape) == 4:
            min_bound_batch = xyR_min_bound.unsqueeze(1).unsqueeze(1)
            max_bound_batch = xyR_max_bound.unsqueeze(1).unsqueeze(1)
        elif len(pose_xyR.shape) == 2:
            min_bound_batch = xyR_min_bound
            max_bound_batch = xyR_max_bound

        scale = max_bound_batch - min_bound_batch
        pose_xyR = (pose_xyR - min_bound_batch) / (scale + 1e-6)
        pose_xyR = 2 * pose_xyR - 1
        pose_xyR = pose_xyR.clamp(-1, 1)

        return pose_xyR


    def descale_xyR_pose(self, pose_xyR, xyR_min_bound, xyR_max_bound):
        """
        descale the pose_xy to the original range
        pose_xy: B * N * H * 3
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

    def xyR_to_T(self, xyR, data_batch):
        # Instead of finding nearest point to acquire translation, we fit a plane model for acceleration
        assert "plane_model" in data_batch
        assert "T_plane" in data_batch
        
        B, N, H, _ = xyR.shape
        xyR_descale = self.descale_xyR_pose(xyR, 
            data_batch["gt_pose_xyR_min_bound"], 
            data_batch["gt_pose_xyR_max_bound"]) # (B, N, H, 3)
        T_plane = data_batch['T_plane'] # (B, 4, 4)
        T_plane = T_plane[:, None, None]
        T = torch.eye(4, dtype=xyR.dtype, device=xyR.device)
        T = T.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, N, H, 1, 1).clone()  # (B, N, H, 4, 4)
        T[..., 0, 0] = torch.cos(xyR_descale[..., 2])  # cosθ
        T[..., 0, 1] = -torch.sin(xyR_descale[..., 2]) # -sinθ
        T[..., 1, 0] = torch.sin(xyR_descale[..., 2])  # sinθ
        T[..., 1, 1] = torch.cos(xyR_descale[..., 2])  # cosθ
        T = T_plane @ T # (B, N, H, 4, 4)

        # Compute the rotation
        plane_model = data_batch['plane_model'] 
        a = plane_model[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        b = plane_model[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        c = plane_model[:, 2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        d = plane_model[:, 3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Compute the translation
        x = xyR_descale[..., 0:1] # (B, N, H, 1)
        y = xyR_descale[..., 1:2] # (B, N, H, 1)
        z = -(a * x + b * y + d) / (c + 1e-8) # (B, N, H, 1)
        t = torch.cat([x, y, z], dim=-1) # ( 
        T[..., :3, 3] = t
        return T
        
 
 
class NonCollisionGuidance_v3(Guidance):
    def __init__(self):
        super(NonCollisionGuidance_v3, self).__init__()

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evalueates all guidance losses and total and individual values.
        - x: (B, N, H, 2) the sampled points to use to compute losses
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0
        voxel_bounds = data_batch["voxel_bounds"]  # [B, 2]
        scene_pc = data_batch["pc_position"] # [B, 2048, 3]
        
        B, N, H = x.size()[:3]
        obj_pc = data_batch['object_pc_position'] # (B, 512, 3) # aligned to z
        N_obj = obj_pc.size(1)
        T = self.xyR_to_T(x, data_batch) # (B, N, H, 4, 4)
        obj_pc = obj_pc.unsqueeze(1).unsqueeze(2).expand(-1, N, H, -1, -1) # (B, N, H, O, 3)
        obj_pc_placed = torch.matmul(obj_pc, T[..., :3, :3].transpose(-1, -2)) + T[..., :3, 3].unsqueeze(-2) # (B, N, H, O, 3)

        ### query tsdf version: [B, N, O*H, 3]
        query_points = obj_pc_placed.view(B, N, -1, 3) # [B, N, O*H, 3]
        voxel_bounds = voxel_bounds.unsqueeze(-1).repeat(1, 1, 3)  # [B, 2, 3]
        voxel_bounds_min = voxel_bounds[:, 0][:, None, None] # [B, 1, 1, 3]
        voxel_bounds_max = voxel_bounds[:, 1][:, None, None] # [B, 1, 1, 3]
        query_points = (query_points - voxel_bounds_min) / (
            voxel_bounds_max - voxel_bounds_min
        )  # [B, N, O*H, 3]
        
        # ### Visualization
        # if t % 10 == 0:
        #     vis = [data_batch["scene_mesh"]]
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(
        #         (query_points[0, 0].cpu().detach().numpy() * 2 - 1)
        #     )
        #     vis.append(pcd)
        #     o3d.visualization.draw_geometries(vis)
        # ### Visualization

        query_grids = query_points * 2 - 1  # [B, N, O*H, 3]
        query_grids = query_grids[..., [2, 1, 0]]  # [B, N, O*H, 3]
        query_grids = query_grids.unsqueeze(-2)  # [B, N, O*H, 1, 3]
        #tsdf = tsdf._tsdf_vol
        #tsdf = torch.tensor(tsdf, dtype=torch.float32).cuda()[None, None]# [B, C, D, H, W]
        tsdf = data_batch["tsdf_vol"][:, None] # [B, 1, D, H, W]  

        query_tsdf = nn.functional.grid_sample(tsdf, query_grids, align_corners=True) # [B, 1, N, O*H, 1]
        query_tsdf = query_tsdf.squeeze(-1).squeeze(1) # [B, N, O*H]
        query_tsdf = query_tsdf.view(B, N, H, N_obj) # [B, N, O*H]
        collision_loss = F.relu(0.1 - query_tsdf).mean(dim=-1) # (B, N, H)

        guide_losses["loss"] = collision_loss 
        
        collision_loss = collision_loss.mean()  
        loss_tot += collision_loss

        return loss_tot, guide_losses

 
class AffordanceGuidance_v3(Guidance):
    """
    convert to pose and calulate the loss

    """
    def __init__(self):
        super(AffordanceGuidance_v3, self).__init__()
        self.topk = 50

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evalueates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the xyR state to use to compute losses
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0
        B, N, H = x.size()[:3]
        obj_pc = data_batch['object_pc_position'] # (B, 512, 3) # aligned to z
        T = self.xyR_to_T(x, data_batch) # (B, N, H, 4, 4)
        obj_pc = obj_pc.unsqueeze(1).unsqueeze(2).expand(-1, N, H, -1, -1) # (B, N, H, O, 3)
        obj_pc_placed = torch.matmul(obj_pc, T[..., :3, :3].transpose(-1, -2)) + T[..., :3, 3].unsqueeze(-2) # (B, N, H, O, 3)

        # find the top k affordance
        if "affordance_fine" in data_batch:
            affordance_ori = data_batch["affordance_fine"] 
        else:
            #print("No affordance_fine in data_batch, using affordance")
            affordance_ori = data_batch["affordance"] # [B, 2048, num_affordance]
        
        # sample topk points according to the affordance
        position = data_batch["pc_position"] # [B, 2048, 3]
        sampled_position = self.topk_sampling(position, affordance_ori, sample_k=self.topk) # [B, K, 3]

        # # visualize the sampled points
        # if t % 10 == 0:
        #     pcd_obj = o3d.geometry.PointCloud()
        #     pcd_obj.points = o3d.utility.Vector3dVector(obj_pc_placed[0, 0, 0, :].cpu().detach().numpy())
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(position[0].cpu().detach().numpy())
        #     pcd.colors = o3d.utility.Vector3dVector(self.get_heatmap(affordance_ori[0].cpu().detach().numpy().squeeze(), invert=False))
        #     vis = [pcd, pcd_obj]
        #     for ii in range(H):
        #         pose_ii = T[0, 0, ii, :, :].cpu().detach().numpy()
        #         axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        #         axis.transform(pose_ii)
        #         vis.append(axis)

        #     for ii in range(sampled_position.size(1)):
        #         pos_vis = o3d.geometry.TriangleMesh.create_sphere()
        #         pos_vis.compute_vertex_normals()
        #         pos_vis.scale(0.03, center=(0, 0, 0))
        #         pos_vis.translate(sampled_position[0, ii, :3].detach().cpu().numpy())
        #         vis_color = [0.5, 0.5, 0.5]
        #         pos_vis.paint_uniform_color(vis_color)
        #         vis.append(pos_vis)
        #     o3d.visualization.draw_geometries(vis)

        expanded_target_points = sampled_position[:, None, :, None, None] # (B, 1, K, 1, 1, 3)
        expanded_predicted_points = obj_pc_placed[:, :, None] # (B, N, 1, H, O, 3)
        affordance_loss = F.mse_loss(expanded_target_points, expanded_predicted_points, reduction="none") # (B, N, K, H, O, 3)
        affordance_loss = affordance_loss.min(dim=2)[0] # (B, N, H, O, 3)
        affordance_loss = affordance_loss.mean(dim=-1).mean(dim=-1)
        guide_losses["distance_error"] = affordance_loss # (B, N, H)
        guide_losses['loss'] = affordance_loss

        loss_tot += affordance_loss.mean()
        
        return loss_tot, guide_losses

    
    def draw_sphere_at_point(self, center, radius=0.06, color=[0.1, 0.1, 0.7]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        sphere.translate(center)
        return sphere

    def weighted_sampling_with_threshold(self, position, affordance, sample_k=10, threshold=0.0):
        """
        position: (B, 2048, 3)
        affordance: (B, 2048, 1)
        """
        weights = affordance.squeeze(-1)
        weights = torch.where(weights > threshold, weights, torch.zeros_like(weights))

        weights_sum = weights.sum(dim=1, keepdim=True) # (B, 1)
        weights = weights / (weights_sum + 1e-8)

        batch_size, num_points = weights.shape
        sampled_indices = torch.multinomial(weights, sample_k, replacement=True) # (B, sample_k)

        sampled_points = torch.gather(position, dim=1, index=sampled_indices.unsqueeze(-1).expand(-1, -1, position.size(-1))) # (B, sample_k, 3)
        return sampled_points
    
    def topk_sampling(self, position, affordance, sample_k=10):
        affordance = affordance.squeeze(-1)
        top_k_values, top_k_indices = torch.topk(affordance, sample_k, dim=1)
        top_k_positions = torch.gather(position, dim=1, index=top_k_indices.unsqueeze(-1).expand(-1, -1, position.size(-1)))
        top_k_values = top_k_values.unsqueeze(-1)

        return top_k_positions    

    def compact_sampled_points(self, sampled_points, thershold = 0.1):
        """
        check if the sampled points are compact or not

        sampled_points: (B, sample-k, 3)
        """
        # method 1: check the distance to the centroid
        #sampled_points = sampled_points[...,:2]
        #centroid = sampled_points.mean(dim=1) # (B, 3)
        #distance = torch.norm(sampled_points - centroid[:, None, :], dim=-1) # (B, sample-k)
        #result = torch.all(distance < thershold) 

        # method 2: DBSCAN
        sampled_points = sampled_points.squeeze(0).cpu().detach().numpy() # FIXME: only for batch size 1
        min_points = 2
        db= DBSCAN(eps=thershold, min_samples=min_points).fit(sampled_points)
        labels = db.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if num_clusters > 1:
            result = False
        else:
            result = True
        return result
        
    def normalize_affordance(self, affordance):

        batch_min = affordance.min(dim=1, keepdim=True)[0]
        batch_max = affordance.max(dim=1, keepdim=True)[0]

        normalized_affordance = (affordance - batch_min) / (batch_max - batch_min)  

        return normalized_affordance


class CompositeGuidance(Guidance):
    def __init__(self):
        super(CompositeGuidance, self).__init__()


    def compute_guidance_loss(self, x, t, data_batch):
        loss_tot = 0
        guide_losses = dict()
        # affordance guidance
        #affordance_guidance = AffordanceGuidance_v2()
        affordance_guidance = AffordanceGuidance_v3()
        affordance_loss, affordance_guide_losses = affordance_guidance.compute_guidance_loss(x, t, data_batch)
        loss_tot += affordance_loss
        guide_losses["affordance_loss"] = affordance_guide_losses["loss"]
        guide_losses["distance_error"] = affordance_guide_losses["distance_error"]


        # collision guidance
        collision_guidance = NonCollisionGuidance_v3()
        collision_loss, collision_guide_losses = collision_guidance.compute_guidance_loss(x, t, data_batch)
        loss_tot += collision_loss
        guide_losses["collision_loss"] = collision_guide_losses["loss"]
        guide_losses["loss"] = guide_losses["affordance_loss"]  * 500 + guide_losses["collision_loss"] * 1000

        print("diffusion step {}, affordance_loss: {}, collision_loss: {}".format(t,  affordance_loss.mean().item(), collision_loss.mean().item()))
        return loss_tot, guide_losses

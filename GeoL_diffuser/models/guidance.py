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
    """
    TODO:
    - use mse
    - use min to get the loss of k affordance
    """
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
        k = 5 # top k affordance values
        topk_affordance, topk_idx = torch.topk(affordance, k, dim=-1)
        topk_positions = torch.gather(position, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, position.size(-1)))
        avg_topk_positions = topk_positions.mean(dim=1)  # (B, 3) unscaled
        avg_topk_positions_debug = avg_topk_positions.clone() # (B, 3) unscaled
        avg_topk_positions = avg_topk_positions[:, None, None].expand(-1, num_samp, num_hypo, -1)  # (B, N, H, 3)

        # scale the topk affordance
        avg_topk_positions = self.scale_xy_pose(avg_topk_positions, data_batch["gt_pose_xy_min_bound"], data_batch["gt_pose_xy_max_bound"]) # (B, N, H, 3)
        #avg_topk_positions = torch.zeros_like(avg_topk_positions) # debug
        affordance_loss = F.mse_loss(
            avg_topk_positions[..., :2], x[..., :2], reduction="none"
        ) # (B, N, H, 3) scaled avg_topk_positions 
        #avg_topk_positions = torch.ones_like(avg_topk_positions) * 0.2# debug
        affordance_loss = torch.norm(avg_topk_positions[..., :2] - x[..., :2], dim=-1) # (B, N, H)
        #affordance_loss = affordance_loss.mean(dim=-1)# (B, N, H)
        #affordance_loss = affordance_loss.values # (B, N, H)
        affordance_loss = affordance_loss * 1
        affordance_loss_debug = affordance_loss.clone()
        guide_losses["affordance_loss"] = affordance_loss # (B, N, H)
        affordance_loss = affordance_loss.sum(dim=-1)  # (B, N)
        
        affordance_loss = affordance_loss.sum() #* 500 # (B,)
        loss_tot += affordance_loss

        ####### DEBUG visualize avg_topk_positions
        avg_topk_positions_debug = self.scale_xy_pose(avg_topk_positions_debug, data_batch["gt_pose_xy_max_bound"], data_batch["gt_pose_xy_min_bound"]) # (B, 3)
        avg_topk_positions_debug = avg_topk_positions_debug[0].cpu().detach().numpy() # (3,) unsacled
        position_debug = position[0].cpu().detach().numpy() # (2048, 3)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(position_debug)
        pc.paint_uniform_color([0.5, 0.5, 0.5])
        avg_topk_sphere = self.draw_sphere_at_point(avg_topk_positions_debug)
        descaled_x = self.descale_xy_pose(x, data_batch["gt_pose_xy_min_bound"], data_batch["gt_pose_xy_max_bound"])
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


class ReceptacelGuidance(Guidance):
    def __init__(self):
        super(ReceptacelGuidance, self).__init__()

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evalueates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the sampled points to use to compute losses
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        bsize, num_samp, num_hypo, _ = x.size() 

        receptacle_height = data_batch["receptacle_height"] # TODO: get table height
        T_plane = data_batch['T_plane']
        plane_model = data_batch['plane_model'] # (B, 4)
        receptacle_height = - plane_model[:, 3] / plane_model[:, 2] # (B,)

        descaled_pred_x = self.descale_xyz_pose(x, data_batch["gt_pose_xyz_min_bound"], data_batch["gt_pose_xyz_max_bound"]) # unscaled # (B, N, H, 3)
        
        descaled_pred_x_aligned = self.transform_points(descaled_pred_x, T_plane) # (B, N, H, 3) aligned to z


        receptacle_height = receptacle_height[:, None, None, None].expand(-1, num_samp, num_hypo, 1) # unscaled
        receptacle_height = self.scale_xyz_pose(receptacle_height, data_batch["gt_pose_xyz_min_bound"], data_batch["gt_pose_xyz_max_bound"]) # scaled
        receptacle_loss =  torch.abs(receptacle_height[..., 0] - descaled_pred_x_aligned[..., 2]) # (B, N, H) # NOTE: abs or not?

        receptacle_loss[receptacle_loss > 0.05] =  1000 # set a large loss for points that are far from the receptacle  

        guide_losses["affordance_loss"] = affordance_loss # (B, N, H)
        affordance_loss = affordance_loss.mean(dim=-1)  # (B, N)
        
        affordance_loss = affordance_loss.mean() * 500 # (B,)
        loss_tot += affordance_loss

        return loss_tot, guide_losses
    
    def transform_points(self, points, transform_matrices):
        """
        points: (B, N, H, 3)
        T: (B, 4, 4)

        return: (B, N, H, 3)
        """
        batch_size, num_samp, num_hypo, _ = points.size()
        points = points.view(batch_size,-1, 3) # (B, N*H, 3)
        ones = torch.ones(*points.shape[:-1], 1, device=points.device)  # Shape: [Batch size, num, 1]
        points_homogeneous = torch.cat([points, ones], dim=-1)  # Shape: [Batch size, num, 4]

        # Step 2: Apply the transformation
        transformed_points_homogeneous = torch.einsum('bij,bnj->bni', transform_matrices, points_homogeneous)  # Shape: [Batch size, num, 4]

        # Step 3: Convert back to 3D
        transformed_points = transformed_points_homogeneous[..., :3] / transformed_points_homogeneous[..., 3:] # Shape: [Batch size, num, 3]
        transformed_points = transformed_points.view(batch_size, num_samp, num_hypo, 3) # (B, N, H, 3)
        
        return transformed_points

        


class NonCollisionGuidance(Guidance):
    def __init__(self):
        super(NonCollisionGuidance, self).__init__()

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evalueates all guidance losses and total and individual values.
        - x: (B, N, H, 2) the sampled points to use to compute losses
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        batchsize, num_samp, num_hypo, _ = x.size()
        guide_losses = dict()
        loss_tot = 0.0
        depth = data_batch["depth"]
        color = data_batch["color_tsdf"]
        intrinsics = data_batch["intrinsics"]
        vol_bnds = data_batch["vol_bnds"]
        vol_bnds = vol_bnds[0]
        vol_bnds[:, 0] = vol_bnds[:, 0].min()
        vol_bnds[:, 1] = vol_bnds[:, 1].max()
        #tsdf = tsdf[:, None, None].expand(-1, num_samp, num_hypo, -1)

        # get the first tsdf
        tsdf = TSDFVolume(vol_bnds.cpu().detach().numpy(), voxel_dim=256, num_margin=5)
        tsdf.integrate(
            color[0].cpu().detach().numpy(), 
            depth[0].cpu().detach().numpy(), 
            intrinsics[0].cpu().detach().numpy(), 
            np.eye(4))
        mesh = tsdf.get_mesh()

        # scene_pc align to tsdf
        scene_pc = data_batch['pc_position']
        scene_pc = (scene_pc - vol_bnds.T[0:1]) / (vol_bnds.T[1:2] - vol_bnds.T[0:1])
        scene_pc = scene_pc * 2 - 1 

        T_plane = data_batch['T_plane'] # (B, 4, 4)
        T_plane_one = T_plane[0].cpu().detach().numpy()
        T_plane_one_tsdf, plane_model_one_tsdf= get_tf_for_scene_rotation(scene_pc[0].cpu().detach().numpy())
        plane_model_one_tsdf[3] = plane_model_one_tsdf[3] + 0.01 # add a small margin to avoid collision with desk
        
        # descaled_pred
        x = self.descale_xy_pose(x, data_batch["gt_pose_xy_min_bound"], data_batch["gt_pose_xy_max_bound"]) # (B, N, H, 3)
        x = (x - vol_bnds.T[0:1]) / (vol_bnds.T[1:2] - vol_bnds.T[0:1]) # (B, N, H, 3)
        x = x * 2 - 1
        x[:,:,:,2] = (-plane_model_one_tsdf[3] - plane_model_one_tsdf[0] * x[:,:,:,0] - plane_model_one_tsdf[1] * x[:,:,:,1]) / plane_model_one_tsdf[2]

        # get obj
        obj_pc = data_batch['object_pc_position'] # (B, 512, 3) # aligned to z
        obj_pc = (obj_pc - vol_bnds.T[0:1]) / (vol_bnds.T[1:2] - vol_bnds.T[0:1]) # (B, 512, 3)
        obj_pc = obj_pc * 2 - 1 # obj point aligned to tsdf mesh
        
        # for the obj aligned to z-axis(positive)
        min_bound = torch.min(obj_pc, dim=1)[0] # (B, 3)
        max_bound = torch.max(obj_pc, dim=1)[0] # (B, 3)
        base_point = torch.empty_like(min_bound) # (B, 3)
        base_point[:, 0] = (min_bound[:, 0] + max_bound[:, 0]) / 2
        base_point[:, 1] = (min_bound[:, 1] + max_bound[:, 1]) / 2
        base_point[:, 2] = max_bound[:, 2] # (B, 3) # NOTE: assuming the object is aligned to z-axis
        base_point = torch.matmul(base_point, torch.tensor(T_plane_one[:3,:3].T).cuda()) # (B, 1, 3)

        # visualize the scene [0]
        scene_pc_one_data = scene_pc[0].cpu().detach().numpy()
        scene_pc_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pc_one_data))        

        # translate the objects from base point to pred_x
        obj_pc_align_plane = torch.matmul(obj_pc, torch.tensor(T_plane_one[:3,:3].T).cuda()) # (B, 512, 3)
        obj_pc_align_plane = torch.matmul(obj_pc_align_plane, torch.tensor([[1., 0., 0.], [0., -1., 0.], [0.,0.,-1.]]).cuda()) # (B, 512, 3)
        obj_pc_align_plane = obj_pc_align_plane - base_point[:, None, :].expand(-1, obj_pc.size(1), -1) # (B, 512, 3)
        bsize, num_samp, num_hypo, _ = x.size() 
        points_for_place = x.reshape(bsize, -1, 3) # (B, N*H, 3)
        obj_pc_align_plane = obj_pc_align_plane.unsqueeze(2) # [B, 512, 1, 3]
        points_for_place = points_for_place.unsqueeze(1) # [B, 1, N*H, 3]
        translated_points = points_for_place + obj_pc_align_plane # [B, 512, N*H, 3]

        # prepare the first object in the first place position for visualization
        obj_pc_one_data = translated_points[0, :, 2, :].cpu().detach().numpy()
        obj_pc_one_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_one_data))

        #obj_pc_all_data = translated_points[0, : , :, :].cpu().detach().numpy()
        #obj_pc_all_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_all_data.reshape(-1, 3)))
        # visualize coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # visualize the obj, scene, coordinate frame, mesh
        o3d.visualization.draw_geometries([obj_pc_one_data_vis, scene_pc_vis, coordinate_frame, mesh])
        #o3d.visualization.draw_geometries([obj_pc_all_data_vis, scene_pc_vis, coordinate_frame, mesh])
    


        ### query tsdf 
        query_points = translated_points.view(bsize, -1, 3) # [B, 512*N*H, 3]
        query_points = query_points[None, None,  ...] # [B, 1, 512*N*H, 3]
        query_points = query_points[..., [2, 1, 0]] # NOTE: change the order
        tsdf = tsdf._tsdf_vol
        tsdf = torch.tensor(tsdf, dtype=torch.float32).cuda()[None, None]# [B, C, D, H, W]
        query_tsdf = nn.functional.grid_sample(tsdf, query_points, align_corners=True) # [B, C, 1, 1, N*H*512]
        query_tsdf = query_tsdf.view(bsize, num_samp * num_hypo, -1) # [B, N*H, 512]
        collision_loss = F.relu(0.1 - query_tsdf).mean(dim=-1)

        guide_losses["collision_loss"] = collision_loss # (B, N*H)
        
        collision_loss = collision_loss.mean()  # (B,)
        loss_tot += collision_loss

        return loss_tot, guide_losses


class NonCollisionGuidance_v2(Guidance):
    def __init__(self):
        super(NonCollisionGuidance_v2, self).__init__()

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evalueates all guidance losses and total and individual values.
        - x: (B, N, H, 2) the sampled points to use to compute losses
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        x = x[..., :2] # only need xy
        batchsize, num_samp, num_hypo, _ = x.size()
        guide_losses = dict()
        loss_tot = 0.0
        depth = data_batch["depth"]
        color = data_batch["color_tsdf"]
        intrinsics = data_batch["intrinsics"]
        vol_bnds = data_batch["vol_bnds"]
        vol_bnds = vol_bnds[0]
        vol_bnds[:, 0] = vol_bnds[:, 0].min()
        vol_bnds[:, 1] = vol_bnds[:, 1].max()


        #get the first tsdf
        # tsdf = TSDFVolume(vol_bnds.cpu().detach().numpy(), voxel_dim=256, num_margin=5)
        # tsdf.integrate(
        #     color[0].cpu().detach().numpy(), 
        #     depth[0].cpu().detach().numpy(), 
        #     intrinsics[0].cpu().detach().numpy(), 
        #     np.eye(4))
        # mesh = tsdf.get_mesh()
        

        # scene_pc align to tsdf
        scene_pc = data_batch['pc_position']
        scene_pc = (scene_pc - vol_bnds.T[0:1]) / (vol_bnds.T[1:2] - vol_bnds.T[0:1])
        scene_pc = scene_pc * 2 - 1 

        T_plane = data_batch['T_plane'] # (B, 4, 4)
        T_plane_one = T_plane[0].cpu().detach().numpy()
        T_plane_one_tsdf, plane_model_one_tsdf= get_tf_for_scene_rotation(scene_pc[0].cpu().detach().numpy())
        plane_model_one_tsdf[3] = plane_model_one_tsdf[3] + 0.000 # add a small margin to avoid collision with desk
        
        # get obj
        obj_pc = data_batch['object_pc_position'] # (B, 512, 3) # aligned to z
        num_pcdobj = obj_pc.size(1)
        obj_pc = (obj_pc - vol_bnds.T[0:1]) / (vol_bnds.T[1:2] - vol_bnds.T[0:1]) # (B, 512, 3)
        obj_pc = obj_pc * 2 - 1 # obj point aligned to tsdf mesh
        obj_pc = obj_pc.unsqueeze(1).unsqueeze(3).expand(-1, num_samp, -1, num_hypo, -1) # (B, N, O, H, 3)
        # descaled_pred

        x = self.descale_xy_pose(x, data_batch["gt_pose_xy_min_bound"], data_batch["gt_pose_xy_max_bound"]) # (B, N, H, 3)
        x = (x - vol_bnds.T[0:1][...,:2]) / (vol_bnds.T[1:2][...,:2] - vol_bnds.T[0:1][...,:2]) # (B, N, H, 2)
        x = x * 2 - 1
        x_z = (-plane_model_one_tsdf[3] - plane_model_one_tsdf[0] * x[:,:,:,0] - plane_model_one_tsdf[1] * x[:,:,:,1]) / plane_model_one_tsdf[2]
        #x[:,:,:,2] = (-plane_model_one_tsdf[3] - plane_model_one_tsdf[0] * x[:,:,:,0] - plane_model_one_tsdf[1] * x[:,:,:,1]) / plane_model_one_tsdf[2]
        x = torch.cat([x, x_z.unsqueeze(-1)], dim=-1) # (B, N, H, 3)



        x = x.unsqueeze(2).expand(-1, -1, num_pcdobj, -1, -1) # (B, N, O, H, 3)

        
        # for the obj aligned to z-axis(positive)
        min_bound = torch.min(obj_pc, dim=2)[0] # (B, N, H, 3)
        max_bound = torch.max(obj_pc, dim=2)[0] # (B, N, H, 3)
        base_point = torch.empty_like(min_bound) # (B, N, H, 3)
        base_point[..., 0] = (min_bound[..., 0] + max_bound[..., 0]) / 2
        base_point[..., 1] = (min_bound[..., 1] + max_bound[..., 1]) / 2
        base_point[..., 2] = max_bound[..., 2]  # NOTE: assuming the object is aligned to z-axis
        #base_point = torch.matmul(base_point, torch.tensor(T_plane_one[:3,:3].T).cuda()) # should use the tsdf plane
        T_plane_one_tsdf = torch.tensor(T_plane_one_tsdf, dtype=torch.float32).cuda()
        #base_point = torch.matmul(base_point, torch.tensor(T_plane_one_tsdf[:3,:3].T).cuda()) 
        base_point = base_point[:, :, None].repeat(1, 1, num_pcdobj, 1, 1)[:, :, :, :1] # (B, N, O, 1, 3)
        

        # visualize the scene [0]
        scene_pc_one_data = scene_pc[0].cpu().detach().numpy()
        scene_pc_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pc_one_data))        

        # translate the objects from base point to pred_x
        obj_pc_align_plane = torch.matmul(obj_pc, torch.tensor(T_plane_one[:3,:3].T).cuda()) # (B, N, O, H, 3)
        obj_pc_align_plane = torch.matmul(obj_pc_align_plane, torch.tensor([[1., 0., 0.], [0., -1., 0.], [0.,0.,-1.]]).cuda()) 
        obj_pc_align_plane = obj_pc
        
        bsize, num_samp, _,  num_hypo, _ = x.size() 
        #points_for_place = x.reshape(bsize, -1, 3) # (B, N*H, 3)
        #obj_pc_align_plane = obj_pc_align_plane.unsqueeze(2) # [B, 512, 1, 3]
        #points_for_place = points_for_place.unsqueeze(1) # [B, 1, N*H, 3]
        #translated_points = obj_pc_align_plane + x - base_point # [B, N, O, H, 3]
        translated_points = obj_pc_align_plane

        # prepare the first object in the first place position for visualization
        obj_pc_one_data = translated_points[0, 0, :, 0, :].cpu().detach().numpy() # (O, 3)
        obj_pc_one_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_one_data))

        obj_pc_all_data = translated_points[0, : , :, :].cpu().detach().numpy()
        obj_pc_all_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_all_data.reshape(-1, 3)))
        # visualize coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # visualize the obj, scene, coordinate frame, mesh
        #create base point sphere o3d
        base_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        base_point_sphere.compute_vertex_normals()
        base_point_sphere.paint_uniform_color([0.1, 0.1, 0.7])
        base_point_sphere.translate(base_point[0, 0, 0, 0, :].cpu().detach().numpy())



        o3d.visualization.draw_geometries([obj_pc_one_data_vis, scene_pc_vis, coordinate_frame, base_point_sphere])
        #o3d.visualization.draw_geometries([obj_pc_all_data_vis, scene_pc_vis, coordinate_frame, mesh])
        #o3d.visualization.draw_geometries([obj_pc_one_data_vis, scene_pc_vis, coordinate_frame])
        #o3d.visualization.draw_geometries([obj_pc_all_data_vis, scene_pc_vis, coordinate_frame])
    
    
        #### debug: tsdf check the tsdf value one by one
        # collision_loss = torch.zeros(bsize, num_samp, num_hypo).cuda()
        # tsdf = tsdf._tsdf_vol
        # tsdf = torch.tensor(tsdf, dtype=torch.float32).cuda()[None, None]# [B, C, D, H, W]
        # for ii in range(num_hypo):

        #     query_points = translated_points[:, :, :, ii, :].view(bsize, num_samp, -1, 3) # [B, N, O*1, 3]
        #     query_points = query_points[..., [2, 1, 0]]
        #     query_points = query_points.unsqueeze(-2) # [B, N*O*1, 1, 3]

        #     query_tsdf = nn.functional.grid_sample(tsdf, query_points, align_corners=True) # [B, 1, N, O*1, 1]
        #     query_tsdf = query_tsdf.squeeze(-1).squeeze(1).view(bsize, num_samp, 1, -1 ) # [B, N, H=1, O]
        #     collision_loss[0,0,ii] = F.relu(0.1 - query_tsdf).sum(dim=-1).squeeze() # (B, N, H)
        #     print("collision_loss: ", collision_loss[0,0,ii] )

        #     obj_pc_one_data = translated_points[0, 0, :, ii, :].cpu().detach().numpy() # (O, 3)
        #     obj_pc_one_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_one_data))
        #     o3d.visualization.draw_geometries([obj_pc_one_data_vis, scene_pc_vis, coordinate_frame, mesh])


        ### query tsdf version: [B, N, O*H, 3]
        query_points = translated_points.view(bsize, num_samp, -1, 3) # [B, N, O*H, 3]
        query_points = query_points[..., [2, 1, 0]] # NOTE: change the order
        query_points = query_points.unsqueeze(-2) # [B, N, O*H, 1, 3]
        #tsdf = tsdf._tsdf_vol
        #tsdf = torch.tensor(tsdf, dtype=torch.float32).cuda()[None, None]# [B, C, D, H, W]
        tsdf = data_batch["tsdf_vol"][0][None, None] # [B, 1, D, H, W]  
        query_tsdf = nn.functional.grid_sample(tsdf, query_points, align_corners=True) # [B, 1, N, O*H, 1]
        #query_tsdf = query_tsdf.squeeze(-1).squeeze(1).view(bsize, num_samp, num_hypo, -1 ) # [B, N, H, O]
        #collision_loss = F.relu(0.1 - query_tsdf).mean(dim=-1) # (B, N, H)
        query_tsdf = query_tsdf.squeeze(-1).squeeze(1).view(bsize, num_samp, -1 , num_hypo,) # [B, N, O, H]
        collision_loss = F.relu(0.2 - query_tsdf).sum(dim=-2) # (B, N, H)

        #### query tsdf version: [B, N*H, O, 3]
        # query_points = translated_points.view(bsize, -1, num_pcdobj, 3) # [B, N*H, O, 3]
        # query_points = query_points[..., [2, 1, 0]] # NOTE: change the order
        # query_points = query_points.unsqueeze(-2) # [B, N*H, O, 1, 3]
        # tsdf = tsdf._tsdf_vol
        # tsdf = torch.tensor(tsdf, dtype=torch.float32).cuda()[None, None]# [B, C, D, H, W]
        # query_tsdf = nn.functional.grid_sample(tsdf, query_points, align_corners=True) # [B, 1, N*H, O, 1]
        # query_tsdf = query_tsdf.squeeze(-1).squeeze(1) # [B, N*H, O]
        # collision_loss = F.relu(0.1 - query_tsdf).mean(dim=-1) # (B, N*H)
        # #query_tsdf_2 = query_tsdf.squeeze(-1).squeeze(1).view(bsize, num_samp, -1 , num_hypo,) # [B, N, O, H]
        # #collision_loss = F.relu(0.1 - query_tsdf).mean(dim=-2) # (B, N, H)

        ### debug; check the collision loss
        # for ii in range(collision_loss.size(2)):
        #     print("collision_loss: ", collision_loss[0, 0, ii])
        #     obj_pc_one_data = translated_points[0, 0, :, ii, :].cpu().detach().numpy() # (O, 3)
        #     obj_pc_one_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_one_data))
        #     obj_pc_all_data = translated_points[0, : , :, :].cpu().detach().numpy()
        #     obj_pc_all_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_all_data.reshape(-1, 3)))
            #o3d.visualization.draw_geometries([obj_pc_one_data_vis, scene_pc_vis, coordinate_frame, mesh])

        guide_losses["loss"] = collision_loss 
        
        collision_loss = collision_loss.mean()  # (B,)
        loss_tot += collision_loss

        return loss_tot, guide_losses
    

class NonCollisionGuidance_v3(Guidance):
    def __init__(self):
        super(NonCollisionGuidance_v3, self).__init__()

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evalueates all guidance losses and total and individual values.
        - x: (B, N, H, 2) the sampled points to use to compute losses
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        print(x.shape)
        x = x[..., :2] # only need xy
        z_r = x[..., -1][..., None] # rotation z axis
        batchsize, num_samp, num_hypo, _ = x.size()
        guide_losses = dict()
        loss_tot = 0.0
        depth = data_batch["depth"]
        color = data_batch["color_tsdf"]
        intrinsics = data_batch["intrinsics"]
        vol_bnds = data_batch["vol_bnds"]
        vol_bnds = vol_bnds[0]
        vol_bnds[:, 0] = vol_bnds[:, 0].min()
        vol_bnds[:, 1] = vol_bnds[:, 1].max()


        #get the first tsdf
        # tsdf = TSDFVolume(vol_bnds.cpu().detach().numpy(), voxel_dim=256, num_margin=5)
        # tsdf.integrate(
        #     color[0].cpu().detach().numpy(), 
        #     depth[0].cpu().detach().numpy(), 
        #     intrinsics[0].cpu().detach().numpy(), 
        #     np.eye(4))
        # mesh = tsdf.get_mesh()
        

        # scene_pc align to tsdf
        scene_pc = data_batch['pc_position']
        scene_pc = (scene_pc - vol_bnds.T[0:1]) / (vol_bnds.T[1:2] - vol_bnds.T[0:1])
        scene_pc = scene_pc * 2 - 1 

        T_plane = data_batch['T_plane'] # (B, 4, 4)
        T_plane_one = T_plane[0].cpu().detach().numpy()
        T_plane_one_tsdf, plane_model_one_tsdf= get_tf_for_scene_rotation(scene_pc[0].cpu().detach().numpy())
        plane_model_one_tsdf[3] = plane_model_one_tsdf[3] + 0.005 # add a small margin to avoid collision with desk
        
        # get obj
        obj_pc = data_batch['object_pc_position'] # (B, 512, 3) # aligned to z
        num_pcdobj = obj_pc.size(1)
        obj_pc = (obj_pc - vol_bnds.T[0:1]) / (vol_bnds.T[1:2] - vol_bnds.T[0:1]) # (B, 512, 3)
        obj_pc = obj_pc * 2 - 1 # obj point aligned to tsdf mesh
        obj_pc = obj_pc.unsqueeze(1).unsqueeze(3).expand(-1, num_samp, -1, num_hypo, -1) # (B, N, O, H, 3)

        # descaled_pred

        x = self.descale_xy_pose(x, data_batch["gt_pose_xy_min_bound"], data_batch["gt_pose_xy_max_bound"]) # (B, N, H, 3)
        x = (x - vol_bnds.T[0:1][...,:2]) / (vol_bnds.T[1:2][...,:2] - vol_bnds.T[0:1][...,:2]) # (B, N, H, 2)
        x = x * 2 - 1
        x_z = (-plane_model_one_tsdf[3] - plane_model_one_tsdf[0] * x[:,:,:,0] - plane_model_one_tsdf[1] * x[:,:,:,1]) / plane_model_one_tsdf[2]
        #x[:,:,:,2] = (-plane_model_one_tsdf[3] - plane_model_one_tsdf[0] * x[:,:,:,0] - plane_model_one_tsdf[1] * x[:,:,:,1]) / plane_model_one_tsdf[2]
        x = torch.cat([x, x_z.unsqueeze(-1)], dim=-1) # (B, N, H, 3) # 0.05 margin

        x = x.unsqueeze(2).expand(-1, -1, num_pcdobj, -1, -1) # (B, N, O, H, 3)

        # for the obj aligned to z-axis(positive)
        min_bound = torch.min(obj_pc, dim=2)[0] # (B, N, H, 3)
        max_bound = torch.max(obj_pc, dim=2)[0] # (B, N, H, 3)
        base_point = torch.empty_like(min_bound) # (B, N, H, 3)
        base_point[..., 0] = (min_bound[..., 0] + max_bound[..., 0]) / 2
        base_point[..., 1] = (min_bound[..., 1] + max_bound[..., 1]) / 2
        base_point[..., 2] = max_bound[..., 2]  # NOTE: assuming the object is aligned to z-axis
        #base_point = torch.matmul(base_point, torch.tensor(T_plane_one[:3,:3].T).cuda()) # should use the tsdf plane
        T_plane_one_tsdf = torch.tensor(T_plane_one_tsdf, dtype=torch.float32).cuda()
        base_point = torch.matmul(base_point, torch.tensor(T_plane_one_tsdf[:3,:3].T).cuda()) 
        base_point = base_point[:, :, None].repeat(1, 1, num_pcdobj, 1, 1)[:, :, :, :1] # (B, N, O, 1, 3)

        # rotation z axis
        z_r = z_r.squeeze(-1)
        z_r = z_r.clamp(-10 * np.pi / 180, 10 * np.pi  / 180)
        cos_theta = torch.cos(z_r)
        sin_theta = torch.sin(z_r)
        B, N, O, H, _ = obj_pc.shape
        R = torch.zeros((B, 1, H, 3, 3), dtype=obj_pc.dtype, device=obj_pc.device)
        R[..., 0, 0] = cos_theta  # cosθ
        R[..., 0, 1] = -sin_theta # -sinθ
        R[..., 1, 0] = sin_theta  # sinθ
        R[..., 1, 1] = cos_theta  # cosθ
        R[..., 2, 2] = 1          # z 轴保持不变
        obj_pc = obj_pc.permute(0, 1, 3, 2, 4)  # (B, H, N, 512, 3)
        obj_pc = obj_pc @ R  # (B, H, N, 512, 3, 3) @ (B, H, 3, 3) -> (B, H, N, 512, 3)
        obj_pc = obj_pc.permute(0, 1,3,2,4)  # (B, N, 512, H, 3)

        # visualize the scene [0]
        scene_pc_one_data = scene_pc[0].cpu().detach().numpy()
        scene_pc_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pc_one_data))        

        # translate the objects from base point to pred_x
        obj_pc = torch.matmul(obj_pc, torch.tensor(T_plane_one_tsdf[:3,:3].T).cuda()) # (B, N, O, H, 3)
        #obj_pc_align_plane = torch.matmul(obj_pc, torch.tensor(T_plane_one[:3,:3].T).cuda()) # (B, N, O, H, 3)
        #obj_pc_align_plane = torch.matmul(obj_pc_align_plane, torch.tensor([[1., 0., 0.], [0., -1., 0.], [0.,0.,-1.]]).cuda()) 

        
        bsize, num_samp, _,  num_hypo, _ = x.size() 
        #points_for_place = x.reshape(bsize, -1, 3) # (B, N*H, 3)
        #obj_pc_align_plane = obj_pc_align_plane.unsqueeze(2) # [B, 512, 1, 3]
        #points_for_place = points_for_place.unsqueeze(1) # [B, 1, N*H, 3]
        translated_points = obj_pc + x - base_point # [B, N, O, H, 3]
        #translated_points = obj_pc_align_plane

        # prepare the first object in the first place position for visualization
        obj_pc_one_data = translated_points[0, 0, :, 0, :].cpu().detach().numpy() # (O, 3)
        obj_pc_one_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_one_data))

        obj_pc_all_data = translated_points[0, : , :, :].cpu().detach().numpy()
        obj_pc_all_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_all_data.reshape(-1, 3)))
        # visualize coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        
        obj_pc_all_data_vis.paint_uniform_color([1,0,0])
        # visualize the obj, scene, coordinate frame, mesh
        #create base point sphere o3d
        base_point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        base_point_sphere.compute_vertex_normals()
        base_point_sphere.paint_uniform_color([0.1, 0.1, 0.7])
        base_point_sphere.translate(base_point[0, 0, 0, 0, :].cpu().detach().numpy())

        o3d.visualization.draw_geometries([obj_pc_one_data_vis, scene_pc_vis, coordinate_frame, base_point_sphere])
        o3d.visualization.draw_geometries([obj_pc_all_data_vis, scene_pc_vis, coordinate_frame])
        # o3d.io.write_point_cloud("Geo_comb/obj_pc.ply", obj_pc_all_data_vis+scene_pc_vis)
        #o3d.visualization.draw_geometries([obj_pc_one_data_vis, scene_pc_vis, coordinate_frame])
        #o3d.visualization.draw_geometries([obj_pc_all_data_vis, scene_pc_vis, coordinate_frame])
    
    
        #### debug: tsdf check the tsdf value one by one
        # collision_loss = torch.zeros(bsize, num_samp, num_hypo).cuda()
        # tsdf = tsdf._tsdf_vol
        # tsdf = torch.tensor(tsdf, dtype=torch.float32).cuda()[None, None]# [B, C, D, H, W]
        # for ii in range(num_hypo):

        #     query_points = translated_points[:, :, :, ii, :].view(bsize, num_samp, -1, 3) # [B, N, O*1, 3]
        #     query_points = query_points[..., [2, 1, 0]]
        #     query_points = query_points.unsqueeze(-2) # [B, N*O*1, 1, 3]

        #     query_tsdf = nn.functional.grid_sample(tsdf, query_points, align_corners=True) # [B, 1, N, O*1, 1]
        #     query_tsdf = query_tsdf.squeeze(-1).squeeze(1).view(bsize, num_samp, 1, -1 ) # [B, N, H=1, O]
        #     collision_loss[0,0,ii] = F.relu(0.1 - query_tsdf).sum(dim=-1).squeeze() # (B, N, H)
        #     print("collision_loss: ", collision_loss[0,0,ii] )

        #     obj_pc_one_data = translated_points[0, 0, :, ii, :].cpu().detach().numpy() # (O, 3)
        #     obj_pc_one_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_one_data))
        #     o3d.visualization.draw_geometries([obj_pc_one_data_vis, scene_pc_vis, coordinate_frame, mesh])


        ### query tsdf version: [B, N, O*H, 3]
        query_points = translated_points.view(bsize, num_samp, -1, 3) # [B, N, O*H, 3]
        query_points = query_points[..., [2, 1, 0]] # NOTE: change the order
        query_points = query_points.unsqueeze(-2) # [B, N, O*H, 1, 3]
        #tsdf = tsdf._tsdf_vol
        #tsdf = torch.tensor(tsdf, dtype=torch.float32).cuda()[None, None]# [B, C, D, H, W]
        tsdf = data_batch["tsdf_vol"][0][None, None] # [B, 1, D, H, W]  
        query_tsdf = nn.functional.grid_sample(tsdf, query_points, align_corners=True) # [B, 1, N, O*H, 1]
        #query_tsdf = query_tsdf.squeeze(-1).squeeze(1).view(bsize, num_samp, num_hypo, -1 ) # [B, N, H, O]
        #collision_loss = F.relu(0.1 - query_tsdf).mean(dim=-1) # (B, N, H)
        query_tsdf = query_tsdf.squeeze(-1).squeeze(1).view(bsize, num_samp, -1 , num_hypo,) # [B, N, O, H]
        collision_loss = F.relu(0.2 - query_tsdf).sum(dim=-2) # (B, N, H)

        #### query tsdf version: [B, N*H, O, 3]
        # query_points = translated_points.view(bsize, -1, num_pcdobj, 3) # [B, N*H, O, 3]
        # query_points = query_points[..., [2, 1, 0]] # NOTE: change the order
        # query_points = query_points.unsqueeze(-2) # [B, N*H, O, 1, 3]
        # tsdf = tsdf._tsdf_vol
        # tsdf = torch.tensor(tsdf, dtype=torch.float32).cuda()[None, None]# [B, C, D, H, W]
        # query_tsdf = nn.functional.grid_sample(tsdf, query_points, align_corners=True) # [B, 1, N*H, O, 1]
        # query_tsdf = query_tsdf.squeeze(-1).squeeze(1) # [B, N*H, O]
        # collision_loss = F.relu(0.1 - query_tsdf).mean(dim=-1) # (B, N*H)
        # #query_tsdf_2 = query_tsdf.squeeze(-1).squeeze(1).view(bsize, num_samp, -1 , num_hypo,) # [B, N, O, H]
        # #collision_loss = F.relu(0.1 - query_tsdf).mean(dim=-2) # (B, N, H)

        ### debug; check the collision loss
        # for ii in range(collision_loss.size(2)):
        #     print("collision_loss: ", collision_loss[0, 0, ii])
        #     obj_pc_one_data = translated_points[0, 0, :, ii, :].cpu().detach().numpy() # (O, 3)
        #     obj_pc_one_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_one_data))
        #     obj_pc_all_data = translated_points[0, : , :, :].cpu().detach().numpy()
        #     obj_pc_all_data_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pc_all_data.reshape(-1, 3)))
            #o3d.visualization.draw_geometries([obj_pc_one_data_vis, scene_pc_vis, coordinate_frame, mesh])

        guide_losses["loss"] = collision_loss 
        
        collision_loss = collision_loss.mean()  # (B,)
        loss_tot += collision_loss

        return loss_tot, guide_losses

class AffordanceGuidance_v2(Guidance):
    """
    TODO:
    - adaptive soften

    """
    def __init__(self):
        super(AffordanceGuidance_v2, self).__init__()

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
        affordance_ori = data_batch["affordance"] # [B, 2048, num_affordance]
        affordance_ori = self.normalize_affordance(affordance_ori) # (B, 2048, num_affordance)
        
        affordance_filtered = affordance_ori.clone()
        affordance_filtered = np.asarray(affordance_filtered.cpu().detach().numpy())
        position = data_batch["pc_position"] # [B, 2048, 3]
        for i in range(affordance_ori.size(-1)):
        # filter out the point affordance on the ground
            plane_model = fit_plane_from_points(np.asarray(position[i].cpu().detach().numpy()))

            # if point is on the ground, set the affordance value to 0
            distances = np.abs(plane_model[0] * position[i].cpu().detach().numpy()[:, 0] + plane_model[1] * position[i].cpu().detach().numpy()[:, 1] + plane_model[2] * position[i].cpu().detach().numpy()[:, 2] + plane_model[3])
            #position = position[distances > 0.01]
            affordance_filtered[i] = affordance_ori[i].cpu().detach().numpy()
            affordance_filtered[i][distances > 0.01] = 0
        
        affordance_filtered = torch.tensor(affordance_filtered).cuda()
        affordance = affordance_filtered
        affordance_ori = affordance_filtered

        
        
        isCompact = False
        soft_coeff = 1
        while not isCompact:
            
            affordance = (affordance_ori * soft_coeff).clamp(0, 1) # (B, 2048, num_affordance)
            position = data_batch["pc_position"] # [B, 2048, 3]
            _, num_points, num_affordance = affordance.size()

            affordance = torch.mean(affordance, dim=-1) # (B, 2048, 1)
            sampled_position = self.topk_sampling(position, affordance, sample_k=5)
            
            # check if the sampled points are compact or not
            isCompact = self.compact_sampled_points(sampled_position)
            soft_coeff += 1

            isCompact = True #NOTE: if wanna compose the affordance, commment this

        # visualize the affordance

        


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(position[0].cpu().detach().numpy())
        pcd.colors = o3d.utility.Vector3dVector(self.get_heatmap(affordance[0].cpu().detach().numpy().squeeze(), invert=False))
        #pcd.colors = o3d.utility.Vector3dVector(self.get_heatmap(affordance, invert=False))

        #o3d.visualization.draw_geometries([pcd])
        vis = [pcd]
        
        # sample k points conditioned on the affordance value
        # option1: weihgted sampling
        #sampled_position = self.weighted_sampling_with_threshold(position, affordance, sample_k=10, threshold=0.0)
        # option2: top k sampling







        # visualize the sampled points
        for ii in range(sampled_position.size(1)):
            pos_vis = o3d.geometry.TriangleMesh.create_sphere()
            pos_vis.compute_vertex_normals()
            pos_vis.scale(0.03, center=(0, 0, 0))
            pos_vis.translate(sampled_position[0, ii, :3].detach().cpu().numpy())
            vis_color = [0.5, 0.5, 0.5]
            pos_vis.paint_uniform_color(vis_color)
            vis.append(pos_vis)

        #if t == 0:
            #o3d.visualization.draw_geometries(vis)


        sampled_position = self.scale_xy_pose(sampled_position[...,:2], data_batch["gt_pose_xy_min_bound"], data_batch["gt_pose_xy_max_bound"])

        expanded_target_points = sampled_position.unsqueeze(1).unsqueeze(3)
        expanded_predicted_points = x.unsqueeze(2)
        affordance_loss = F.mse_loss(expanded_target_points[...,:2], expanded_predicted_points[...,:2], reduction="none") # (B, N, num_targets, H, 2)
        affordance_loss = affordance_loss.min(dim=2)[0] # (B, N, H, 2)
        affordance_loss = affordance_loss.mean(dim=-1) # (B, N, H)
        guide_losses["distance_error"] = affordance_loss # (B, N, H)

        affordance_loss = affordance_loss * 500
        guide_losses['loss'] = affordance_loss

        loss_tot += affordance_loss.mean()
        
        return loss_tot, guide_losses
    
    

        k = 20 # top k affordance values
        topk_affordance, topk_idx = torch.topk(affordance, k, dim=-1)
        topk_positions = torch.gather(position, dim=1, index=topk_idx.unsqueeze(-1).expand(-1, -1, position.size(-1)))
        avg_topk_positions = topk_positions.mean(dim=1)  # (B, 3) unscaled
        avg_topk_positions_debug = avg_topk_positions.clone() # (B, 3) unscaled
        avg_topk_positions = avg_topk_positions[:, None, None].expand(-1, num_samp, num_hypo, -1)  # (B, N, H, 3)
        avg_topk_positions = avg_topk_positions[..., :2]
        # scale the topk affordance
        avg_topk_positions = self.scale_xy_pose(avg_topk_positions, data_batch["gt_pose_xy_min_bound"], data_batch["gt_pose_xy_max_bound"]) # (B, N, H, 3)
        #avg_topk_positions = torch.zeros_like(avg_topk_positions) # debug
        affordance_loss = F.mse_loss(
            avg_topk_positions[..., :2], x[..., :2], reduction="none"
        ) # (B, N, H, 3) scaled avg_topk_positions, [B, N, H, 3], [B, N, H, 3]
        #avg_topk_positions = torch.ones_like(avg_topk_positions) * 0.2# debug
        # affordance_loss = torch.norm(avg_topk_positions[..., :2] - x[..., :2], dim=-1) # (B, N, H)
        #affordance_loss = affordance_loss.mean(dim=-1)# (B, N, H)
        #affordance_loss = affordance_loss.values # (B, N, H)

        #### set second sphere in origin, and use mse, min to get the affordance loss
        #second_positions_debug = np.array([0.001, -0.01, 1.26]) # (3,)
        #second_positions_debug = np.array([0.061, -0.273, 1.455])
        #second_positions_debug[...,2] = (-plane_model[3] - plane_model[0] * second_positions_debug[..., 0] - plane_model[1] * second_positions_debug[..., 1]) / plane_model[2]
        #second_sphere = self.draw_sphere_at_point(second_positions_debug)
        #second_positions_debug = torch.tensor(second_positions_debug).cuda()
        #second_positions_debug = second_positions_debug[None, None, None].expand(bsize, num_samp, num_hypo, -1) # (B, N, H, 3)
        #second_positions_debug = self.scale_xyz_pose(second_positions_debug, data_batch["gt_pose_xyz_min_bound"], data_batch["gt_pose_xyz_max_bound"]) # (B, N, H, 3)
        #target_points = torch.cat([avg_topk_positions, second_positions_debug], dim=1) # (B, N=2, H, 3)
        #affordance_loss = torch.norm(target_points[..., :2] - x[..., :2], dim=-1) # (B, N, H)
        #affordance_loss = torch.min(affordance_loss, dim=1)[0] # (B, H)

        guide_losses["distance_error"] =  affordance_loss.mean(dim=-1) # (B, N, H)
        affordance_loss = affordance_loss.mean(dim=-1) * 300
        affordance_loss_debug = affordance_loss.clone()
        guide_losses["affordance_loss"] = affordance_loss # (B, N, H)
        affordance_loss = affordance_loss.mean(dim=-1)  # (B, N)
        
        affordance_loss = affordance_loss.mean() #* 500 # (B,)
        loss_tot += affordance_loss

        ####### DEBUG visualize avg_topk_positions
        #avg_topk_positions_debug = self.scale_xyz_pose(avg_topk_positions_debug, data_batch["gt_pose_xyz_min_bound"], data_batch["gt_pose_xyz_max_bound"]) # (B, 3)
        avg_topk_positions_debug = avg_topk_positions_debug[0].cpu().detach().numpy() # (3,) unsacled
        position_debug = position[0].cpu().detach().numpy() # (2048, 3)
        T_plane, plane_model = get_tf_for_scene_rotation(position_debug)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(position_debug)
        pc.paint_uniform_color([0.5, 0.5, 0.5])
        avg_topk_sphere = self.draw_sphere_at_point(avg_topk_positions_debug)
        descaled_x = self.descale_xy_pose(x, data_batch["gt_pose_xy_min_bound"], data_batch["gt_pose_xy_max_bound"])
        pred_points = descaled_x[0].reshape(-1, 2).cpu().detach().numpy()
        distances = np.sqrt(
            ((position_debug[:, :2][:, None, :] - pred_points[:, :2]) ** 2).sum(axis=2)
        ) # x, y distance 

        scenepts_to_anchor_dist = np.min(distances, axis=1)  # [num_points]
        scenepts_to_anchor_id = np.argmin(distances, axis=1)  # [num_points]
        topk_points_id = np.argsort(scenepts_to_anchor_dist, axis=0)[: pred_points.shape[0]]
        tokk_points_id_corr_anchor = scenepts_to_anchor_id[topk_points_id]

        guide_cost = affordance_loss_debug[0].flatten().cpu().detach().numpy() # [N*H]
        #guide_cost = guide_cost[tokk_points_id_corr_anchor]
        guide_cost_color =  self.get_heatmap(guide_cost[None], invert=False)[0]

        #points_for_place= position_debug[topk_points_id] # option1: use the nearest points

        #pred_points_align = pred_points.copy() # option2: use the plane model
        #pred_points_align[...,2] = (-plane_model[3] - plane_model[0] * pred_points_align[..., 0] - plane_model[1] * pred_points_align[..., 1]) / plane_model[2]
        #points_for_place = pred_points_align



        vis = [pc, avg_topk_sphere]
        #vis = [pc, avg_topk_sphere, second_sphere]

        # for ii, pos in enumerate(points_for_place):
        #     pos_vis = o3d.geometry.TriangleMesh.create_sphere()
        #     pos_vis.compute_vertex_normals()
        #     pos_vis.scale(0.03, center=(0, 0, 0))
        #     pos_vis.translate(pos[:3])
        #     vis_color = guide_cost_color[ii]
        #     pos_vis.paint_uniform_color(vis_color)
        #     vis.append(pos_vis)


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
        affordance_guidance = AffordanceGuidance_v2()
        affordance_loss, affordance_guide_losses = affordance_guidance.compute_guidance_loss(x, t, data_batch)
        loss_tot += affordance_loss
        guide_losses["affordance_loss"] = affordance_guide_losses["loss"]
        guide_losses["distance_error"] = affordance_guide_losses["distance_error"]


        # collision guidance
        collision_guidance = NonCollisionGuidance_v3()
        collision_loss, collision_guide_losses = collision_guidance.compute_guidance_loss(x, t, data_batch)
        loss_tot += collision_loss
        guide_losses["collision_loss"] = collision_guide_losses["loss"]

        #
        guide_losses["loss"] = guide_losses["affordance_loss"] + guide_losses["collision_loss"]

        return loss_tot, guide_losses

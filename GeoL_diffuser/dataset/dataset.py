import os
import open3d as o3d
import json
# Ignore warnings
import warnings
from typing import Any

#import albumentations as A
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet2_ops import pointnet2_utils
import cv2
from GeoL_net.core.registry import registry
import trimesh
from GeoL_diffuser.models.helpers import TSDFVolume, get_view_frustum
from GeoL_diffuser.dataset.visualize_bbox import create_bounding_box
from GeoL_diffuser.models.utils.fit_plane import *
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points

def visualize_sphere_o3d(center, color=[1, 0, 0], size=0.03):
    # center
    center_o3d = o3d.geometry.TriangleMesh.create_sphere()
    center_o3d.compute_vertex_normals()
    center_o3d.scale(size, [0, 0, 0])
    center_o3d.translate(center)
    center_o3d.paint_uniform_color(color)
    return center_o3d

warnings.filterwarnings("ignore")

def is_red(color, tolerance=0.1):
    return (color[0] > 1 - tolerance and color[1] < tolerance and color[2] < tolerance)


def get_top_affordance_points(fps_colors, fps_points, sample_num):
    """

    params:
        fps_colors: numpy.ndarray, (N, 3)
        fps_points: numpy.ndarray, (N, 3)
        sample_num: int.

    Return:
        top_points: torch.Tensor, (sample_num, 3)
    """
    green_values = fps_colors[:, 1]

    top_indices = np.argsort(green_values)[-sample_num:][::-1]  # 从高到低排序并取前 sample_num 个

    top_points = fps_points[top_indices]

    return torch.tensor(top_points, dtype=torch.float32)

@registry.register_dataset(name="Pose_bproc")
class PoseDataset(Dataset):
    """ 
    The dataset for the pose estimation task
    
    Update:
        1. add the offset to the x,y in gt_pose_4d
    """

    def __init__(self,
                 split:str,
                 affordance_threshold:float = 0.001,
                 gt_pose_samples:int = 80,
                 root_dir = "dataset/scene_RGBD_mask_v2_kinect_cfg") -> None:
        super().__init__()

        self.split = split
        self.root_dir = root_dir
        self.folder_path = self.root_dir
        self.affordance_threshold = affordance_threshold

        self.files = []
        self.gt_pose_samples = gt_pose_samples
        items = os.listdir(self.root_dir)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item) # e.g. 'dataset/scene_RGBD_mask_v2/id164_1'
            #print(sub_folder_path)
            sub_items =[f for f in os.listdir(sub_folder_path) if os.path.isdir(os.path.join(sub_folder_path, f))] # e.g. ['printer_0001_normal', 'printer_0001_normal', 'printer_0001_normal', 'printer_0001_normal']

            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item) # 'dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal'
                for sub_sub_item in os.listdir(sub_sub_folder_path):
                    # if end with ply
                    #if sub_sub_item.endswith('_Behind.ply') or sub_sub_item.endswith('_Front.ply') or sub_sub_item.endswith('_Left.ply') or sub_sub_item.endswith('_Right.ply'):
                    if sub_sub_item.endswith('mask_Left.ply') or sub_sub_item.endswith('mask_Right.ply') or sub_sub_item.endswith('mask_Front.ply') or sub_sub_item.endswith('mask_Behind.ply')\
                        or sub_sub_item.endswith('mask_Left Front.ply') or sub_sub_item.endswith('mask_Left Behind.ply') or sub_sub_item.endswith('mask_Right Front.ply') or sub_sub_item.endswith('mask_Right Behind.ply'):
                        one_dataset_pcd_path = os.path.join(sub_sub_folder_path, sub_sub_item)
                        self.files.extend([one_dataset_pcd_path])
                

        #print(self.files)
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, index) -> Any:
        """
        fps_points_scene_from_original: points after FPS [4096*3]
        fps_colors_scene_from_original: colors of points after FPS [4096*3]

        reference_obj: name of the reference obj, text
        reference_position: position of the reference obj in the PC [3]
        phrase: guidance text, text
        """
        pc_path = self.files[index] # "dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal/mask_Right.ply"
        json_path = os.path.join(pc_path.rsplit('/',2)[0], 'text_guidance.json')
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])

        intrinsics = np.array(
                [
                    [607.09912 / 2, 0.0, 636.85083 / 2],
                    [0.0, 607.05212 / 2, 367.35952 / 2],
                    [0.0, 0.0, 1.0],
                ]
            )

        # get scene pcd 
        scene_pcd = o3d.io.read_point_cloud(pc_path) # use red mask instead of mask.ply

        # scene pcd points and colors
        scene_pcd_points_ori = np.asarray(scene_pcd.points) / 1000 # convert to meters
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position, whose normal is not aligned with z-axis (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)

        # get color and depth
        rgb_img_path = os.path.join(pc_path.rsplit('/',1)[0], "no_obj/test_pbr/000000/rgb/000000.jpg")
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1)) 
        rgb_image = rgb_image / 255.0 # FIXME: Normalize the image to [0, 1]
        assert rgb_image.max() <= 1.0 and rgb_image.min() >= 0.0

        depth_path = pc_path.rsplit("/", 1)[0] + "/no_obj/test_pbr/000000/depth/000000.png"
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32)
        depth = depth / 1000.0 # depth shall be in the unit of meters
        depth[depth > 2] = 0 # remove the invalid depth values
        scene_depth_cloud, _ = backproject(depth, intrinsics, np.logical_and(depth > 0, depth < 2))
        scene_depth_cloud = visualize_points(scene_depth_cloud)
        scene_depth_cloud_points = np.asarray(scene_depth_cloud.points)


        ##########visualize 
        #scene_pcd.points = o3d.utility.Vector3dVector(scene_pcd_points)
        #o3d.visualization.draw_geometries([scene_depth_cloud, scene_pcd])
        #scene_depth_cloud = np.asarray(scene_depth_cloud.points)
        #assert depth.max() <= 2.0 and depth.min() >= 0.0
        #scene_points_perfect, scene_points_id = backproject(depth, intrinsics, depth > 0)
        
        # make the plane noraml align with z-axis
            # get the T_plane and plane_model
        T_plane, plane_model = get_tf_for_scene_rotation(scene_pcd_points)
        #scene_pcd_points_z = np.dot(scene_pcd_points, T_plane[:3, :3]) + T_plane[:3, 3] # norm z-axis
        
        # visualize
        #scene_pcd_z = o3d.geometry.PointCloud()
        #scene_pcd_ori = o3d.geometry.PointCloud()
        #scene_pcd_z.points = o3d.utility.Vector3dVector(scene_pcd_points_z)
        #scene_pcd_z.colors = o3d.utility.Vector3dVector(scene_pcd_colors) # use the same color as the original point cloud
        #scene_pcd_ori.points = o3d.utility.Vector3dVector(scene_pcd_points)
        #scene_pcd_ori.colors = o3d.utility.Vector3dVector(scene_pcd_colors)
        #coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        #o3d.visualization.draw_geometries([scene_pcd_z, coordinate_frame, scene_pcd_ori])

        # Convert points and colors to tensors
        scene_pcd_tensor = torch.tensor(scene_pcd_points, dtype=torch.float32).unsqueeze(0)
        scene_color_tensor = torch.tensor(scene_pcd_colors, dtype=torch.float32).unsqueeze(0)

        # Move to CUDA if necessary
        scene_pcd_tensor = scene_pcd_tensor.to("cuda")
        scene_color_tensor = scene_color_tensor.to("cuda")

        # Perform furthest point sampling (FPS)
        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor.contiguous(), 2048)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = scene_pcd_points[fps_indices_scene_np]
        fps_colors_scene_from_original = scene_pcd_colors[fps_indices_scene_np]

        # Get the label (grenn channel color) 
        fps_mask_mapped = fps_colors_scene_from_original[:,1].reshape(-1, 1) # get label from green mask directly
        
        # find the point position with the highest affordance (G channel value)
        max_green_index = np.argmax(fps_colors_scene_from_original[:,1])
        max_green_point = torch.tensor(fps_points_scene_from_original[max_green_index], dtype=torch.float32)

        # gt_pose_4d_min_bound find the min bound of the whole pc
        min_bound = np.append(np.min(fps_points_scene_from_original, axis=0), -180)
        max_bound = np.append(np.max(fps_points_scene_from_original, axis=0), 180)

        min_xyz_bound = np.min(scene_depth_cloud_points, axis=0) 
        max_xyz_bound = np.max(scene_depth_cloud_points, axis=0) 


        # sample points with affordance value higher than the threshold
        affordance_threshold = self.affordance_threshold
        fps_points_scene_affordance = fps_points_scene_from_original[fps_colors_scene_from_original[:,1] > affordance_threshold] # [F, 3]
        if fps_points_scene_affordance.shape[0] == 0:
            fps_points_scene_affordance = fps_points_scene_from_original
        fps_points_scene_affordance_perfect = fps_points_scene_affordance
        # fps_points_to_perfect_scene_dist = np.linalg.norm(fps_points_scene_affordance[:, None] - scene_points_perfect[None], axis=-1) # [F, N]
        # fps_points_to_perfect_scene_dist = np.min(fps_points_to_perfect_scene_dist, axis=1) # [F,]
        # fps_points_scene_affordance_perfect = fps_points_scene_affordance[fps_points_to_perfect_scene_dist < 0.1] # [F, 3]


        bound_affordance_z_mid = np.median(fps_points_scene_affordance_perfect[:, 2])
        min_bound_affordance = np.append(np.min(fps_points_scene_affordance_perfect, axis=0), -180)
        max_bound_affordance = np.append(np.max(fps_points_scene_affordance_perfect, axis=0), 180)
        min_bound_affordance[2] = bound_affordance_z_mid * 0.9
        max_bound_affordance[2] = bound_affordance_z_mid * 0.9
        min_xyz_bound[2] = bound_affordance_z_mid * 0.9
        max_xyz_bound[2] = bound_affordance_z_mid * 0.9


        # sample 512 points from fps_points_scene_affordance
        fps_points_scene_affordance = fps_points_scene_affordance[np.random.choice(fps_points_scene_affordance.shape[0], 512, replace=True)] # [512, 3]

        # sample num_samples from scene for non_condition
        #gt_non_cond = fps_points_scene_from_original[np.random.choice(fps_points_scene_from_original.shape[0], self.gt_pose_samples, replace=True)] # [num_samples, 3]
        x1 = min_xyz_bound[0]
        x2 = max_xyz_bound[0]
        y1 = min_xyz_bound[1]
        y2 = max_xyz_bound[1]
        z1 = min_xyz_bound[2]
        z2 = max_xyz_bound[2]

        bound_points = [
            [x1, y1, z1],
            [x2, y1, z1],
            [x2, y2, z1],
            [x1, y2, z1],
        ]
        points_per_group = self.gt_pose_samples // 4
        gt_non_cond = np.repeat(bound_points, points_per_group, axis=0)
        # get the object pc position (rotated, scaled, translated to the origin point)
        scene_id = pc_path.split('/')[2]
        obj_name = pc_path.split('/')[3]
        json_path = os.path.join("dataset/scene_gen/scene_mesh_json_kinect", f"{scene_id}.json") #e.g. "dataset/scene_mesh_json_kinect/id3.json"
        # find the key which includes obj_name
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for key in data.keys():
                if obj_name in key:
                    obj_json = data[key]
                    break
        obj_mesh_path = key
        obj_json = data[obj_mesh_path]
        obj_scale = obj_json[1]
        obj_rotation = torch.tensor(obj_json[2], dtype=torch.float32).unsqueeze(0)
        obj_mesh = trimesh.load(obj_mesh_path)
        #with open("GeoL_net/dataset_gen/obj_size.json", 'r') as f:
        #    obj_size = json.load(f)
        #obj_size = obj_size[obj_name]
        #obj_mesh_extent = obj_mesh.bounds.to_extents()
        #obj_scale = obj_size / obj_mesh_extent
        obj_mesh.apply_scale(obj_scale)
        obj_pc = obj_mesh.sample(512)

        # ################### DEBUG Acqurie the scene pcd ###################
        #scene_colors = rgb_image[:, scene_points_id[0], scene_points_id[1]].T
        #scene_pcd2 = o3d.geometry.PointCloud()
        #scene_pcd2.points = o3d.utility.Vector3dVector(scene_points_perfect)
        # scene_pcd2.colors = o3d.utility.Vector3dVector(scene_colors)

        #fps_pcd = o3d.geometry.PointCloud()
        #fps_pcd.points = o3d.utility.Vector3dVector(fps_points_scene_affordance_perfect)
        #fps_pcd.paint_uniform_color([1, 0, 0])

        #min_bound_vis = min_xyz_bound[:3]
        #max_bound_vis = max_xyz_bound[:3]
        #min_bound_vis_o3d = visualize_sphere_o3d(min_bound_vis, color=[0, 1, 0])
        #max_bound_vis_o3d = visualize_sphere_o3d(max_bound_vis, color=[0, 1, 0])
        #vis = [min_bound_vis_o3d, max_bound_vis_o3d, fps_pcd]
        #o3d.visualization.draw(vis)
        # ################### DEBUG Acqurie the scene pcd ###################


        # tsdf grid
        
        #intrinsics = np.array([[607.09912 / 2, 0.0, 636.85083 / 2],
        #        [0.0, 607.05212 / 2, 367.35952 / 2],
        #        [0.0, 0.0, 1.0]])
        #intrinsics[0, 2] = depth.shape[1] / 2
        #intrinsics[1, 2] = depth.shape[0] / 2
        #inv_intrinsics = np.linalg.pinv(intrinsics)
        #vol_bnds = np.zeros((3,2))
        #view_frust_pts = get_view_frustum(depth, intrinsics, np.eye(4))
        #vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1)).min()
        #vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1)).max()

        #tsdf = TSDFVolume(vol_bnds, voxel_dim=256, num_margin=30)
        #tsdf.integrate(rgb_image, depth, intrinsics, np.eye(4))
        #tsdf_grid = tsdf.get_tsdf_volume()
        

        # add noise
        noise_scale = 0.05
        noise = torch.randn(self.gt_pose_samples,4, dtype=torch.float32) * noise_scale

        noise_4d = noise
        noise_4d[:, 2:] = 0
        noise_xyR = noise[:, :3]
        noise_xyz = noise[:, :3]
        # Prepare the final sample
        sample = {
            
            "pc_position": fps_points_scene_from_original, #[num_points, 3]
            "affordance": fps_mask_mapped,  # [num_points, 1]New mask based on turbo colormap
            "object_name": obj_name,
            "object_pc_position": obj_pc, # [num_obj_points, 3]
            "gt_pose_4d": torch.cat((max_green_point, obj_rotation), dim=0).unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_4d, 
            "gt_pose_4d_min_bound":min_bound, #[4,]
            "gt_pose_4d_max_bound":max_bound, #[4,]
            "pc_position_xy_affordance": fps_points_scene_affordance[:, :2], #[num_affordance, 2]
            "gt_pose_xyR": torch.cat((max_green_point[:2], obj_rotation), dim=0).unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_xyR, #[pose_samples, 3]
            "gt_pose_xyR_min_bound": np.delete(min_bound_affordance, 2, axis=0), #[3,] 
            "gt_pose_xyR_max_bound": np.delete(max_bound_affordance, 2, axis=0), #[3,] 
            "gt_pose_xyz": max_green_point.unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_xyz, #[pose_samples, 3]
            'gt_pose_xy': max_green_point[:2].unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_xyR[:, :2], #[pose_samples, 2]
            "gt_pose_xyz_for_non_cond": gt_non_cond, #[pos_samples, 3]
            "gt_pose_xy_for_non_cond": gt_non_cond[:, :2], #[pos_samples, 2]
            #"gt_pose_xyz_min_bound": np.delete(min_bound_affordance, 3, axis=0), #[3,]
            #"gt_pose_xyz_max_bound": np.delete(max_bound_affordance, 3, axis=0), #[3,]
            "gt_pose_xyz_min_bound": min_xyz_bound, #[3,]
            "gt_pose_xyz_max_bound": max_xyz_bound, #[3,]
            "gt_pose_xy_min_bound": min_xyz_bound[:2], #[2,]
            "gt_pose_xy_max_bound": max_xyz_bound[:2], #[2,]
            #"tsdf_grid": tsdf_grid, 
            "depth": depth,
            "image": rgb_image,
            "T_plane": T_plane,
            "plane_model": plane_model
        }

        return sample

@registry.register_dataset(name="Pose_bproc_top")
class PoseDataset_top(Dataset):
    """ 
    The dataset for the pose estimation task
    
    Update:
        1. use topk point positions as the gt
    """

    def __init__(self,
                 split:str,
                 affordance_threshold:float = 0.001,
                 gt_pose_samples:int = 80,
                 root_dir = "dataset/scene_RGBD_mask_v2_kinect_cfg") -> None:
        super().__init__()

        self.split = split
        self.root_dir = root_dir
        self.folder_path = self.root_dir
        self.affordance_threshold = affordance_threshold

        self.files = []
        self.gt_pose_samples = gt_pose_samples
        items = os.listdir(self.root_dir)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item) # e.g. 'dataset/scene_RGBD_mask_v2/id164_1'
            #print(sub_folder_path)
            sub_items =[f for f in os.listdir(sub_folder_path) if os.path.isdir(os.path.join(sub_folder_path, f))] # e.g. ['printer_0001_normal', 'printer_0001_normal', 'printer_0001_normal', 'printer_0001_normal']

            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item) # 'dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal'
                for sub_sub_item in os.listdir(sub_sub_folder_path):
                    # if end with ply
                    #if sub_sub_item.endswith('_Behind.ply') or sub_sub_item.endswith('_Front.ply') or sub_sub_item.endswith('_Left.ply') or sub_sub_item.endswith('_Right.ply'):
                    if sub_sub_item.endswith('mask_Left.ply') or sub_sub_item.endswith('mask_Right.ply') or sub_sub_item.endswith('mask_Front.ply') or sub_sub_item.endswith('mask_Behind.ply')\
                        or sub_sub_item.endswith('mask_Left Front.ply') or sub_sub_item.endswith('mask_Left Behind.ply') or sub_sub_item.endswith('mask_Right Front.ply') or sub_sub_item.endswith('mask_Right Behind.ply'):
                        one_dataset_pcd_path = os.path.join(sub_sub_folder_path, sub_sub_item)
                        self.files.extend([one_dataset_pcd_path])
                

        #print(self.files)
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, index) -> Any:
        """
        fps_points_scene_from_original: points after FPS [4096*3]
        fps_colors_scene_from_original: colors of points after FPS [4096*3]

        reference_obj: name of the reference obj, text
        reference_position: position of the reference obj in the PC [3]
        phrase: guidance text, text
        """
        torch.manual_seed(42)
        np.random.seed(42)

        pc_path = self.files[index] # "dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal/mask_Right.ply"
        json_path = os.path.join(pc_path.rsplit('/',2)[0], 'text_guidance.json')
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])

        intrinsics = np.array(
                [
                    [607.09912 / 2, 0.0, 636.85083 / 2],
                    [0.0, 607.05212 / 2, 367.35952 / 2],
                    [0.0, 0.0, 1.0],
                ]
            )

        # get scene pcd 
        scene_pcd = o3d.io.read_point_cloud(pc_path) # use red mask instead of mask.ply

        # scene pcd points and colors
        scene_pcd_points_ori = np.asarray(scene_pcd.points) / 1000 # convert to meters
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position, whose normal is not aligned with z-axis (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)

        # get color and depth
        rgb_img_path = os.path.join(pc_path.rsplit('/',1)[0], "no_obj/test_pbr/000000/rgb/000000.jpg")
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1)) 
        rgb_image = rgb_image / 255.0 # FIXME: Normalize the image to [0, 1]
        assert rgb_image.max() <= 1.0 and rgb_image.min() >= 0.0

        depth_path = pc_path.rsplit("/", 1)[0] + "/no_obj/test_pbr/000000/depth/000000.png"
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32)
        depth = depth / 1000.0 # depth shall be in the unit of meters
        depth[depth > 2] = 0 # remove the invalid depth values
        scene_depth_cloud, _ = backproject(depth, intrinsics, np.logical_and(depth > 0, depth < 2))
        scene_depth_cloud = visualize_points(scene_depth_cloud)
        scene_depth_cloud_points = np.asarray(scene_depth_cloud.points)


        ##########visualize 
        #scene_pcd.points = o3d.utility.Vector3dVector(scene_pcd_points)
        #o3d.visualization.draw_geometries([scene_depth_cloud, scene_pcd])
        #scene_depth_cloud = np.asarray(scene_depth_cloud.points)
        #assert depth.max() <= 2.0 and depth.min() >= 0.0
        #scene_points_perfect, scene_points_id = backproject(depth, intrinsics, depth > 0)
        
        # make the plane noraml align with z-axis
            # get the T_plane and plane_model
        T_plane, plane_model = get_tf_for_scene_rotation(scene_pcd_points)
        #scene_pcd_points_z = np.dot(scene_pcd_points, T_plane[:3, :3]) + T_plane[:3, 3] # norm z-axis
        
        
        # Convert points and colors to tensors
        scene_pcd_tensor = torch.tensor(scene_pcd_points, dtype=torch.float32).unsqueeze(0)
        scene_color_tensor = torch.tensor(scene_pcd_colors, dtype=torch.float32).unsqueeze(0)

        # Move to CUDA if necessary
        scene_pcd_tensor = scene_pcd_tensor.to("cuda")
        scene_color_tensor = scene_color_tensor.to("cuda")

        # Perform furthest point sampling (FPS)
        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor.contiguous(), 2048)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = scene_pcd_points[fps_indices_scene_np]
        fps_colors_scene_from_original = scene_pcd_colors[fps_indices_scene_np]

        #fps_colors_scene_from_original_for_non_cond = torch.randn_like(fps_colors_scene_from_original) # get the non_cond color
        fps_colors_scene_from_original_for_non_cond = np.random.rand(*fps_colors_scene_from_original.shape)
        # Get the label (green channel color) 
        fps_mask_mapped = fps_colors_scene_from_original[:,1].reshape(-1, 1) # get label from green mask directly
        fps_mask_for_non_cond = fps_colors_scene_from_original_for_non_cond[:,1].reshape(-1, 1) # get label from green mask directly
        
        
        # find the point position with the highest affordance (G channel value)
        topk_green_points = get_top_affordance_points(fps_colors_scene_from_original, fps_points_scene_from_original, self.gt_pose_samples)
        topk_green_points_avg = torch.mean(topk_green_points, dim=0).repeat(self.gt_pose_samples, 1)
        topk_green_points_for_non_cond = get_top_affordance_points(fps_colors_scene_from_original_for_non_cond, fps_points_scene_from_original, self.gt_pose_samples)
        #max_green_index = np.argmax(fps_colors_scene_from_original[:,1])
        #max_green_point = torch.tensor(fps_points_scene_from_original[max_green_index], dtype=torch.float32)

        # gt_pose_4d_min_bound find the min bound of the whole pc
        min_bound = np.append(np.min(fps_points_scene_from_original, axis=0), -180)
        max_bound = np.append(np.max(fps_points_scene_from_original, axis=0), 180)

        min_xyz_bound = np.min(scene_depth_cloud_points, axis=0) 
        max_xyz_bound = np.max(scene_depth_cloud_points, axis=0) 


        # sample points with affordance value higher than the threshold
        affordance_threshold = self.affordance_threshold
        fps_points_scene_affordance = fps_points_scene_from_original[fps_colors_scene_from_original[:,1] > affordance_threshold] # [F, 3]
        if fps_points_scene_affordance.shape[0] == 0:
            fps_points_scene_affordance = fps_points_scene_from_original
        fps_points_scene_affordance_perfect = fps_points_scene_affordance



        bound_affordance_z_mid = np.median(fps_points_scene_affordance_perfect[:, 2])
        min_bound_affordance = np.append(np.min(fps_points_scene_affordance_perfect, axis=0), -180)
        max_bound_affordance = np.append(np.max(fps_points_scene_affordance_perfect, axis=0), 180)
        min_bound_affordance[2] = bound_affordance_z_mid * 0.9
        max_bound_affordance[2] = bound_affordance_z_mid * 0.9
        min_xyz_bound[2] = bound_affordance_z_mid * 0.9
        max_xyz_bound[2] = bound_affordance_z_mid * 0.9


        # sample 512 points from fps_points_scene_affordance
        fps_points_scene_affordance = fps_points_scene_affordance[np.random.choice(fps_points_scene_affordance.shape[0], 512, replace=True)] # [512, 3]

        # sample num_samples from scene for non_condition
        #gt_non_cond = fps_points_scene_from_original[np.random.choice(fps_points_scene_from_original.shape[0], self.gt_pose_samples, replace=True)] # [num_samples, 3]
        # x1 = min_xyz_bound[0]
        # x2 = max_xyz_bound[0]
        # y1 = min_xyz_bound[1]
        # y2 = max_xyz_bound[1]
        # z1 = min_xyz_bound[2]
        # z2 = max_xyz_bound[2]

        # bound_points = [
        #     [x1, y1, z1],
        #     [x2, y1, z1],
        #     [x2, y2, z1],
        #     [x1, y2, z1],
        # ]
        # points_per_group = self.gt_pose_samples // 4
        # gt_non_cond = np.repeat(bound_points, points_per_group, axis=0)
        # get the object pc position (rotated, scaled, translated to the origin point)
        scene_id = pc_path.split('/')[2]
        obj_name = pc_path.split('/')[3]
        json_path = os.path.join("dataset/scene_gen/scene_mesh_json_aug", f"{scene_id}.json") #e.g. "dataset/scene_mesh_json_kinect/id3.json"
        # find the key which includes obj_name
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for key in data.keys():
                if obj_name in key:
                    obj_json = data[key]
                    break
        obj_mesh_path = key
        obj_json = data[obj_mesh_path]
        obj_scale = obj_json[1]
        obj_rotation = torch.tensor(obj_json[2], dtype=torch.float32).unsqueeze(0)
        obj_mesh = trimesh.load(obj_mesh_path)
        #with open("GeoL_net/dataset_gen/obj_size.json", 'r') as f:
        #    obj_size = json.load(f)
        #obj_size = obj_size[obj_name]
        #obj_mesh_extent = obj_mesh.bounds.to_extents()
        #obj_scale = obj_size / obj_mesh_extent
        obj_mesh.apply_scale(obj_scale)
        obj_pc = obj_mesh.sample(512)

        
        # add noise
        noise_scale = 0.05
        noise = torch.randn(self.gt_pose_samples,4, dtype=torch.float32) * noise_scale

        noise_4d = noise
        noise_4d[:, 2:] = 0
        noise_xyR = noise[:, :3]
        noise_xyz = noise[:, :3]
        # Prepare the final sample
        sample = {
            
            "pc_position": fps_points_scene_from_original, #[num_points, 3]
            "affordance": fps_mask_mapped,  # [num_points, 1]New mask based on turbo colormap
            "affordance_for_non_cond": fps_mask_for_non_cond,
            "object_name": obj_name,
            "object_pc_position": obj_pc, # [num_obj_points, 3]
            #"gt_pose_4d": torch.cat((max_green_point, obj_rotation), dim=0).unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_4d, 
            "gt_pose_4d_min_bound":min_bound, #[4,]
            "gt_pose_4d_max_bound":max_bound, #[4,]
            "pc_position_xy_affordance": fps_points_scene_affordance[:, :2], #[num_affordance, 2]
            #"gt_pose_xyR": torch.cat((max_green_point[:2], obj_rotation), dim=0).unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_xyR, #[pose_samples, 3]
            "gt_pose_xyR_min_bound": np.delete(min_bound_affordance, 2, axis=0), #[3,] 
            "gt_pose_xyR_max_bound": np.delete(max_bound_affordance, 2, axis=0), #[3,] 
            "gt_pose_xyz": topk_green_points, #[pose_samples, 3] same point repeated 80
            "gt_pose_xy": topk_green_points[:, :2], #[pose_samples, 2]
            "gt_pose_xyz_for_non_cond": topk_green_points_for_non_cond, #[pos_samples, 3]
            "gt_pose_xy_for_non_cond": topk_green_points_for_non_cond[:, :2], #[pos_samples, 2]
            #"gt_pose_xyz_min_bound": np.delete(min_bound_affordance, 3, axis=0), #[3,]
            #"gt_pose_xyz_max_bound": np.delete(max_bound_affordance, 3, axis=0), #[3,]
            "gt_pose_xyz_min_bound": min_xyz_bound, #[3,]
            "gt_pose_xyz_max_bound": max_xyz_bound, #[3,]
            "gt_pose_xy_min_bound": min_xyz_bound[:2], #[2,]
            "gt_pose_xy_max_bound": max_xyz_bound[:2], #[2,]
            #"tsdf_grid": tsdf_grid, 
            "depth": depth,
            "image": rgb_image,
            "T_plane": T_plane,
            "plane_model": plane_model
        }

        return sample

@registry.register_dataset(name="Pose_bproc_top_plane")
class PoseDataset_top_plane(Dataset):
    """ 
    The dataset for the pose estimation task
    
    Update:
        1. use 'fit the plane' function to get the plane and plane model
    """

    def __init__(self,
                 split:str,
                 affordance_threshold:float = 0.001,
                 gt_pose_samples:int = 80,
                 root_dir = "dataset/scene_RGBD_mask_v2_kinect_cfg") -> None:
        super().__init__()

        self.split = split
        self.root_dir = root_dir
        self.folder_path = self.root_dir
        self.affordance_threshold = affordance_threshold

        self.files = []
        self.gt_pose_samples = gt_pose_samples
        items = os.listdir(self.root_dir)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item) # e.g. 'dataset/scene_RGBD_mask_v2/id164_1'
            #print(sub_folder_path)
            sub_items =[f for f in os.listdir(sub_folder_path) if os.path.isdir(os.path.join(sub_folder_path, f))] # e.g. ['printer_0001_normal', 'printer_0001_normal', 'printer_0001_normal', 'printer_0001_normal']

            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item) # 'dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal'
                for sub_sub_item in os.listdir(sub_sub_folder_path):
                    # if end with ply
                    #if sub_sub_item.endswith('_Behind.ply') or sub_sub_item.endswith('_Front.ply') or sub_sub_item.endswith('_Left.ply') or sub_sub_item.endswith('_Right.ply'):
                    if sub_sub_item.endswith('mask_Left.ply') or sub_sub_item.endswith('mask_Right.ply') or sub_sub_item.endswith('mask_Front.ply') or sub_sub_item.endswith('mask_Behind.ply')\
                        or sub_sub_item.endswith('mask_Left Front.ply') or sub_sub_item.endswith('mask_Left Behind.ply') or sub_sub_item.endswith('mask_Right Front.ply') or sub_sub_item.endswith('mask_Right Behind.ply'):
                        one_dataset_pcd_path = os.path.join(sub_sub_folder_path, sub_sub_item)
                        self.files.extend([one_dataset_pcd_path])
                

        #print(self.files)
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, index) -> Any:
        """
        fps_points_scene_from_original: points after FPS [4096*3]
        fps_colors_scene_from_original: colors of points after FPS [4096*3]

        reference_obj: name of the reference obj, text
        reference_position: position of the reference obj in the PC [3]
        phrase: guidance text, text
        """
        pc_path = self.files[index] # "dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal/mask_Right.ply"
        json_path = os.path.join(pc_path.rsplit('/',2)[0], 'text_guidance.json')
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])

        intrinsics = np.array(
                [
                    [607.09912 / 2, 0.0, 636.85083 / 2],
                    [0.0, 607.05212 / 2, 367.35952 / 2],
                    [0.0, 0.0, 1.0],
                ]
            )

        # get scene pcd 
        scene_pcd = o3d.io.read_point_cloud(pc_path) # use red mask instead of mask.ply

        # scene pcd points and colors
        scene_pcd_points_ori = np.asarray(scene_pcd.points) / 1000 # convert to meters
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position, whose normal is not aligned with z-axis (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)

        # get color and depth
        rgb_img_path = os.path.join(pc_path.rsplit('/',1)[0], "no_obj/test_pbr/000000/rgb/000000.jpg")
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1)) 
        rgb_image = rgb_image / 255.0 # FIXME: Normalize the image to [0, 1]
        assert rgb_image.max() <= 1.0 and rgb_image.min() >= 0.0

        depth_path = pc_path.rsplit("/", 1)[0] + "/no_obj/test_pbr/000000/depth/000000.png"
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32)
        depth = depth / 1000.0 # depth shall be in the unit of meters
        depth[depth > 2] = 0 # remove the invalid depth values
        scene_depth_cloud, _ = backproject(depth, intrinsics, np.logical_and(depth > 0, depth < 2))
        scene_depth_cloud = visualize_points(scene_depth_cloud)
        scene_depth_cloud_points = np.asarray(scene_depth_cloud.points)


        ##########visualize 
        #scene_pcd.points = o3d.utility.Vector3dVector(scene_pcd_points)
        #o3d.visualization.draw_geometries([scene_depth_cloud, scene_pcd])
        #scene_depth_cloud = np.asarray(scene_depth_cloud.points)
        #assert depth.max() <= 2.0 and depth.min() >= 0.0
        #scene_points_perfect, scene_points_id = backproject(depth, intrinsics, depth > 0)
        
        # make the plane noraml align with z-axis
            # get the T_plane and plane_model
        T_plane, plane_model = get_tf_for_scene_rotation(scene_pcd_points)
        scene_pcd_points_z = np.dot(scene_pcd_points, T_plane[:3, :3]) + T_plane[:3, 3] # norm z-axis
        T_plane_align, plane_model_align = get_tf_for_scene_rotation(scene_pcd_points_z)

        # visualize fit plane
        scene_pcd_z = o3d.geometry.PointCloud()
        scene_pcd_ori = o3d.geometry.PointCloud()
        scene_pcd_z.points = o3d.utility.Vector3dVector(scene_pcd_points_z)
        scene_pcd_z.colors = o3d.utility.Vector3dVector(scene_pcd_colors) # use the same color as the original point cloud
        scene_pcd_ori.points = o3d.utility.Vector3dVector(scene_pcd_points)
        scene_pcd_ori.colors = o3d.utility.Vector3dVector(scene_pcd_colors)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        plane_height = -1 * plane_model_align[3] / plane_model_align[2]
        plane_height_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1).translate([0, 0, plane_height]).paint_uniform_color([1, 0, 0]).compute_vertex_normals()
        plane_height_sphere_2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1).translate([0, 1, plane_height]).paint_uniform_color([0, 1, 0]).compute_vertex_normals()
        plane_height_sphere_3 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1).translate([1, 0, plane_height]).paint_uniform_color([0, 0, 1]).compute_vertex_normals()
        o3d.visualization.draw_geometries([scene_pcd_z, coordinate_frame, scene_pcd_ori, plane_height_sphere, plane_height_sphere_2, plane_height_sphere_3])

        # Convert points and colors to tensors
        scene_pcd_tensor = torch.tensor(scene_pcd_points, dtype=torch.float32).unsqueeze(0)
        scene_color_tensor = torch.tensor(scene_pcd_colors, dtype=torch.float32).unsqueeze(0)

        # Move to CUDA if necessary
        scene_pcd_tensor = scene_pcd_tensor.to("cuda")
        scene_color_tensor = scene_color_tensor.to("cuda")

        # Perform furthest point sampling (FPS)
        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor.contiguous(), 2048)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = scene_pcd_points[fps_indices_scene_np]
        fps_colors_scene_from_original = scene_pcd_colors[fps_indices_scene_np]

        # Get the label (green channel color) 
        fps_mask_mapped = fps_colors_scene_from_original[:,1].reshape(-1, 1) # get label from green mask directly
        
        # find the point position with the highest affordance (G channel value)
        topk_green_points = get_top_affordance_points(fps_colors_scene_from_original, fps_points_scene_from_original, self.gt_pose_samples)
        #max_green_index = np.argmax(fps_colors_scene_from_original[:,1])
        #max_green_point = torch.tensor(fps_points_scene_from_original[max_green_index], dtype=torch.float32)

        # gt_pose_4d_min_bound find the min bound of the whole pc
        min_bound = np.append(np.min(fps_points_scene_from_original, axis=0), -180)
        max_bound = np.append(np.max(fps_points_scene_from_original, axis=0), 180)

        min_xyz_bound = np.min(scene_depth_cloud_points, axis=0) 
        max_xyz_bound = np.max(scene_depth_cloud_points, axis=0) 


        # sample points with affordance value higher than the threshold
        affordance_threshold = self.affordance_threshold
        fps_points_scene_affordance = fps_points_scene_from_original[fps_colors_scene_from_original[:,1] > affordance_threshold] # [F, 3]
        if fps_points_scene_affordance.shape[0] == 0:
            fps_points_scene_affordance = fps_points_scene_from_original
        fps_points_scene_affordance_perfect = fps_points_scene_affordance



        bound_affordance_z_mid = np.median(fps_points_scene_affordance_perfect[:, 2])
        min_bound_affordance = np.append(np.min(fps_points_scene_affordance_perfect, axis=0), -180)
        max_bound_affordance = np.append(np.max(fps_points_scene_affordance_perfect, axis=0), 180)
        min_bound_affordance[2] = bound_affordance_z_mid * 0.9
        max_bound_affordance[2] = bound_affordance_z_mid * 0.9
        min_xyz_bound[2] = bound_affordance_z_mid * 0.9
        max_xyz_bound[2] = bound_affordance_z_mid * 0.9


        # sample 512 points from fps_points_scene_affordance
        fps_points_scene_affordance = fps_points_scene_affordance[np.random.choice(fps_points_scene_affordance.shape[0], 512, replace=True)] # [512, 3]

        # sample num_samples from scene for non_condition
        #gt_non_cond = fps_points_scene_from_original[np.random.choice(fps_points_scene_from_original.shape[0], self.gt_pose_samples, replace=True)] # [num_samples, 3]
        x1 = min_xyz_bound[0]
        x2 = max_xyz_bound[0]
        y1 = min_xyz_bound[1]
        y2 = max_xyz_bound[1]
        z1 = min_xyz_bound[2]
        z2 = max_xyz_bound[2]

        bound_points = [
            [x1, y1, z1],
            [x2, y1, z1],
            [x2, y2, z1],
            [x1, y2, z1],
        ]
        points_per_group = self.gt_pose_samples // 4
        gt_non_cond = np.repeat(bound_points, points_per_group, axis=0)
        # get the object pc position (rotated, scaled, translated to the origin point)
        scene_id = pc_path.split('/')[2]
        obj_name = pc_path.split('/')[3]
        json_path = os.path.join("dataset/scene_gen/scene_mesh_json_kinect", f"{scene_id}.json") #e.g. "dataset/scene_mesh_json_kinect/id3.json"
        # find the key which includes obj_name
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for key in data.keys():
                if obj_name in key:
                    obj_json = data[key]
                    break
        obj_mesh_path = key
        obj_json = data[obj_mesh_path]
        obj_scale = obj_json[1]
        obj_rotation = torch.tensor(obj_json[2], dtype=torch.float32).unsqueeze(0)
        obj_mesh = trimesh.load(obj_mesh_path)
        #with open("GeoL_net/dataset_gen/obj_size.json", 'r') as f:
        #    obj_size = json.load(f)
        #obj_size = obj_size[obj_name]
        #obj_mesh_extent = obj_mesh.bounds.to_extents()
        #obj_scale = obj_size / obj_mesh_extent
        obj_mesh.apply_scale(obj_scale)
        obj_pc = obj_mesh.sample(512)

        
        # add noise
        noise_scale = 0.05
        noise = torch.randn(self.gt_pose_samples,4, dtype=torch.float32) * noise_scale

        noise_4d = noise
        noise_4d[:, 2:] = 0
        noise_xyR = noise[:, :3]
        noise_xyz = noise[:, :3]
        # Prepare the final sample
        sample = {
            
            "pc_position": fps_points_scene_from_original, #[num_points, 3]
            "affordance": fps_mask_mapped,  # [num_points, 1]New mask based on turbo colormap
            "object_name": obj_name,
            "object_pc_position": obj_pc, # [num_obj_points, 3]
            #"gt_pose_4d": torch.cat((max_green_point, obj_rotation), dim=0).unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_4d, 
            "gt_pose_4d_min_bound":min_bound, #[4,]
            "gt_pose_4d_max_bound":max_bound, #[4,]
            "pc_position_xy_affordance": fps_points_scene_affordance[:, :2], #[num_affordance, 2]
            #"gt_pose_xyR": torch.cat((max_green_point[:2], obj_rotation), dim=0).unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_xyR, #[pose_samples, 3]
            "gt_pose_xyR_min_bound": np.delete(min_bound_affordance, 2, axis=0), #[3,] 
            "gt_pose_xyR_max_bound": np.delete(max_bound_affordance, 2, axis=0), #[3,] 
            "gt_pose_xyz": topk_green_points, #[pose_samples, 3]
            "gt_pose_xyz_for_non_cond": gt_non_cond, #[pos_samples, 3]
            #"gt_pose_xyz_min_bound": np.delete(min_bound_affordance, 3, axis=0), #[3,]
            #"gt_pose_xyz_max_bound": np.delete(max_bound_affordance, 3, axis=0), #[3,]
            "gt_pose_xyz_min_bound": min_xyz_bound, #[3,]
            "gt_pose_xyz_max_bound": max_xyz_bound, #[3,]
            #"tsdf_grid": tsdf_grid, 
            "depth": depth,
            "image": rgb_image,
            "T_plane": T_plane, # from the original point cloud
            "plane_model": plane_model,
            "T_plane_align": T_plane_align, # from the plane aligned with z-axis
            "plane_model_align": plane_model_align
        }

        return sample
    
@registry.register_dataset(name="Pose_bproc_3D")
class PoseDataset_3D(Dataset):
    """ 
    The dataset for the pose estimation task
    
    Update:
        1. add the offset to the x,y in gt_pose_4d
    """

    def __init__(self,
                 split:str,
                 affordance_threshold:float = 0.001,
                 gt_pose_samples:int = 80,
                 root_dir = "dataset/scene_RGBD_mask_v2_kinect_cfg") -> None:
        super().__init__()

        self.split = split
        self.root_dir = root_dir
        self.folder_path = self.root_dir
        self.affordance_threshold = affordance_threshold

        self.files = []
        self.gt_pose_samples = gt_pose_samples
        items = os.listdir(self.root_dir)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item) # e.g. 'dataset/scene_RGBD_mask_v2/id164_1'
            #print(sub_folder_path)
            sub_items =[f for f in os.listdir(sub_folder_path) if os.path.isdir(os.path.join(sub_folder_path, f))] # e.g. ['printer_0001_normal', 'printer_0001_normal', 'printer_0001_normal', 'printer_0001_normal']

            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item) # 'dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal'
                for sub_sub_item in os.listdir(sub_sub_folder_path):
                    # if end with ply
                    #if sub_sub_item.endswith('_Behind.ply') or sub_sub_item.endswith('_Front.ply') or sub_sub_item.endswith('_Left.ply') or sub_sub_item.endswith('_Right.ply'):
                    if sub_sub_item.endswith('mask_Left.ply') or sub_sub_item.endswith('mask_Right.ply') or sub_sub_item.endswith('mask_Front.ply') or sub_sub_item.endswith('mask_Behind.ply')\
                        or sub_sub_item.endswith('mask_Left Front.ply') or sub_sub_item.endswith('mask_Left Behind.ply') or sub_sub_item.endswith('mask_Right Front.ply') or sub_sub_item.endswith('mask_Right Behind.ply'):
                        one_dataset_pcd_path = os.path.join(sub_sub_folder_path, sub_sub_item)
                        self.files.extend([one_dataset_pcd_path])
                

        #print(self.files)
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, index) -> Any:
        """
        fps_points_scene_from_original: points after FPS [4096*3]
        fps_colors_scene_from_original: colors of points after FPS [4096*3]

        reference_obj: name of the reference obj, text
        reference_position: position of the reference obj in the PC [3]
        phrase: guidance text, text
        """
        pc_path = self.files[index] # "dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal/mask_Right.ply"
        json_path = os.path.join(pc_path.rsplit('/',2)[0], 'text_guidance.json')
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])

        intrinsics = np.array(
                [
                    [607.09912 / 2, 0.0, 636.85083 / 2],
                    [0.0, 607.05212 / 2, 367.35952 / 2],
                    [0.0, 0.0, 1.0],
                ]
            )

        # get scene pcd 
        scene_pcd = o3d.io.read_point_cloud(pc_path) # use red mask instead of mask.ply

        # scene pcd points and colors
        scene_pcd_points_ori = np.asarray(scene_pcd.points) / 1000 # convert to meters
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position, whose normal is not aligned with z-axis (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)

        # get color and depth
        rgb_img_path = os.path.join(pc_path.rsplit('/',1)[0], "no_obj/test_pbr/000000/rgb/000000.jpg")
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1)) 
        rgb_image = rgb_image / 255.0 # FIXME: Normalize the image to [0, 1]
        assert rgb_image.max() <= 1.0 and rgb_image.min() >= 0.0

        depth_path = pc_path.rsplit("/", 1)[0] + "/no_obj/test_pbr/000000/depth/000000.png"
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32)
        depth = depth / 1000.0 # depth shall be in the unit of meters
        depth[depth > 2] = 0 # remove the invalid depth values
        scene_depth_cloud, _ = backproject(depth, intrinsics, np.logical_and(depth > 0, depth < 2))
        scene_depth_cloud = visualize_points(scene_depth_cloud)
        scene_depth_cloud_points = np.asarray(scene_depth_cloud.points)


        ##########visualize 
        #scene_pcd.points = o3d.utility.Vector3dVector(scene_pcd_points)
        #o3d.visualization.draw_geometries([scene_depth_cloud, scene_pcd])
        #scene_depth_cloud = np.asarray(scene_depth_cloud.points)
        #assert depth.max() <= 2.0 and depth.min() >= 0.0
        #scene_points_perfect, scene_points_id = backproject(depth, intrinsics, depth > 0)
        
        # make the plane noraml align with z-axis
            # get the T_plane and plane_model
        T_plane, plane_model = get_tf_for_scene_rotation(scene_pcd_points)
        #scene_pcd_points_z = np.dot(scene_pcd_points, T_plane[:3, :3]) + T_plane[:3, 3] # norm z-axis
        
        # visualize
        #scene_pcd_z = o3d.geometry.PointCloud()
        #scene_pcd_ori = o3d.geometry.PointCloud()
        #scene_pcd_z.points = o3d.utility.Vector3dVector(scene_pcd_points_z)
        #scene_pcd_z.colors = o3d.utility.Vector3dVector(scene_pcd_colors) # use the same color as the original point cloud
        #scene_pcd_ori.points = o3d.utility.Vector3dVector(scene_pcd_points)
        #scene_pcd_ori.colors = o3d.utility.Vector3dVector(scene_pcd_colors)
        #coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        #o3d.visualization.draw_geometries([scene_pcd_z, coordinate_frame, scene_pcd_ori])

        # Convert points and colors to tensors
        scene_pcd_tensor = torch.tensor(scene_pcd_points, dtype=torch.float32).unsqueeze(0)
        scene_color_tensor = torch.tensor(scene_pcd_colors, dtype=torch.float32).unsqueeze(0)

        # Move to CUDA if necessary
        scene_pcd_tensor = scene_pcd_tensor.to("cuda")
        scene_color_tensor = scene_color_tensor.to("cuda")

        # Perform furthest point sampling (FPS)
        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor.contiguous(), 2048)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = scene_pcd_points[fps_indices_scene_np]
        fps_colors_scene_from_original = scene_pcd_colors[fps_indices_scene_np]

        # Get the label (grenn channel color) 
        fps_mask_mapped = fps_colors_scene_from_original[:,1].reshape(-1, 1) # get label from green mask directly
        
        # find the point position with the highest affordance (G channel value)
        max_green_index = np.argmax(fps_colors_scene_from_original[:,1])
        max_green_point = torch.tensor(fps_points_scene_from_original[max_green_index], dtype=torch.float32)

        # gt_pose_4d_min_bound find the min bound of the whole pc
        min_bound = np.append(np.min(fps_points_scene_from_original, axis=0), -180)
        max_bound = np.append(np.max(fps_points_scene_from_original, axis=0), 180)

        min_xyz_bound = np.min(scene_depth_cloud_points, axis=0) 
        max_xyz_bound = np.max(scene_depth_cloud_points, axis=0) 


        # sample points with affordance value higher than the threshold
        affordance_threshold = self.affordance_threshold
        fps_points_scene_affordance = fps_points_scene_from_original[fps_colors_scene_from_original[:,1] > affordance_threshold] # [F, 3]
        if fps_points_scene_affordance.shape[0] == 0:
            fps_points_scene_affordance = fps_points_scene_from_original
        fps_points_scene_affordance_perfect = fps_points_scene_affordance
        # fps_points_to_perfect_scene_dist = np.linalg.norm(fps_points_scene_affordance[:, None] - scene_points_perfect[None], axis=-1) # [F, N]
        # fps_points_to_perfect_scene_dist = np.min(fps_points_to_perfect_scene_dist, axis=1) # [F,]
        # fps_points_scene_affordance_perfect = fps_points_scene_affordance[fps_points_to_perfect_scene_dist < 0.1] # [F, 3]


        bound_affordance_z_mid = np.median(fps_points_scene_affordance_perfect[:, 2])
        min_bound_affordance = np.append(np.min(fps_points_scene_affordance_perfect, axis=0), -180)
        max_bound_affordance = np.append(np.max(fps_points_scene_affordance_perfect, axis=0), 180)
        min_bound_affordance[2] = bound_affordance_z_mid * 0.9
        max_bound_affordance[2] = bound_affordance_z_mid * 0.9
        #min_xyz_bound[2] = bound_affordance_z_mid * 0.9
        #max_xyz_bound[2] = bound_affordance_z_mid * 0.9


        # sample 512 points from fps_points_scene_affordance
        fps_points_scene_affordance = fps_points_scene_affordance[np.random.choice(fps_points_scene_affordance.shape[0], 512, replace=True)] # [512, 3]

        # sample num_samples from scene for non_condition
        gt_non_cond = fps_points_scene_from_original[np.random.choice(fps_points_scene_from_original.shape[0], self.gt_pose_samples, replace=True)] # [num_samples, 3]

        # get the object pc position (rotated, scaled, translated to the origin point)
        scene_id = pc_path.split('/')[2]
        obj_name = pc_path.split('/')[3]
        json_path = os.path.join("dataset/scene_gen/scene_mesh_json_kinect", f"{scene_id}.json") #e.g. "dataset/scene_mesh_json_kinect/id3.json"
        # find the key which includes obj_name
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for key in data.keys():
                if obj_name in key:
                    obj_json = data[key]
                    break
        obj_mesh_path = key
        obj_json = data[obj_mesh_path]
        obj_scale = obj_json[1]
        obj_rotation = torch.tensor(obj_json[2], dtype=torch.float32).unsqueeze(0)
        obj_mesh = trimesh.load(obj_mesh_path)
        #with open("GeoL_net/dataset_gen/obj_size.json", 'r') as f:
        #    obj_size = json.load(f)
        #obj_size = obj_size[obj_name]
        #obj_mesh_extent = obj_mesh.bounds.to_extents()
        #obj_scale = obj_size / obj_mesh_extent
        obj_mesh.apply_scale(obj_scale)
        obj_pc = obj_mesh.sample(512)

        # ################### DEBUG Acqurie the scene pcd ###################
        #scene_colors = rgb_image[:, scene_points_id[0], scene_points_id[1]].T
        #scene_pcd2 = o3d.geometry.PointCloud()
        #scene_pcd2.points = o3d.utility.Vector3dVector(scene_points_perfect)
        # scene_pcd2.colors = o3d.utility.Vector3dVector(scene_colors)

        #fps_pcd = o3d.geometry.PointCloud()
        #fps_pcd.points = o3d.utility.Vector3dVector(fps_points_scene_affordance_perfect)
        #fps_pcd.paint_uniform_color([1, 0, 0])

        #min_bound_vis = min_xyz_bound[:3]
        #max_bound_vis = max_xyz_bound[:3]
        #min_bound_vis_o3d = visualize_sphere_o3d(min_bound_vis, color=[0, 1, 0])
        #max_bound_vis_o3d = visualize_sphere_o3d(max_bound_vis, color=[0, 1, 0])
        #vis = [min_bound_vis_o3d, max_bound_vis_o3d, fps_pcd]
        #o3d.visualization.draw(vis)
        # ################### DEBUG Acqurie the scene pcd ###################


        # tsdf grid
        
        #intrinsics = np.array([[607.09912 / 2, 0.0, 636.85083 / 2],
        #        [0.0, 607.05212 / 2, 367.35952 / 2],
        #        [0.0, 0.0, 1.0]])
        #intrinsics[0, 2] = depth.shape[1] / 2
        #intrinsics[1, 2] = depth.shape[0] / 2
        #inv_intrinsics = np.linalg.pinv(intrinsics)
        #vol_bnds = np.zeros((3,2))
        #view_frust_pts = get_view_frustum(depth, intrinsics, np.eye(4))
        #vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1)).min()
        #vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1)).max()

        #tsdf = TSDFVolume(vol_bnds, voxel_dim=256, num_margin=30)
        #tsdf.integrate(rgb_image, depth, intrinsics, np.eye(4))
        #tsdf_grid = tsdf.get_tsdf_volume()
        

        # add noise
        noise_scale = 0.10
        noise = torch.randn(self.gt_pose_samples,4, dtype=torch.float32) * noise_scale

        noise_4d = noise
        noise_4d[:, 2:] = 0
        noise_xyR = noise[:, :3]
        noise_xyz = noise[:, :3]
        # Prepare the final sample
        sample = {
            
            "pc_position": fps_points_scene_from_original, #[num_points, 3]
            "affordance": fps_mask_mapped,  # [num_points, 1]New mask based on turbo colormap
            "object_name": obj_name,
            "object_pc_position": obj_pc, # [num_obj_points, 3]
            "gt_pose_4d": torch.cat((max_green_point, obj_rotation), dim=0).unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_4d, 
            "gt_pose_4d_min_bound":min_bound, #[4,]
            "gt_pose_4d_max_bound":max_bound, #[4,]
            "pc_position_xy_affordance": fps_points_scene_affordance[:, :2], #[num_affordance, 2]
            "gt_pose_xyR": torch.cat((max_green_point[:2], obj_rotation), dim=0).unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_xyR, #[pose_samples, 3]
            "gt_pose_xyR_min_bound": np.delete(min_bound_affordance, 2, axis=0), #[3,] 
            "gt_pose_xyR_max_bound": np.delete(max_bound_affordance, 2, axis=0), #[3,] 
            "gt_pose_xyz": max_green_point.unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_xyz, #[pose_samples, 3]
            "gt_pose_xyz_for_non_cond": gt_non_cond, #[pos_samples, 3]
            #"gt_pose_xyz_min_bound": np.delete(min_bound_affordance, 3, axis=0), #[3,]
            #"gt_pose_xyz_max_bound": np.delete(max_bound_affordance, 3, axis=0), #[3,]
            "gt_pose_xyz_min_bound": min_xyz_bound, #[3,] # NOTE: now it's xyz-bound
            "gt_pose_xyz_max_bound": max_xyz_bound, #[3,]
            #"tsdf_grid": tsdf_grid, 
            "depth": depth,
            "image": rgb_image,
            "T_plane": T_plane,
            "plane_model": plane_model
        }

        return sample

@registry.register_dataset(name="Pose_bproc_fit_plane")
class PoseDataset_plane(Dataset):
    """ 
    The dataset for the pose estimation task
    
    Update:
        1. fit the plane
        2. z-axis of world coordinate align with the plane normal
    """

    def __init__(self,
                 split:str,
                 affordance_threshold:float = 0.001,
                 gt_pose_samples:int = 80,
                 root_dir = "dataset/scene_RGBD_mask_v2_kinect_cfg") -> None:
        super().__init__()

        self.split = split
        self.root_dir = root_dir
        self.folder_path = self.root_dir
        self.affordance_threshold = affordance_threshold

        self.files = []
        self.gt_pose_samples = gt_pose_samples
        items = os.listdir(self.root_dir)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item) # e.g. 'dataset/scene_RGBD_mask_v2/id164_1'
            #print(sub_folder_path)
            sub_items =[f for f in os.listdir(sub_folder_path) if os.path.isdir(os.path.join(sub_folder_path, f))] # e.g. ['printer_0001_normal', 'printer_0001_normal', 'printer_0001_normal', 'printer_0001_normal']

            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item) # 'dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal'
                for sub_sub_item in os.listdir(sub_sub_folder_path):
                    # if end with ply
                    #if sub_sub_item.endswith('_Behind.ply') or sub_sub_item.endswith('_Front.ply') or sub_sub_item.endswith('_Left.ply') or sub_sub_item.endswith('_Right.ply'):
                    if sub_sub_item.endswith('mask_Left.ply') or sub_sub_item.endswith('mask_Right.ply') or sub_sub_item.endswith('mask_Front.ply') or sub_sub_item.endswith('mask_Behind.ply')\
                        or sub_sub_item.endswith('mask_Left Front.ply') or sub_sub_item.endswith('mask_Left Behind.ply') or sub_sub_item.endswith('mask_Right Front.ply') or sub_sub_item.endswith('mask_Right Behind.ply'):
                        one_dataset_pcd_path = os.path.join(sub_sub_folder_path, sub_sub_item)
                        self.files.extend([one_dataset_pcd_path])
                

        #print(self.files)
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, index) -> Any:

        pc_path = self.files[index] # "dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal/mask_Right.ply"
        json_path = os.path.join(pc_path.rsplit('/',2)[0], 'text_guidance.json')
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])

        intrinsics = np.array(
                [
                    [607.09912 / 2, 0.0, 636.85083 / 2],
                    [0.0, 607.05212 / 2, 367.35952 / 2],
                    [0.0, 0.0, 1.0],
                ]
            )

        # get scene pcd 
        scene_pcd = o3d.io.read_point_cloud(pc_path) # use red mask instead of mask.ply

        # scene pcd points and colors
        scene_pcd_points_ori = np.asarray(scene_pcd.points) / 1000 # convert to meters
        scene_pcd_points_not_aligned = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position, whose normal is not aligned with z-axis (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)

        # get color and depth
        rgb_img_path = os.path.join(pc_path.rsplit('/',1)[0], "no_obj/test_pbr/000000/rgb/000000.jpg")
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1)) 
        rgb_image = rgb_image / 255.0 # FIXME: Normalize the image to [0, 1]
        assert rgb_image.max() <= 1.0 and rgb_image.min() >= 0.0

        depth_path = pc_path.rsplit("/", 1)[0] + "/no_obj/test_pbr/000000/depth/000000.png"
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32)
        depth = depth / 1000.0 # depth shall be in the unit of meters
        depth[depth > 2] = 0 # remove the invalid depth values
        scene_depth_cloud_not_aligned, _ = backproject(depth, intrinsics, np.logical_and(depth > 0, depth < 2))
        scene_depth_cloud_not_aligned = visualize_points(scene_depth_cloud_not_aligned) # from depth backproject to point cloud, depth < 2
        scene_depth_cloud_points_not_aligned = np.asarray(scene_depth_cloud_not_aligned.points)



        ##########visualize 
        #scene_pcd.points = o3d.utility.Vector3dVector(scene_pcd_points)
        #o3d.visualization.draw_geometries([scene_depth_cloud, scene_pcd])
        #scene_depth_cloud = np.asarray(scene_depth_cloud.points)
        #assert depth.max() <= 2.0 and depth.min() >= 0.0
        #scene_points_perfect, scene_points_id = backproject(depth, intrinsics, depth > 0)
        
        # make the plane noraml align with z-axis
            # get the T_plane and plane_model
        T_plane, plane_model = get_tf_for_scene_rotation(scene_pcd_points_not_aligned)
        scene_pcd_points_aligned = np.dot(scene_pcd_points_not_aligned, T_plane[:3, :3]) + T_plane[:3, 3] # norm z-axis
        scene_depth_cloud_points_aligned = np.dot(scene_depth_cloud_points_not_aligned, T_plane[:3, :3]) + T_plane[:3, 3] # norm z-axis
        
        # visualize to check is aligned scene pcd has been aligned to z-axis
        scene_pcd_aligned = o3d.geometry.PointCloud()
        scene_pcd_not_aligned = o3d.geometry.PointCloud()
        scene_pcd_aligned.points = o3d.utility.Vector3dVector(scene_pcd_points_aligned)
        scene_pcd_aligned.colors = o3d.utility.Vector3dVector(scene_pcd_colors) # use the same color as the original point cloud
        scene_pcd_not_aligned.points = o3d.utility.Vector3dVector(scene_pcd_points_not_aligned)
        scene_pcd_not_aligned.colors = o3d.utility.Vector3dVector(scene_pcd_colors)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([scene_pcd_aligned, coordinate_frame, scene_pcd_not_aligned])

        # Convert points and colors to tensors
        scene_pcd_aligned_tensor = torch.tensor(scene_pcd_points_aligned, dtype=torch.float32).unsqueeze(0).to("cuda")
        scene_color_tensor = torch.tensor(scene_pcd_colors, dtype=torch.float32).unsqueeze(0).to("cuda")

        # Perform furthest point sampling (FPS)
        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_aligned_tensor.contiguous(), 2048)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_aligned = scene_pcd_points_aligned[fps_indices_scene_np] # 
        fps_colors_scene_aligned = scene_pcd_colors[fps_indices_scene_np]

        # Get the label (grenn channel color) 
        fps_mask_mapped = fps_colors_scene_aligned[:,1].reshape(-1, 1) # get label from green mask directly
        
        # find the point position with the highest affordance (G channel value)
        max_green_index = np.argmax(fps_colors_scene_aligned[:,1])
        max_green_point = torch.tensor(fps_points_scene_aligned[max_green_index], dtype=torch.float32)

        # gt_pose_4d_min_bound find the min bound of the whole pc
        min_bound = np.append(np.min(fps_points_scene_aligned, axis=0), -180)
        max_bound = np.append(np.max(fps_points_scene_aligned, axis=0), 180)

        min_xyz_bound = np.min(scene_depth_cloud_points_aligned, axis=0) 
        max_xyz_bound = np.max(scene_depth_cloud_points_aligned, axis=0) 


        # sample points with affordance value higher than the threshold
        affordance_threshold = self.affordance_threshold
        fps_points_scene_affordance = fps_points_scene_aligned[fps_colors_scene_aligned[:,1] > affordance_threshold] # [F, 3]
        if fps_points_scene_affordance.shape[0] == 0:
            fps_points_scene_affordance = fps_points_scene_aligned
        fps_points_scene_affordance_perfect = fps_points_scene_affordance
        # fps_points_to_perfect_scene_dist = np.linalg.norm(fps_points_scene_affordance[:, None] - scene_points_perfect[None], axis=-1) # [F, N]
        # fps_points_to_perfect_scene_dist = np.min(fps_points_to_perfect_scene_dist, axis=1) # [F,]
        # fps_points_scene_affordance_perfect = fps_points_scene_affordance[fps_points_to_perfect_scene_dist < 0.1] # [F, 3]


        bound_affordance_z_mid = np.median(fps_points_scene_affordance_perfect[:, 2])
        min_bound_affordance = np.append(np.min(fps_points_scene_affordance_perfect, axis=0), -180)
        max_bound_affordance = np.append(np.max(fps_points_scene_affordance_perfect, axis=0), 180)
        min_bound_affordance[2] = bound_affordance_z_mid * 0.9
        max_bound_affordance[2] = bound_affordance_z_mid * 0.9
        min_xyz_bound[2] = bound_affordance_z_mid * 0.9
        max_xyz_bound[2] = bound_affordance_z_mid * 0.9


        # sample 512 points from fps_points_scene_affordance
        fps_points_scene_affordance = fps_points_scene_affordance[np.random.choice(fps_points_scene_affordance.shape[0], 512, replace=True)] # [512, 3]

        # sample num_samples from scene for non_condition
        gt_non_cond = fps_points_scene_aligned[np.random.choice(fps_points_scene_aligned.shape[0], self.gt_pose_samples, replace=True)] # [num_samples, 3]

        # get the object pc position (rotated, scaled, translated to the origin point)
        scene_id = pc_path.split('/')[2]
        obj_name = pc_path.split('/')[3]
        json_path = os.path.join("dataset/scene_gen/scene_mesh_json_kinect", f"{scene_id}.json") #e.g. "dataset/scene_mesh_json_kinect/id3.json"
        # find the key which includes obj_name
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for key in data.keys():
                if obj_name in key:
                    obj_json = data[key]
                    break
        obj_mesh_path = key
        obj_json = data[obj_mesh_path]
        obj_scale = obj_json[1]
        obj_rotation = torch.tensor(obj_json[2], dtype=torch.float32).unsqueeze(0)
        obj_mesh = trimesh.load(obj_mesh_path)
        obj_mesh.apply_scale(obj_scale)
        obj_pc = obj_mesh.sample(512)

        # ################### DEBUG Acqurie the scene pcd ###################
        #scene_colors = rgb_image[:, scene_points_id[0], scene_points_id[1]].T
        #scene_pcd2 = o3d.geometry.PointCloud()
        #scene_pcd2.points = o3d.utility.Vector3dVector(scene_points_perfect)
        # scene_pcd2.colors = o3d.utility.Vector3dVector(scene_colors)

        #fps_pcd = o3d.geometry.PointCloud()
        #fps_pcd.points = o3d.utility.Vector3dVector(fps_points_scene_affordance_perfect)
        #fps_pcd.paint_uniform_color([1, 0, 0])

        #min_bound_vis = min_xyz_bound[:3]
        #max_bound_vis = max_xyz_bound[:3]
        #min_bound_vis_o3d = visualize_sphere_o3d(min_bound_vis, color=[0, 1, 0])
        #max_bound_vis_o3d = visualize_sphere_o3d(max_bound_vis, color=[0, 1, 0])
        #vis = [min_bound_vis_o3d, max_bound_vis_o3d, fps_pcd]
        #o3d.visualization.draw(vis)
        # ################### DEBUG Acqurie the scene pcd ###################


        # tsdf grid
        '''
        intrinsics = np.array([[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]])
        intrinsics[0, 2] = depth.shape[1] / 2
        intrinsics[1, 2] = depth.shape[0] / 2
        inv_intrinsics = np.linalg.pinv(intrinsics)
        vol_bnds = np.zeros((3,2))
        view_frust_pts = get_view_frustum(depth, intrinsics, np.eye(4))
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1)).min()
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1)).max()

        tsdf = TSDFVolume(vol_bnds, voxel_dim=256, num_margin=30)
        tsdf.integrate(color, depth, intrinsics, np.eye(4))
        tsdf_grid = tsdf.get_tsdf_volume()
        '''

        # add noise
        noise_scale = 0.05
        noise = torch.randn(self.gt_pose_samples,4, dtype=torch.float32) * noise_scale

        noise_4d = noise
        noise_4d[:, 2:] = 0
        noise_xyR = noise[:, :3]
        noise_xyz = noise[:, :3]
        # Prepare the final sample
        sample = {
            
            "pc_position": fps_points_scene_aligned, #[num_points, 3]
            "affordance": fps_mask_mapped,  # [num_points, 1]New mask based on turbo colormap
            "object_name": obj_name,
            "object_pc_position": obj_pc, # [num_obj_points, 3]
            "gt_pose_4d": torch.cat((max_green_point, obj_rotation), dim=0).unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_4d, 
            "gt_pose_4d_min_bound":min_bound, #[4,]
            "gt_pose_4d_max_bound":max_bound, #[4,]
            "pc_position_xy_affordance": fps_points_scene_affordance[:, :2], #[num_affordance, 2]
            "gt_pose_xyR": torch.cat((max_green_point[:2], obj_rotation), dim=0).unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_xyR, #[pose_samples, 3]
            "gt_pose_xyR_min_bound": np.delete(min_bound_affordance, 2, axis=0), #[3,] 
            "gt_pose_xyR_max_bound": np.delete(max_bound_affordance, 2, axis=0), #[3,] 
            "gt_pose_xyz": max_green_point.unsqueeze(0).repeat(self.gt_pose_samples, 1) + noise_xyz, #[pose_samples, 3]
            "gt_pose_xyz_for_non_cond": gt_non_cond, #[pos_samples, 3]

            "gt_pose_xyz_min_bound": min_xyz_bound, #[3,]
            "gt_pose_xyz_max_bound": max_xyz_bound, #[3,]
            #"tsdf_grid": tsdf_grid, 
            "depth": depth,
            "image": rgb_image,
            "T_plane": T_plane,
            "plane_model": plane_model
        }

        return sample


def visualize_bound(batch):
    '''
    Given the batch, visualize the point cloud and the bound
    '''
    assert 'pc_position' in batch.keys() and 'gt_pose_4d_min_bound' in batch.keys() and 'gt_pose_4d_affordance_min_bound' in batch.keys()
    for i in range((batch['pc_position'].shape[0])):
        pc_position = batch['pc_position'][i]
        gt_pose_4d_min_bound = batch['gt_pose_4d_min_bound'][i]
        gt_pose_4d_max_bound = batch['gt_pose_4d_max_bound'][i]
        gt_pose_4d_affordance_min_bound = batch['gt_pose_4d_affordance_min_bound'][i]
        gt_pose_4d_affordance_max_bound = batch['gt_pose_4d_affordance_max_bound'][i]

        # visualize the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_position)

        # visualize the bound
        bounding_box = create_bounding_box(gt_pose_4d_min_bound, gt_pose_4d_max_bound)
        bounding_box_affordance = create_bounding_box(gt_pose_4d_affordance_min_bound, gt_pose_4d_affordance_max_bound)

        o3d.visualization.draw_geometries([bounding_box, bounding_box_affordance, pcd])


if __name__ == "__main__":

    def project_3d(point, intr):
        """
        Project 3D points to 2D
        Args:
            point: [num_points, 3]
            intr: [3, 3]
        Returns:
            uv: [num_points, 2]
        """
        point = point / point[..., 2:]
        uv = point @ intr.T
        uv = uv[..., :2]
        return uv

    dataset_cls = PoseDataset_top_plane(split="train", root_dir="dataset/scene_RGBD_mask_v2_kinect_cfg", gt_pose_samples=80, affordance_threshold=-0.001)
    train_loader = DataLoader(dataset_cls, batch_size=1)
    len(dataset_cls)
    print("dataset length: ", len(dataset_cls))
    intrinsics = np.array(
            [
                [607.09912 / 2, 0.0, 636.85083 / 2],
                [0.0, 607.05212 / 2, 367.35952 / 2],
                [0.0, 0.0, 1.0],
            ]
        )
    for i, batch in enumerate(train_loader):
        # Read the data
        gt_pose_xyR = batch['gt_pose_xyz'].cpu().numpy()[0] 
        gt_pose_xyz_min_bound = batch['gt_pose_xyz_min_bound'].cpu().numpy()[0, :3]
        gt_pose_xyz_max_bound = batch['gt_pose_xyz_max_bound'].cpu().numpy()[0, :3]


        image = batch['image'].cpu().numpy()[0] # [3, H, W]
        image = (np.transpose(image, (1, 2, 0)) * 255).astype(np.uint8)
        depth = batch['depth'].cpu().numpy()[0] # [H, W]
        points_scene, _ = backproject(depth, intrinsics, depth>0) # [num_points, 3]

        # Measure the pairwise distance between the points
        distances = np.sqrt(
            ((points_scene[:, :2][:, None, :] - gt_pose_xyR[:, :2]) ** 2).sum(axis=2)
        )

        # Find the topk scene points that are closest to the anchor points (gt_pose_xyR)
        scenepts_to_anchor_dist = np.min(distances, axis=1)  # [num_points]
        scenepts_to_anchor_id = np.argmin(distances, axis=1)  # [num_points]
        topk_points_id = np.argsort(scenepts_to_anchor_dist, axis=0)[: gt_pose_xyR.shape[0]]
        tokk_points_id_corr_anchor = scenepts_to_anchor_id[topk_points_id]
        points_for_place = points_scene[topk_points_id]
        points_for_place_bound = np.stack([gt_pose_xyz_min_bound, gt_pose_xyz_max_bound], axis=0)
        # points_for_place = gt_pose_xyR.copy()
        # points_for_place[:, 2] = 1.0

        # Visualize the image
        uv_for_place = project_3d(points_for_place, intrinsics) # [80, 2]
        uv_for_place_bound = project_3d(points_for_place_bound, intrinsics)


        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for i in range(uv_for_place.shape[0]):
            uv_color = np.random.randint(0, 255, 3).astype(np.uint8)
            cv2.circle(image, 
                    (int(uv_for_place[i][0]), int(uv_for_place[i][1])), 
                    5, 
                    (int(uv_color[0]), int(uv_color[1]), int(uv_color[2])), -1)
        cv2.circle(image, 
                    (int(uv_for_place_bound[0, 0]), int(uv_for_place_bound[0, 1])), 
                    8, 
                    (255, 0, 0), -1)
        cv2.circle(image, 
                    (int(uv_for_place_bound[1, 0]), int(uv_for_place_bound[1, 1])), 
                    8, 
                    (0, 0, 255), -1)
        cv2.rectangle(image, 
                    (int(uv_for_place_bound[0, 0]), int(uv_for_place_bound[0, 1])), 
                    (int(uv_for_place_bound[1, 0]), int(uv_for_place_bound[1, 1])),
                    (0, 255, 0), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)

        # breakpoint()



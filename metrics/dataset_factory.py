import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import os
import json
from PIL import Image
import numpy as np
import cv2
import open3d as o3d
from pointnet2_ops import pointnet2_utils
from GeoL_diffuser.models.helpers import TSDFVolume, get_view_frustum
from GeoL_diffuser.models.utils.fit_plane import *
import trimesh


class BlendprocDesktopDataset(Dataset):
    def __init__(
            self, 
            split:str ="test", 
            root_dir:str = "/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data"
            ):
        self.split = split
        self.folder_path = root_dir
        
        self.files = []

        self.files = []
        items = os.listdir(self.folder_path)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item)
            #print(sub_folder_path)
            sub_items = os.listdir(sub_folder_path)
            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item)
                
                if os.path.isdir(sub_sub_folder_path):
                    self.files.extend([sub_sub_folder_path])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_path = self.files[index]

        # text _guidance
        text_guidance = os.path.join(file_path.rsplit('/', 1)[0], "text_guidance.json")
        text_guidance_data = json.load(open(text_guidance))

        # cam_rotation_matrix
        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])        
        
        # object_name
        object_name = file_path.rsplit('/', 1)[1].rsplit('_',2)[0]
        object_color = file_path.rsplit('/', 1)[1].split('_')[-1]
        object_full = f'the {object_color} {object_name}'

        # direction
        direction = text_guidance_data[object_full][3]
        if direction == "Front Left":
            direction = "Left Front"
        if direction == "Front Right":
            direction = "Right Front"
        if direction == "Behind Left":
            direction = "Left Behind"
        if direction == "Behind Right":
            direction = "Right Behind"

        # anchor obj name 
        anchor_obj_name = text_guidance_data[object_full][0]

        # image_with_obj
        mask_with_obj_path = os.path.join(file_path, 'mask_with_obj.png')

        # image_without_obj
        mask_without_obj_path = os.path.join(file_path, 'mask_no_obj.png')

        # depth_with_obj
        depth_with_obj_path = os.path.join(file_path, 'with_obj/test_pbr/000000/depth/000000.png')

        # depth_without_obj
        depth_without_obj_path = os.path.join(file_path, 'no_obj/test_pbr/000000/depth/000000.png')

        # image
        image_without_obj_path = os.path.join(file_path, 'no_obj/test_pbr/000000/rgb/000000.jpg')

        # image with obj
        image_with_obj_path = os.path.join(file_path, 'with_obj/test_pbr/000000/rgb/000000.jpg')

        # hdf5, for the desk mask
        hdf5_path = os.path.join(file_path, 'no_obj/0.hdf5')

        ######## get the data only for our model
        # find the ply corresponding to the direction
        ply_path = os.path.join(file_path, f'mask_{direction}.ply')
        scene_pcd = o3d.io.read_point_cloud(ply_path)
        scene_pcd_points_ori = np.asarray(scene_pcd.points) / 1000.0
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)

        scene_pcd_tensor = torch.tensor(scene_pcd_points, dtype=torch.float32).unsqueeze(0)
        scene_color_tensor = torch.tensor(scene_pcd_colors, dtype=torch.float32).unsqueeze(0)

        scene_pcd_tensor = scene_pcd_tensor.to("cuda")
        scene_color_tensor = scene_color_tensor.to("cuda")

        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor.contiguous(), 2048)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = scene_pcd_points[fps_indices_scene_np]
        fps_colors_scene_from_original = scene_pcd_colors[fps_indices_scene_np]

        rgb_img_path = os.path.join(file_path, "no_obj/test_pbr/000000/rgb/000000.jpg")
        rgb_image_GeoL = Image.open(rgb_img_path).convert("RGB")
        rgb_image_GeoL = np.array(rgb_image_GeoL).astype(float)
        rgb_image_GeoL = np.transpose(rgb_image_GeoL, (2, 0, 1))  # Change to (C, H, W) format for PyTorch
        rgb_image_GeoL = rgb_image_GeoL / 255.0 # FIXME: Normalize the image to [0, 1]
        assert rgb_image_GeoL.max() <= 1.0 and rgb_image_GeoL.min() >= 0.0        

        # Load the depth image
        depth_img_path = os.path.join(file_path, "no_obj/test_pbr/000000/depth/000000.png")
        depth_GeoL = np.array(cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
        depth_GeoL = depth_GeoL / 1000.0 # depth shall be in the unit of meters
        depth_GeoL[depth_GeoL > 2] = 0 # remove the invalid depth values
        assert depth_GeoL.max() <= 2.0 and depth_GeoL.min() >= 0.0

        # anchor name
        des = file_path.split('_')[-1]
        obj_name = file_path.split("/")[-1].rsplit('_', 2)[:-2][0]
        obj_to_place = f"the {des} {obj_name}"
        target_name = text_guidance_data[obj_to_place][0]


        # object_point_cloud_path
        scene_id = file_path.split('/')[-2]
        obj_name = file_path.split('/')[-1]
        scene_json_path = os.path.join("/home/stud/zhoy/MasterThesis_zhoy/dataset/scene_gen/scene_mesh_json_kinect", f"{scene_id}.json")        
        with open(scene_json_path, 'r') as f:
            scene_data = json.load(f)
            for key in scene_data.keys():
                if obj_name in key:
                    obj_json = scene_data[key]
                    break
        obj_mesh_path = key
        obj_json = scene_data[obj_mesh_path]
        obj_mesh_path = os.path.join("/home/stud/zhoy/MasterThesis_zhoy/", key)
        obj_scale = obj_json[1]
        obj_rotation = obj_json[2]
        obj_mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh.apply_scale(obj_scale)
        obj_points_sampled = obj_mesh.sample(2000)
        obj_rotation_matrix = np.array([[np.cos(obj_rotation), -np.sin(obj_rotation), 0], [np.sin(obj_rotation), np.cos(obj_rotation), 0], [0, 0, 1]])
        obj_points_sampled = obj_points_sampled @ obj_rotation_matrix.T



        # get the full mesh of the scene without table
        scene_mesh_path = f"/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_mesh/with_obj_to_place/{scene_id}/mesh.obj"



        intrinsics = np.array([[607.09912 / 2, 0.0, 636.85083 / 2],
               [0.0, 607.05212 / 2, 367.35952 / 2],
               [0.0, 0.0, 1.0]])
        # use depth and imge with obj 
        depth = cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        color = cv2.imread(image_without_obj_path).astype(np.float32)

        intrinsics[0, 2] = depth.shape[1] / 2
        intrinsics[1, 2] = depth.shape[0] / 2
        inv_intrinsics = np.linalg.pinv(intrinsics)
        vol_bnds = np.zeros((3,2))
        view_frust_pts = get_view_frustum(depth, intrinsics, np.eye(4))
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
        vol_bnds[:, 0] = vol_bnds[:, 0].min()
        vol_bnds[:, 1] = vol_bnds[:, 1].max()

        color_tsdf = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        tsdf = TSDFVolume(vol_bnds, voxel_dim=256, num_margin=2)
        tsdf.integrate(color_tsdf, depth, intrinsics, np.eye(4))

        # get the T plane
        # scene pcd points and colors
        T_plane, plane_model = get_tf_for_scene_rotation(scene_pcd_points)



        sample = {
            "file_path": file_path,
            "direction": direction,
            "object_name": object_name,
            "object_color": object_color,
            "anchor_obj_name": anchor_obj_name,
            "image_without_obj_path": image_without_obj_path,
            "image_with_obj_path": image_with_obj_path,
            "mask_with_obj_path": mask_with_obj_path,
            "mask_without_obj_path": mask_without_obj_path,
            "depth_with_obj_path": depth_with_obj_path,
            "depth_without_obj_path": depth_without_obj_path,
            "hdf5_path": hdf5_path,
            "fps_points_scene": fps_points_scene_from_original,
            "fps_colors_scene": fps_colors_scene_from_original,
            #"anchor_position": None, # Geol, generate by GroundingDINO
            "depth": depth_GeoL, # Geol
            "image": rgb_image_GeoL, # Geol
            "scene_mesh_path": scene_mesh_path,
            "obj_mesh_path": obj_mesh_path,
            "tsdf_vol": tsdf._tsdf_vol,
            "vol_bnds": vol_bnds,
            "T_plane": T_plane, 
            "color_tsdf": color_tsdf,
            "intrinsics": intrinsics,
            "obj_points": obj_points_sampled,
        }
        return sample
    
class BlendprocDesktopDataset_incompleted(Dataset):
    """
    compared to the BlendprocDesktopDataset, this dataset is used for the incompleted mesh
    the only difference is the mesh path
    """
    def __init__(
            self, 
            split:str ="test", 
            root_dir:str = "/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data"
            ):
        self.split = split
        self.folder_path = root_dir
        
        self.files = []

        self.files = []
        items = os.listdir(self.folder_path)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item)
            #print(sub_folder_path)
            sub_items = os.listdir(sub_folder_path)
            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item)
                
                if os.path.isdir(sub_sub_folder_path):
                    self.files.extend([sub_sub_folder_path])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_path = self.files[index]

        # text _guidance
        text_guidance = os.path.join(file_path.rsplit('/', 1)[0], "text_guidance.json")
        text_guidance_data = json.load(open(text_guidance))

        # cam_rotation_matrix
        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])        
        
        # object_name
        object_name = file_path.rsplit('/', 1)[1].rsplit('_',2)[0]
        object_color = file_path.rsplit('/', 1)[1].split('_')[-1]
        object_full = f'the {object_color} {object_name}'

        # direction
        direction = text_guidance_data[object_full][3]
        if direction == "Front Left":
            direction = "Left Front"
        if direction == "Front Right":
            direction = "Right Front"
        if direction == "Behind Left":
            direction = "Left Behind"
        if direction == "Behind Right":
            direction = "Right Behind"

        # anchor obj name 
        anchor_obj_name = text_guidance_data[object_full][0]

        # image_with_obj
        mask_with_obj_path = os.path.join(file_path, 'mask_with_obj.png')

        # image_without_obj
        mask_without_obj_path = os.path.join(file_path, 'mask_no_obj.png')

        # depth_with_obj
        depth_with_obj_path = os.path.join(file_path, 'with_obj/test_pbr/000000/depth/000000.png')

        # depth_without_obj
        depth_without_obj_path = os.path.join(file_path, 'no_obj/test_pbr/000000/depth/000000.png')

        # image
        image_without_obj_path = os.path.join(file_path, 'no_obj/test_pbr/000000/rgb/000000.jpg')

        # image with obj
        image_with_obj_path = os.path.join(file_path, 'with_obj/test_pbr/000000/rgb/000000.jpg')

        # hdf5, for the desk mask
        hdf5_path = os.path.join(file_path, 'no_obj/0.hdf5')

        ######## get the data only for our model
        # find the ply corresponding to the direction
        ply_path = os.path.join(file_path, f'mask_{direction}.ply')
        scene_pcd = o3d.io.read_point_cloud(ply_path)
        scene_pcd_points_ori = np.asarray(scene_pcd.points) / 1000.0
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)

        scene_pcd_tensor = torch.tensor(scene_pcd_points, dtype=torch.float32).unsqueeze(0)
        scene_color_tensor = torch.tensor(scene_pcd_colors, dtype=torch.float32).unsqueeze(0)

        scene_pcd_tensor = scene_pcd_tensor.to("cuda")
        scene_color_tensor = scene_color_tensor.to("cuda")

        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor.contiguous(), 2048)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = scene_pcd_points[fps_indices_scene_np]
        fps_colors_scene_from_original = scene_pcd_colors[fps_indices_scene_np]

        rgb_img_path = os.path.join(file_path, "no_obj/test_pbr/000000/rgb/000000.jpg")
        rgb_image_GeoL = Image.open(rgb_img_path).convert("RGB")
        rgb_image_GeoL = np.array(rgb_image_GeoL).astype(float)
        rgb_image_GeoL = np.transpose(rgb_image_GeoL, (2, 0, 1))  # Change to (C, H, W) format for PyTorch
        rgb_image_GeoL = rgb_image_GeoL / 255.0 # FIXME: Normalize the image to [0, 1]
        assert rgb_image_GeoL.max() <= 1.0 and rgb_image_GeoL.min() >= 0.0        

        # Load the depth image
        depth_img_path = os.path.join(file_path, "no_obj/test_pbr/000000/depth/000000.png")
        depth_GeoL = np.array(cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
        depth_GeoL = depth_GeoL / 1000.0 # depth shall be in the unit of meters
        depth_GeoL[depth_GeoL > 2] = 0 # remove the invalid depth values
        assert depth_GeoL.max() <= 2.0 and depth_GeoL.min() >= 0.0

        # anchor name
        des = file_path.split('_')[-1]
        obj_name = file_path.split("/")[-1].rsplit('_', 2)[:-2][0]
        obj_to_place = f"the {des} {obj_name}"
        target_name = text_guidance_data[obj_to_place][0]


        # object_point_cloud_path
        scene_id = file_path.split('/')[-2]
        obj_name = file_path.split('/')[-1]
        scene_json_path = os.path.join("/home/stud/zhoy/MasterThesis_zhoy/dataset/scene_gen/scene_mesh_json_kinect", f"{scene_id}.json")        
        with open(scene_json_path, 'r') as f:
            scene_data = json.load(f)
            for key in scene_data.keys():
                if obj_name in key:
                    obj_json = scene_data[key]
                    break
        obj_mesh_path = key
        obj_json = scene_data[obj_mesh_path]
        obj_mesh_path = os.path.join("/home/stud/zhoy/MasterThesis_zhoy/", key)
        obj_scale = obj_json[1]
        obj_rotation = obj_json[2]
        obj_mesh = trimesh.load_mesh(obj_mesh_path)
        obj_mesh.apply_scale(obj_scale)
        obj_points_sampled = obj_mesh.sample(2000)
        obj_rotation_matrix = np.array([[np.cos(obj_rotation), -np.sin(obj_rotation), 0], [np.sin(obj_rotation), np.cos(obj_rotation), 0], [0, 0, 1]])
        obj_points_sampled = obj_points_sampled @ obj_rotation_matrix.T



        # get the full mesh of the scene without table
        #scene_mesh_path = f"/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_mesh/with_obj_to_place/{scene_id}/mesh.obj"
        scene_mesh_path = os.path.join(file_path, "mesh.obj")


        intrinsics = np.array([[607.09912 / 2, 0.0, 636.85083 / 2],
               [0.0, 607.05212 / 2, 367.35952 / 2],
               [0.0, 0.0, 1.0]])
        # use depth and imge with obj 
        depth = cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        color = cv2.imread(image_without_obj_path).astype(np.float32)

        intrinsics[0, 2] = depth.shape[1] / 2
        intrinsics[1, 2] = depth.shape[0] / 2
        inv_intrinsics = np.linalg.pinv(intrinsics)
        vol_bnds = np.zeros((3,2))
        view_frust_pts = get_view_frustum(depth, intrinsics, np.eye(4))
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
        vol_bnds[:, 0] = vol_bnds[:, 0].min()
        vol_bnds[:, 1] = vol_bnds[:, 1].max()

        color_tsdf = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        tsdf = TSDFVolume(vol_bnds, voxel_dim=256, num_margin=2)
        tsdf.integrate(color_tsdf, depth, intrinsics, np.eye(4))

        # get the T plane
        # scene pcd points and colors
        T_plane, plane_model = get_tf_for_scene_rotation(scene_pcd_points)



        sample = {
            "file_path": file_path,
            "direction": direction,
            "object_name": object_name,
            "object_color": object_color,
            "anchor_obj_name": anchor_obj_name,
            "image_without_obj_path": image_without_obj_path,
            "image_with_obj_path": image_with_obj_path,
            "mask_with_obj_path": mask_with_obj_path,
            "mask_without_obj_path": mask_without_obj_path,
            "depth_with_obj_path": depth_with_obj_path,
            "depth_without_obj_path": depth_without_obj_path,
            "hdf5_path": hdf5_path,
            "fps_points_scene": fps_points_scene_from_original,
            "fps_colors_scene": fps_colors_scene_from_original,
            #"anchor_position": None, # Geol, generate by GroundingDINO
            "depth": depth_GeoL, # Geol
            "image": rgb_image_GeoL, # Geol
            "scene_mesh_path": scene_mesh_path,
            "obj_mesh_path": obj_mesh_path,
            "tsdf_vol": tsdf._tsdf_vol,
            "vol_bnds": vol_bnds,
            "T_plane": T_plane, 
            "color_tsdf": color_tsdf,
            "intrinsics": intrinsics,
            "obj_points": obj_points_sampled,
        }
        return sample


class BlendprocDesktopDataset_incompleted_sparse(Dataset):
    """
    compared to the BlendprocDesktopDataset, this dataset is used for the incompleted mesh
    the only difference is the mesh path
    """
    def __init__(
            self, 
            split:str ="test", 
            root_dir:str = "/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse"
            ):
        self.split = split
        self.folder_path = root_dir
        
        self.files = []

        self.files = []
        items = os.listdir(self.folder_path)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item)
            #print(sub_folder_path)
            sub_items = os.listdir(sub_folder_path)
            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item)
                
                if os.path.isdir(sub_sub_folder_path):
                    self.files.extend([sub_sub_folder_path])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_path = self.files[index]

        # text _guidance
        text_guidance = os.path.join(file_path.rsplit('/', 1)[0], "text_guidance.json")
        text_guidance_data = json.load(open(text_guidance))

        # cam_rotation_matrix
        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])        
        
        # object_name
        object_name = file_path.rsplit('/', 1)[1].rsplit('_',2)[0]
        object_color = file_path.rsplit('/', 1)[1].split('_')[-1]
        object_full = f'the {object_color} {object_name}'

        # direction
        direction = text_guidance_data[object_full][3]
        if direction == "Front Left":
            direction = "Left Front"
        if direction == "Front Right":
            direction = "Right Front"
        if direction == "Behind Left":
            direction = "Left Behind"
        if direction == "Behind Right":
            direction = "Right Behind"

        # anchor obj name 
        anchor_obj_name = text_guidance_data[object_full][0]

        # image_with_obj
        mask_with_obj_path = os.path.join(file_path, 'mask_with_obj.png')

        # image_without_obj
        mask_without_obj_path = os.path.join(file_path, 'mask_no_obj.png')

        # depth_with_obj
        depth_with_obj_path = os.path.join(file_path, 'with_obj/test_pbr/000000/depth/000000.png')

        # depth_without_obj
        depth_without_obj_path = os.path.join(file_path, 'no_obj/test_pbr/000000/depth/000000.png')

        # image
        image_without_obj_path = os.path.join(file_path, 'no_obj/test_pbr/000000/rgb/000000.jpg')

        # image with obj
        image_with_obj_path = os.path.join(file_path, 'with_obj/test_pbr/000000/rgb/000000.jpg')

        # hdf5, for the desk mask
        hdf5_path = os.path.join(file_path, 'no_obj/0.hdf5')

        ######## get the data only for our model
        # find the ply corresponding to the direction
        ply_path = os.path.join(file_path, f'mask_{direction}.ply')
        scene_pcd = o3d.io.read_point_cloud(ply_path)
        scene_pcd_points_ori = np.asarray(scene_pcd.points) / 1000.0
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)

        scene_pcd_tensor = torch.tensor(scene_pcd_points, dtype=torch.float32).unsqueeze(0)
        scene_color_tensor = torch.tensor(scene_pcd_colors, dtype=torch.float32).unsqueeze(0)

        scene_pcd_tensor = scene_pcd_tensor.to("cuda")
        scene_color_tensor = scene_color_tensor.to("cuda")

        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor.contiguous(), 2048)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = scene_pcd_points[fps_indices_scene_np]
        fps_colors_scene_from_original = scene_pcd_colors[fps_indices_scene_np]

        rgb_img_path = os.path.join(file_path, "no_obj/test_pbr/000000/rgb/000000.jpg")
        rgb_image_GeoL = Image.open(rgb_img_path).convert("RGB")
        rgb_image_GeoL = np.array(rgb_image_GeoL).astype(float)
        rgb_image_GeoL = np.transpose(rgb_image_GeoL, (2, 0, 1))  # Change to (C, H, W) format for PyTorch
        rgb_image_GeoL = rgb_image_GeoL / 255.0 # FIXME: Normalize the image to [0, 1]
        assert rgb_image_GeoL.max() <= 1.0 and rgb_image_GeoL.min() >= 0.0        

        # Load the depth image
        depth_img_path = os.path.join(file_path, "no_obj/test_pbr/000000/depth/000000.png")
        depth_GeoL = np.array(cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
        depth_GeoL = depth_GeoL / 1000.0 # depth shall be in the unit of meters
        depth_GeoL[depth_GeoL > 2] = 0 # remove the invalid depth values
        assert depth_GeoL.max() <= 2.0 and depth_GeoL.min() >= 0.0

        # anchor name
        des = file_path.split('_')[-1]
        obj_category = file_path.split("/")[-1].rsplit('_', 2)[:-2][0]
        obj_to_place = f"the {des} {obj_category}"
        target_name = text_guidance_data[obj_to_place][0]


        # object_point_cloud_path
        scene_id = file_path.split('/')[-2]
        obj_name = file_path.split('/')[-1]
        scene_json_path = os.path.join("/home/stud/zhoy/MasterThesis_zhoy/dataset/scene_gen/scene_mesh_json_aug", f"{scene_id}.json")        
        with open(scene_json_path, 'r') as f:
            scene_data = json.load(f)
            for key in scene_data.keys():
                if obj_name in key:
                    obj_json = scene_data[key]
                    break
        obj_mesh_path = key
        obj_json = scene_data[obj_mesh_path]
        obj_mesh_path = os.path.join("/home/stud/zhoy/MasterThesis_zhoy/", key)
        #obj_scale = obj_json[1]
        with open("/home/stud/zhoy/MasterThesis_zhoy/GeoL_net/dataset_gen/obj_size.json", 'r') as f:
            obj_target_size_json = json.load(f)
        target_size = obj_target_size_json[obj_category]

        obj_rotation = obj_json[2]
        obj_mesh = trimesh.load_mesh(obj_mesh_path)
        current_size = obj_mesh.bounds[1] - obj_mesh.bounds[0]
        scale_x, scale_y, scale_z = [target_size[0]/current_size[0], target_size[1]/current_size[1], target_size[2]/current_size[2]]
        
        obj_mesh.apply_scale([scale_x, scale_y, scale_z])
        obj_points_sampled = obj_mesh.sample(2000)
        obj_rotation_matrix = np.array([[np.cos(obj_rotation), -np.sin(obj_rotation), 0], [np.sin(obj_rotation), np.cos(obj_rotation), 0], [0, 0, 1]])
        obj_points_sampled = obj_points_sampled @ obj_rotation_matrix.T



        # get the full mesh of the scene without table
        #scene_mesh_path = f"/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_mesh/with_obj_to_place/{scene_id}/mesh.obj"
        scene_mesh_path = os.path.join(file_path, "mesh.obj")


        intrinsics = np.array([[607.09912 / 2, 0.0, 636.85083 / 2],
               [0.0, 607.05212 / 2, 367.35952 / 2],
               [0.0, 0.0, 1.0]])
        # use depth and imge with obj 
        depth = cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        color = cv2.imread(image_without_obj_path).astype(np.float32)

        intrinsics[0, 2] = depth.shape[1] / 2
        intrinsics[1, 2] = depth.shape[0] / 2
        inv_intrinsics = np.linalg.pinv(intrinsics)
        vol_bnds = np.zeros((3,2))
        view_frust_pts = get_view_frustum(depth, intrinsics, np.eye(4))
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
        vol_bnds[:, 0] = vol_bnds[:, 0].min()
        vol_bnds[:, 1] = vol_bnds[:, 1].max()

        color_tsdf = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        tsdf = TSDFVolume(vol_bnds, voxel_dim=256, num_margin=5)
        tsdf.integrate(color_tsdf, depth, intrinsics, np.eye(4))

        # get the T plane
        # scene pcd points and colors
        T_plane, plane_model = get_tf_for_scene_rotation(scene_pcd_points)



        sample = {
            "file_path": file_path,
            "direction": direction,
            "object_name": object_name,
            "object_color": object_color,
            "anchor_obj_name": anchor_obj_name,
            "image_without_obj_path": image_without_obj_path,
            "image_with_obj_path": image_with_obj_path,
            "mask_with_obj_path": mask_with_obj_path,
            "mask_without_obj_path": mask_without_obj_path,
            "depth_with_obj_path": depth_with_obj_path,
            "depth_without_obj_path": depth_without_obj_path,
            "hdf5_path": hdf5_path,
            "fps_points_scene": fps_points_scene_from_original,
            "fps_colors_scene": fps_colors_scene_from_original,
            #"anchor_position": None, # Geol, generate by GroundingDINO
            "depth": depth_GeoL, # Geol
            "image": rgb_image_GeoL, # Geol
            "scene_mesh_path": scene_mesh_path,
            "obj_mesh_path": obj_mesh_path,
            "tsdf_vol": tsdf._tsdf_vol,
            "vol_bnds": vol_bnds,
            "T_plane": T_plane, 
            "color_tsdf": color_tsdf,
            "intrinsics": intrinsics,
            "obj_points": obj_points_sampled,
        }
        return sample
    
if __name__ == "__main__":
    dataset = BlendprocDesktopDataset_incompleted()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(len(dataset))
    for i, data_batch in enumerate(dataloader):
        print(i, data_batch)
        img_path = data_batch["image_without_obj_path"][0]
        image = Image.open(img_path)
        object_to_place = data_batch["object_name"][0]
        direction = data_batch["direction"][0]
        anchor_obj_name = data_batch["anchor_obj_name"][0]

        depth_with_obj_path = data_batch["depth_with_obj_path"][0]
        depth_with_obj = np.array(cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED))


        
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
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points

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


class BlendprocDesktopDataset_incompleted_mult_cond(Dataset):
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
                    #self.files.extend(["dataset/benchmark_bproc_data/id9_1/phone_0000_normal"])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_path = self.files[index]

        # text _guidance
        text_guidance = os.path.join(file_path.rsplit('/', 1)[0], "text_guidance_mult_cond.json")
        text_guidance_data = json.load(open(text_guidance))

        # cam_rotation_matrix
        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])        
        
        # object_name
        object_name = file_path.rsplit('/', 1)[1].rsplit('_',2)[0]
        object_des = file_path.rsplit('/', 1)[1].split('_')[-1]
        object_full = f'the {object_des} {object_name}'

        # check if it is mult condition
        mult_conditions = [text_guidance_data[object_full][1]]
        if not text_guidance_data[object_full][-1].startswith("dataset"):
            mult_conditions.append(text_guidance_data[object_full][-1])
        if not text_guidance_data[object_full][-2].startswith("dataset"):
            mult_conditions.append(text_guidance_data[object_full][-2])



        # direction and anchor object list
        direction_list = []
        anchor_obj_name_list = [] # red cup
        #direction_list.append(text_guidance_data[object_full][3])
        for i in range(len(mult_conditions)):
            direction_list.append(mult_conditions[i].split(" the ")[0])
            anchor_obj_name_list.append(mult_conditions[i].split(" the ")[1])

            #adjust some directions
        for i in range(len(direction_list)):
            if direction_list[i] == "Front Left":
                direction_list[i] = "Left Front"
            if direction_list[i] == "Front Right":
                direction_list[i] = "Right Front"
            if direction_list[i] == "Behind Left":
                direction_list[i] = "Left Behind"
            if direction_list[i] == "Behind Right":
                direction_list[i] = "Right Behind"
        
        # get anchor object positions: find the image and depth base on the phrase and get the gt
            # find the image and depth on the phrase
        anchor_obj_file_list = []
        anchor_obj_position_list = []
        for i in range(len(mult_conditions)):
            if i == 0:
                continue # exclude the first one
            anchor_obj_name = anchor_obj_name_list[i]
            obj_name = anchor_obj_name.split(' ')[-1]
            obj_des = anchor_obj_name.split(' ')[0]

            for file in os.listdir(file_path.rsplit("/", 1)[0]):
                if file.startswith(obj_name) and file.endswith(obj_des):
                    anchor_obj_file_list.append(os.path.join(file_path.rsplit("/", 1)[0], file))
        # find the anchor obj poition from the main file
        anchor_obj_position = find_anchor_obj_position(file_path, color=[1, 0, 0])
        anchor_obj_position_list.append(anchor_obj_position)

        # find the anchor obj position from other files
        for anchor_obj_file in anchor_obj_file_list:
            anchor_obj_position = find_anchor_obj_position(anchor_obj_file)
            anchor_obj_position_list.append(anchor_obj_position)

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
        ply_path = os.path.join(file_path, f'mask_{direction_list[0]}.ply')
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
        obj_points_sampled = obj_mesh.sample(512)
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
            "direction": direction_list,
            "object_name": object_name,
            "object_color": object_des,
            "anchor_obj_name": anchor_obj_name_list,
            "anchor_obj_positions": torch.tensor(anchor_obj_position_list),
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
                if sub_sub_folder_path not in [
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id414_id317_0_0/lamp_0004_orange',				
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id295_id157_0_0/notebook_0003_green',		
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id330_id136_0_0/eye_glasses_0002_black',				
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id61_id102_0_0/cup_0003_green',				
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id19_id44_0_0/bowl_0003_white',				
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id64_id20_0_0/laptop_0005_black',				
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id432_id4_0_0/cup_0004_white',				
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id68_id231_0_0/monitor_0012_white',				
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id68_id231_0_0/phone_0000_normal',				
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id68_id231_0_0/pencil_0006_orange',				
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id68_id231_0_0/bottle_0002_mixture',				
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id323_id8_0_0/clock_0001_normal',			
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id323_id8_0_0/bottle_0002_mixture',				
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id39_id338_0_0/vase_0003_red'
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id39_id338_0_0/eye_glasses_0004_black',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id68_id231_0_0/clock_0001_normal',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id323_id8_0_0/phone_0000_normal',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id295_id157_0_0/bottle_0003_cola',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id39_id68_0_0/cup_0004_white',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id39_id68_0_0/phone_0000_normal',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id19_id318_0_0/plant_0008_normal',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id317_id108_0_0/eraser_0002_normal',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id432_id321_0_0/clock_0001_normal',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id6_id435_0_0/phone_0000_normal',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id6_id435_0_0/bottle_0001_plastic',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id6_id435_0_0/bottle_0004_brown',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id295_id5_0_0/eye_glasses_0002_black',
                    '/home/stud/zhoy/MasterThesis_zhoy/dataset/benchmark_bproc_data_sparse/id19_id318_0_0/clock_0001_normal',
                    ]:
                
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
        obj_points_sampled = obj_mesh.sample(512)
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


def find_anchor_obj_position(directory, color=[0, 1, 0]):
    """extract the specific color obj from the image and depth in the directory
    """
    intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                [  0.     , 607.05212/2, 367.35952/2],
                [  0.     ,   0.     ,   1.     ]])
    
    cam_rotation_matrix = np.array([[1, 0, 0],[0,0.8,-0.6],[0,0.6,0.8]])
    
    depth_file = os.path.join(directory, "with_obj/test_pbr/000000/depth/000000.png")
    mask_rgb_file = os.path.join(directory, "mask_with_obj.png")
    depth = np.array(cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)) / 1000
    pts_with_obj, idx_with_obj = backproject(depth, intrinsics, np.logical_and(depth > 0, depth >0))
    mask_rgb = cv2.imread(mask_rgb_file, cv2.COLOR_BGR2RGB)[idx_with_obj[0], idx_with_obj[1]]
    pts = visualize_points(pts_with_obj, mask_rgb/255)
    pts.points =o3d.utility.Vector3dVector(np.asarray(pts.points)) # now it is horizional

    gt_obj_position = catch_specific_color_obj(pts, color)

    return gt_obj_position


def catch_specific_color_obj(scene_point, color=[0,0,0]):
    """
    catch the specific color obj from the scene_point
    """
    scene_colors  = np.asarray(scene_point.colors)
    mask = np.all(scene_colors == color, axis=-1)
    scene_position = np.asarray(scene_point.points)
    masked_scene_position = scene_position[mask]

    if len(masked_scene_position) == 0:
        return [100, 100, 100]

    x_min, x_max = masked_scene_position[:,0].min(), masked_scene_position[:,0].max()
    y_min, y_max = masked_scene_position[:,1].min(), masked_scene_position[:,1].max()
    z_min, z_max = masked_scene_position[:,2].min(), masked_scene_position[:,2].max()

    gt_position = np.array([(x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2])
    gt_position[2] = z_max

    ### for visualization
    # masked_scene_colors = scene_colors[mask]
    # specified_point = o3d.geometry.PointCloud()
    # specified_point.points = o3d.utility.Vector3dVector(masked_scene_position)
    # specified_point.colors = o3d.utility.Vector3dVector(masked_scene_colors)
    # o3d.visualization.draw_geometries([specified_point])

    return gt_position




class realworld_dataset(Dataset):
    def __init__(self, root_dir="dataset/realworld_2103/json"):
        self.root_dir = root_dir
       
        self.files = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        intrinsics = np.array([[911.09, 0.0, 657.44],
                                 [0.0, 910.68, 346.58],
                                 [0.0, 0.0, 1.0]])


        json_file = self.files[index]
        with open(json_file, 'r') as f:
            data = json.load(f)
        rgb_image_file_path = data["rgb_image_file_path"]
        depth_img_file_path = data["depth_image_file_path"]
        mask_file_path = data["mask_file_path"]
        ref_objects = data["ref_objects"]
        for i, ref_obj in enumerate(ref_objects):
            if ref_obj == "monitor" or ref_obj == "Monitor":
                ref_objects[i] = "purple screen"
        directions = data["directions"]
        obj_to_place = data["obj_to_place"]

        color_image = cv2.imread(rgb_image_file_path).astype(np.float32)
        depth = cv2.imread(depth_img_file_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        pts, idx = backproject(depth, intrinsics, np.logical_and(depth > 0, depth <2))
        mask_rgb = color_image[idx[0], idx[1]]
        pcd = visualize_points(pts, mask_rgb/255)
        T_plane, plane_model = get_tf_for_scene_rotation(np.asarray(pcd.points))

        scene_pcd_tensor = torch.tensor(np.asarray(pcd.points), dtype=torch.float32).unsqueeze(0)
        scene_color_tensor = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32).unsqueeze(0)

        scene_pcd_tensor = scene_pcd_tensor.to("cuda")
        scene_color_tensor = scene_color_tensor.to("cuda")

        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor.contiguous(), 2048)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = np.asarray(pcd.points)[fps_indices_scene_np]
        fps_colors_scene_from_original = np.asarray(pcd.colors)[fps_indices_scene_np]
    
        vol_bnds = np.zeros((3,2))
        view_frust_pts = get_view_frustum(depth, intrinsics, np.eye(4))
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
        vol_bnds[:, 0] = vol_bnds[:, 0].min()
        vol_bnds[:, 1] = vol_bnds[:, 1].max()


        #color_tsdf = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        #tsdf = TSDFVolume(vol_bnds, voxel_dim=256, num_margin=5)
        #tsdf.integrate(color_tsdf, depth, intrinsics, np.eye(4))

        # find the obj mesh path
        #obj_to_place = os.path.join("dataset/obj/mesh", obj_to_place)
        #obj_mesh = trimesh.load_mesh(obj_to_place)

        #o3d.vsualization.draw_geometries([obj_mesh])


        obj_mesh = trimesh.load_mesh("dataset/obj/mesh/cup/cup_0001_red/mesh.obj") # NOTE: temporary use the cup mesh 
        target_size = [0.1, 0.1, 0.1]
        current_size = obj_mesh.bounds[1] - obj_mesh.bounds[0]
        scale_x, scale_y, scale_z = [target_size[0]/current_size[0], target_size[1]/current_size[1], target_size[2]/current_size[2]]
        obj_mesh.apply_scale([scale_x, scale_y, scale_z])
        obj_points_sampled = obj_mesh.sample(512)

        batch = {
            'json_file': json_file,
            "rgb_image_file_path": rgb_image_file_path,
            "depth_img_file_path": depth_img_file_path,
            "mask_file_path": mask_file_path,
            "ref_objects": ref_objects,
            "directions": directions,
            "obj_to_place": obj_to_place,
            "depth": depth,
            "color_image": color_image,
            "intrinsics": intrinsics,
            "fps_points_scene": fps_points_scene_from_original,
            "fps_colors_scene": fps_colors_scene_from_original,
            #"tsdf_vol": tsdf._tsdf_vol,
            "vol_bnds": vol_bnds,
            "T_plane": T_plane,
            #"color_tsdf": color_tsdf,
            "obj_points": obj_points_sampled,
            "obj_mesh_path": "dataset/obj/mesh/cup/cup_0001_red/mesh.obj",
            "obj_bbox_file_path":rgb_image_file_path.replace("color", "obj_bbox").replace("png", "npz"),

        }
        return batch


    
if __name__ == "__main__":
    #dataset = BlendprocDesktopDataset_incompleted_mult_cond()
    dataset = realworld_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(len(dataset))
    for i, data_batch in enumerate(dataloader):
        print(i, data_batch)
        
    # print(len(dataset))
    # for i, data_batch in enumerate(dataloader):
    #     print(i, data_batch)
    #     img_path = data_batch["image_without_obj_path"][0]
    #     image = Image.open(img_path)
    #     object_to_place = data_batch["object_name"][0]
    #     direction = data_batch["direction"][0]
    #     anchor_obj_name = data_batch["anchor_obj_name"][0]

    #     depth_with_obj_path = data_batch["depth_with_obj_path"][0]
    #     depth_with_obj = np.array(cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED))




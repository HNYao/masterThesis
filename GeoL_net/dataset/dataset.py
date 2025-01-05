import os
import open3d as o3d
import json
from clip.model import build_model, load_clip, tokenize
# Ignore warnings
import warnings
from collections import defaultdict
from typing import Any
from PIL import Image
from typing import List
from matplotlib import cm
#import albumentations as A
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pointnet2_ops import pointnet2_utils
import cv2

from GeoL_net.core.logger import logger
from GeoL_net.core.registry import registry
from GeoL_net.utils.utils import (
    decode_rle_mask,
    load_image,
    load_json,
    load_pickle,
)
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")




def collate_fn(batch):
    observations = defaultdict(list)

    for sample in batch:
        for key, val in sample.items():
            observations[key].append(val)

    observations_batch = {}
    for key, val in observations.items():
        if "target_category" in key:
            observations_batch[key] = val
            continue
        observations_batch[key] = torch.stack(val)

        # NHWC -> NCHW for augmentations
        if len(observations_batch[key].shape) == 4:
            observations_batch[key] = observations_batch[key].permute(
                0, 3, 1, 2
            )
    return observations_batch


def is_yellow(color, tolerance=0.1):
    return (color[0] > 1 - tolerance and color[1] > 1 - tolerance and color[2] < tolerance)

def is_green(color, tolerance=0.1):
    return (color[0] < tolerance and color[1] > 1 - tolerance and color[2] < tolerance)

def is_red(color, tolerance=0.1):
    return (color[0] > 1 - tolerance and color[1] < tolerance and color[2] < tolerance)

@registry.register_dataset(name="GeoL_dataset_full")
class GeoLPlacementDataset(Dataset):
    def __init__(self,
                 split:str,
                 root_dir:str) -> None:
        super().__init__()

        self.split = split
        self.root_dir = root_dir
        self.folder_path = self.root_dir

        self.files = []
        items = os.listdir(self.root_dir)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item)
            #print(sub_folder_path)
            sub_items = os.listdir(sub_folder_path)
            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item)
                
                if os.path.isdir(sub_sub_folder_path):
                    self.files.extend([sub_sub_folder_path])
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
        file_path = self.files[index]

        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])

        # get scene pcd
        scene_pcd = o3d.io.read_point_cloud(os.path.join(file_path, "mask_red.ply")) # use red mask instead of mask.ply

        # scene pcd points and colors
        scene_pcd_points_ori = np.asarray(scene_pcd.points)
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)
        
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

        # Get the label color 
        fps_mask_mapped = fps_colors_scene_from_original[:,0].reshape(-1, 1) # get label from red mask directly

        # Load the RGB image
        rgb_image = Image.open(os.path.join(file_path, "no_obj/test_pbr/000000/rgb/000000.jpg")).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # Change to (C, H, W) format for PyTorch
        
        # Load the text guidance and reference object from the JSON file
        parent_dir = os.path.dirname(file_path)
        json_path = os.path.join(parent_dir, "text_guidance.json")
        removed_obj_name = file_path.split("/")[-1]
        name = '_'.join(removed_obj_name.rsplit('_', 2)[:-2])
        des = removed_obj_name.split("_")[-1]
        target_name = f"the {des} {name}"

        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        reference_obj = data[target_name][0] # e.g the red cup
        phrase = data[target_name][1] # e.g Left Behind of cup
        reference_obj_short = data[target_name][2] # e.g cup
        direction_text = data[target_name][3] # e.g left

        # Prepare the final sample
        sample = {
            "fps_points_scene": fps_points_scene_from_original,
            "fps_colors_scene": fps_colors_scene_from_original,
            "phrase": phrase,
            "image": rgb_image,
            "mask": fps_mask_mapped,  # [num_points, 1]New mask based on turbo colormap
            "file_path": file_path,
            "reference_obj": reference_obj,
            "direction_text": direction_text
        }

        return sample
    
# for direction prompt training
@registry.register_dataset(name="GeoL_dataset_direction")
class GeoLPlacementDataset_direction(Dataset):
    def __init__(self,
                 split:str,
                 root_dir = "dataset/scene_RGBD_mask_direction") -> None:
        super().__init__()

        self.split = split
        self.root_dir = root_dir
        self.folder_path = self.root_dir

        self.files = []
        items = os.listdir(self.root_dir)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item)
            #print(sub_folder_path)
            sub_items = os.listdir(sub_folder_path)
            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item) # 'dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal'
                for sub_sub_item in os.listdir(sub_sub_folder_path):
                    # if end with ply
                    if sub_sub_item.endswith('.ply'):
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

        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])

        # get scene pcd
        scene_pcd = o3d.io.read_point_cloud(pc_path) # use red mask instead of mask.ply

        # scene pcd points and colors
        scene_pcd_points_ori = np.asarray(scene_pcd.points)
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)
        
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

        # Get the label color 
        fps_mask_mapped = fps_colors_scene_from_original[:,0].reshape(-1, 1) # get label from red mask directly

        # Load the RGB image
        rgb_img_path = os.path.join(pc_path.rsplit('/',1)[0], "img_without_removed_obj.jpg")
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # Change to (C, H, W) format for PyTorch

        # Load the text guidance and reference object from the JSON file
        json_path = os.path.join(pc_path.rsplit('/',1)[0], "text_guidance.json")
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        direction_word_in_ply = pc_path.split('/')[-1].split('_')[1].split('.')[0]

        target_name = data[direction_word_in_ply][0] # e.g the red cup

        phrase = data[direction_word_in_ply][1] # e.g Left Behind of cup
        direction_text = data[direction_word_in_ply][3] # e.g left

        # Prepare the final sample
        sample = {
            "fps_points_scene": fps_points_scene_from_original,
            "fps_colors_scene": fps_colors_scene_from_original,
            "phrase": phrase,
            "image": rgb_image,
            "mask": fps_mask_mapped,  # [num_points, 1]New mask based on turbo colormap
            "file_path": pc_path,
            "reference_obj": target_name,
            "direction_text": direction_text
        }

        return sample
    

@registry.register_dataset(name="GeoL_dataset_direction_mult")
class GeoLPlacementDataset_direction_mult(Dataset):
    def __init__(self,
                 split:str,
                 root_dir = "dataset/scene_RGBD_mask_v2_kinect_cfg") -> None:
        super().__init__()

        self.split = split
        self.root_dir = root_dir
        self.folder_path = self.root_dir

        self.files = []
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

        # get scene pcd
        scene_pcd = o3d.io.read_point_cloud(pc_path) # use red mask instead of mask.ply

        # scene pcd points and colors
        scene_pcd_points_ori = np.asarray(scene_pcd.points) / 1000.0
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)

        # 从旋转后的points中提取出红色点，计算红色点的位置的均值
        red_mask = np.apply_along_axis(is_red, 1, scene_pcd_colors)
        red_points = scene_pcd_points[red_mask]
        red_pcd_center = np.mean(red_points, axis=0)
        
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


        # Get the label color 
        fps_mask_mapped = fps_colors_scene_from_original[:,1].reshape(-1, 1) # get label from green mask directly

        # Load the RGB image
        rgb_img_path = os.path.join(pc_path.rsplit('/',1)[0], "no_obj/test_pbr/000000/rgb/000000.jpg")
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # Change to (C, H, W) format for PyTorch
        rgb_image = rgb_image / 255.0 # FIXME: Normalize the image to [0, 1]
        assert rgb_image.max() <= 1.0 and rgb_image.min() >= 0.0

        # Load the depth image
        depth_img_path = os.path.join(pc_path.rsplit('/',1)[0], "no_obj/test_pbr/000000/depth/000000.png")
        depth = np.array(cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)).astype(np.float32)
        depth = depth / 1000.0 # depth shall be in the unit of meters
        depth[depth > 2] = 0 # remove the invalid depth values
        assert depth.max() <= 2.0 and depth.min() >= 0.0

        # anchor name
        des = pc_path.split('/')[-2].split('_')[-1]
        obj_name = pc_path.split("/")[-2].rsplit('_', 2)[:-2][0]
        target_name = f"the {des} {obj_name}"
        target_name = data[target_name][0]
        
        # direction text
        pc_path_split = pc_path.split('/')[-1]
        direction_text = pc_path_split.split('_')[1].split('.')[0]

        phrase = f"{direction_text} the {target_name}"        
        # Prepare the final sample
        sample = {
            
            "fps_points_scene": fps_points_scene_from_original,
            "fps_colors_scene": fps_colors_scene_from_original,
            "anchor_position": red_pcd_center,
            "phrase": phrase,
            "image": rgb_image,
            "mask": fps_mask_mapped,  # [num_points, 1]New mask based on turbo colormap
            "file_path": pc_path,
            "reference_obj": target_name,
            "direction_text": direction_text,
            'depth': depth
        }

        return sample


@registry.register_dataset(name="GeoL_dataset_direction_mult_4096")
class GeoLPlacementDataset_direction_mult_4096(Dataset):
    def __init__(self,
                 split:str,
                 root_dir = "dataset/scene_RGBD_mask_direction_mult") -> None:
        super().__init__()

        self.split = split
        self.root_dir = root_dir
        self.folder_path = self.root_dir

        self.files = []
        items = os.listdir(self.root_dir)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item)
            #print(sub_folder_path)
            sub_items = os.listdir(sub_folder_path)
            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item) # 'dataset/scene_RGBD_mask_direction/id164_1/printer_0001_normal'
                for sub_sub_item in os.listdir(sub_sub_folder_path):
                    # if end with ply
                    if sub_sub_item.endswith('_Behind.ply') or sub_sub_item.endswith('_Front.ply') or sub_sub_item.endswith('_Left.ply') or sub_sub_item.endswith('_Right.ply'):
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
        json_path = pc_path.rsplit('/',2)[:-1] + '/text_guidance.json'
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])

        # get scene pcd
        scene_pcd = o3d.io.read_point_cloud(pc_path) # use red mask instead of mask.ply

        # scene pcd points and colors
        scene_pcd_points_ori = np.asarray(scene_pcd.points)
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T  # reverse rotation to original position (can be aligned to image directly)
        scene_pcd_colors = np.asarray(scene_pcd.colors)

        # 从旋转后的points中提取出绿色的点，计算绿色点的位置的均值
        green_mask = np.apply_along_axis(is_green, 1, scene_pcd_colors)
        green_points = scene_pcd_points[green_mask]
        green_pcd_center = np.mean(green_points, axis=0)
        
        # Convert points and colors to tensors
        scene_pcd_tensor = torch.tensor(scene_pcd_points, dtype=torch.float32).unsqueeze(0)
        scene_color_tensor = torch.tensor(scene_pcd_colors, dtype=torch.float32).unsqueeze(0)

        # Move to CUDA if necessary
        scene_pcd_tensor = scene_pcd_tensor.to("cuda")
        scene_color_tensor = scene_color_tensor.to("cuda")

        # Perform furthest point sampling (FPS)
        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor.contiguous(), 4096)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = scene_pcd_points[fps_indices_scene_np]
        fps_colors_scene_from_original = scene_pcd_colors[fps_indices_scene_np]

        # Get the label color 
        fps_mask_mapped = fps_colors_scene_from_original[:,0].reshape(-1, 1) # get label from red mask directly

        # Load the RGB image
        rgb_img_path = os.path.join(pc_path.rsplit('/',1)[0], "img.jpg")
        rgb_image = Image.open(rgb_img_path).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # Change to (C, H, W) format for PyTorch


        # anchor name
        des = pc_path.split('/')[-2].split('_')[-1]
        obj_name = pc_path.split("/")[-2].rsplit('_', 2)[:-2][0]
        target_name = f"the {des} {obj_name}"
        
        # direction text
        pc_path_split = pc_path.split('/')[-1]
        direction_text = pc_path_split.split('_')[1].split('.')[0]

        phrase = f"{direction_text} the {target_name}"        
        # Prepare the final sample
        sample = {
            
            "fps_points_scene": fps_points_scene_from_original,
            "fps_colors_scene": fps_colors_scene_from_original,
            "anchor_position": green_pcd_center,
            "phrase": phrase,
            "image": rgb_image,
            "mask": fps_mask_mapped,  # [num_points, 1]New mask based on turbo colormap
            "file_path": pc_path,
            "reference_obj": data[target_name][0],
            "direction_text": direction_text
        }

        return sample


if __name__ == "__main__":
    dataset_cls = GeoLPlacementDataset_direction_mult(split="train", root_dir="dataset/scene_RGBD_mask_v2_kinect_cfg")
    train_loader = DataLoader(dataset_cls, batch_size=2)
    len(dataset_cls)
    print("dataset length: ", len(dataset_cls))
    for i, batch in enumerate(train_loader):
        batch = batch
        fps_points = batch["fps_points_scene"]
        mask = batch["mask"]
        instrinsic = np.array([[591.0125 ,   0.     , 322.525  ],
                               [  0.     , 590.16775, 244.11084],
                               [  0.     ,   0.     ,   1.     ]])
        '''
        for i in range(batch['fps_points_scene'].shape[0]):
            depth = batch['depth'][i].numpy()
            fps_points = batch['fps_points_scene'][i].numpy()
            fps_colors = batch['fps_colors_scene'][i].numpy()
            points_scene, _ = backproject(depth, instrinsic, np.logical_and(depth > 0, depth > 0), NOCS_convention=False)
            
            distances = cdist(points_scene, fps_points)
            nearest_idx = np.argmin(distances, axis=1)
            colors_scene = fps_colors[nearest_idx]
            pcd_scene = visualize_points(points_scene, colors_scene)
            o3d.visualization.draw_geometries([pcd_scene])
        '''
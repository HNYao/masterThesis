import os
import random
import open3d as o3d
import json
import clip
from clip.model import build_model, load_clip, tokenize
# Ignore warnings
import warnings
from collections import defaultdict
from typing import Any
from PIL import Image
from typing import List

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset
from pointnet2_ops import pointnet2_utils

from GeoL_net.core.logger import logger
from GeoL_net.core.registry import registry
from GeoL_net.utils.utils import (
    decode_rle_mask,
    load_image,
    load_json,
    load_pickle,
)

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
        print(self.files)

    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, index) -> Any:
        """
        fps_points_scene_from_original: points after FPS [4096*3]
        fps_colors_scene_from_original: colors of points after FPS [4096*3]
        colors_modified: colors label of posints after FPS [4096 * 4]
        reference_obj: name of the reference obj, text
        reference_position: position of the reference obj in the PC [3]
        phrase: guidance text, text
        """
        file_path = self.files[index]
        
        cam_rotation_matrix = np.array([
        [1, 0, 0],
        [0,0.8,-0.6],
        [0,0.6,0.8]
        ])  


        # get scene pcd
        scene_pcd = o3d.io.read_point_cloud(os.path.join(file_path, "mask.ply"))

        # scene pcd points and colors
        scene_pcd_points_ori = np.asarray(scene_pcd.points)
        scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ scene_pcd_points_ori.T).T #reverse rotation

        scene_pcd_colors = np.asarray(scene_pcd.colors)
        
        # get label color 
        label_pcd_colors = np.asarray(scene_pcd.colors)

        scene_pcd_tensor = torch.tensor(scene_pcd_points, dtype=torch.float32).unsqueeze(0)
        scene_color_tensor = torch.tensor(scene_pcd_colors, dtype=torch.float32).unsqueeze(0)

        # move to cuda
        scene_pcd_tensor = scene_pcd_tensor.to("cuda")
        scene_color_tensor = scene_color_tensor.to("cuda")

        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor.contiguous(), 512)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = scene_pcd_points[fps_indices_scene_np]
        fps_colors_scene_from_original = scene_pcd_colors[fps_indices_scene_np]

        # get the color of fps-pc
        fps_colors_from_original = label_pcd_colors[fps_indices_scene_np]

        #get the color image
        rgb_image = Image.open(os.path.join(file_path, "no_obj/test_pbr/000000/rgb/000000.jpg")).convert("RGB")
        rgb_image = np.array(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))

        # get the depth image
        depth_image = Image.open(os.path.join(file_path, "no_obj/test_pbr/000000/depth/000000.png"))
        depth_image = np.array(depth_image).astype(float)
        
        # get a fake mask
        fake_mask = np.random.rand(480, 640)
        fake_mask_2 = np.random.rand(4096, 4)


        # 4 classes
        colors_modified_4cls = np.zeros((fps_colors_from_original.shape[0],4))
        for i in range(fps_colors_from_original.shape[0]):
            if (fps_colors_from_original[i] == [0.,1.,0.]).all(): #green
                colors_modified_4cls[i] = [0.,1.,0.,0.]
            elif (fps_colors_from_original[i] == [1.,0.,0.]).all(): # red
                colors_modified_4cls[i] = [1.,0.,0.,0.]
            elif (fps_colors_from_original[i] == [0.,0.,0.]).all(): #black
                colors_modified_4cls[i] = [0.,0.,0.,1]
            elif (fps_colors_from_original[i] == [0.,0.,1.]).all(): #blue
                colors_modified_4cls[i] = [0.,0.,1.,0.]

        # 2 classes
        colors_modified_2cls = np.zeros((fps_colors_from_original.shape[0],2))
        for i in range(fps_colors_from_original.shape[0]):
            if (fps_colors_from_original[i] == [0.,1.,0.]).all(): #green
                colors_modified_2cls[i] = [0.,1.]
            elif (fps_colors_from_original[i] == [1.,0.,0.]).all(): # red
                colors_modified_2cls[i] = [1.,0.]
            elif (fps_colors_from_original[i] == [0.,0.,0.]).all(): #black
                colors_modified_2cls[i] = [1.,0.]
            elif (fps_colors_from_original[i] == [0.,0.,1.]).all(): #blue
                colors_modified_2cls[i] = [1.,0.]



        
        # read json and the phrase
        parent_dir = os.path.dirname(file_path)
        json_path = os.path.join(parent_dir, "text_guidance.json")
        removed_obj_name = file_path.split("/")[-1]
        name, _, des = removed_obj_name.split("_")
        target_name = f"the {des} {name}"
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        reference_obj = data[target_name][0]
        phrase = data[target_name][1]

        # reference position
        scene_ref_pcd = o3d.io.read_point_cloud(os.path.join(file_path, "mask_ref.ply"))
        points_ref = np.asarray(scene_ref_pcd.points)
        colors_ref = np.asarray(scene_ref_pcd.colors)
        yellow_indices = [i for i, color in enumerate(colors_ref) if is_yellow(color)]
        yellow_points = points_ref[yellow_indices]
        yellow_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(yellow_points))
        ref_center = yellow_bbox.get_center()


        sample = {
            "fps_points_scene": fps_points_scene_from_original,
            "fps_colors_scene": fps_colors_scene_from_original,
            "ref_center": ref_center,
            "colors_modified": colors_modified_4cls,
            "ref_obj": reference_obj,
            "phrase":phrase,
            "image": rgb_image,
            "mask": colors_modified_4cls
            
        }

    
        return sample      
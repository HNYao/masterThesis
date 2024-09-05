import json
from tqdm import tqdm
import numpy as np
import open3d as o3d
import os
import torch
from torch.utils.data import DataLoader, Dataset
from pointnet2_ops import pointnet2_utils


def is_yellow(color, tolerance=0.1):
    return (color[0] > 1 - tolerance and color[1] > 1 - tolerance and color[2] < tolerance)

# ply_scene_without_obj, phrase, reference_postionn_in_ply
class AffordanceDataset(Dataset):
    def __init__(
            self, 
            folder_path='dataset/scene_RGBD_mask',
            obj_path = None, # need to add
            label_path = None
            ) -> None:
        self.folder_path = folder_path
        self.files = []
        items = os.listdir(self.folder_path)
        for item in items:
            sub_folder_path = os.path.join(self.folder_path, item)
            #print(sub_folder_path)
            sub_items = os.listdir(sub_folder_path)
            for sub_item in sub_items:
                sub_sub_folder_path = os.path.join(sub_folder_path, sub_item)
                #print(sub_sub_folder_path)
                if os.path.isdir(sub_sub_folder_path):
                    self.files.extend([sub_sub_folder_path])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        """
        fps_points_scene_from_original: points after FPS [4096*3]
        fps_colors_scene_from_original: colors of points after FPS [4096*3]
        colors_modified: colors label of posints after FPS [4096 * 4]
        reference_obj: name of the reference obj, text
        reference_position: position of the reference obj in the PC [3]
        phrase: guidance text, text
        """
        file_path = self.files[index]
        
        # get scene pcd
        scene_pcd = o3d.io.read_point_cloud(os.path.join(file_path, "mask.ply"))

        # scene pcd points and colors
        scene_pcd_points = np.asarray(scene_pcd.points)
        scene_pcd_colors = np.asarray(scene_pcd.colors)
        
        # get label color 
        label_pcd_colors = np.asarray(scene_pcd.colors)

        # 转换为 PyTorch 张量，并添加一个维度以匹配 Batch 的维度
        scene_pcd_tensor = torch.tensor(scene_pcd_points, dtype=torch.float32).unsqueeze(0)
        scene_color_tensor = torch.tensor(scene_pcd_colors, dtype=torch.float32).unsqueeze(0)

        # 将点云移动到 GPU（如果可用的话）
        scene_pcd_tensor = scene_pcd_tensor.to("cuda")
        scene_color_tensor = scene_color_tensor.to("cuda")

        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor, 4096)

        # 将 FPS 索引转换为 NumPy 数组
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()

        # 使用 FPS 索引从原始点云中获取对应的点坐标
        fps_points_scene_from_original = scene_pcd_points[fps_indices_scene_np]
        fps_colors_scene_from_original = scene_pcd_colors[fps_indices_scene_np]

        # 获取 FPS 后的点云颜色
        fps_colors_from_original = label_pcd_colors[fps_indices_scene_np]

                
                # 对颜色张量进行逐元素比较和赋值
        colors_modified = np.zeros((fps_colors_from_original.shape[0],4))
        for i in range(fps_colors_from_original.shape[0]):
            if (fps_colors_from_original[i] == [0.,1.,0.]).all(): #green
                colors_modified[i] = [0.,1.,0.,0.]
            elif (fps_colors_from_original[i] == [1.,0.,0.]).all(): # red
                colors_modified[i] = [1.,0.,0.,0.]
            elif (fps_colors_from_original[i] == [0.,0.,0.]).all(): #black
                colors_modified[i] = [0.,0.,0.,1]
            elif (fps_colors_from_original[i] == [0.,0.,1.]).all(): #blue
                colors_modified[i] = [0.,0.,1.,0.]
        
        # read json and the phrase
        parent_dir = os.path.dirname(file_path)
        json_path = os.path.join(parent_dir, "text_guidance.json")
        removed_obj_name = file_path.split("/")[-1]
        name, _, des = removed_obj_name.split("-")
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


    
        return fps_points_scene_from_original, fps_colors_scene_from_original, ref_center, colors_modified, reference_obj, phrase

if __name__ == "__main__":
    folder_path = "dataset/scene_RGBD_mask"
    files = []
    items = os.listdir(folder_path)
    for item in items:
        sub_folder_path = os.path.join(folder_path, item)
        #print(sub_folder_path)
        sub_items = os.listdir(sub_folder_path)
        for sub_item in sub_items:
            sub_sub_folder_path = os.path.join(sub_folder_path, sub_item)
            if os.path.isdir(sub_sub_folder_path): 
                #print(sub_sub_folder_path)
                files.extend([sub_sub_folder_path])
    #print(files)
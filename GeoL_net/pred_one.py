"""
    Prediction the affordance on a user-defined case
"""
from GeoL_net.core.registry import registry
import open3d as o3d
import torch
import os
import numpy as np
from pointnet2_ops import pointnet2_utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json


class GeoLPlacementDataset_pred_one(Dataset):
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
        return 1
    
    
    def __getitem__(self, index):
        """
        fps_points_scene_from_original: points after FPS [4096*3]
        fps_colors_scene_from_original: colors of points after FPS [4096*3]
        colors_modified: colors label of posints after FPS [4096 * 4]
        reference_obj: name of the reference obj, text
        reference_position: position of the reference obj in the PC [3]
        phrase: guidance text, text
        """
        file_path = "dataset/scene_RGBD_mask/id164_2/keyboard_0008_blue"
        
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

        fps_indices_scene = pointnet2_utils.furthest_point_sample(scene_pcd_tensor.contiguous(), 2048)
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
        name = '_'.join(removed_obj_name.rsplit('_', 2)[:-2])
        des = removed_obj_name.split("_")[-1]
        target_name = f"the {des} {name}"
        #print(target_name)
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        reference_obj = data[target_name][0]
        phrase = data[target_name][1]
        #phrase = "put the bottle behind the glasses"

        # reference position
        #scene_ref_pcd = o3d.io.read_point_cloud(os.path.join(file_path, "mask_ref.ply"))
        #points_ref = np.asarray(scene_ref_pcd.points)
        #colors_ref = np.asarray(scene_ref_pcd.colors)
        #yellow_indices = [i for i, color in enumerate(colors_ref) if is_yellow(color)]
        #yellow_points = points_ref[yellow_indices]
        #yellow_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(yellow_points))
        #ref_center = yellow_bbox.get_center()


        sample = {
            "fps_points_scene": fps_points_scene_from_original,
            "fps_colors_scene": fps_colors_scene_from_original,
            #"ref_center": ref_center,
            "colors_modified": colors_modified_4cls,
            "ref_obj": reference_obj,
            "phrase":phrase,
            "image": rgb_image,
            "mask": colors_modified_4cls,
            "file_path": file_path # e.g dataset/scene_RGBD_mask/id167/eye_glasses_0003_black
            
        }

    
        return sample     

# initial model
MyModel = registry.get_affordance_model("GeoL_net")
model = MyModel(input_shape= (3, 480, 640), target_input_shape = (3, 128, 128)).to("cuda")
state_dict = torch.load("outputs/checkpoints/GeoL_1K_0907/ckpt_71.pth", map_location="cpu")
ckpt_dict = (state_dict["ckpt_dict"] if "ckpt_dict" in state_dict else state_dict)
model.load_state_dict(state_dict=ckpt_dict)
model.eval()
# user_defined test case
dataset = GeoLPlacementDataset_pred_one(split="train", root_dir="dataset/scene_RGBD_mask")
dataloader = DataLoader(dataset, batch_size=1)
for i, batch in enumerate(dataloader):

    for key, val in batch.items():
        if type(val) == list:
            continue
        batch[key] = val.float().to("cuda")

    outputs = model(batch=batch)["affordance"].squeeze(1)
    model.inference_heatmap_4cls(epoch="test")


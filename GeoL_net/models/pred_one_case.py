'''
    predict one case
'''

from GeoL_net.core.registry import registry
import open3d as o3d
import torch
from PIL import Image
from pointnet2_ops import pointnet2_utils
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

def is_green(color, tolerance=0.1):
    return (color[0] < tolerance and color[1] > 1 - tolerance and color[2] < tolerance)

class pred_one_case_dataset(Dataset):
    def __init__(self, scene_pcd_file_path, rgb_image_file_path, target_name, direction_text):
        self.scene_pcd_file_path = scene_pcd_file_path
        self.rgb_image_file_path = rgb_image_file_path
        self.target_name = target_name
        self.direction_text = direction_text

        cam_rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0.8, -0.6],
            [0, 0.6, 0.8]
        ])

        self.scene_pcd = o3d.io.read_point_cloud(scene_pcd_file_path)
        self.scene_pcd_points = np.asarray(self.scene_pcd.points)
        self.scene_pcd_points = (np.linalg.inv(cam_rotation_matrix) @ self.scene_pcd_points.T).T
        self.scene_pcd_tensor = torch.tensor(self.scene_pcd_points, dtype=torch.float32).unsqueeze(0)



        # 从旋转后的points中提取出绿色的点，计算绿色点的位置的均值
        self.scene_pcd_colors = np.asarray(self.scene_pcd.colors)
        green_mask = np.apply_along_axis(is_green, 1, self.scene_pcd_colors)
        green_points = self.scene_pcd_points[green_mask]
        self.green_pcd_center = np.mean(green_points, axis=0)


        self.scene_pcd_tensor = self.scene_pcd_tensor.to("cuda")

        fps_indices_scene = pointnet2_utils.furthest_point_sample(self.scene_pcd_tensor.contiguous(), 2048)
        fps_indices_scene_np = fps_indices_scene.squeeze(0).cpu().numpy()
        fps_points_scene_from_original = self.scene_pcd_points[fps_indices_scene_np]
        self.fps_points_scene_from_original = fps_points_scene_from_original

        rgb_image = Image.open(rgb_image_file_path).convert("RGB")
        if rgb_image.height != 480 or rgb_image.width != 640:
            rgb_image = rgb_image.resize(640, 480)
        rgb_image = np.asarray(rgb_image).astype(float)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))

        self.rgb_image = rgb_image
    
    def __len__(self):
        return 2
    
    def __getitem__(self, index):
        sample = {
            "fps_points_scene": self.fps_points_scene_from_original,
            "fps_colors_scene": "no colors in testing mode",
            "phrase": "no phrase in testing mode",
            "image": self.rgb_image,
            "mask": "no mask in testing mode",
            "file_path": "dataset/scene_RGBD_mask/id000/book_0000_fake", #fake
            "reference_obj": self.target_name,
            "direction_text": self.direction_text,
            "anchor_position": self.green_pcd_center
        }
        return sample




#congiguration
scene_pcd_file_path = "dataset/scene_RGBD_mask_direction_mult/id10_1/clock_0001_normal/mask_Behind.ply"
rgb_image_file_path = "dataset/scene_RGBD_mask_direction_mult/id10_1/clock_0001_normal/img.jpg"
target_name = "the red lamp"
direction_text = "Behind"

# load the model
model_cls=  registry.get_affordance_model("GeoL_net_v8")
model = model_cls(input_shape=(3,480,640),target_input_shape=(3,128,128)).to("cuda")

# load the checkpoint
state_dict = torch.load("outputs/checkpoints/GeoL_v8_8direction_0.0001lr/ckpt_441.pth", map_location="cpu")
model.load_state_dict(state_dict["ckpt_dict"])

# create the dataset
dataset = pred_one_case_dataset(scene_pcd_file_path, rgb_image_file_path, target_name, direction_text)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

model.eval()
for i ,batch in enumerate(data_loader):
    for key, val in batch.items():
        if type(val) == list:
            continue
        batch[key] = val.float().to("cuda")
    with torch.no_grad():
        result = model(batch=batch)["affordance"].squeeze(1)
        model.generate_heatmap(epoch=1)
        img_rgb_list, file_name_list, phrase_list = model.pcdheatmap2img()
    for img_rgb in img_rgb_list:
        img_rgb.save("outputs/testing/heatmap_behind.png")
        #img_rgb.show()

# visualize or save the result
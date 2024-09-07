from GeoL_net.core.registry import registry
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import torch
import torch.nn as nn
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from torch.multiprocessing import  set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

dataset_name = "GeoL_dataset_full"
dataset_dir = "dataset/scene_RGBD_mask/"
subset_size = 10
batch_size = 1
num_workers = 1

model_name = "GeoL_net"
ckpt_path = "outputs/checkpoints/GeoL_1K/ckpt_81.pth"
# 1. init dataset
dataset_cls = registry.get_dataset(dataset_name)
print("dataset name:",dataset_name)
train_dataset = dataset_cls(split="train",
                                root_dir=dataset_dir)

    #Subset
subset_indice = list(range(subset_size))
train_dataset = Subset(train_dataset, subset_indice)

    # Initialize data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=True,
    pin_memory=True,
        )

# 2. init model
model_cls = registry.get_affordance_model(model_name)
model = model_cls(
    input_shape=(3, 480, 640), 
    target_input_shape=(3, 128, 128)
).to("cuda")
pretrained_state = defaultdict(int)

def load_state(model, path, ckpt_only: bool = False):
        state_dict = torch.load(path, map_location="cpu")
        ckpt_dict = (
            state_dict["ckpt_dict"] if "ckpt_dict" in state_dict else state_dict
        )
        missing_keys = model.load_state_dict(ckpt_dict)
        return model
model = load_state(model, ckpt_path, ckpt_only=True)
model.eval()

with torch.no_grad():

    for i, batch in enumerate(train_loader):
        for key, val in batch.items():
            if type(val) == list:
                continue
            batch[key] = val.float().to('cuda')

        # Make predictions for this batch
        output_affordance = model(batch=batch)["affordance"].squeeze(1)
        scene_pcs = batch["fps_points_scene"]
        texts = batch["phrase"]
        file_name = batch["file_path"]
        
        pcs = scene_pcs
        feat = output_affordance.permute(0, 2, 1)

        # cls result
        color_map = torch.tensor([
                [1,0,0],
                [0,1,0],
                [0,0,1],
                [0,0,0]
            ]).cpu()
        predicted_color = torch.argmax(feat, dim=-1).cpu()
        colors = color_map[predicted_color]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcs[0].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors[0].cpu().numpy())
        scene_id = file_name[0].split("/")[-2]
        obj_id = file_name[0].split("/")[-1]
        o3d.io.write_point_cloud(f"outputs/model_output/point_cloud_pred/{scene_id}-{obj_id}.ply", point_cloud)

        class_1_feat = feat[:, :, 1].cpu()  # [B, Num_points]
        max_value = torch.max(class_1_feat)
        min_value = torch.min(class_1_feat)
        normalized_class_1_feat = (class_1_feat - min_value) / (max_value - min_value)
        flattened = normalized_class_1_feat.detach().numpy().flatten()
        cmap = plt.get_cmap('turbo')
        cmap = plt.get_cmap("viridis")
        color_mapped = cmap(flattened)[:, :3]  # 获取 RGB 值，忽略 alpha 通道
        colors = color_mapped.reshape(normalized_class_1_feat.shape[0], normalized_class_1_feat.shape[1], 3)

        colors = colors.astype(np.float64)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcs[0].cpu().numpy())
        point_cloud.colors = o3d.utility.Vector3dVector(colors[0])
        pc_heatmap = point_cloud
        scene_id = file_name[0].split("/")[-2]
        obj_id = file_name[0].split("/")[-1]
        o3d.io.write_point_cloud(f"outputs/model_output/point_cloud_pred/{scene_id}-{obj_id}-heatmap.ply", point_cloud)
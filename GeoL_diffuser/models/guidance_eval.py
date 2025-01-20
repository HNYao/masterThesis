"""
    Evaluation script for guidances (Affordance, Noncollision)
"""

from GeoL_net.core.registry import registry
import open3d as o3d
import torch
from PIL import Image
from pointnet2_ops import pointnet2_utils
import os
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, Subset
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import numpy as np
import torch
import json
import os
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
from GeoL_net.models.modules import ProjectColorOntoImage_v3, ProjectColorOntoImage
from scipy.spatial.distance import cdist
from matplotlib import cm
import torchvision.transforms as T
from GeoL_net.gpt.gpt import chatgpt_condition
from GeoL_diffuser.algos.pose_algos import PoseDiffusionModel
from GeoL_diffuser.models.guidance import *
from GeoL_diffuser.models.utils.fit_plane import *
from GeoL_diffuser.dataset.dataset import PoseDataset_top
import yaml
from omegaconf import OmegaConf
import trimesh
from GeoL_diffuser.models.helpers import TSDFVolume, get_view_frustum
seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# initialization
eval_guidance = "affordance" # "affordance" or "noncollision"
class_free_guide_w = 0
apply_guidance = True
guide_clean = True

# initialize diffusion model and load checkpoint
with open("config/baseline/diffusion.yaml", "r") as file:
    yaml_data = yaml.safe_load(file)
config_diffusion = OmegaConf.create(yaml_data)
model_diffuser_cls = PoseDiffusionModel
model_diffuser = model_diffuser_cls(config_diffusion.model).to("cuda")
state_diffusion_dict = torch.load("outputs/checkpoints/GeoL_diffuser_v0__topk_1K/ckpt_21.pth", map_location="cpu") # test
model_diffuser.load_state_dict(state_diffusion_dict["ckpt_dict"])
model_diffuser.eval()

# set guidance
if eval_guidance == "affordance":
    guidance = AffordanceGuidance_v2()
elif eval_guidance == "noncollision":
    guidance = NonCollisionGuidance_v2()
model_diffuser.nets["policy"].set_guidance(guidance)

# set dataset, temporary, use the PoseDataset_top
dataset = PoseDataset_top(split="train", root_dir="dataset/scene_RGBD_mask_v2_kinect_cfg")
subset_size = 100
random_indices = random.sample(range(len(dataset)), subset_size)
subset = Subset(dataset, random_indices)
dataloader = DataLoader(subset, batch_size=24, shuffle=False)

guide_loss_mean = []
guide_loss_max = []
guide_loss_min = []
for i, batch in enumerate(dataloader):
    for key, val in batch.items():
        if type(val) == list:
            continue
        batch[key] = val.float().to("cuda")
    
    with torch.no_grad():
        pred = model_diffuser(batch, class_free_guide_w=class_free_guide_w, apply_guidance=apply_guidance, guide_clean=guide_clean)
        if eval_guidance == "affordance":
            guide_loss_mean.append(pred['guide_losses']['affordance_error'].mean().item())
            guide_loss_max.append(pred['guide_losses']['affordance_error'].max().item())
            guide_loss_min.append(pred['guide_losses']['affordance_error'].min().item())

        elif eval_guidance == "noncollision":
            None

print(f"Mean: {np.mean(guide_loss_mean)}")
print(f"Max: {np.mean(guide_loss_max)}")
print(f"Min: {np.mean(guide_loss_min)}")

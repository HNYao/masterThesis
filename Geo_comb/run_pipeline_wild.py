import open3d as o3d
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import stretch_utils.data_utils as DataUtils
# import time
import numpy as np
# import torch
# import cv2
from scipy.spatial.transform import Rotation as SciR
from Geo_comb.full_pipeline import full_pipeline_v2, retrieve_obj_mesh,seed_everything,predict_depth
from GeoL_net.core.registry import registry
import open3d as o3d
import torch
from PIL import Image
from pointnet2_ops import pointnet2_utils
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from GroundingDINO.groundingdino.util.inference import load_model
import cv2
import numpy as np
import torch
import json
import os
import yaml
from omegaconf import OmegaConf
import matplotlib.pylab as plt
import copy 
import argparse
from GeoL_diffuser.models.guidance import CompositeGuidance
from GeoL_diffuser.algos.pose_algos import PoseDiffusionModel
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points

def load_models(args):
    intrinsics = np.loadtxt(args.intr_path)
    model_detection = load_model(
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
        "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )
    model_detection = model_detection.to("cuda")
    model_detection.eval()
    
    # affordance model
    model_affordance_cls = registry.get_affordance_model("GeoL_net_v9")
    model_affordance = model_affordance_cls(
        input_shape=(3, 720, 1280),
        target_input_shape=(3, 128, 128),
        intrinsics=intrinsics,
    ).to("cuda")
    state_affordance_dict = torch.load("data_and_weights/ckpt_11.pth", map_location="cpu")
    model_affordance.load_state_dict(state_affordance_dict["ckpt_dict"])
    model_affordance.eval()
    
    # diffuser model
    guidance = CompositeGuidance()
    with open("config/baseline/diffusion.yaml", "r") as file:
        yaml_data = yaml.safe_load(file)
    config_diffusion = OmegaConf.create(yaml_data)
    model_diffuser_cls = PoseDiffusionModel
    model_diffuser = model_diffuser_cls(config_diffusion.model).to("cuda")
    state_diffusion_dict = torch.load("data_and_weights/ckpt_93.pth", map_location="cpu")
    model_diffuser.load_state_dict(state_diffusion_dict["ckpt_dict"])
    model_diffuser.nets["policy"].set_guidance(guidance)
    model_diffuser.eval()
    
    # depth model
    if args.use_m3d:    
        model_depth = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
        model_depth = model_depth.to("cuda")
        model_depth.eval()
    else:
        model_depth = None
    
    return model_detection, model_affordance, model_diffuser, model_depth

def run(args):
    
    # Load the models
    model_detection, model_affordance, model_diffuser, model_depth = load_models(args)

    # Read the images
    rgb_image = cv2.imread(args.color_path)
    depth_image_raw = cv2.imread(args.depth_path, cv2.IMREAD_ANYDEPTH)
    intr = np.loadtxt(args.intr_path)
    obj_mesh = retrieve_obj_mesh(args.mesh_category, target_size=args.target_size)
    if args.use_m3d:
        depth_save_path = args.depth_path.replace(".png", "_m3d.png")
        # if os.path.exists(depth_save_path):
        #     depth_image_pred = cv2.imread(depth_save_path, cv2.IMREAD_ANYDEPTH)
        # else:
        depth_image_pred, _ = predict_depth(model_depth, rgb_image, intr)
        depth_image_pred = (depth_image_pred * 1000).astype(np.uint16)
        cv2.imwrite(depth_save_path, depth_image_pred)
    else:
        depth_image = depth_image_raw
    
    # Fill the holes in the depth image
    if args.use_m3d:
        depth_image_raw[depth_image_raw == 0] = depth_image_pred[depth_image_raw == 0]
        depth_image_raw[depth_image_raw > 1500] = 0
        depth_image_pred[depth_image_pred > 1500] = 0
    depth_mask = np.zeros_like(depth_image_pred)
    padding_size = 50
    depth_mask[padding_size:-padding_size, padding_size:-padding_size] = 1
    depth_image = depth_image_pred
    depth_image[depth_image < 500] = 0
    depth_image = depth_image * depth_mask
    
    # Viualize the point cloud
    pc, scene_ids = backproject(depth_image / 1000, intr, depth_image / 1000 > 0)
    pc_colors = rgb_image[scene_ids[0], scene_ids[1]][..., [2, 1, 0]] / 255.0
    pcd = visualize_points(pc, pc_colors)
    o3d.visualization.draw_geometries([pcd])
    
    # Do the inference
    seed_everything(42)
    pred_xyz_all, pred_r_all, pred_cost = full_pipeline_v2(
        model_detection=model_detection,
        model_affordance=model_affordance,
        model_diffuser=model_diffuser,
        rgb_image=rgb_image,
        depth_image=depth_image,
        obj_mesh=obj_mesh,
        intrinsics=intr,
        target_names=args.target_names,
        direction_texts=args.direction_texts,
        use_vlm=args.use_vlm,
        fast_vlm_detection=args.fast_vlm_detection,
        use_kmeans=args.use_kmeans,
        visualize_affordance=args.visualize_affordance,
        visualize_diff=args.visualize_diff,
        visualize_final_obj=args.visualize_final_obj,
        rendering=args.rendering,
    )

    # TODO: Save npz files for rendering
    results = {}
    results["pred_xyz_all"] = pred_xyz_all
    results["pred_r_all"] = pred_r_all
    results["pred_cost"] = pred_cost
    results["rgb_image"] = rgb_image
    results["depth_image_raw"] = depth_image_raw
    results["depth_image"] = depth_image
    results["intrinsics"] = intr
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_m3d", action="store_true")
    parser.add_argument("--intr_path", type=str, default=".tmp/wild_intr.txt")
    parser.add_argument("--color_path", type=str, default=".tmp/wild_color.png")
    parser.add_argument("--depth_path", type=str, default=".tmp/wild_depth.png")
    parser.add_argument("-c", "--mesh_category",  type=str, default="phone")
    parser.add_argument("-s", "--target_size", type=float, default=0.1)
    parser.add_argument("-v", "--visualize_final_obj", action="store_true")

    parser.add_argument("--use_vlm", type=bool, default=True)
    parser.add_argument("--fast_vlm_detection", type=bool, default=True)
    parser.add_argument("--use_kmeans", type=bool, default=True)
    
    parser.add_argument("--target_names", type=str, default=["Mouse"])
    parser.add_argument("--direction_texts", type=str, default=["Right"])
    
    parser.add_argument("--visualize_affordance", action="store_true")
    parser.add_argument("--visualize_diff", action="store_true")
    parser.add_argument("--rendering", action="store_true")
    args = parser.parse_args()

    run(args)

    

import open3d as o3d

from controller_base import ControllerBase
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import stretch_utils.data_utils as DataUtils
# import time
import numpy as np
# import torch
# import cv2
from scipy.spatial.transform import Rotation as SciR
from Geo_comb.full_pipeline import full_pipeline_v2
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
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
from GeoL_net.models.modules import ProjectColorOntoImage_v3, ProjectColorOntoImage
from GeoL_diffuser.models.guidance import *
from GeoL_diffuser.models.utils.fit_plane import *
from scipy.spatial.distance import cdist
from matplotlib import cm
import torchvision.transforms as T
from GeoL_net.gpt.gpt import chatgpt_condition, chatgpt_select_id
from GeoL_diffuser.algos.pose_algos import PoseDiffusionModel
import yaml
from omegaconf import OmegaConf
import matplotlib.pylab as plt
import copy 

INTRINSICS_HEAD = np.array([
            [910.68, 0, 626.58],
            [0, 911.09, 377.44],
            [0, 0, 1],
            ])

ROTATION_MATRIX_X180 = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )

def predict_depth(depth_model, rgb_origin, intr, input_size = (616, 1064)):
    intrinsic = [intr[0, 0], intr[1, 1],
                 intr[0, 2], intr[1, 2]]  # fx, fy, cx, cy
    # ajust input size to fit pretrained model
    # keep ratio resize
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic, hold depth
    intrinsic = [intrinsic[0] * scale, intrinsic[1] *
                 scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half,
                             pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    # normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()

    ###################### canonical camera space ######################
    # inference
    with torch.no_grad():
        pred_depth, confidence, output_dict = depth_model.inference({
                                                                    'input': rgb})

    # un pad
    pred_depth = pred_depth.squeeze()
    confidence = confidence.squeeze()
    pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] -
                            pad_info[1], pad_info[2]: pred_depth.shape[1] - pad_info[3]]
    confidence = confidence[pad_info[0]: confidence.shape[0] -
                            pad_info[1], pad_info[2]: confidence.shape[1] - pad_info[3]]

    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='nearest').squeeze()
    confidence = torch.nn.functional.interpolate(
        confidence[None, None, :, :], rgb_origin.shape[:2], mode='nearest').squeeze()
    ###################### canonical camera space ######################

    # de-canonical transform
    # 1000.0 is the focal length of canonical camera
    canonical_to_real_scale = intrinsic[0] / 1000.0
    pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 5)
    pred_depth = pred_depth.cpu().numpy()
    confidence = confidence.cpu().numpy()
    return pred_depth, confidence


class HephaisbotPlacementController(ControllerBase):
    def __init__(self, cfg=None, use_monodepth=True, dummy_place=False):
        robot_calib = DataUtils.load_json("./config_base_cam.json")
        T_base_headcam = np.array(robot_calib["T_base_headcam"]).reshape(4,4)
        self.T_base_headcam = T_base_headcam
        self.raw_intr = INTRINSICS_HEAD
        super().__init__(cfg=cfg)   
        print("Done with controller initialization!")     
        self.use_monodepth = use_monodepth
        self.dummy_place = dummy_place
        
        if not self.dummy_place:
            # Detection model
            model_detection = load_model(
                "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth"
            )
            self.model_detection = model_detection.to("cuda")
            self.model_detection.eval()
            
            # affordance model
            model_affordance_cls = registry.get_affordance_model("GeoL_net_v9")
            self.model_affordance = model_affordance_cls(
                input_shape=(3, 720, 1280),
                target_input_shape=(3, 128, 128),
                intrinsics=INTRINSICS_HEAD,
            ).to("cuda")
            state_affordance_dict = torch.load("data_and_weights/ckpt_11.pth", map_location="cpu")
            self.model_affordance.load_state_dict(state_affordance_dict["ckpt_dict"])
            self.model_affordance.eval()
            
            # diffuser model
            guidance = CompositeGuidance()
            with open("config/baseline/diffusion.yaml", "r") as file:
                yaml_data = yaml.safe_load(file)
            config_diffusion = OmegaConf.create(yaml_data)
            model_diffuser_cls = PoseDiffusionModel
            self.model_diffuser = model_diffuser_cls(config_diffusion.model).to("cuda")
            state_diffusion_dict = torch.load("data_and_weights/ckpt_93.pth", map_location="cpu")
            self.model_diffuser.load_state_dict(state_diffusion_dict["ckpt_dict"])
            self.model_diffuser.nets["policy"].set_guidance(guidance)
            self.model_diffuser.eval()
            
        if use_monodepth:
            self.depth_model = torch.hub.load(
                'yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
            self.depth_model.cuda().eval()
        else:
            self.depth_model = None

    def inference(self, T_object_hand, obj_mesh, height_offset=0.12, cut_mode="center", verbose=True, debug=False):
        color, depth, intr, T_calib = self._subscribe_image(cut_mode)
        if self.use_monodepth:
            print("Using Metric3D model for depth prediction")
            img_mask = color.sum(-1) > 0
            img_mask = cv2.erode(img_mask.astype(np.uint8), np.ones((25, 25)))
            depth, _ = predict_depth(self.depth_model, color, intr)
            depth[~img_mask] = 0
            depth[depth < 0.3] = 0
        points_scene, scene_ids = DataUtils.backproject(
            depth,
            intr,
            depth < 1.5,
            NOCS_convention=False,
        )
        colors_scene = color[scene_ids[0], scene_ids[1]] / 255.0
        obj_mesh_vis =  copy.deepcopy(obj_mesh)
        obj_mesh_vis.compute_vertex_normals()
        obj_mesh_vis.rotate(ROTATION_MATRIX_X180[:3, :3])
        T_base_headcam = self.T_base_headcam @ T_calib
        
        ##### Dummy inference by manual selection ####
        if debug and self.dummy_place:
            place_pos_samples = DataUtils.pick_points_in_viewer(
                points_scene, colors_scene
            )
            place_pos = place_pos_samples.mean(axis=0)
            # place_pos = np.array([-0.072, 0.23, 1])
            place_ang = 10
        else:
            color = color[..., ::-1].copy().astype(np.uint8)
            depth = (depth * 1000).astype(np.uint16)
            T_camera_plane = np.linalg.inv(T_base_headcam)
            pred_xyz_all, pred_r_all, pred_cost = full_pipeline_v2(
                model_detection=self.model_detection,
                model_affordance=self.model_affordance,
                model_diffuser=self.model_diffuser,
                rgb_image=color,
                depth_image=depth,
                obj_mesh=obj_mesh,
                intrinsics=intr,
                target_name=["Monitor"],
                direction_text=["Front"],
                use_vlm=False,
                use_kmeans=True,
                visualize_affordance=True,
                visualize_diff=False,
                visualize_final_obj=True,
                rendering = False,
            )
            place_pos = pred_xyz_all[np.argmin(pred_cost)]
            place_ang = pred_r_all[np.argmin(pred_cost)]
        
        ##### Dummy inference by manual selection ####
        dR_object = SciR.from_euler("Z", -place_ang, degrees=True).as_matrix()
        # Solve for T_base_object
        T_headcam_object = np.eye(4)
        T_headcam_object[:3, 3] = place_pos
        T_base_object = T_base_headcam @ T_headcam_object
        
        Rx_base_obj, Ry_base_obj, Rz_base_obj = T_base_object[:3, 0], T_base_object[:3, 1], T_base_object[:3, 2]
        Rz_base_obj = np.array([0, 0, 1])
        Ry_base_obj = np.cross(Rz_base_obj, Rx_base_obj)
        Rx_base_obj = np.cross(Ry_base_obj, Rz_base_obj)
        R_base_object = np.stack([Rx_base_obj, Ry_base_obj, Rz_base_obj], axis=1)
        T_base_object[:3, :3] = R_base_object @ dR_object 
        T_base_object[2, 3] += height_offset
        T_base_hand = T_base_object @ T_object_hand

        if verbose:
            # current_size = mesh_obj.get_axis_aligned_bounding_box().get_extent()
            # obj_target_size = [0.8, 0.2, 0.01]
            # obj_scale = np.array([obj_target_size[0]/ current_size[0], 
            #                       obj_target_size[1]/ current_size[1], 
            #                       obj_target_size[2]/ current_size[2]])
            # obj_scale_matrix = np.array([
            #     [obj_scale[0], 0, 0, 0],
            #     [0, obj_scale[1], 0, 0],
            #     [0, 0, obj_scale[2], 0],
            #     [0,0,0,1]
            # ])
            # obj_inverse_matrix = np.array(
            #     [
            #         [1, 0, 0],
            #         [0, -1, 0],
            #         [0, 0, -1],
            #     ]
            # )
            # mesh_obj.paint_uniform_color([0,1,0])
            # mesh_obj.compute_vertex_normals()
            # mesh_obj.transform(obj_scale_matrix)
            # mesh_obj.rotate(obj_inverse_matrix, center=[0, 0, 0])  # rotate obj mesh
            # mesh_obj.rotate(SciR.from_euler("X", 180, degrees=True).as_matrix())
            
            # Transform mesh to the target configuration
            obj_mesh_vis.transform(T_base_object)
            
            # place_pos_in_base = place_pos @ T_base_headcam[:3, :3].T + T_base_headcam[:3, 3]
            # place_pos_in_base[2] += 0.05 # Compensation
            print("... Visualize the inference results ...")
            points_scene_in_base = points_scene @ T_base_headcam[:3, :3].T + T_base_headcam[:3, 3]
            pcd_scene = DataUtils.visualize_points(points_scene_in_base, colors_scene)
            vis_place_pos_in_base = DataUtils.visualize_sphere_o3d(T_headcam_object[:3, 3], size=0.05)
            axis_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            axis_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            axis_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            axis_hand = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

            axis_cam.transform(T_base_headcam)
            axis_obj.transform(T_base_object)
            axis_hand.transform(T_base_hand)
            vis = [pcd_scene, vis_place_pos_in_base, axis_base, axis_cam, axis_obj, axis_hand, obj_mesh_vis]
            o3d.visualization.draw(vis)
        
        # place_pos_in_base += TRAJ_OFFSET
        # return place_pos_in_base
        return T_base_object
        
if __name__ == "__main__":
    TRAJ_OFFSET = np.array([0, -0.23, 0.0])

    zrot_180 = SciR.from_euler("z", 180, degrees=True).as_matrix()
    yrot_90 = SciR.from_euler("y", 90, degrees=True).as_matrix()

    T_object_hand = np.eye(4)
    T_object_hand[:3, :3] = yrot_90 @ zrot_180 @ T_object_hand[:3, :3]
    T_object_hand[:3, 3] = TRAJ_OFFSET
    controller_cfg = {
        "config_network": "./network_config.yaml"
    }
    controller_cfg = edict(controller_cfg)
    controller = HephaisbotPlacementController(controller_cfg, use_monodepth=True)

    # while True:
    obj_mesh = o3d.io.read_triangle_mesh("data_and_weights/mesh/keyboard/keyboard_0001_black/mesh.obj")
    obj_target_size = [0.8, 0.2, 0.01] # H W D
    obj_mesh.compute_vertex_normals()
    current_size = obj_mesh.get_axis_aligned_bounding_box().get_extent()
    obj_scale = np.array([obj_target_size[0]/ current_size[0], obj_target_size[1]/ current_size[1], obj_target_size[2]/ current_size[2]])
    obj_scale_matrix = np.array([
        [obj_scale[0], 0, 0, 0],
        [0, obj_scale[1], 0, 0],
        [0, 0, obj_scale[2], 0],
        [0,0,0,1]
    ])

    obj_mesh.transform(obj_scale_matrix)
    obj_mesh.rotate(ROTATION_MATRIX_X180, center=[0, 0, 0])  # rotate obj mesh
    
    controller.inference(T_object_hand, obj_mesh, height_offset=0.05, cut_mode="full")

    # controller.run(
    #     instruction="pickup thing",
    #     object_name="ball",
    #     click_contact=True,
    #     verbose=True,
    #     save_name="demo_ball",
    #     traj_scale=0.5
    # )
    # controller._publish_trajectory(np.zeros([81,3]))
# scp hello-robot@192.168.1.2:/home/hello-robot/stretch_manip/config_base_cam.json /home/cvai/hanzhi_ws/egoprior-diffuser/policy_server
# export PYTHONPATH="${PYTHONPATH}:$PWD"
# export PYTHONPATH="${PYTHONPATH}:/usr/lib/python3/dist-packages"     

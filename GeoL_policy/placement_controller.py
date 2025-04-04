import open3d as o3d

from GeoL_policy.controller_base import ControllerBase
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import stretch_utils.data_utils as DataUtils
import numpy as np
from scipy.spatial.transform import Rotation as SciR
from Geo_comb.full_pipeline import full_pipeline_v2, predict_depth, retrieve_obj_mesh 
from GeoL_net.core.registry import registry
import open3d as o3d
import torch
from PIL import Image
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
from GeoL_diffuser.algos.pose_algos import PoseDiffusionModel
from GeoL_diffuser.models.guidance import CompositeGuidance
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

    def inference(self, 
                T_object_hand, 
                obj_mesh, 
                target_names=["Monitor"],
                direction_texts=["Front"],
                use_vlm=False,
                fast_vlm_detection=False,
                use_kmeans=True,
                visualize_affordance=False,
                visualize_diff=False,
                visualize_final_obj=True,
                rendering = False,
                height_offset=0.12, 
                cut_mode="center", 
                verbose=True, 
                debug=False):
        color, depth, intr, T_calib = self._subscribe_image(cut_mode)
        if self.use_monodepth:
            print("Using Metric3D model for depth prediction")
            if cut_mode == "full":
                offset = 25
                padding = (1280 - 405) // 2
                img_mask = np.zeros_like(color[..., 0])
                img_mask[:, padding + offset: padding + 405 - offset] = 1
            else:
                img_mask = np.ones_like(color[..., 0])
            # color = (color * img_mask[..., None]).astype(np.uint8)
            depth, _ = predict_depth(self.depth_model, color, intr)
            depth = depth * img_mask
            depth[depth < 0.5] = 0
            
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
            place_ang = 10
        else:
            # pcd_scene = visualize_points(points_scene, colors_scene)
            # o3d.visualization.draw([pcd_scene])
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
                target_names=target_names,
                direction_texts=direction_texts,
                use_vlm=use_vlm,
                fast_vlm_detection=fast_vlm_detection,
                use_kmeans=use_kmeans,
                visualize_affordance=visualize_affordance,
                visualize_diff=visualize_diff,
                visualize_final_obj=visualize_final_obj,
                rendering = rendering,
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
    obj_mesh = retrieve_obj_mesh("phone", target_size=0.1)
    controller.inference(T_object_hand, 
                         obj_mesh, 
                         target_names=["Keyboard", ],
                         direction_texts=["Right", ],
                         use_vlm=True,
                         use_kmeans=True,
                         fast_vlm_detection=True,
                         visualize_affordance=False,
                         visualize_diff=False,
                         visualize_final_obj=True,
                         height_offset=0.05, 
                         cut_mode="full",
                         rendering=True)

 
# scp hello-robot@192.168.1.2:/home/hello-robot/stretch_manip/config_base_cam.json /home/cvai/hanzhi_ws/egoprior-diffuser/policy_server
# export PYTHONPATH="${PYTHONPATH}:$PWD"
# export PYTHONPATH="${PYTHONPATH}:/usr/lib/python3/dist-packages"     

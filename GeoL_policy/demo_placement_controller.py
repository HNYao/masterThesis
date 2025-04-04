import open3d as o3d
import sys
sys.path.append("thirdpart/GroundingDINO")
from GeoL_policy.controller_base import ControllerBase
from omegaconf import OmegaConf
from easydict import EasyDict as edict
import stretch_utils.data_utils as DataUtils
# import time
import numpy as np
# import torch
# import cv2
from scipy.spatial.transform import Rotation as SciR
import matplotlib.pyplot as plt
INTRINSICS_HEAD = np.array([
    [910.68, 0, 626.58],
    [0, 911.09, 377.44],
    [0, 0, 1]
    ])

class DemoPlacementController(ControllerBase):
    def __init__(self, cfg=None):
        robot_calib = DataUtils.load_json("./stretch_config/config_base_cam.json")
        T_base_headcam = np.array(robot_calib["T_base_headcam"]).reshape(4,4)
        self.T_base_headcam = T_base_headcam
        self.raw_intr = INTRINSICS_HEAD
        super().__init__(cfg=cfg)   
        print("Done with controller initialization!")     
    
    def inference(self, T_object_hand, height_offset=0.12, cut_mode="full", verbose=True):
        color, depth, intr, T_calib = self._subscribe_image(cut_mode)
        print(intr)
        points, scene_ids = DataUtils.backproject(depth, intr, depth<5, False)
        points_scene, scene_ids = DataUtils.backproject(
            depth,
            intr,
            depth < 1.5,
            NOCS_convention=False,
        )
        colors_scene = color[scene_ids[0], scene_ids[1]] / 255.0
        
        ##### Dummy inference by manual selection ####
        place_pos_samples = DataUtils.pick_points_in_viewer(
            points_scene, colors_scene
        )
        place_pos = place_pos_samples.mean(axis=0)
        # place_pos = np.array([-0.072, 0.23, 1])
        place_ang = 30

        ##### Dummy inference by manual selection ####
        
        dR_object = SciR.from_euler("Z", place_ang, degrees=True).as_matrix()
        # Solve for T_base_object
        T_base_headcam = self.T_base_headcam @ T_calib
        T_headcam_object = np.eye(4)
        T_headcam_object[:3, 3] = place_pos
        # T_headcam_object[:3, :3] = SciR.from_euler("Z", 20, degrees=True).as_matrix()
        T_base_object = T_base_headcam @ T_headcam_object
        
        Rx_base_obj, Ry_base_obj, Rz_base_obj = T_base_object[:3, 0], T_base_object[:3, 1], T_base_object[:3, 2]
        Rz_base_obj = np.array([0, 0, 1])
        Ry_base_obj = np.cross(Rz_base_obj, Rx_base_obj)
        Rx_base_obj = np.cross(Ry_base_obj, Rz_base_obj)
        R_base_object = np.stack([Rx_base_obj, Ry_base_obj, Rz_base_obj], axis=1)
        T_base_object[:3, :3] = R_base_object @ dR_object 
        T_base_object[2, 3] += height_offset
        T_base_hand = T_base_object @ T_object_hand

        mesh_obj = o3d.io.read_triangle_mesh("mesh.obj")
        mesh_obj.scale(0.001, center=[0, 0, 0])    
        mesh_obj.paint_uniform_color([0,1,0])
        mesh_obj.compute_vertex_normals()
        mesh_obj.transform(T_base_object)
        # place_pos_in_base = place_pos @ T_base_headcam[:3, :3].T + T_base_headcam[:3, 3]
        # place_pos_in_base[2] += 0.05 # Compensation
        if verbose:
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
            vis = [pcd_scene, vis_place_pos_in_base, axis_base, axis_cam, axis_obj, axis_hand, mesh_obj]
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
    controller = DemoPlacementController(controller_cfg)
    while True:
        controller.inference(T_object_hand)
   
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
# hostname -I
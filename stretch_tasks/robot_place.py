from GeoL_policy.placement_controller import HephaisbotPlacementController
from GeoL_policy.demo_placement_controller import DemoPlacementController
from Geo_comb.full_pipeline import retrieve_obj_mesh
import stretch_utils.data_utils as DataUtils
import time
import numpy as np
import torch
import cv2
import open3d as o3d
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation as SciR
import argparse

TRAJ_ROT_INIT = np.array([-np.pi/2, 0, -np.pi])
TRAJ_ROT_INIT = SciR.from_euler("xyz", TRAJ_ROT_INIT, degrees=False).as_matrix()
TRAJ_OFFSET = np.array([0, -0.28, 0.03])
TRAJ_TRA_INIT = np.array([0.0, -0.393, 0.8]) - TRAJ_OFFSET
GRIPPER_OPENNING = 0.3

# Pre-defined transformations
yrot_90 = SciR.from_euler("y", 80, degrees=True).as_matrix()
xrot_10 = SciR.from_euler("x", -20, degrees=True).as_matrix()
zrot_90 = SciR.from_euler("z", 100, degrees=True).as_matrix()
zrot_180 = SciR.from_euler("z", 180, degrees=True).as_matrix()
T_yrot, T_xrot, T_zrot90, T_zrot180 = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
T_yrot[:3, :3] = yrot_90
T_xrot[:3, :3] = xrot_10
T_zrot90[:3, :3] = zrot_90
T_zrot180[:3, :3] = zrot_180

# Initialize the controller
controller_cfg = {
    "config_network": "./stretch_config/network_config.yaml"
}
controller_cfg = edict(controller_cfg)
controller = HephaisbotPlacementController(controller_cfg, use_monodepth=False)
time.sleep(2)
print(" ====================== Done with controller initialization! ====================== ")
def publish_action(controller, T_base_hand, openning=1):
    traj_rot = SciR.from_matrix(T_base_hand[:3, :3]).as_euler("xyz", False)
    traj_grip = np.ones([1]).astype(np.int32) * openning # [1]
    traj = np.concatenate([T_base_hand[:3, 3], traj_rot, traj_grip], axis=-1)
    traj = traj[None]
    time.sleep(0.2)
    controller._publish_trajectory(traj)
    time.sleep(0.2)
    return traj

def main(args):
    vert_grasp = args.vert_grasp
    mesh_category = args.mesh_category
    target_size = args.target_size
    disable_rotation = args.disable_rotation
    
    # Compute the initial T_object_hand configuration
    T_base_hand, T_object_hand = np.eye(4), np.eye(4)
    T_object_hand[:3, :3] = zrot_180 @ T_object_hand[:3, :3]
    T_object_hand[:3, 3] = TRAJ_OFFSET
    if vert_grasp:
        # rotate the object hand around y_axis
        T_object_hand = T_yrot @ T_object_hand

    # Step 0: Rotate the gripper around so that the gripper is not visible
    T_base_hand = np.eye(4)
    T_base_hand[:3, :3] = T_zrot90[:3, :3] @ TRAJ_ROT_INIT
    T_base_hand[:3, 3] = TRAJ_TRA_INIT
    if args.verbose:
        breakpoint()
    else:
        time.sleep(5)
    print(" ====================== Step 0: Rotate the gripper around so that the gripper is not visible ======================")
    publish_action(controller, T_base_hand)

    # Step 1: Parse the scene for the placement configuration
    print(" ====================== Step 1: Parse the scene for the placement configuration ====================== ")
    obj_mesh = retrieve_obj_mesh(mesh_category, target_size=target_size)
    # T_base_object_to_place = controller.inference(T_object_hand, height_offset=0.07, cut_mode="full")
    T_base_object_to_place = controller.inference(
                            T_object_hand, 
                            obj_mesh, 
                            target_names=["Monitor", ],
                            direction_texts=["Right Front", ],
                            use_vlm=True,
                            use_kmeans=True,
                            fast_vlm_detection=True,
                            visualize_affordance=False,
                            visualize_diff=False,
                            visualize_final_obj=True,
                            height_offset=args.height_offset, 
                            disable_rotation=disable_rotation,
                            cut_mode="full",
                            rendering=True,
                            verbose=True,
                            debug=False)

    # Step 2: Grasp the object, rotate gripper around y_axis if necessary
    T_base_object = np.linalg.inv(T_zrot180)
    T_base_object[:3, 3] = TRAJ_TRA_INIT + TRAJ_OFFSET
    T_base_hand = T_base_object @ T_object_hand
    T_base_hand[:3, :3] = T_base_hand[:3, :3] @ TRAJ_ROT_INIT
    Y_MIN, Y_MAX = -0.7, T_base_hand[1, 3]
    time.sleep(2)
    traj = publish_action(controller, T_base_hand)
    time.sleep(2)
    breakpoint()
    print(" ====================== Step 2: Grasp the object ======================")
    controller._publish_gripper([GRIPPER_OPENNING])

    # Step 3: Go to the placement configuration
    T_base_object_to_place[1, 3] = np.clip(T_base_object_to_place[1, 3], a_min=Y_MIN, a_max=Y_MAX)
    if vert_grasp:
        T_object_hand = T_xrot @ T_object_hand
    T_base_hand = T_base_object_to_place @ T_object_hand
    T_base_hand[:3, :3] = T_base_hand[:3, :3] @ TRAJ_ROT_INIT
    print(" =================== START RECORDING =================== ")
    print(" =================== START RECORDING =================== ")
    print(" =================== START RECORDING =================== ")
    print(" =================== START RECORDING =================== ")
    print(" =================== START RECORDING =================== ")
    print(" =================== START RECORDING =================== ")
    print(" =================== START RECORDING =================== ")
    print(" ====================== Step 3: Go to the refined placement configuration ====================== ")
    breakpoint()
    traj = publish_action(controller, T_base_hand, GRIPPER_OPENNING)
    time.sleep(2)

    # Step 4: Open the gripper
    traj_post = traj.copy()
    traj_post[:, -1] = 1
    if args.verbose:
        breakpoint()
    else:
        time.sleep(2)
    print(" ====================== Step 4: Open the gripper to release the object  ====================== ")
    traj = controller._publish_trajectory(traj_post)

    # Step 5: Slowly release the object
    T_base_hand[:3, 3] += np.array([0, 0.2, 0.0])
    print(" ====================== Step 5: Make sure the object is released  ====================== ")
    time.sleep(2)
    traj = publish_action(controller, T_base_hand)
    if args.verbose:
        breakpoint()
    else:
        time.sleep(2)

    # Step 6: Home
    T_base_hand, T_object_hand = np.eye(4), np.eye(4)
    T_object_hand[:3, :3] = zrot_180 @ T_object_hand[:3, :3]
    T_object_hand[:3, 3] = TRAJ_OFFSET
    if vert_grasp:
        T_object_hand = T_xrot @ T_yrot @ T_object_hand
    T_base_object = np.linalg.inv(T_zrot180)
    T_base_object[:3, 3] = TRAJ_TRA_INIT + TRAJ_OFFSET
    T_base_hand = T_base_object @ T_object_hand
    T_base_hand[:3, :3] = T_base_hand[:3, :3] @ TRAJ_ROT_INIT
    print("Step 6: Robot home!")
    if args.verbose:
        breakpoint()
    else:
        time.sleep(2)
    traj = publish_action(controller, T_base_hand)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--mesh_category",  type=str, default="phone")
    parser.add_argument("-s", "--target_size", type=float, default=0.1)
    parser.add_argument("-o", "--height_offset", type=float, default=0.05)
    parser.add_argument("-v", "--vert_grasp", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable_rotation", action="store_true")
    args = parser.parse_args()
    main(args)

    

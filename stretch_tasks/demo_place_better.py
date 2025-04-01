from demo_placement_controller import DemoPlacementController
import stretch_utils.data_utils as DataUtils
import time
import numpy as np
import torch
import cv2
import open3d as o3d
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation as SciR

TRAJ_ROT_INIT = np.array([-np.pi/2, 0, -np.pi])
TRAJ_ROT_INIT = SciR.from_euler("xyz", TRAJ_ROT_INIT, degrees=False).as_matrix()
TRAJ_OFFSET = np.array([0, -0.23, 0.03])
GRIPPER_OPENNING = 0.3

HEIGHT_OFFSET = 0.12
VERT_GRASP = True

HEIGHT_OFFSET = 0.07
VERT_GRASP = False

def publish_action(controller, T_base_hand, openning=1):
    traj_rot = SciR.from_matrix(T_base_hand[:3, :3]).as_euler("xyz", False)
    traj_grip = np.ones([1]).astype(np.int32) * openning # [1]
    traj = np.concatenate([T_base_hand[:3, 3], traj_rot, traj_grip], axis=-1)
    traj = traj[None]
    time.sleep(0.2)
    controller._publish_trajectory(traj)
    time.sleep(0.2)
    return traj

yrot_90 = SciR.from_euler("y", 80, degrees=True).as_matrix()
xrot_10 = SciR.from_euler("x", -10, degrees=True).as_matrix()
zrot_180 = SciR.from_euler("z", 180, degrees=True).as_matrix()
T_yrot, T_xrot, T_zrot = np.eye(4), np.eye(4), np.eye(4)
T_yrot[:3, :3] = yrot_90
T_xrot[:3, :3] = xrot_10
T_zrot[:3, :3] = zrot_180

# Initialize the controller
controller_cfg = {
"config_network": "./network_config.yaml"
}
controller_cfg = edict(controller_cfg)
controller = DemoPlacementController(controller_cfg)

# # Close the gripper to grasp the object
# breakpoint()
# controller._publish_gripper([GRIPPER_OPENNING])
# breakpoint()

# Step 1: Init T_object_hand
T_base_hand, T_object_hand = np.eye(4), np.eye(4)
T_object_hand[:3, :3] = zrot_180 @ T_object_hand[:3, :3]
T_object_hand[:3, 3] = TRAJ_OFFSET
if VERT_GRASP:
    T_object_hand = T_yrot @ T_object_hand

# Step 2: Grasp the object, rotate gripper around y_axis if necessary
T_base_object = np.linalg.inv(T_zrot)
T_base_object[:3, 3] = np.array([-0.0, 0.23, 0.7])
T_base_hand = T_base_object @ T_object_hand
T_base_hand[:3, :3] = T_base_hand[:3, :3] @ TRAJ_ROT_INIT
traj = publish_action(controller, T_base_hand)
time.sleep(2)
breakpoint()
controller._publish_gripper([GRIPPER_OPENNING])
breakpoint()

# Step 3: Go to the placement configuration
T_base_object = controller.inference(T_object_hand, height_offset=0.07, cut_mode="bottom")
T_base_hand = T_base_object @ T_object_hand
T_base_hand[:3, :3] = T_base_hand[:3, :3] @ TRAJ_ROT_INIT
traj = publish_action(controller, T_base_hand, GRIPPER_OPENNING)
time.sleep(2)
breakpoint()

# Step 4: Go to the refined placement configuration
if VERT_GRASP:
    T_object_hand = T_xrot @ T_object_hand

T_base_hand = T_base_object @ T_object_hand
T_base_hand[:3, :3] = T_base_hand[:3, :3] @ TRAJ_ROT_INIT
traj = publish_action(controller, T_base_hand, GRIPPER_OPENNING)
time.sleep(2)
breakpoint()

# Step 5: Open the gripper
traj_post = traj.copy()
traj_post[:, -1] = 1
traj = controller._publish_trajectory(traj_post)
breakpoint()

# Step 6: Home
T_base_hand, T_object_hand = np.eye(4), np.eye(4)
T_object_hand[:3, :3] = zrot_180 @ T_object_hand[:3, :3]
T_object_hand[:3, 3] = TRAJ_OFFSET
if VERT_GRASP:
    T_object_hand = T_xrot @ T_yrot @ T_object_hand
T_base_object = np.linalg.inv(T_zrot)
T_base_object[:3, 3] = np.array([-0.0, 0.23, 0.7])
T_base_hand = T_base_object @ T_object_hand
T_base_hand[:3, :3] = T_base_hand[:3, :3] @ TRAJ_ROT_INIT
traj = publish_action(controller, T_base_hand)

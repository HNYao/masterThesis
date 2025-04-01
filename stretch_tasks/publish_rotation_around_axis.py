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
TRAJ_OFFSET = np.array([0, -0.23, 0.03])
GRIPPER_OPENNING = 1
yrot_90 = SciR.from_euler("y", 90, degrees=True).as_matrix()
xrot_20 = SciR.from_euler("x", -20, degrees=True).as_matrix()
zrot_180 = SciR.from_euler("z", 180, degrees=True).as_matrix()
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

# Step 1: rotate around y axis
T_base_hand, T_object_hand = np.eye(4), np.eye(4)
T_object_hand[:3, :3] = xrot_20 @ yrot_90 @ zrot_180 @ T_object_hand[:3, :3]
T_object_hand[:3, 3] = xrot_20 @ yrot_90 @ TRAJ_OFFSET

# T_object_hand[:3, :3] = yrot_90 @ T_object_hand[:3, :3] 
# T_object_hand[:3, 3] = 

T_base_object = controller.inference(T_object_hand, cut_mode="bottom")
T_base_hand = T_base_object @ T_object_hand
T_base_hand[:3, :3] = T_base_hand[:3, :3] @ SciR.from_euler("xyz", TRAJ_ROT_INIT, degrees=False).as_matrix()

# Publish the rotation
traj_rot = SciR.from_matrix(T_base_hand[:3, :3]).as_euler("xyz", False)
traj_grip = np.ones([1]).astype(np.int32) * GRIPPER_OPENNING # [1]
traj = np.concatenate([T_base_hand[:3, 3], traj_rot, traj_grip], axis=-1)
traj = traj[None]
controller._publish_trajectory(traj)

# # Open the gripper
# time.sleep(2)
# traj_post = traj.copy()
# traj_post[:, -1] = 1
# controller._publish_trajectory(traj_post)

# # Go to the home
# time.sleep(5)
# controller.home()
# time.sleep(2)
# controller.home()

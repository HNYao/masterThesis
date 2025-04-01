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
TRAJ_OFFSET = np.array([0, 0.23, 0.03])
GRIPPER_OPENNING = 0.5

# Initialize the controller
controller_cfg = {
"config_network": "./network_config.yaml"
}
controller_cfg = edict(controller_cfg)
controller = DemoPlacementController(controller_cfg)

# Close the gripper to grasp the object
breakpoint()
controller._publish_gripper([GRIPPER_OPENNING])
breakpoint()

# Inference the placing position by the network
place_pos_in_base = controller.inference(cut_mode="bottom")

# Build the full trajectory
traj_tra = place_pos_in_base # [3]
traj_rot_mat = SciR.from_euler("xyz", TRAJ_ROT_INIT, degrees=False).as_matrix() # [3]
traj_rot = SciR.from_matrix(traj_rot_mat).as_euler("xyz", False)
traj_grip = np.ones([1]).astype(np.int32) * GRIPPER_OPENNING # [1]
traj = np.concatenate([traj_tra, traj_rot, traj_grip], axis=-1)
traj = traj[None]

# Go to the placement target
controller._publish_trajectory(traj)

# Open the gripper
time.sleep(2)
traj_post = traj.copy()
traj_post[:, -1] = 1
controller._publish_trajectory(traj_post)

# Go to the home
time.sleep(5)
controller.home()
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
GRIPPER_OPENNING = 0.5

# Initialize the controller
controller_cfg = {
"config_network": "./network_config.yaml"
}
controller_cfg = edict(controller_cfg)
controller = DemoPlacementController(controller_cfg)


# Go to the home
time.sleep(1)
controller.home()
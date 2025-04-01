from stretch_utils.zmq_utils import (
    ZMQKeypointPublisher,
    create_request_socket
)
import numpy as np
from easydict import EasyDict as edict
from omegaconf import OmegaConf
import time

config_network_path = "./network_config.yaml"
cfg_network = edict(OmegaConf.to_container(OmegaConf.load(config_network_path)))

def get_home_param(
    h=0.3,
    y=0.0,
    x=0.0,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    gripper=1.0,
    closing_threshold=None,
    reopening_threshold=None,
    stretch_gripper_max=None,
    stretch_gripper_min=None,
    stretch_gripper_tight=None,
    sticky_gripper=None,
    # Below the first value, it will close, above the second value it will open
    gripper_threshold_post_grasp_list=None,
):
    """
    Returns a list of home parameters
    """
    return [
        h,
        y,
        x,
        yaw,
        pitch,
        roll,
        gripper,
        stretch_gripper_max,
        stretch_gripper_min,
        stretch_gripper_tight,
        sticky_gripper,
        closing_threshold,
        reopening_threshold,
        gripper_threshold_post_grasp_list,
    ]


def pub_home_params():
    flag_socket.send(b"") # I send
    time.sleep(0.3)
    action_publisher.pub_keypoints(get_home_param(0.3, gripper=-0.3), "params")
    time.sleep(0.3)
    flag_socket.recv()  
    

def home():
    flag_socket.send(b"") # I send
    time.sleep(0.3)
    action_publisher.pub_keypoints([1], "home")
    time.sleep(0.3)
    flag_socket.recv()  
    

action_publisher = ZMQKeypointPublisher(
    cfg_network.remote_address,
    cfg_network.action_port,            
)

flag_socket = create_request_socket(
            cfg_network.host_address, 
            cfg_network.flag_port
        )
  
# # Back to home
# pub_home_params()
# time.sleep(0.5)
# home()
from stretch_utils.zmq_utils import (
    ZMQCameraSubscriber,
    ZMQTrajectoryPublisher,
    create_request_socket,
    ZMQKeypointPublisher
)
from stretch_utils.data_utils import visualize_scores, preprocess_stretch_head_image

import cv2
import open3d as o3d
import numpy as np
import os
from omegaconf import OmegaConf

import torch
import numpy as np
import cv2

import open3d as o3d
from easydict import EasyDict as edict
import time
from transformations import rotation_matrix

class ControllerBase:
    def __init__(self, cfg=None):
        config_network_path = cfg.config_network

        self.cfg = cfg
        self.cfg_network = edict(OmegaConf.to_container(OmegaConf.load(config_network_path)))

        # Initialize subscriber and publisher
        self.camera_subscriber = ZMQCameraSubscriber(
            self.cfg_network.host_address,
            self.cfg_network.camera_port,
            "RGBD",
        )
        self.trajectory_publisher = ZMQTrajectoryPublisher(
            self.cfg_network.remote_address,
            self.cfg_network.trajectory_port,            
        )
        self.action_publisher = ZMQKeypointPublisher(
            self.cfg_network.remote_address,
            self.cfg_network.action_port,            
        )
        self.flag_socket = create_request_socket(
            self.cfg_network.host_address, 
            self.cfg_network.flag_port
        )

    def _subscribe_image(self, cut_mode="center"):
        color, depth, timestamp = self.camera_subscriber.recv_image_and_depth()
        color, depth, intr, T_calib = preprocess_stretch_head_image(color, depth, self.raw_intr, cut_mode=cut_mode)
        # depth[depth > 1.5] == 0
        color = color[..., [2,1,0]].copy()
        return color, depth, intr, T_calib

    def _publish_trajectory(self, trajectory):
        self.flag_socket.send(b"") # I send
        time.sleep(0.02)
        self.trajectory_publisher.pub_trajectory(trajectory, "post_trajectory")
        self.flag_socket.recv()    

    def _publish_action(self, action):
        self.flag_socket.send(b"") # I send
        time.sleep(0.02)
        self.action_publisher.pub_keypoints(action, "robot_action")
        self.flag_socket.recv()    

    def _publish_home_params(self, params):
        print("Publish new home parameters: ", params)
        self.flag_socket.send(b"") # I send
        time.sleep(0.02)
        self.action_publisher.pub_keypoints(params, "params")
        self.flag_socket.recv()  
    
    def _publish_gripper(self, gripper):
        self.flag_socket.send(b"") # I send
        time.sleep(0.02)
        self.action_publisher.pub_keypoints(gripper, "gripper")
        self.flag_socket.recv()  
    
    def home(self):
        self.flag_socket.send(b"") # I send
        time.sleep(0.02)
        self.action_publisher.pub_keypoints([1], "home")
        self.flag_socket.recv()    
        

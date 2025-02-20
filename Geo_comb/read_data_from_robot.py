import cv2
import numpy as np
import torch
import json
import os
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
import open3d as o3d

INTRINSICS = np.array([[619.0125 ,   0.     , 640  ],[  0.     , 619.16775, 360],[  0.     ,   0.     ,   1.     ]])
INTRINSICS = np.array([[619.0125 ,   0.     , 360  ],[  0.     , 619.16775, 640],[  0.     ,   0.     ,   1.     ]])

# data from robot camera
rgb_image_file_path = "dataset/data_from_robot/data_from_robot/img/img_16.jpg"
depth_image_file_path = "dataset/data_from_robot/data_from_robot/depth/depth_16.png"

depth = cv2.imread(depth_image_file_path, cv2.IMREAD_UNCHANGED)
depth = depth.astype(np.float32) / 1000.0
color_no_obj_scene = cv2.imread(rgb_image_file_path)
color_no_obj = color_no_obj_scene.astype(np.float32) / 255.0

intr = INTRINSICS
points_no_obj_scene, scene_no_obj_idx = backproject(
    depth,
    intr,
    np.logical_and(depth > 0, depth < 2),
    NOCS_convention=False,
)
colors_no_obj_scene = color_no_obj[scene_no_obj_idx[0], scene_no_obj_idx[1]]
pcd_no_obj_scene = visualize_points(points_no_obj_scene, colors_no_obj_scene)

o3d.visualization.draw_geometries([pcd_no_obj_scene])
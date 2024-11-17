"""
transfer the RGBD to a mask pointcloud
Compared to the previous version, this version aguments the dataset by generating 8 directions of the mask point cloud
scene_RGBD_mask_v2
"""
from GeoL_net.dataset_gen.hdf52png import hdf52png
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
import math
import cv2
import numpy as np
import open3d as o3d
import os
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import copy


def determine_direction(p1, p2):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    
    angle = math.degrees(math.atan2(y_diff, x_diff))
    if angle < 0:
        angle += 360
    
    if 0 <= angle < 22.5 or 337.5 <= angle < 360:
        return "Left"
    elif 22.5 <= angle < 67.5:
        return "Left Behind"
    elif 67.5 <= angle < 112.5:
        return "Behind"
    elif 112.5 <= angle < 157.5:
        return "Right Behind"
    elif 157.5 <= angle < 202.5:
        return "Right"
    elif 202.5 <= angle < 247.5:
        return "Right Front"
    elif 247.5 <= angle < 292.5:
        return "Front"
    elif 292.5 <= angle < 337.5:
        return "Left Front"



def RGBD2MaskPC(depth_path, mask_hd5f_path, output_dir, depth_removed_obj = None, mask_hd5f_removed_obj = None):
    """ transfer deph and mask hd5f to point cloud(mask)
    """

    # pc_mask_path
    pc_mask_path = f"{output_dir}/mask.ply"
    pcd_mask_preprocessing = pcd_mask_preprocessing_red_label
    # get the mask png (with obj)
    mask_img_path = f"{output_dir}/mask_with_obj.png"
    mask_image = hdf52png(hdf5_path=mask_hd5f_path, output_dir=mask_img_path)

    #get the mask png (without obj)
    assert depth_removed_obj is not None
    assert mask_hd5f_removed_obj is not None

    mask_img_no_obj_path = f"{output_dir}/mask_no_obj.png"
    mask_no_obj_image = hdf52png(hdf5_path=mask_hd5f_removed_obj, output_dir=mask_img_no_obj_path)

    #get the depth 
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = np.array(depth_image)

        #get the depth no image
    if (depth_removed_obj is not None) and (mask_hd5f_removed_obj is not None):
        depth_no_obj_image = cv2.imread(depth_removed_obj, cv2.IMREAD_UNCHANGED)
        depth_no_obj = np.array(depth_no_obj_image)
    
    # get the mask
    color_image = mask_image
    color = np.array(color_image) / 255.0

        # get the mask no image
    color_no_obj_image = mask_no_obj_image
    color_no_obj = np.array(color_no_obj_image) / 255.0


    # camera intrinsc matrix kinect    
    intr = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                [  0.     , 607.05212/2, 367.35952/2],
                [  0.     ,   0.     ,   1.     ]])

    # backprojection
    points_scene, scene_idx = backproject(
                depth,
                intr,
                np.logical_and(depth > 0, depth > 0),
                NOCS_convention=False,
            )
    
        # backprojection no obj
    if (depth_removed_obj is not None) and (mask_hd5f_removed_obj is not None):
        points_no_obj_scene, scene_no_obj_idx = backproject(
                depth_no_obj,
                intr,
                np.logical_and(depth > 0, depth > 0),
                NOCS_convention=False,
            )
    # camera rotation 
    cam_rotation_matrix = np.array([
        [1, 0, 0],
        [0,0.8,-0.6],
        [0,0.6,0.8]
    ])
    points_scene = (cam_rotation_matrix @ points_scene.T).T
    #centroid = np.mean(points_scene, axis=0)
    #print("centroid:", centroid)
    #points_scene = points_scene - centroid
    colors_scene = color[scene_idx[0], scene_idx[1]]
    pcd_scene = visualize_points(points_scene, colors_scene)

        # camera rotation no obj
    if (depth_removed_obj is not None) and (mask_hd5f_removed_obj is not None):
        pass
        points_no_obj_scene = (cam_rotation_matrix @ points_no_obj_scene.T).T
        colors_no_obj_scene = color_no_obj[scene_no_obj_idx[0], scene_no_obj_idx[1]]
        pcd_no_obj_scene = visualize_points(points_no_obj_scene, colors_no_obj_scene)
        

        pcd_no_obj_scene = remove_ground(pcd_no_obj_scene)

        points_scene_list = pcd_mask_preprocessing(points_scene=pcd_scene, points_no_obj_scene=pcd_no_obj_scene)
    
    if points_scene_list is None:
        return None # 缺少红色和绿色点 提前return

    
    # remove the ground floor
    for direction in points_scene_list.keys():
        if points_scene_list[direction] is not None:
            pcd_scene = points_scene_list[direction]
            pc_mask_path = f"{output_dir}/mask_{direction}.ply"
    
        o3d.io.write_point_cloud(pc_mask_path, pcd_scene)
        print(f"{pc_mask_path} is done")

   

def is_blue(color, threshold=0.9):
    return color[0] < (1 - threshold) and color[1] < (1 - threshold) and color[2] > threshold

def is_black(color, threshold=0.1):
    return color[0] < threshold and color[1] < threshold and color[2] < threshold

def is_green(color, threshold=0.9):
    return color[0] < (1 - threshold) and color[1] > threshold and color[2] < (1 - threshold)

def is_red(color, threshold=0.9):
    return color[0] > threshold and color[1] < (1 - threshold) and color[2] < (1 - threshold)


def pcd_mask_preprocessing_red_label(points_scene, points_no_obj_scene=None):
    "generate red label"

    colors = np.asarray(points_scene.colors)
    points = np.asarray(points_scene.points)

    # 提取绿色点云 removed obj
    green_mask = np.apply_along_axis(is_green, 1, colors)
    green_points = points[green_mask]


    # 提取红色点云 anchor obj
    red_mask = np.apply_along_axis(is_red, 1, colors)
    red_points = points[red_mask]

    # TODO: 处理红色点0的情况
    # TODO: 处理绿色点0的情况
    if red_points.shape[0] == 0 or green_points.shape[0] == 0:
        print("lack of red or green points")
        return None


    # 计算绿色点位置均值
    green_center = np.mean(green_points, axis=0) # removed obj position
    # 计算红色点均值
    red_center = np.mean(red_points, axis=0) # anchor obj position

    # 计算 x y 平面上的距离 
    offset = np.linalg.norm(red_center[:2] - green_center[:2])
    #offset = random.uniform(300, 400) # NOTE: hardcode
    direction = determine_direction(green_center[:2], red_center[:2])
    if direction == "Front" or direction == "Behind" or direction == "Left" or direction == "Right":
        offset = offset
    else:
        offset = offset / math.sqrt(2)
    exist_direction_dict = {"Left": None, "Left Behind": None, "Behind": None, "Right Behind": None, "Right": None, "Right Front": None, "Front": None, "Left Front": None}

    if points_no_obj_scene is not None:
        points_scene = points_no_obj_scene
        colors = np.asarray(points_no_obj_scene.colors)
        points = np.asarray(points_no_obj_scene.points)
    

    # 提取蓝色点云 other objs
    blue_mask = np.apply_along_axis(is_blue, 1, colors)
    blue_points = points[blue_mask]
    blue_pcd = o3d.geometry.PointCloud()
    blue_pcd.points = o3d.utility.Vector3dVector(blue_points)

    # 提取黑色点云 
    black_mask = np.apply_along_axis(is_black, 1, colors)
    z_max = blue_pcd.get_axis_aligned_bounding_box().get_max_bound()[2]
    z_center = blue_pcd.get_axis_aligned_bounding_box().get_center()[2]
    z_range_mask = (points[:, 2] <= z_max + 75) & (points[:, 2] >= z_center)

    # 目标点云
    red_mask = np.apply_along_axis(is_red, 1, colors)
    target_mask =  ~red_mask#(z_range_mask & black_mask)
    target_points = points[target_mask]

    # 计算所有target点与绿色点云中心的距离
    dists = np.linalg.norm(target_points[:, :2] - green_center[:2], axis=1)
    dists = dists ** 0.95  # 可调整距离的幂次以改变效果

    # 初始化颜色矩阵（全为0，RGB）
    heatmap_colors = colors.copy()

    # 对于红色点进行处理
    valid_dists = dists.copy()
    valid_dists[valid_dists > 200] = 200  # 将所有大于150的距离值截断为150

    # 将距离映射到G通道值，距离为0时R=1，距离为150时R=0
    g_channel_values = 1 - (valid_dists / 200)
    
    # 映射成绿色，R B为0
    heatmap_colors[target_mask, 0] = 0  
    heatmap_colors[target_mask, 1] = g_channel_values
    heatmap_colors[target_mask, 2] = 0

    

    # 更新点云颜色
    points_scene.colors = o3d.utility.Vector3dVector(heatmap_colors)

    exist_direction_dict[direction] = points_scene

    for direction in exist_direction_dict.keys():
        if exist_direction_dict[direction] is not None:
            continue
        heatmap_colors[target_mask, 0] = 0  
        heatmap_colors[target_mask, 1] = 0
        heatmap_colors[target_mask, 2] = 0
        points_scene_copy = copy.deepcopy(points_scene)
        points_scene_copy.colors = o3d.utility.Vector3dVector(heatmap_colors) # 去掉 green mask
        
        new_target_points = red_center.copy()
        if direction == "Left":
            new_target_points[0] -= offset
        elif direction == "Right":
            new_target_points[0] += offset
        elif direction == "Front":
            new_target_points[1] += offset
        elif direction == "Behind":
            new_target_points[1] -= offset
        elif direction == "Left Front":
            new_target_points[0] -= offset
            new_target_points[1] += offset
        elif direction == "Left Behind":
            new_target_points[0] -= offset
            new_target_points[1] -= offset
        elif direction == "Right Front":
            new_target_points[0] += offset
            new_target_points[1] += offset
        elif direction == "Right Behind":
            new_target_points[0] += offset
            new_target_points[1] -= offset
        
        dists = np.linalg.norm(target_points[:, :2] - new_target_points[:2], axis=1)
        dists = dists ** 0.95  
        # 初始化颜色矩阵（全为0，RGB）
        heatmap_colors = colors.copy()

        # 对于红色点进行处理
        valid_dists = dists.copy()
        valid_dists[valid_dists > 250] = 250  # 将所有大于150的距离值截断为150

        # 将距离映射到G通道值，距离为0时R=1，距离为150时R=0
        g_channel_values = 1 - (valid_dists / 250)
        
        # 映射成绿色，R B为0
        heatmap_colors[target_mask, 0] = 0  
        heatmap_colors[target_mask, 1] = g_channel_values
        heatmap_colors[target_mask, 2] = 0

        points_scene_copy.colors = o3d.utility.Vector3dVector(heatmap_colors)

        exist_direction_dict[direction] = points_scene_copy

    return exist_direction_dict


def remove_ground(point_cloud):
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    z_min = np.min(points[:,2])
    z_max = np.max(points[:,2])
    height = z_max - z_min

    z_thershold = z_max - height /50 # remove 2%
    mask = points[:, 2] <= z_thershold
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
    pcd_filtered.colors = o3d.utility.Vector3dVector(filtered_colors)
    return pcd_filtered

def RGB2RefMaskPC(depth_path, ref_image_path, out_dir):

    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = np.array(depth_image)

    # camera intrinsc matrix
    intr = np.array([[591.0125 ,   0.     , 322.525  ],
                     [  0.     , 590.16775, 244.11084],
                     [  0.     ,   0.     ,   1.     ]])

    points_scene, scene_idx = backproject(
            depth,
            intr,
            np.logical_and(depth > 0, depth > 0),
            NOCS_convention=False,
        )
    colors_image = cv2.imread(ref_image_path, cv2.IMREAD_UNCHANGED)
    color = np.array(colors_image) / 255.0
    colors_scene = color[scene_idx[0], scene_idx[1]]

        # camera rotation 
    cam_rotation_matrix = np.array([
        [1, 0, 0],
        [0,0.8,-0.6],
        [0,0.6,0.8]
    ])

    points_scene = (cam_rotation_matrix @ points_scene.T).T
    
    pcd_scene = visualize_points(points_scene, colors_scene)
    # remove the ground floor
    pcd_scene = remove_ground(pcd_scene)
    pc_mask_path = f"{out_dir}/mask_ref.ply"
    o3d.io.write_point_cloud(pc_mask_path, pcd_scene)

if __name__ == "__main__":
    # ps: 获得的点云 倒置
    # input depth and hdf5, get the ply
    # dataset/scene_RGBD_mask/scene_name/remove_obj/ply

    # config
    
    folder_path = "dataset/scene_RGBD_mask_v2_kinect_cfg"
    amount_scene = 0
    for root,dirs,files in os.walk(folder_path):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            for sub_root, sub_dirs, sub_files in os.walk(subfolder_path):
                for sub_dir_name in sub_dirs:
                    sub_subfolder_path = os.path.join(sub_root, sub_dir_name)
                    #print(sub_subfolder_path)
                    amount_scene = amount_scene + 1
                    #print(amount_scene)

                    base = sub_subfolder_path # e.g. 'dataset/scene_RGBD_mask_v2/id121_1/bottle_0003_green'
                    image_path = base + "/with_obj/test_pbr/000000/rgb/000000.jpg"
                    depth_path = base + "/with_obj/test_pbr/000000/depth_noise/000000.png"
                    mask_hdf5_path = base + "/with_obj/0.hdf5"
                    output_dir = base
                    depth_removed_obj = base + "/no_obj/test_pbr/000000/depth_noise/000000.png"
                    mask_hdf5_removed_obj = base + "/no_obj/0.hdf5"
                    start_time = time.time()
                    ply_path = os.path.join(base, "mask_red.ply")
                    if os.path.exists(ply_path):
                        print(f"{output_dir}/mask_red.ply already exists")
                        continue
                    else:
                        RGBD2MaskPC(depth_path=depth_path,  # 'dataset/scene_RGBD_mask_v2/id121_1/bottle_0003_green/with_obj/test_pbr/000000/depth_noise/000000.png'
                                    mask_hd5f_path=mask_hdf5_path,
                                    output_dir=output_dir,
                                    depth_removed_obj= depth_removed_obj, # 'dataset/scene_RGBD_mask_v2/id121_1/bottle_0003_green/no_obj/test_pbr/000000/depth_noise/000000.png'
                                    mask_hd5f_removed_obj=mask_hdf5_removed_obj)
                    end_time = time.time()
                    print(f"{output_dir} is done, comsuming {end_time - start_time} s")
                    
                    print(f"{amount_scene} scene are done")
               
                break
            
        break
    
    

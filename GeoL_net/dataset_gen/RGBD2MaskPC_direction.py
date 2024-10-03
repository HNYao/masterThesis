"""
transfer the RGBD to a mask pointcloud
only keep the anchor obj
and gerenate the mask pointcloud of 8 directions"""
from GeoL_net.dataset_gen.hdf52png import hdf52png
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
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
import shutil
import json

def RGBD2MaskPC_direction(depth_path, mask_hd5f_path, output_dir, reprocessing_flag=False, depth_removed_obj = None, mask_hd5f_removed_obj = None):
    """ transfer deph and mask hd5f to point cloud(mask)
    """

    # pc_mask_path
    pc_mask_path = f"{output_dir}/mask_red.ply"
    # check if the file exists
    if os.path.exists(pc_mask_path):
        print(f"{output_dir}/mask_red.ply already exists")
        return None
    #pcd_mask_preprocessing = pcd_mask_preprocessing
    pcd_mask_preprocessing = pcd_mask_preprocessing_direction_red_label
    # get the mask png (with obj)
    mask_img_path = f"{output_dir}/mask_with_obj.png"
    mask_image = hdf52png(hdf5_path=mask_hd5f_path, output_dir=mask_img_path)

        #get the mask png (without obj)
    if (depth_removed_obj is not None) and (mask_hd5f_removed_obj is not None):
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

    # camera intrinsc matrix
    intr = np.array([[591.0125 ,   0.     , 322.525  ],
                     [  0.     , 590.16775, 244.11084],
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
    colors_scene = color[scene_idx[0], scene_idx[1]]
    pcd_scene = visualize_points(points_scene, colors_scene)

        # camera rotation no obj
    if (depth_removed_obj is not None) and (mask_hd5f_removed_obj is not None):
        pass
        points_no_obj_scene = (cam_rotation_matrix @ points_no_obj_scene.T).T
        colors_no_obj_scene = color_no_obj[scene_no_obj_idx[0], scene_no_obj_idx[1]]
        pcd_no_obj_scene = visualize_points(points_no_obj_scene, colors_no_obj_scene)
        pcd_no_obj_scene = remove_ground(pcd_no_obj_scene)
    
    if reprocessing_flag == True:
        if (depth_removed_obj is not None) and (mask_hd5f_removed_obj is not None):
            pcd_mask_preprocessing(points_scene=pcd_scene, points_no_obj_scene=pcd_no_obj_scene, output_dir=output_dir)
        else:
            pcd_mask_preprocessing(points_scene=pcd_scene)


   

def is_blue(color, threshold=0.9):
    return color[0] < (1 - threshold) and color[1] < (1 - threshold) and color[2] > threshold

def is_black(color, threshold=0.1):
    return color[0] < threshold and color[1] < threshold and color[2] < threshold

def is_green(color, threshold=0.9):
    return color[0] < (1 - threshold) and color[1] > threshold and color[2] < (1 - threshold)

def is_red(color, threshold=0.9):
    return color[0] > threshold and color[1] < (1 - threshold) and color[2] < (1 - threshold)


def compute_distance_matrix(points_a, points_b):
    distances = np.sqrt(np.sum((points_a[:, None] - points_b[None, :])**2, axis=2))
    return distances

def pcd_mask_preprocessing(points_scene, points_no_obj_scene = None):
    " generate turbo heatmap label"

    colors = np.asarray(points_scene.colors)
    points = np.asarray(points_scene.points)

    # 提取绿色点云 removed obj
    green_mask = np.apply_along_axis(is_green, 1, colors)
    green_points = points[green_mask]
    green_pcd = o3d.geometry.PointCloud()
    green_pcd.points = o3d.utility.Vector3dVector(green_points)

    if points_no_obj_scene is not None:
        points_scene = points_no_obj_scene
        colors = np.asarray(points_no_obj_scene.colors)
        points = np.asarray(points_no_obj_scene.points)
    green_center = np.mean(green_points, axis=0)
    
    # 提取蓝色点云 other objs
    blue_mask = np.apply_along_axis(is_blue, 1, colors)
    blue_points = points[blue_mask]
    blue_pcd = o3d.geometry.PointCloud()
    blue_pcd.points = o3d.utility.Vector3dVector(blue_points)
    
    # 提取black点云 
    black_mask = np.apply_along_axis(is_black, 1, colors)
    z_max = blue_pcd.get_axis_aligned_bounding_box().get_max_bound()[2]
    z_center = blue_pcd.get_axis_aligned_bounding_box().get_center()[2]
    z_range_mask = (points[:, 2] <= z_max + 10) & (points[:, 2] >= z_center)
    target_mask = z_range_mask & black_mask
    colors[target_mask] = [1, 0, 0]  # 将这些点的颜色设置为红色 表示桌面
    points_scene.colors = o3d.utility.Vector3dVector(colors)
    
    # 提取红色点
    red_mask = np.all(colors == [1, 0, 0], axis=1)
    red_points = points[red_mask]

    # 重新提取
    black_mask = np.apply_along_axis(is_black, 1, colors)
    green_mask = np.apply_along_axis(is_green, 1, colors)
    red_mask = np.all(colors == [1, 0, 0], axis=1)
    # 初始化 heatmap 值为 0
    heatmap = np.zeros(len(points))

    # 计算所有红色点与绿色点云中心的距离
    dists = np.linalg.norm(red_points[:,:2] - green_center[:2], axis=1)
    dists = dists ** 0.95

    # 对于蓝色或黑色点，直接将其 heatmap 设为 0
    invalid_mask = blue_mask[red_mask] | black_mask[red_mask]

    # 计算有效点的 heatmap 值 (距离为 0 时为 1，距离大于 10 时为 0，0 到 10 之间线性分布)
    valid_mask = ~invalid_mask
    valid_dists = dists[valid_mask]

    heatmap_values = np.zeros_like(dists)
    heatmap_values[valid_dists <= 150] = 1 - valid_dists[valid_dists <= 150] / 150
    heatmap_values[valid_dists == 0] = 1  # 距离为 0 的点赋值为 1
    heatmap[red_mask] = heatmap_values

    # 使用 Turbo colormap
    turbo_colormap = plt.get_cmap('turbo')

    # 对 heatmap 值为 0 的点（蓝色和黑色点以及距离大于 10 的红色点），设置颜色为 Turbo colormap 对应 0 值的颜色
    heatmap_colors = np.zeros((len(points), 3))
    heatmap_colors[red_mask] = turbo_colormap(heatmap_values)[:, :3]  # 将 heatmap 值转换为颜色

    # 设置蓝色和黑色点的颜色为 Turbo colormap 中对应 0 的颜色
    zero_color = turbo_colormap(0)[:3]
    heatmap_colors[blue_mask | black_mask] = zero_color

    # 更新点云颜色
    points_scene.colors = o3d.utility.Vector3dVector(heatmap_colors)
    return points_scene

def pcd_mask_preprocessing_direction_red_label(points_scene, points_no_obj_scene = None, output_dir = None):
    " generate turbo heatmap label in 8 directions"
    " generate red label"

    colors = np.asarray(points_no_obj_scene.colors)
    points = np.asarray(points_no_obj_scene.points)

    # get the blue points
    blue_mask = np.apply_along_axis(is_blue, 1, colors)
    blue_points = points[blue_mask]
    blue_pcd = o3d.geometry.PointCloud()
    blue_pcd.points = o3d.utility.Vector3dVector(blue_points)

    # get the black points
    black_mask = np.apply_along_axis(is_black, 1, colors)
    z_max = blue_pcd.get_axis_aligned_bounding_box().get_max_bound()[2]
    z_center = blue_pcd.get_axis_aligned_bounding_box().get_center()[2]
    z_range_mask = (points[:, 2] <= z_max + 10) & (points[:, 2] >= z_center)
    target_mask = z_range_mask & black_mask
    colors[target_mask] = [1, 0, 0]  # set the color of these points to red, representing the desktop
    points_no_obj_scene.colors = o3d.utility.Vector3dVector(colors)

    # get the red points
    red_mask = np.all(colors == [1, 0, 0], axis=1)
    red_points = points[red_mask]

    # re-extract
    black_mask = np.apply_along_axis(is_black, 1, colors)
    blue_mask = np.apply_along_axis(is_blue, 1, colors)
    red_mask = np.all(colors == [1, 0, 0], axis=1)

    # initialize the heatmap value to 0
    heatmap = np.zeros(len(points))

    # determin the target point cloud center. center of the blue point cloud bbox
    blue_pcd_center = blue_pcd.get_center()
    offset = random.uniform(100, 200)
    pcd_mask_8_direction_list = []

    for direction in ["Left", "Right", "Front", "Behind", "Left Front", "Left Behind", "Right Front", "Right Behind"]:
        target_pos = blue_pcd_center.copy()
        if direction == "Left":
            target_pos[0] -= offset
        elif direction == "Right":
            target_pos[0] += offset
        elif direction == "Front":
            target_pos[1] += offset
        elif direction == "Behind":
            target_pos[1] -= offset
        elif direction == "Left Front":
            target_pos[0] -= offset
            target_pos[1] += offset
        elif direction == "Left Behind":
            target_pos[0] -= offset
            target_pos[1] -= offset
        elif direction == "Right Front":
            target_pos[0] += offset
            target_pos[1] += offset
        elif direction == "Right Behind":
            target_pos[0] += offset
            target_pos[1] -= offset
        
        # calculate the distance between all red points and the target point cloud center
        dists = np.linalg.norm(red_points[:,:2] - target_pos[:2], axis=1)
        dists = dists ** 0.95

            # 初始化颜色矩阵（全为0，RGB）
        heatmap_colors = np.zeros((len(points), 3))

            # 对于红色点进行处理
        valid_dists = dists.copy()
        valid_dists[valid_dists > 150] = 150  # 将所有大于150的距离值截断为150

            # 将距离映射到R通道值，距离为0时R=1，距离为150时R=0
        r_channel_values = 1 - (valid_dists / 150)
        # 将映射后的R通道值赋给红色点的R通道，G和B通道保持为0
        heatmap_colors[red_mask, 0] = r_channel_values  # 只设置红色通道
        # 对蓝色和黑色点，保持颜色为0（默认）
        heatmap_colors[blue_mask | black_mask] = [0, 0, 0]
        
        '''
        # for blue or black points, set their heatmap to 0 directly
        invalid_mask = blue_mask[red_mask] | black_mask[red_mask]
        
        # calculate the heatmap value of valid points (1 when the distance is 0, 0 when the distance is greater than 10, linear distribution between 0 and 10)
        valid_mask = ~invalid_mask
        valid_dists = dists[valid_mask]

        heatmap_values = np.zeros_like(dists)
        heatmap_values[valid_dists <= 150] = 1 - valid_dists[valid_dists <= 150] / 150
        heatmap_values[valid_dists == 0] = 1  # set the points with distance 0 to 1
        heatmap[red_mask] = heatmap_values

        # use Turbo colormap
        turbo_colormap = plt.get_cmap('turbo')

        # set the color of points with heatmap value 0 (blue and black points and red points with distance greater than 10) to the color corresponding to 0 in the Turbo colormap
        heatmap_colors = np.zeros((len(points), 3))
        heatmap_colors[red_mask] = turbo_colormap(heatmap_values)[:, :3]  # convert the heatmap value to color
        
        # set the color of blue and black points to the color corresponding to 0 in the Turbo colormap
        zero_color = turbo_colormap(0)[:3]
        heatmap_colors[blue_mask | black_mask] = zero_color
        '''
        # update the point cloud color
        points_no_obj_scene.colors = o3d.utility.Vector3dVector(heatmap_colors)
        #o3d.visualization.draw_geometries([points_no_obj_scene])
        # add the point cloud to the list
        pcd_without_ground = remove_ground(points_no_obj_scene)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        o3d.io.write_point_cloud(f"{output_dir}/mask_{direction}.ply", pcd_without_ground)
        print(f"{output_dir}/mask_{direction}.ply is done")

    return pcd_mask_8_direction_list

def pcd_mask_preprocessing_red_label(points_scene, points_no_obj_scene=None):
    "generate red label"

    colors = np.asarray(points_scene.colors)
    points = np.asarray(points_scene.points)

    # 提取绿色点云 removed obj
    green_mask = np.apply_along_axis(is_green, 1, colors)
    green_points = points[green_mask]
    green_pcd = o3d.geometry.PointCloud()
    green_pcd.points = o3d.utility.Vector3dVector(green_points)

    if points_no_obj_scene is not None:
        points_scene = points_no_obj_scene
        colors = np.asarray(points_no_obj_scene.colors)
        points = np.asarray(points_no_obj_scene.points)
    green_center = np.mean(green_points, axis=0)

    # 提取蓝色点云 other objs
    blue_mask = np.apply_along_axis(is_blue, 1, colors)
    blue_points = points[blue_mask]
    blue_pcd = o3d.geometry.PointCloud()
    blue_pcd.points = o3d.utility.Vector3dVector(blue_points)

    # 提取黑色点云 
    black_mask = np.apply_along_axis(is_black, 1, colors)
    z_max = blue_pcd.get_axis_aligned_bounding_box().get_max_bound()[2]
    z_center = blue_pcd.get_axis_aligned_bounding_box().get_center()[2]
    z_range_mask = (points[:, 2] <= z_max + 10) & (points[:, 2] >= z_center)
    target_mask = z_range_mask & black_mask
    colors[target_mask] = [1, 0, 0]  # 将这些点的颜色设置为红色 表示桌面
    points_scene.colors = o3d.utility.Vector3dVector(colors)

    # 提取红色点
    red_mask = np.all(colors == [1, 0, 0], axis=1)
    red_points = points[red_mask]

    # 重新提取
    black_mask = np.apply_along_axis(is_black, 1, colors)
    green_mask = np.apply_along_axis(is_green, 1, colors)
    red_mask = np.all(colors == [1, 0, 0], axis=1)

    # 计算所有红色点与绿色点云中心的距离
    dists = np.linalg.norm(red_points[:, :2] - green_center[:2], axis=1)
    dists = dists ** 0.95  # 可调整距离的幂次以改变效果

    # 初始化颜色矩阵（全为0，RGB）
    heatmap_colors = np.zeros((len(points), 3))

    # 对于红色点进行处理
    valid_dists = dists.copy()
    valid_dists[valid_dists > 150] = 150  # 将所有大于150的距离值截断为150

    # 将距离映射到R通道值，距离为0时R=1，距离为150时R=0
    r_channel_values = 1 - (valid_dists / 150)
    
    # 将映射后的R通道值赋给红色点的R通道，G和B通道保持为0
    heatmap_colors[red_mask, 0] = r_channel_values  # 只设置红色通道

    # 对蓝色和黑色点，保持颜色为0（默认）
    heatmap_colors[blue_mask | black_mask] = [0, 0, 0]

    # 更新点云颜色
    points_scene.colors = o3d.utility.Vector3dVector(heatmap_colors)

    return points_scene


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
    # 目前先从 scene_RGBD_mask_sequence 中读取数据 save到 scene_RGBD_mask_direction
    # 结构为
    # scene_RGBD_mask_direction
    #   scene_name
    #       anchor obj
    #           mask_left.ply
    #           mask_right.ply
    #           mask_front.ply
    #           mask_behind.ply
    #           mask_left_front.ply
    #           mask_left_behind.ply
    #           mask_right_front.ply
    #           mask_right_behind.ply
    #           text_guidance.txt
    
    folder_path = "dataset/scene_RGBD_mask_sequence"

    # 遍历每个ID文件夹并查找第一个子文件夹的路径
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            id_folder_path = os.path.join(root, dir_name)
            
            # 获取每个ID文件夹下的所有子文件夹
            sub_dirs = [d for d in os.listdir(id_folder_path) if os.path.isdir(os.path.join(id_folder_path, d))]
            
            if sub_dirs:  # 检查是否有子文件夹
                first_sub_dir = sub_dirs[0]
                full_path = os.path.join(id_folder_path, first_sub_dir)
                
                print(f"ID文件夹: {dir_name}, 第一个子文件夹路径: {full_path}")

            # 仅处理第一个ID文件夹

            # 读取第一个子文件夹中的json文件 找到anchor obj
            json_file = os.path.join(id_folder_path, "text_guidance.json")
            with open(json_file, 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    anchor_obj = value[0] #eg: "the red cup"
                    anchor_obj_short = value[2] #eg: "cup"
                    break
            
            data_dict = {}
            data_dict["Left"] = [anchor_obj, f"Left {anchor_obj}", anchor_obj_short, "Left"]
            data_dict["Right"] = [anchor_obj, f"Right {anchor_obj}", anchor_obj_short, "Right"]
            data_dict["Front"] = [anchor_obj, f"Front {anchor_obj}", anchor_obj_short, "Front"]
            data_dict["Behind"] = [anchor_obj, f"Behind {anchor_obj}", anchor_obj_short, "Behind"]
            data_dict["Left Front"] = [anchor_obj, f"Left Front {anchor_obj}", anchor_obj_short, "Left Front"]
            data_dict["Left Behind"] = [anchor_obj, f"Left Behind {anchor_obj}", anchor_obj_short, "Left Behind"]
            data_dict["Right Front"] = [anchor_obj, f"Right Front {anchor_obj}", anchor_obj_short, "Right Front"]
            data_dict["Right Behind"] = [anchor_obj, f"Right Behind {anchor_obj}", anchor_obj_short, "Right Behind"]

            
            id_dir = dir_name
            base = full_path
            image_path = base + "/with_obj/test_pbr/000000/rgb/000000.jpg"
            depth_path = base + "/with_obj/test_pbr/000000/depth_noise/000000.png"
            mask_hdf5_path = base + "/with_obj/0.hdf5"
            output_dir = base.replace("scene_RGBD_mask_sequence", "scene_RGBD_mask_direction")
            depth_removed_obj = base + "/no_obj/test_pbr/000000/depth_noise/000000.png"
            mask_hdf5_removed_obj = base + "/no_obj/0.hdf5"
            # save the image without removed obj
            source_image_path = base + "/no_obj/test_pbr/000000/rgb/000000.jpg"
            target_image_path = base.replace("scene_RGBD_mask_sequence", "scene_RGBD_mask_direction") + "/img_without_removed_obj.jpg"
            destination_dir = os.path.dirname(target_image_path)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

            shutil.copyfile(source_image_path, target_image_path)
            print(f"copy {source_image_path} to {target_image_path}")

            with open(f"{output_dir}/text_guidance.json", "w", encoding='utf-8') as f:
                f.write(json.dumps(data_dict))
            print(f"write {output_dir}/text_guidance.json")

            RGBD2MaskPC_direction(depth_path=depth_path, 
            mask_hd5f_path=mask_hdf5_path,
            output_dir=output_dir,
            reprocessing_flag=True,
            depth_removed_obj= depth_removed_obj,
            mask_hd5f_removed_obj=mask_hdf5_removed_obj)
            
        
        # 因为我们只需要查找ID文件夹，所以在找到第一个层级的子文件夹后就可以停止递归
        break

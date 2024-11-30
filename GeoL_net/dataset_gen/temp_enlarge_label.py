"""
temporarily enlarge the red and ignoring the obj
"""
import open3d as o3d
import numpy as np
import random

def is_blue(color, threshold=0.9):
    return color[0] < (1 - threshold) and color[1] < (1 - threshold) and color[2] > threshold

def is_black(color, threshold=0.1):
    return color[0] < threshold and color[1] < threshold and color[2] < threshold

def is_green(color, threshold=0.9):
    return color[0] < (1 - threshold) and color[1] > threshold and color[2] < (1 - threshold)

def is_red(color, threshold=0.9):
    return color[0] > threshold and color[1] < (1 - threshold) and color[2] < (1 - threshold)

def color_non_green_neighbors_after_black(ply_file_path):
    # 读取点云数据
    points_scene = o3d.io.read_point_cloud(ply_file_path)

    colors = np.asarray(points_scene.colors)
    points = np.asarray(points_scene.points)

    # get the mean of the green points in the points scene
    green_mask = np.apply_along_axis(is_green, 1, colors)
    green_points = points[green_mask]
    green_pcd = o3d.geometry.PointCloud()
    green_pcd.points = o3d.utility.Vector3dVector(green_points)
    green_pcd_center = green_pcd.get_center()




    # get the black points
    black_mask = ~green_mask
    z_max = green_pcd.get_axis_aligned_bounding_box().get_max_bound()[2]
    z_center = green_pcd.get_axis_aligned_bounding_box().get_center()[2]
    z_range_mask = (points[:, 2] <= z_max + 40) & (points[:, 2] >= z_center-40)
    target_mask = z_range_mask & black_mask

    colors[target_mask] = [1, 0, 0]  # set the color of these points to red, representing the desktop
    points_scene.colors = o3d.utility.Vector3dVector(colors)

    # get the red points
    red_mask = np.all(colors == [1, 0, 0], axis=1)
    red_points = points[red_mask]

    # re-extract 
    black_mask = np.apply_along_axis(is_black, 1, colors) # groudn and table
    blue_mask = np.apply_along_axis(is_blue, 1, colors) # obj
    red_mask = np.all(colors == [1, 0, 0], axis=1) # tabletop
   
    offset = random.uniform(200, 300)

    for direction in ["Left", "Right", "Front", "Behind", "Left Front", "Left Behind", "Right Front", "Right Behind"]:
        target_pos = green_pcd_center.copy()
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
        valid_dists[valid_dists > 200] = 200  # 将所有大于150的距离值截断为150

            # 将距离映射到R通道值，距离为0时R=1，距离为150时R=0
        r_channel_values = 1 - (valid_dists / 200)
        # 将映射后的R通道值赋给红色点的R通道，G和B通道保持为0
        heatmap_colors[red_mask, 0] = r_channel_values  # 只设置红色通道

        # 保留绿色点的颜色
        heatmap_colors[green_mask] = colors[green_mask]
          
        # update the point cloud color
        points_scene.colors = o3d.utility.Vector3dVector(heatmap_colors)
        output_ply_file = ply_file_path.replace(".ply", f"_{direction}.ply")
        o3d.io.write_point_cloud(output_ply_file, points_scene)
        

ply_file_path = "dataset/test/scene_RGBD_mask_direction_mult/id117_2/phone_0000_normal/mask.ply"

avg_green_position = color_non_green_neighbors_after_black(ply_file_path)

if avg_green_position is not None:
    print(f"绿色点的平均位置: {avg_green_position}")
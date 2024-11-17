"""
    Generate the dataset for the direction estimation task from scene_RGBD
"""
import cv2
import os
import numpy as np
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
import open3d as o3d
import random
import shutil
from GeoL_net.dataset_gen.RGBD2MaskPC_direction import remove_ground
def RGBD_mask2direction_mult(id_obj_file_dir, out_put_dir):
    """
    Generate the dataset for the direction estimation task from scene_RGBD

    Params:
    id_obj_file_dir: the directory of the id_obj_file, e,g. dataset/scene_RGBD_mask/id0/bottle_0001_plastic
    out_put_dir: the output directory

    Return:
    None
    """
    # 进入到id_obj_file_dir 读取其中的mask_with_obj.jpg
    img_mask_with_obj = cv2.imread(os.path.join(id_obj_file_dir, "mask_with_obj.png"))
    color = np.array(img_mask_with_obj) / 255.0

    # 读取depth.img
    depth_path = os.path.join(id_obj_file_dir, "with_obj/test_pbr/000000/depth_noise/000000.png")
    depth = np.array(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED))
  

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

        # camera rotation 
    cam_rotation_matrix = np.array([
        [1, 0, 0],
        [0,0.8,-0.6],
        [0,0.6,0.8]
    ])

    # rotate the points
    points_scene = (cam_rotation_matrix @ points_scene.T).T
    colors_scene = color[scene_idx[0], scene_idx[1]]
    pcd_scene = visualize_points(points_scene, colors_scene)

    #o3d.io.write_point_cloud(os.path.join(out_put_dir, "scene.ply"), pcd_scene)
    pcd_mask_preprocessing_direction_red_label(pcd_scene, out_put_dir)


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

def pcd_mask_preprocessing_direction_red_label(points_scene, output_dir = None):
    " generate turbo heatmap label in 8 directions"
    " generate red label"

    colors = np.asarray(points_scene.colors)
    points = np.asarray(points_scene.points)

    # get the mean of the green points in the points scene
    green_mask = np.apply_along_axis(is_green, 1, colors)
    green_points = points[green_mask]
    green_pcd = o3d.geometry.PointCloud()
    green_pcd.points = o3d.utility.Vector3dVector(green_points)
    green_pcd_center = green_pcd.get_center()


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
    points_scene.colors = o3d.utility.Vector3dVector(colors)

    # get the red points
    red_mask = np.all(colors == [1, 0, 0], axis=1)
    red_points = points[red_mask]

    # re-extract 
    black_mask = np.apply_along_axis(is_black, 1, colors) # groudn and table
    blue_mask = np.apply_along_axis(is_blue, 1, colors) # obj
    red_mask = np.all(colors == [1, 0, 0], axis=1) # tabletop
   
    offset = random.uniform(100, 300)

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
        valid_dists[valid_dists > 150] = 150  # 将所有大于150的距离值截断为150

            # 将距离映射到R通道值，距离为0时R=1，距离为150时R=0
        r_channel_values = 1 - (valid_dists / 150)
        # 将映射后的R通道值赋给红色点的R通道，G和B通道保持为0
        heatmap_colors[red_mask, 0] = r_channel_values  # 只设置红色通道

        # 保留绿色点的颜色
        heatmap_colors[green_mask] = colors[green_mask]
          
        # update the point cloud color
        points_scene.colors = o3d.utility.Vector3dVector(heatmap_colors)
        #o3d.visualization.draw_geometries([points_no_obj_scene])
        # add the point cloud to the list
        pcd_without_ground = remove_ground(points_scene)
        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)
        o3d.io.write_point_cloud(f"{output_dir}/mask_{direction}.ply", pcd_without_ground)
        print(f"{output_dir}/mask_{direction}.ply is done")


if __name__ == "__main__":
    id_obj_file_dir = "dataset/scene_RGBD_mask/id695_2/monitor_0011_white"
    output_dir = id_obj_file_dir.replace("scene_RGBD_mask", "scene_RGBD_mask_direction_mult")
    # check if the out_put_dir exists
    # 遍历得到scene_RGBD_mask下的所有文件夹中的子文件夹 例如id695_2/monitor_0011_white
    dir_name = "dataset/scene_RGBD_mask"
    id_obj_dir_list = []
    amount = 0
    for root, dirs, files in os.walk(dir_name):
        for dir in dirs:
            id_obj_file_dir = os.path.join(dir_name, dir)
            for root, dirs, files in os.walk(id_obj_file_dir):
                sub_amout = 0 # 3 cases each id 
                for dir in dirs:
                    id_obj_dir = os.path.join(id_obj_file_dir, dir)
                    #id_obj_dir_list.append(id_obj_dir)
                    output_dir = id_obj_dir.replace("scene_RGBD_mask", "test/scene_RGBD_mask_direction_mult")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    RGBD_mask2direction_mult(id_obj_dir, output_dir)
                    img_path = os.path.join(id_obj_dir, "with_obj/test_pbr/000000/rgb/000000.jpg")
                    target_path = os.path.join(output_dir, "img.jpg")
                    shutil.copy(img_path, target_path)
                    amount += 1
                    print(f"{amount} is done")
                    #print(id_obj_dir_list)
                    sub_amout += 1
                    print(sub_amout)
                    if sub_amout == 3: # 3 cases each id 
                        break
                break
        break

    """
    while amount < 200:
        id_obj_dir = random.choice(id_obj_dir_list)
        output_dir = id_obj_dir.replace("scene_RGBD_mask", "scene_RGBD_mask_direction_mult")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        RGBD_mask2direction_mult(id_obj_dir, output_dir)
        img_path = os.path.join(id_obj_dir, "with_obj/test_pbr/000000/rgb/000000.jpg")
        target_path = os.path.join(output_dir, "img.jpg")
        shutil.copy(img_path, target_path)
        amount += 1
        print(f"{amount} is done")
    """
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    #RGBD_mask2direction_mult(id_obj_file_dir, output_dir)
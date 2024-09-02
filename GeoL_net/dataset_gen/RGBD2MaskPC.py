"""
transfer the RGBD to a mask pointcloud"""
from GeoL_net.dataset_gen.hdf52png import hdf52png
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
from GroundingDINO.RGB_dect import rgb_obj_dect
import cv2
import numpy as np
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points
import random
from tqdm import tqdm

def RGBD2MaskPC(depth_path, mask_hd5f_path, output_dir, reprocessing_flag=False, depth_removed_obj = None, mask_hd5f_removed_obj = None):
    """ transfer deph and mask hd5f to point cloud(mask)
    """
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
    #centroid = np.mean(points_scene, axis=0)
    #print("centroid:", centroid)
    #points_scene = points_scene - centroid
    colors_scene = color[scene_idx[0], scene_idx[1]]
    pcd_scene = visualize_points(points_scene, colors_scene)

        # camera rotation no obj
    if (depth_removed_obj is not None) and (mask_hd5f_removed_obj is not None):
        points_no_obj_scene = (cam_rotation_matrix @ points_no_obj_scene.T).T
        #points_no_obj_scene = points_no_obj_scene - centroid
        colors_no_obj_scene = color_no_obj[scene_no_obj_idx[0], scene_no_obj_idx[1]]
        pcd_no_obj_scene = visualize_points(points_no_obj_scene, colors_no_obj_scene)
        
        # save the pcd no obj scene
        pcd_no_obj_scene = remove_ground(pcd_no_obj_scene)
        pcd_ori_path = f"{output_dir}/pc_ori.ply"
        o3d.io.write_point_cloud(pcd_ori_path, pcd_no_obj_scene)

    if reprocessing_flag == True:
        if (depth_removed_obj is not None) and (mask_hd5f_removed_obj is not None):
            points_scene = pcd_mask_preprocessing(points_scene=pcd_scene, points_no_obj_scene=pcd_no_obj_scene)
        else:
            points_scene = pcd_mask_preprocessing(points_scene=pcd_scene)
    
    # remove the ground floor
    points_scene = remove_ground(points_scene)
    
    # pc_mask_path
    pc_mask_path = f"{output_dir}/mask.ply"
    o3d.io.write_point_cloud(pc_mask_path, points_scene)

def is_blue(color, threshold=0.9):
    return color[0] < (1 - threshold) and color[1] < (1 - threshold) and color[2] > threshold

def is_black(color, threshold=0.1):
    return color[0] < threshold and color[1] < threshold and color[2] < threshold

def is_green(color, threshold=0.9):
    return color[0] < (1 - threshold) and color[1] > threshold and color[2] < (1 - threshold)

def is_red(color, threshold=0.9):
    return color[0] > threshold and color[1] < (1 - threshold) and color[2] < (1 - threshold)

def check_collision(green_pcd, other_pcd, threshold=0.001):
    # 计算绿色点云每个点到最近邻点的距离
    distances = green_pcd.compute_point_cloud_distance(other_pcd)
    # 如果最近邻点的最小距离小于阈值，则认为存在碰撞
    #print("distance:", np.min(distances))
    return np.min(distances) <= threshold

def check_collisiton_batch(green_pcds, other_pcds, threshold=0.001):
    """
    check the collision in batch

    Paras:
        green_pcds: B * N1* 3
        other_pcds: B * N2 * 3
    Return:
        is_collision: B
    """


def compute_distance_matrix(points_a, points_b):
    distances = np.sqrt(np.sum((points_a[:, None] - points_b[None, :])**2, axis=2))
    return distances

def pcd_mask_preprocessing(points_scene, points_no_obj_scene = None):

    colors = np.asarray(points_scene.colors)
    points = np.asarray(points_scene.points)

    # 提取绿色点云
    green_mask = np.apply_along_axis(is_green, 1, colors)
    green_points = points[green_mask]
    green_pcd = o3d.geometry.PointCloud()
    green_pcd.points = o3d.utility.Vector3dVector(green_points)

    if points_no_obj_scene is not None:
        points_scene = points_no_obj_scene
        colors = np.asarray(points_no_obj_scene.colors)
        points = np.asarray(points_no_obj_scene.points)
    
    # 提取蓝色点云
    blue_mask = np.apply_along_axis(is_blue, 1, colors)
    blue_points = points[blue_mask]
    blue_pcd = o3d.geometry.PointCloud()
    blue_pcd.points = o3d.utility.Vector3dVector(blue_points)
    bbox = blue_pcd.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()
    
    
    # 提取black点云
    black_mask = np.apply_along_axis(is_black, 1, colors)
    z_max = blue_pcd.get_axis_aligned_bounding_box().get_max_bound()[2]
    z_center = blue_pcd.get_axis_aligned_bounding_box().get_center()[2]
    z_range_mask = (points[:, 2] <= z_max + 10) & (points[:, 2] >= z_center)
    target_mask = z_range_mask & black_mask
    colors[target_mask] = [1, 1, 1]  # 将这些点的颜色设置为白色
    points_scene.colors = o3d.utility.Vector3dVector(colors)
    
    # 提取白色点
    white_mask = np.all(colors == [1, 1, 1], axis=1)
    white_points = points[white_mask]
    
    # 在白色点中随机选择
    number_selected = 4000
    if len(white_points) > number_selected:
        selected_indices = random.sample(range(len(white_points)), number_selected)
        selected_white_points = white_points[selected_indices]
    else:
        selected_white_points = white_points
    
    # 计算绿色点云的包围盒最高的中心点
    green_bbox = green_pcd.get_axis_aligned_bounding_box()
    green_box_extent = green_bbox.get_extent()
    collision_thershold = np.sqrt(green_box_extent[0]**2 + green_box_extent[1]**2 ) / 5
    green_bbox_max_center = green_bbox.get_center()
    green_bbox_max_center[2] = green_bbox.get_max_bound()[2]
    
    '''
    # 处理每个选定的白色点 for 
    for white_point in tqdm(selected_white_points, desc="Processing white points"):
        translation_vector = white_point  - green_bbox_max_center
        green_pcd.translate(translation_vector)
             
        if not check_collision(green_pcd, blue_pcd, threshold=10): # 阈值需要大 因为点其实残缺和稀疏
            colors[np.all(points == white_point, axis=1)] = [1, 0, 0]  # 红色
            if np.sqrt((white_point[0] - green_bbox_max_center[0])**2 + (white_point[1] - green_bbox_max_center[1])**2) < 200:
                colors[np.all(points == white_point, axis=1)] = [0, 1, 0]  # green
            #print("red")
        else:
            colors[np.all(points == white_point, axis=1)] = [0, 0, 0]  # 黑色
        # 恢复绿色点云位置
    
        green_pcd.translate(-translation_vector)
    '''
    # 
    distance_vectors = selected_white_points - green_bbox_max_center
    green_points_tensor = torch.tensor(green_points)
    green_points_expanded = green_points_tensor.unsqueeze(0).cuda()
    green_points_fps, _ = sample_farthest_points(green_points_expanded, K=64)
    distance_vectors_expanded = distance_vectors[:, np.newaxis, :]
    moved_green_points = green_points_fps + torch.tensor(distance_vectors_expanded).cuda()
    moved_green_points_tensor_fps = torch.tensor(moved_green_points, dtype=torch.float32)
    blue_points_tensor = torch.tensor(blue_points, dtype=torch.float32).unsqueeze(0).cuda()
    blue_points_tensor_fps, _ = sample_farthest_points(blue_points_tensor, K=4096)
    blue_points_tensor_fps = blue_points_tensor_fps.expand(number_selected, -1, -1)
 
    distances = torch.cdist(moved_green_points_tensor_fps, blue_points_tensor_fps)
    min_distances = torch.min(distances, dim=-1).values
    final_min_dists = torch.min(min_distances, dim=-1).values
    final_min_dists = final_min_dists.cpu().numpy()

    
    # 筛选白色点
    is_white_point = np.any(np.all(points[:, np.newaxis] == selected_white_points, axis=2), axis=1)

    # 找到与白色点匹配的索引
    white_point_indices = np.where(is_white_point)[0]
    filtered_white_points = points[white_point_indices]

    # 基于坐标匹配找出相应的最小距离
    dist_mask = np.all(filtered_white_points[:, np.newaxis, :] == selected_white_points, axis=2)
    matched_dists = np.dot(dist_mask, final_min_dists)

    # 修改颜色 (无碰撞 -> 红色)
    colors[white_point_indices] = [1, 0, 0]

    # 进一步调整颜色 (接近绿色盒子中心 -> 绿色)
    distances_to_center = np.linalg.norm(filtered_white_points[:, :2] - green_bbox_max_center[:2], axis=1)
    close_to_center_mask = distances_to_center < max(collision_thershold*5, 200)
    colors[white_point_indices[close_to_center_mask]] = [0, 1, 0]

    # 修改颜色 (有碰撞 -> 黑色)
    collision_mask = matched_dists <= collision_thershold
    colors[white_point_indices[collision_mask]] = [0, 0, 0]
    '''
    for i in tqdm(range(number_selected)):
        white_point = selected_white_points[i]
        final_min_dist = final_min_dists[i]
        if final_min_dist > 50: # no collision
            colors[np.all(points == white_point, axis=1)] = [1, 0, 0]  # 红色
            if np.sqrt((white_point[0] - green_bbox_max_center[0])**2 + (white_point[1] - green_bbox_max_center[1])**2) < 200:
                colors[np.all(points == white_point, axis=1)] = [0, 1, 0]  # green
            #print("red")
        else:
            colors[np.all(points == white_point, axis=1)] = [0, 0, 0]  # 黑色
        # 恢复绿色点云位置
    #distances, norm = chamfer_distance(moved_green_points_tensor_fps, blue_points_tensor_fps)
    #min_distances_per_points = torch.min(distances, dim=-1).values
    '''

    # 更新点云颜色
    points_scene.colors = o3d.utility.Vector3dVector(colors)
    colors = np.asarray(points_scene.colors)
    points = np.asarray(points_scene.points)
    #colors = np.asarray(points_scene.colors)
    #red_mask = np.apply_along_axis(is_red, 1, colors)
    #print(np.any(red_mask))
    # 从点云中删除所有白色点
    not_white_mask = np.any(colors != [1, 1, 1], axis=1)
    points_scene.points = o3d.utility.Vector3dVector(points[not_white_mask])
    points_scene.colors = o3d.utility.Vector3dVector(colors[not_white_mask])
    return points_scene

def remove_ground(point_cloud):
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    z_min = np.min(points[:,2])
    z_max = np.max(points[:,2])
    height = z_max - z_min

    z_thershold = z_max - height /4 # remove 25%
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
    base = "dataset/scene_RGBD_mask/id162_1/lamp_0004_orange" # only part need to be changed

    image_path = base + "/with_obj/test_pbr/000000/rgb/000000.jpg"
    depth_path = base + "/with_obj/test_pbr/000000/depth/000000.png"
    mask_hdf5_path = base + "/with_obj/0.hdf5"
    output_dir = base
    depth_removed_obj = base + "/no_obj/test_pbr/000000/depth/000000.png"
    mask_hdf5_removed_obj = base + "/no_obj/0.hdf5"
    ref_image_path = base + "/RGB_ref.jpg"


    
    #center_x, center_y = rgb_obj_dect(image_path, text_prompt)
    #print("rgb_obj_dect is done")
    RGBD2MaskPC(depth_path=depth_path, 
                mask_hd5f_path=mask_hdf5_path,
                output_dir=output_dir,
                reprocessing_flag=True,
                depth_removed_obj= depth_removed_obj,
                mask_hd5f_removed_obj=mask_hdf5_removed_obj
               )
    print("-----RGBD2MaskPC done-----")
    RGB2RefMaskPC(depth_path=depth_path,
                  ref_image_path=ref_image_path,
                  out_dir=output_dir)
    print("-----RGBD2RefMaskPC done-----")
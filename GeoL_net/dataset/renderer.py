"""
    infer the position in blender from the position in pointcloud
"""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import json
import cv2
from scipy.spatial.transform import Rotation as R
def backproject(depth, intrinsics, instance_mask, NOCS_convention=False):
    """
        depth: np.array, [H,W]
        intrinsics: np.array, [3, 3]
        instance_mask: np.array, [H, W]; (np.logical_and(depth>0, depth<2))
    """
    intrinsics_inv = np.linalg.inv(intrinsics)
    non_zero_mask = depth > 0
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    if NOCS_convention:
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]

    return pts, idxs

def find_max_value_point(ply_path):
    point_cloud = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    cmap = plt.cm.get_cmap("turbo", 256)
    cmap_colors = cmap(np.linspace(0, 1, 256))[:, :3]

    distances = np.linalg.norm(colors[:, None, :] - cmap_colors[None, :, :], axis=2)

    closest_indices = np.argmin(distances, axis=1)

    values = closest_indices / 255

    max_value_index = np.argmax(values)

    return points[max_value_index], values[max_value_index]

def transform_points_to_world_fromRGBD(points, json_file):
    """
    transform point cloud to world coordinates
    
    Params:
        points(numpy.ndarray): (N, 3)
        json_file
    
    Return:
        numpy.ndarray
    """
    cam_paras = load_camera_params(json_file=json_file)
    cam_K = cam_paras["cam_K"]
    cam_R = cam_paras["cam_R"]
    cam_t = cam_paras["cam_t"]

    
    transformed_points = np.dot(cam_R, points.T).T + cam_t

    return transformed_points

def load_camera_params(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    cam_K = np.array(data["0"]['cam_K']).reshape(3,3)
    cam_R = np.array(data['0']['cam_R_w2c']).reshape(3,3)

    cam_t = np.array(data['0']['cam_t_w2c'])


    camera_dict = {"cam_K": cam_K,
                   "cam_R": cam_R,
                   "cam_t": cam_t
                   }

    return camera_dict

def pixel_to_world(u, v, depth_map, K, R, T):
    """
    transform u, v in RGB to world coord 

    Params:
    u, v: 像素坐标
    depth_map: depth image
    K: instrinsic
    R: rotation  w2c
    T: translation w2c

    Return:
    world_point: 世界坐标系中的3D点 (3x1)
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    x = (u - cx) / fx
    y = (v - cy) / fy

    Z = depth_map[u,v]/1000

    X = x * Z# depth scale = 1
    Y = y * Z

    camera_coords = np.array([X, Y, Z])  

    
    world_coords = np.dot(R.T, (camera_coords - T/1000)) # camera coords to world_coords
    
    return world_coords

def set_pixel_red(image, u, v):
    """
    set (u,v) color as red

    Params:
    image: original image
    u, v: coords
    
    Return:
    image: modified image
    """
    # 检查像素坐标是否在图片范围内
    if v < 0 or v >= image.shape[0] or u < 0 or u >= image.shape[1]:
        raise ValueError("out of range")

    # 将像素点 (u, v) 设为红色 (在OpenCV中BGR格式，所以要设置为 [0, 0, 255])
    image[v, u] = [0, 0, 255]

    return image


if __name__ == "__main__":

    world_file = "dataset/scene_RGBD_mask/id163_2/monitor_0012_white/no_obj/test_pbr/000000/scene_camera.json"
    pose_file = "dataset/scene_gen/scene_mesh_json/id163_2.json"
    # pose_file2 = "dataset/scene_RGBD_mask/id136_2/bowl_0001_wooden/no_obj/test_pbr/000000/scene_gt.json"
    with open(world_file, 'r') as f:
        cam_data = json.load(f)
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    # with open(pose_file2, 'r') as f:
    #     pose_data2 = json.load(f)

    cam_K = np.array(cam_data["0"]['cam_K']).reshape(3,3)
    cam_R = np.array(cam_data['0']['cam_R_w2c']).reshape(3,3)

    cam_t = np.array(cam_data['0']['cam_t_w2c'])
    T_cw = np.eye(4)
    T_cw[:3, :3] = cam_R
    T_cw[:3, 3] = cam_t / 1000.
    T_wc = np.linalg.inv(T_cw)
    depth_path = "dataset/scene_RGBD_mask/id163_2/monitor_0012_white/with_obj/test_pbr/000000/depth/000000.png"
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0

    image = cv2.imread("dataset/scene_RGBD_mask/id163_2/monitor_0012_white/with_obj/test_pbr/000000/rgb/000000.jpg")
    image = image[:, :, ::-1]

    points, scene_ids = backproject(depth_image, cam_K, depth_image>0)
    points_in_world = points @ T_wc[:3, :3].T + T_wc[:3, 3]

    point_colors = image[scene_ids[0], scene_ids[1]] / 255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)   
    to_vis = [pcd]
    for k, v in pose_data.items():
        T_wo = np.eye(4)
        z_angle = v[2]
        T_wo[:3,  3] = np.array(v[0])
        T_wo[2,3] = 0 # set z to 0 (as what we did in blender)
        T_wo[:3, :3] = R.from_euler('z', z_angle, degrees=True).as_matrix()
        T_co = T_cw @ T_wo
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        axis.transform(T_co)
        to_vis.append(axis)
    
    # for k, v in pose_data2.items():
    #     for ni in range(len(v)):
    #         T_co = np.eye(4)
    #         T_co[:3, :3] = np.array(v[ni]["cam_R_m2c"]).reshape(3,3).T
    #         T_co[:3, 3] = np.array(v[ni]["cam_t_m2c"]) / 1000.
    #         T_wo = T_wc @ T_co
    #         pose = T_wo
    #         axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    #         axis.transform(T_wo)
    #         axis.paint_uniform_color([1,0,0])
    #         to_vis.append(axis)

    o3d.visualization.draw(to_vis)


    # world_point = pixel_to_world(u, v, depth_map, K=cam_K, R=cam_R, T=cam_t)
    # print(world_point)

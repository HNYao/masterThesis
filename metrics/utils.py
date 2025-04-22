import numpy as np
import cv2
from PIL import Image, ImageDraw
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
import open3d as o3d
import math
from GeoL_diffuser.models.utils.fit_plane import fit_plane_from_points, get_tf_for_scene_rotation
import h5py
import trimesh
import json
import copy

def hdf52png_table(hdf5_path, output_dir=None):
    """
    This function is used to get table mask from category_id seg_maps. only extract table mask
    """
    category_colors = {
        0: (0, 0, 0),         
        1: (0, 0, 0),   #Obj Black   
        2: (255, 255, 255),   #Table - White
        3: (0, 0, 0),   #Ground - Black
        4: (0, 0, 0),  # removed obj - Green 
        5: (0, 0, 0) # anchor obj - Red  
    }

    file_path =  hdf5_path
    with h5py.File(file_path, 'r') as hdf5_file:
        segmap = hdf5_file['/category_id_segmaps'][...]
        segmap = segmap.astype(np.int32)
        rgb_image = np.zeros((segmap.shape[0], segmap.shape[1], 3), dtype=np.uint8)
        for category, color in category_colors.items():
            rgb_image[segmap == category] = color
    
        #cv2.imwrite("outputs/img_output/table_mask.png",rgb_image)
    return rgb_image

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

def visualize_points(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def get_tf_for_scene_rotation(points, axis="z"):
    """
    Get the transformation matrix for rotating the scene to align with the plane normal.

    Args:
        points (np.ndarray): The points of the scene.
        axis (str): The axis to align with the plane normal.
    
    Returns:
        T_plane: The transformation matrix.
        plane_model: The plane model.

    """

    points_filtered = points
    plane_model = fit_plane_from_points(points_filtered)
    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(points_filtered)


    plane_dir = -plane_model[:3]
    plane_dir = plane_dir / np.linalg.norm(plane_dir)
    T_plane = np.eye(4)
    if axis == "y":
        T_plane[:3, 1] = -plane_dir
        T_plane[:3, 1] /= np.linalg.norm(T_plane[:3, 1])
        T_plane[:3, 2] = -np.cross([1, 0, 0], plane_dir)
        T_plane[:3, 2] /= np.linalg.norm(T_plane[:3, 2])
        T_plane[:3, 0] = np.cross(T_plane[:3, 1], T_plane[:3, 2])
    elif axis == "z":
        T_plane[:3, 2] = -plane_dir 
        T_plane[:3, 2] /= np.linalg.norm(T_plane[:3, 2]) 
        T_plane[:3, 0] = -np.cross([0, 1, 0], plane_dir) 
        T_plane[:3, 0] /= np.linalg.norm(T_plane[:3, 0])
        T_plane[:3, 1] = np.cross(T_plane[:3, 2], T_plane[:3, 0])

    return T_plane, plane_model

def fit_plane_from_points(points, threshold=0.01, ransac_n=3, num_iterations=2000):
    """
    Fit a plane from the points.

    Args:
        points (np.ndarray): The points.
        threshold (float): The threshold for RANSAC.
        ransac_n (int): The number of points to sample for RANSAC.
        num_iterations (int): The number of iterations for RANSAC.
    
    Returns:
        plane_model: The plane model.
    
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = o3d.geometry.PointCloud.segment_plane(
    pcd,
    distance_threshold=threshold,
    ransac_n=ransac_n,
    num_iterations=num_iterations,
    )
    return plane_model

def sample_points_in_bbox(image, bbox: list, n: int = 20):
    """
    Randomly samples n points inside a bounding box in an image and updates the image such that:
    - Sampled points are colored red.
    - All other pixels are set to black.

    Args:
        image (numpy.ndarray): Input image as a NumPy array (H, W, C) or (H, W).
        bbox (list or tuple): Bounding box coordinates [x_min, y_min, x_max, y_max],
                              values are fractions (0-1) of image dimensions.
        n (int): Number of points to sample.

    Returns:
        numpy.ndarray: Updated image with sampled points in red and other pixels black.
    """
    # Validate the bounding box
    x_min, y_min, x_max, y_max = bbox
    if not (0 <= x_min < x_max <= 1) or not (0 <= y_min < y_max <= 1):
        raise ValueError("Bounding box coordinates should be fractions between 0 and 1.")

    # Get image dimensions
    height, width = image.shape[:2]

    # Convert fractional bounding box to pixel coordinates
    x_min_px = int(x_min * width)
    y_min_px = int(y_min * height)
    x_max_px = int(x_max * width)
    y_max_px = int(y_max * height)

    # Generate n random points within the bounding box
    x_coords = np.random.randint(x_min_px, x_max_px, size=n)
    y_coords = np.random.randint(y_min_px, y_max_px, size=n)

    # Stack x and y coordinates together
    points = np.stack((x_coords, y_coords), axis=1)

    # Create a new image with all pixels set to black
    updated_image = np.zeros_like(image)

    # Set the sampled points to red
    for point in points:
        if len(updated_image.shape) == 3:  # Color image
            updated_image[point[1], point[0]] = [255, 0, 0]  # Red in BGR format
        else:  # Grayscale image
            updated_image[point[1], point[0]] = 255  # White in grayscale

    return updated_image




def process_direction_metrics(
        image_sampled_point: np.ndarray, 
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
        ):
    """
    Processes the metric for a single image pair.
    especially for the case of predicting through VLM

    image_sampled_point: np.ndarray, [H, W, C] in red
    """
    # intrinsics
    if intrinsics is None:
        intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                    [  0.     , 607.05212/2, 367.35952/2],
                    [  0.     ,   0.     ,   1.     ]])

    # get the points_cloud of with and without object
    depth_with_obj = np.array(cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED))
    depth_without_obj = np.array(cv2.imread(depth_without_obj_path, cv2.IMREAD_UNCHANGED))
    pts_with_obj, idx_with_obj = backproject(depth_with_obj, intrinsics, np.logical_and(depth_with_obj > 0, depth_with_obj >0))
    pts_without_obj, idx_without_obj = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj > 0))
    color_with_obj = cv2.imread(mask_with_obj_path, cv2.COLOR_BGR2RGB)[idx_with_obj[0], idx_with_obj[1]]
    color_without_obj = cv2.imread(mask_without_obj_path, cv2.COLOR_BGR2RGB)[idx_without_obj[0], idx_without_obj[1]]
    pts_with_obj = visualize_points(pts_with_obj, color_with_obj/255)
    pts_without_obj = visualize_points(pts_without_obj, color_without_obj/255)

    # rotate the pts to algin with xy plane
    T_plane_with_obj, plane_model_with_obj = get_tf_for_scene_rotation(np.asarray(pts_with_obj.points), axis="z")
    T_plane_without_obj, plane_model_without_obj = get_tf_for_scene_rotation(np.asarray(pts_without_obj.points), axis="z")
    pts_with_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_with_obj.points) @ T_plane_with_obj[:3, :3])
    pts_without_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_without_obj.points) @ T_plane_without_obj[:3, :3])
    
    # metrics 1: position relateve
    # get ground truth
    bbox_achor = find_color_and_bbox(pts_without_obj, target_color=[1, 0, 0])
    bbox_achor_center = bbox_achor.get_center()

    # get the red points, which are the sampled points
    pts_sampled_point = o3d.geometry.PointCloud()
    pts_sampled_point = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj > 0))[0]
    color_sampled_point = image_sampled_point[idx_without_obj[0], idx_without_obj[1]]
    pts_sampled_point = visualize_points(pts_sampled_point, color_sampled_point/255)
    pts_sampled_point.points = o3d.utility.Vector3dVector(np.asarray(pts_sampled_point.points) @ T_plane_without_obj[:3, :3])

    colors = np.asarray(pts_sampled_point.colors)
    points = np.asarray(pts_sampled_point.points)

    is_red = (colors[:, 0] >= 0.9) & (colors[:, 1] <= 0.1) & (colors[:, 2] <= 0.1)
    sampled_points = points[is_red]

    #test_point = [1000, -1300, 825]
    results = get_relative_direction(sampled_points, bbox_achor_center, direction=direction)
    #test_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=100)
    #test_sphere.translate(test_point)
    
    #### visualize for dubugging
    # add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])

    # o3d.visualization.draw_geometries([pts_with_obj, coordinate_frame, test_sphere])
    # o3d.visualization.draw_geometries([pts_without_obj, coordinate_frame])
    # pts_gt = o3d.io.read_point_cloud("dataset/scene_RGBD_mask_v2_kinect_cfg/id1_1/bowl_0002_glass/mask_Right.ply")
    # #o3d.visualization.draw_geometries([pts_gt, pts_with_obj])
    vis = [coordinate_frame, pts_without_obj]
    for point in sampled_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=100)
        sphere.translate(point)
        vis.append(sphere)
        #o3d.visualization.draw_geometries([pts_without_obj, sphere])
    #o3d.visualization.draw_geometries(vis)
    return results


    # pass

def find_color_and_bbox(point_cloud, target_color, color_tolerance=0.1):
    """
    find the specific color in the point cloud and return the bounding box of the color

    Args:
        point_cloud (numpy.ndarray): The point cloud.
        target_color (list): The target color.
        color_tolerance (float): The color tolerance.
    
    Returns:
        bbox: The bounding box of the color.
    """

    colors = np.asarray(point_cloud.colors)
    points = np.asarray(point_cloud.points)

    lower_bound = np.array(target_color) - color_tolerance
    upper_bound = np.array(target_color) + color_tolerance

    mask = np.all(colors >= lower_bound, axis=1) & np.all(colors <= upper_bound, axis=1)

    filtered_points = points[mask]

    if len(filtered_points) == 0:
        raise ValueError("No points found with the target color.")
    
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

    bbox = filtered_point_cloud.get_axis_aligned_bounding_box()

    #### visualize for dubugging
    # add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    bbox.color = [1, 0, 0]
    #o3d.visualization.draw_geometries([point_cloud, coordinate_frame, bbox])

    return bbox

def get_relative_direction(points:np.ndarray, target_point:np.ndarray, direction:str):
    """
    Get the relative direction of the points with respect to the target point.

    Args:
        points (numpy.ndarray): The points, (n, 3).
        target_point (numpy.ndarray): The target point (3,).
        direction: e.g "Left".
    
    Returns:
        list of boll: if each point is in the specified direction of the target point.
"""
    if not isinstance(points, np.ndarray):
        raise ValueError("points should be a numpy.ndarray.")
    if not isinstance(target_point, np.ndarray):
        raise ValueError("target_point should be a numpy.ndarray.")
    if direction not in ['Left', 'Right', 'Front', 'Behind', 'Left Behind', 'Right Behind', 'Left Front', 'Right Front']:
        raise ValueError("direction should be one of ['Left', 'Right', 'Front', 'Behind', 'Left Behind', 'Right Behind', 'Left Front', 'Right Front']")
    
    # get the xy of points
    points_xy = points[:, :2]
    target_xy = target_point[:2]

    offsets = target_xy - points_xy

    results = []

    angle_disturb = 22.5

    for offset in offsets:
        angel = math.degrees(math.atan2(offset[1], offset[0]))

        if angel < 0:
            angel += 360
        print(angel)
    
        satisfies_direction = False
        if direction == "Left" and 337.5-angle_disturb <= angel <= 360 + angle_disturb or 0 - angle_disturb <= angel <= 22.5 + angle_disturb or 360-angle_disturb<=angel:
            satisfies_direction = True
        elif direction == "Left Front" and 292.5 - angle_disturb < angel <= 337.5 + angle_disturb:
            satisfies_direction = True
        elif direction == "Front" and 247.5-angle_disturb < angel <= 292.5+angle_disturb:
            satisfies_direction = True
        elif direction == "Right Front" and 202.5-angle_disturb < angel <= 247.5+angle_disturb:
            satisfies_direction = True
        elif direction == "Right" and 157.5-angle_disturb < angel <= 202.5+angle_disturb:
            satisfies_direction = True
        elif direction == "Rgiht Behind" and 112.5-angle_disturb < angel <= 157.5+angle_disturb:
            satisfies_direction = True
        elif direction == "Behind" and 67.5-angle_disturb < angel <= 112.5+angle_disturb:
            satisfies_direction = True
        elif direction == "Left Behind" and 22.5-angle_disturb < angel <= 67.5+angle_disturb:
            satisfies_direction = True
        
        results.append(satisfies_direction)
    
    return results


def process_receptacle_metrics(
        image_sampled_point: np.ndarray, 
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
        ):
    """
    Processes the metric for a single image pair."""

    if intrinsics is None:
        intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                    [  0.     , 607.05212/2, 367.35952/2],
                    [  0.     ,   0.     ,   1.     ]])
    
    # get the points_cloud of without object
    depth_with_obj = np.array(cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED))
    depth_without_obj = np.array(cv2.imread(depth_without_obj_path, cv2.IMREAD_UNCHANGED))
    pts_with_obj, idx_with_obj = backproject(depth_with_obj, intrinsics, np.logical_and(depth_with_obj > 0, depth_with_obj > 0))
    pts_without_obj, idx_without_obj = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj > 0))
    color_with_obj = cv2.imread(mask_with_obj_path, cv2.COLOR_BGR2RGB)[idx_with_obj[0], idx_with_obj[1]]
    color_without_obj = cv2.imread(mask_without_obj_path, cv2.COLOR_BGR2RGB)[idx_without_obj[0], idx_without_obj[1]]
    pts_with_obj = visualize_points(pts_with_obj, color_with_obj/255)
    pts_without_obj = visualize_points(pts_without_obj, color_without_obj/255)

    # rotate the pts to algin with xy plane
    T_plane_with_obj, plane_model_with_obj = get_tf_for_scene_rotation(np.asarray(pts_with_obj.points), axis="z")
    T_plane_without_obj, plane_model_without_obj = get_tf_for_scene_rotation(np.asarray(pts_without_obj.points), axis="z")
    pts_with_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_with_obj.points) @ T_plane_with_obj[:3, :3])
    pts_without_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_without_obj.points) @ T_plane_without_obj[:3, :3])
    
    # get the ground truth of the receptacle
    # get the mask of the table 
    table_mask_img = hdf52png_table(hdf5_path=hdf5_path, ) # white mask for table / desk, without obj
    color_table_mask = table_mask_img[idx_without_obj[0], idx_without_obj[1]]
    pts_table_mask = visualize_points(pts_without_obj.points, color_table_mask/255) # already rotated
    withe_mask_thershold = 0.98
    points = np.asarray(pts_table_mask.points)
    colors = np.asarray(pts_table_mask.colors)
    is_white_mask = np.all(colors > withe_mask_thershold, axis=1)
    white_points = points[is_white_mask]
    white_colors = colors[is_white_mask]

    desk_pcd = o3d.geometry.PointCloud()
    desk_pcd.points = o3d.utility.Vector3dVector(white_points)
    desk_pcd.colors = o3d.utility.Vector3dVector(white_colors)
    desk_bbox = desk_pcd.get_axis_aligned_bounding_box()
    x_min, y_min, z_min = desk_bbox.get_min_bound()
    x_max, y_max, z_max = desk_bbox.get_max_bound()

    # visualize for debugging
    desk_bbox.color = [1, 0, 0]
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])
    #o3d.visualization.draw_geometries([pts_table_mask, coordinate_frame, desk_bbox])

    # get the red points, which are the sampled points
    pts_sampled_point = o3d.geometry.PointCloud()
    pts_sampled_point = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj > 0))[0]
    color_sampled_point = image_sampled_point[idx_without_obj[0], idx_without_obj[1]]
    pts_sampled_point = visualize_points(pts_sampled_point, color_sampled_point/255)
    pts_sampled_point.points = o3d.utility.Vector3dVector(np.asarray(pts_sampled_point.points) @ T_plane_without_obj[:3, :3])

    colors_sampled_point = np.asarray(pts_sampled_point.colors)
    points_sampled_point = np.asarray(pts_sampled_point.points)

    is_red = (colors_sampled_point[:, 0] >= 0.9) & (colors_sampled_point[:, 1] <= 0.1) & (colors_sampled_point[:, 2] <= 0.1)
    sampled_points = points_sampled_point[is_red]

    # check if the points in bbox
    is_in_bbox = []
    sampled_sphere = [pts_sampled_point, coordinate_frame, desk_bbox]
    for point in sampled_points:
        x, y, z = point
        desk_height = z_min
        if x_min <= x <= x_max and y_min <= y <= y_max :
            is_in_bbox.append(True)
        else:
            is_in_bbox.append(False)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=100)
        sphere.translate(point)
        sampled_sphere.append(sphere)
    
    #o3d.visualization.draw_geometries(sampled_sphere)
    return is_in_bbox
    
    
def process_collision_metrics(
        image_sampled_point: np.ndarray, 
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
):
    """
    Process the metrics for checking collision rate.
    """
    obj_mesh_path = kwargs.get("obj_mesh_path", None)
    scene_mesh_path = kwargs.get("scene_mesh_path", None)
    scene_id = scene_mesh_path.split("/")[-3]
    scene_json_path = f'/home/stud/zhoy/MasterThesis_zhoy/dataset/scene_gen/scene_mesh_json_aug/{scene_id}.json'
    scene_json = json.load(open(scene_json_path, "r"))
    #obj_scale = scene_json[obj_mesh_path.replace("/home/stud/zhoy/MasterThesis_zhoy/", "")][1] # for dense scene
    with open("/home/stud/zhoy/MasterThesis_zhoy/GeoL_net/dataset_gen/obj_size.json", 'r') as f:
        obj_target_size_json = json.load(f)
    obj_category = obj_mesh_path.split("/")[-2].rsplit("_", 2)[0]
    target_size = obj_target_size_json[obj_category]
    obj_rotation_degree = scene_json[obj_mesh_path.replace("/home/stud/zhoy/MasterThesis_zhoy/", "")][2]
    obj_rotation = np.radians(obj_rotation_degree)

    if intrinsics is None:
        intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                    [  0.     , 607.05212/2, 367.35952/2],
                    [  0.     ,   0.     ,   1.     ]])
    
    # get the points_cloud of without object
    depth_with_obj = np.array(cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED))
    depth_without_obj = np.array(cv2.imread(depth_without_obj_path, cv2.IMREAD_UNCHANGED))
    pts_with_obj, idx_with_obj = backproject(depth_with_obj, intrinsics, np.logical_and(depth_with_obj > 0, depth_with_obj > 0))
    pts_without_obj, idx_without_obj = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj > 0))
    color_with_obj = cv2.imread(mask_with_obj_path, cv2.COLOR_BGR2RGB)[idx_with_obj[0], idx_with_obj[1]]
    color_without_obj = cv2.imread(mask_without_obj_path, cv2.COLOR_BGR2RGB)[idx_without_obj[0], idx_without_obj[1]]
    pts_with_obj = visualize_points(pts_with_obj, color_with_obj/255)
    pts_without_obj = visualize_points(pts_without_obj, color_without_obj/255)

    # get the obj mesh and scene mesh(completed scene)
    obj_mesh = trimesh.load_mesh(obj_mesh_path)
    current_size = obj_mesh.bounds[1] - obj_mesh.bounds[0]
    obj_scale = [target_size[0]/current_size[0], target_size[1]/current_size[1], target_size[2]/current_size[2]]
    obj_mesh.apply_scale(obj_scale)
    scene_mesh = trimesh.load_mesh(scene_mesh_path)
    obj_points_sampled = obj_mesh.sample(2000)
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points_sampled * 1000) # scale obj
    scene_points_sampled = scene_mesh.sample(10000)
    scene_comp_pcd = o3d.geometry.PointCloud()
    scene_comp_pcd.points = o3d.utility.Vector3dVector(scene_points_sampled * 1000) 
    
    # algin the obj and scene mesh points
    scene_rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    scene_comp_pcd.rotate(scene_rotation_matrix, center=(0,0,0))
    scene_comp_pcd.translate([0, -1000, 1000])
    obj_rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ np.array([[np.cos(obj_rotation), -np.sin(obj_rotation), 0], [np.sin(obj_rotation), np.cos(obj_rotation), 0], [0, 0, 1]])
    obj_pcd.rotate(obj_rotation_matrix, center=(0,0,0)) # rotate obj mesh
    obj_pcd.translate([0, -1000, 1000])
    #o3d.visualization.draw_geometries([scene_comp_pcd])
    #o3d.visualization.draw_geometries([obj_pcd])
    #o3d.visualization.draw_geometries([obj_pcd, scene_comp_pcd])


    # rotate the pts to algin with xy plane
    T_plane_with_obj, plane_model_with_obj = get_tf_for_scene_rotation(np.asarray(pts_with_obj.points), axis="z")
    T_plane_without_obj, plane_model_without_obj = get_tf_for_scene_rotation(np.asarray(pts_without_obj.points), axis="z")
    pts_with_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_with_obj.points) @ T_plane_with_obj[:3, :3])
    pts_without_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_without_obj.points) @ T_plane_without_obj[:3, :3])
    
    # get the ground truth of the receptacle
    # get the mask of the table 

    #o3d.visualization.draw_geometries([pts_with_obj, pts_without_obj, obj_pcd, scene_comp_pcd])

    # get the red points, which are the sampled points
    pts_sampled_point = o3d.geometry.PointCloud()
    pts_sampled_point = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj > 0))[0]
    color_sampled_point = image_sampled_point[idx_without_obj[0], idx_without_obj[1]]
    pts_sampled_point = visualize_points(pts_sampled_point, color_sampled_point/255)
    pts_sampled_point.points = o3d.utility.Vector3dVector(np.asarray(pts_sampled_point.points) @ T_plane_without_obj[:3, :3])

    colors = np.asarray(pts_sampled_point.colors)
    points = np.asarray(pts_sampled_point.points)

    is_red = (colors[:, 0] >= 0.999) & (colors[:, 1] <= 0.1) & (colors[:, 2] <= 0.1)
    sampled_points = points[is_red]

    # check collision
    non_collision = []
    sampled_sphere = [pts_without_obj, scene_comp_pcd]
    for point in sampled_points:
        # move the obj to the point
        obj_max_bound = obj_pcd.get_max_bound()
        obj_min_bound = obj_pcd.get_min_bound()
        obj_bottom_center = (obj_max_bound + obj_min_bound) / 2
        obj_bottom_center[2] = obj_max_bound[2]  # attention: the z axis is reversed
        
        obj_pcd.translate(point - obj_bottom_center)
        obj_pcd.paint_uniform_color([1, 1, 0])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
        sphere.translate(point)
        sampled_sphere.append(sphere)
        noncollision = isNonCollision(obj_pcd, scene_comp_pcd)
        #print(noncollision)
        non_collision.append(noncollision)
        #o3d.visualization.draw_geometries([obj_pcd, scene_comp_pcd])  
    #o3d.visualization.draw_geometries(sampled_sphere)
    collision_rate = sum(non_collision) / (len(non_collision) + 1e-6)
    #print(collision_rate)
    return non_collision

def isNonCollision(obj_pcd: o3d.geometry.PointCloud, scene_pcd: o3d.geometry.PointCloud, threshold: float = 10):
    """
    Check if the object point cloud collides with the scene point cloud.

    Args:
        obj_pcd (o3d.geometry.PointCloud): The object point cloud.
        scene_pcd (o3d.geometry.PointCloud): The scene point cloud.
        threshold (float): The collision threshold.
    
    Returns:
        bool: True if there is a collision, False otherwise.
    """
    # Create a KD tree for the scene point cloud
    scene_kdtree = o3d.geometry.KDTreeFlann(scene_pcd)

    # Check for collisions
    for point in np.asarray(obj_pcd.points):
        [_, idx, _] = scene_kdtree.search_knn_vector_3d(point, 1)
        if np.linalg.norm(point - np.asarray(scene_pcd.points)[idx[0]]) < threshold:
            return False

    return True


def process_direction_metrics_GeoL(
        fps_pcd, 
        affordance_value,
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
        ):
    """
    Processes the metric for a single image pair.
    especially for the case of predicting through VLM

    image_sampled_point: np.ndarray, [H, W, C] in red
    """
    # intrinsics
    if intrinsics is None:
        intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                    [  0.     , 607.05212/2, 367.35952/2],
                    [  0.     ,   0.     ,   1.     ]])

    # get the points_cloud of with and without object
    depth_with_obj = np.array(cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED)) 
    depth_without_obj = np.array(cv2.imread(depth_without_obj_path, cv2.IMREAD_UNCHANGED)) 
    pts_with_obj, idx_with_obj = backproject(depth_with_obj, intrinsics, np.logical_and(depth_with_obj > 0, depth_with_obj < 2000))
    pts_without_obj, idx_without_obj = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj < 2000))
    color_with_obj = cv2.imread(mask_with_obj_path, cv2.COLOR_BGR2RGB)[idx_with_obj[0], idx_with_obj[1]]
    color_without_obj = cv2.imread(mask_without_obj_path, cv2.COLOR_BGR2RGB)[idx_without_obj[0], idx_without_obj[1]]
    pts_with_obj = visualize_points(pts_with_obj, color_with_obj/255)
    pts_without_obj = visualize_points(pts_without_obj, color_without_obj/255)

    # rotate the pts to algin with xy plane
    T_plane_with_obj, plane_model_with_obj = get_tf_for_scene_rotation(np.asarray(pts_with_obj.points), axis="z")
    T_plane_without_obj, plane_model_without_obj = get_tf_for_scene_rotation(np.asarray(pts_without_obj.points), axis="z")
    pts_with_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_with_obj.points) @ T_plane_with_obj[:3, :3])
    pts_without_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_without_obj.points) @ T_plane_without_obj[:3, :3])
    
    # metrics 1: position relateve
    # get ground truth
    bbox_achor = find_color_and_bbox(pts_without_obj, target_color=[1, 0, 0])
    bbox_achor_center = bbox_achor.get_center()

    # get the points with high affordance value
    sampled_points = kwargs.get("pred_points", None)

    if sampled_points is None:
        points = np.asarray(pts_without_obj.points)
        fps_pcd.points = o3d.utility.Vector3dVector(np.asarray(fps_pcd.points) @ T_plane_without_obj[:3, :3] * 1000)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])
        #o3d.visualization.draw_geometries([pts_without_obj, fps_pcd, coordinate_frame])
        #o3d.visualization.draw_geometries([fps_pcd, coordinate_frame])
        
        # find the most topk affordance value points
        topk_afffordance = 5
        affordance_value = np.asarray(affordance_value.squeeze())
        topk_idx = np.argsort(affordance_value)[-topk_afffordance:]
        affordance_points = np.asarray(fps_pcd.points)
        topk_points = affordance_points[topk_idx]

        #### visualize for debugging 
        vis = [pts_without_obj, fps_pcd, coordinate_frame]
        for points in topk_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
            sphere.translate(points)
            vis.append(sphere)
        #o3d.visualization.draw_geometries(vis)


        sampled_points = topk_points
    #test_point = [1000, -1300, 825]
    #sampled_points = sampled_points @ T_plane_without_obj[:3, :3]
    results = get_relative_direction(sampled_points, bbox_achor_center, direction=direction)
    #test_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=100)
    #test_sphere.translate(test_point)
    
    #### visualize for dubugging
    # add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # o3d.visualization.draw_geometries([pts_with_obj, coordinate_frame, test_sphere])
    # o3d.visualization.draw_geometries([pts_without_obj, coordinate_frame])
    # pts_gt = o3d.io.read_point_cloud("dataset/scene_RGBD_mask_v2_kinect_cfg/id1_1/bowl_0002_glass/mask_Right.ply")
    # #o3d.visualization.draw_geometries([pts_gt, pts_with_obj])
    vis = [coordinate_frame, pts_without_obj]
    
    for point in sampled_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere()
        sphere.compute_vertex_normals()
        sphere.scale(50, [0,0,0])
        sphere.translate(point)
        vis.append(sphere)
        #o3d.visualization.draw_geometries([pts_without_obj, sphere])
    #o3d.visualization.draw_geometries(vis)
    #print(results)
    #print(direction)

    return results

def process_direction_metrics_GeoL_completed(
        affordance_pred, 
        affordance_value,
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
        ):
    """
    Processes the metric for a single image pair.
    especially for the case of predicting through VLM

    image_sampled_point: np.ndarray, [H, W, C] in red
    """
    # intrinsics
    if intrinsics is None:
        intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                    [  0.     , 607.05212/2, 367.35952/2],
                    [  0.     ,   0.     ,   1.     ]])

    # get the points_cloud of with and without object
    depth_with_obj = np.array(cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED)) / 1000
    depth_without_obj = np.array(cv2.imread(depth_without_obj_path, cv2.IMREAD_UNCHANGED)) /1000
    pts_with_obj, idx_with_obj = backproject(depth_with_obj, intrinsics, np.logical_and(depth_with_obj > 0, depth_with_obj < 2))
    pts_without_obj, idx_without_obj = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj < 2))
    color_with_obj = cv2.imread(mask_with_obj_path, cv2.COLOR_BGR2RGB)[idx_with_obj[0], idx_with_obj[1]]
    color_without_obj = cv2.imread(mask_without_obj_path, cv2.COLOR_BGR2RGB)[idx_without_obj[0], idx_without_obj[1]]
    pts_with_obj = visualize_points(pts_with_obj, color_with_obj/255)
    pts_without_obj = visualize_points(pts_without_obj, color_without_obj/255)

    # rotate the pts to algin with xy plane
    T_plane_with_obj, plane_model_with_obj = get_tf_for_scene_rotation(np.asarray(pts_with_obj.points), axis="z")
    T_plane_without_obj, plane_model_without_obj = get_tf_for_scene_rotation(np.asarray(pts_without_obj.points), axis="z")
    pts_with_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_with_obj.points) @ T_plane_with_obj[:3, :3])
    pts_without_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_without_obj.points) @ T_plane_without_obj[:3, :3])
    
    # metrics 1: position relateve
    # get ground truth
    bbox_achor = find_color_and_bbox(pts_without_obj, target_color=[1, 0, 0])
    bbox_achor_center = bbox_achor.get_center()

    # get the points with high affordance value
    sampled_points = kwargs.get("pred_points", None)

    if sampled_points is None:
        points = np.asarray(pts_without_obj.points)
        affordance_pred.points = o3d.utility.Vector3dVector(np.asarray(affordance_pred.points) @ T_plane_without_obj[:3, :3] * 1000)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pts_without_obj, affordance_pred, coordinate_frame])
        
        # find the most topk affordance value points
        topk_afffordance = 5
        affordance_value = np.asarray(affordance_value.squeeze())
        topk_idx = np.argsort(affordance_value)[-topk_afffordance:]
        affordance_points = np.asarray(affordance_pred.points)
        topk_points = affordance_points[topk_idx]

        #### visualize for debugging 
        vis = [pts_without_obj, affordance_pred, coordinate_frame]
        for points in topk_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
            sphere.translate(points)
            vis.append(sphere)
        #o3d.visualization.draw_geometries(vis)


        sampled_points = topk_points
    #test_point = [1000, -1300, 825]
    sampled_points = sampled_points @ T_plane_without_obj[:3, :3]
    results = get_relative_direction(sampled_points, bbox_achor_center, direction=direction)
    #test_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=100)
    #test_sphere.translate(test_point)
    
    #### visualize for dubugging
    # add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # o3d.visualization.draw_geometries([pts_with_obj, coordinate_frame, test_sphere])
    # o3d.visualization.draw_geometries([pts_without_obj, coordinate_frame])
    # pts_gt = o3d.io.read_point_cloud("dataset/scene_RGBD_mask_v2_kinect_cfg/id1_1/bowl_0002_glass/mask_Right.ply")
    # #o3d.visualization.draw_geometries([pts_gt, pts_with_obj])
    vis = [coordinate_frame, pts_without_obj]
    
    for point in sampled_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere()
        sphere.compute_vertex_normals()
        sphere.scale(0.1, [0,0,0])
        sphere.translate(point)
        vis.append(sphere)
        #o3d.visualization.draw_geometries([pts_without_obj, sphere])
    #o3d.visualization.draw_geometries(vis)

    return results

def process_receptacle_metrics_GeoL(
        fps_pcd, 
        affordance_value,
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
        ):
    """
    Processes the metric for a single image pair."""

    if intrinsics is None:
        intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                    [  0.     , 607.05212/2, 367.35952/2],
                    [  0.     ,   0.     ,   1.     ]])
    
    # get the points_cloud of without object
    depth_with_obj = np.array(cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED))
    depth_without_obj = np.array(cv2.imread(depth_without_obj_path, cv2.IMREAD_UNCHANGED))
    pts_with_obj, idx_with_obj = backproject(depth_with_obj, intrinsics, np.logical_and(depth_with_obj > 0, depth_with_obj < 2000))
    pts_without_obj, idx_without_obj = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj < 2000))
    color_with_obj = cv2.imread(mask_with_obj_path, cv2.COLOR_BGR2RGB)[idx_with_obj[0], idx_with_obj[1]]
    color_without_obj = cv2.imread(mask_without_obj_path, cv2.COLOR_BGR2RGB)[idx_without_obj[0], idx_without_obj[1]]
    pts_with_obj = visualize_points(pts_with_obj, color_with_obj/255)
    pts_without_obj = visualize_points(pts_without_obj, color_without_obj/255)

    # rotate the pts to algin with xy plane
    T_plane_with_obj, plane_model_with_obj = get_tf_for_scene_rotation(np.asarray(pts_with_obj.points), axis="z")
    T_plane_without_obj, plane_model_without_obj = get_tf_for_scene_rotation(np.asarray(pts_without_obj.points), axis="z")
    pts_with_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_with_obj.points) @ T_plane_with_obj[:3, :3])
    pts_without_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_without_obj.points) @ T_plane_without_obj[:3, :3])
    
    # get the ground truth of the receptacle
    # get the mask of the table 
    table_mask_img = hdf52png_table(hdf5_path=hdf5_path, ) # white mask for table / desk, without obj
    color_table_mask = table_mask_img[idx_without_obj[0], idx_without_obj[1]]
    pts_table_mask = visualize_points(pts_without_obj.points, color_table_mask/255) # already rotated
    withe_mask_thershold = 0.98
    points = np.asarray(pts_table_mask.points)
    colors = np.asarray(pts_table_mask.colors)
    is_white_mask = np.all(colors > withe_mask_thershold, axis=1)
    white_points = points[is_white_mask]
    white_colors = colors[is_white_mask]

    desk_pcd = o3d.geometry.PointCloud()
    desk_pcd.points = o3d.utility.Vector3dVector(white_points)
    desk_pcd.colors = o3d.utility.Vector3dVector(white_colors)
    desk_bbox = desk_pcd.get_axis_aligned_bounding_box()
    x_min, y_min, z_min = desk_bbox.get_min_bound()
    x_max, y_max, z_max = desk_bbox.get_max_bound()

    # visualize for debugging
    desk_bbox.color = [1, 0, 0]
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])
    #o3d.visualization.draw_geometries([pts_table_mask, coordinate_frame, desk_bbox])

    sampled_points = kwargs.get("pred_points", None)

    if sampled_points is None:
    # get the red points, which are the sampled points
        points = np.asarray(pts_without_obj.points)
        #fps_pcd.points = o3d.utility.Vector3dVector(np.asarray(fps_pcd.points))
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])
        #o3d.visualization.draw_geometries([pts_without_obj, affordance_pred, coordinate_frame])
        
        # find the most topk affordance value points
        topk_afffordance = 5
        affordance_value = np.asarray(affordance_value.squeeze())
        topk_idx = np.argsort(affordance_value)[-topk_afffordance:]
        affordance_points = np.asarray(fps_pcd.points)
        topk_points = affordance_points[topk_idx]
        sampled_points = topk_points
    
    #sampled_points = sampled_points @ T_plane_without_obj[:3, :3]
    # check if the points in bbox
    sampled_sphere = [desk_bbox, coordinate_frame]
    is_in_bbox = []
    for point in sampled_points:
        x, y, z = point
        desk_height = z_min
        if x_min <= x <= x_max and y_min <= y <= y_max and z_min-10 <= z <= z_min+10:
            is_in_bbox.append(True)
        else:
            is_in_bbox.append(False)
        sphere = o3d.geometry.TriangleMesh.create_sphere()
        sphere.compute_vertex_normals()
        sphere.scale(50, [0,0,0])
        sphere.translate(point)
        sampled_sphere.append(sphere)
    
    #o3d.visualization.draw_geometries(sampled_sphere)
    return is_in_bbox

def process_receptacle_metrics_GeoL_completed(
        affordance_pred, 
        affordance_value,
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
        ):
    """
    Processes the metric for a single image pair."""

    if intrinsics is None:
        intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                    [  0.     , 607.05212/2, 367.35952/2],
                    [  0.     ,   0.     ,   1.     ]])
    
    # get the points_cloud of without object
    depth_with_obj = np.array(cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED))/1000
    depth_without_obj = np.array(cv2.imread(depth_without_obj_path, cv2.IMREAD_UNCHANGED))/1000
    pts_with_obj, idx_with_obj = backproject(depth_with_obj, intrinsics, np.logical_and(depth_with_obj > 0, depth_with_obj < 2000))
    pts_without_obj, idx_without_obj = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj < 2000))
    color_with_obj = cv2.imread(mask_with_obj_path, cv2.COLOR_BGR2RGB)[idx_with_obj[0], idx_with_obj[1]]
    color_without_obj = cv2.imread(mask_without_obj_path, cv2.COLOR_BGR2RGB)[idx_without_obj[0], idx_without_obj[1]]
    pts_with_obj = visualize_points(pts_with_obj, color_with_obj/255)
    pts_without_obj = visualize_points(pts_without_obj, color_without_obj/255)

    # rotate the pts to algin with xy plane
    T_plane_with_obj, plane_model_with_obj = get_tf_for_scene_rotation(np.asarray(pts_with_obj.points), axis="z")
    T_plane_without_obj, plane_model_without_obj = get_tf_for_scene_rotation(np.asarray(pts_without_obj.points), axis="z")
    pts_with_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_with_obj.points) @ T_plane_with_obj[:3, :3])
    pts_without_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_without_obj.points) @ T_plane_without_obj[:3, :3])
    
    # get the ground truth of the receptacle
    # get the mask of the table 
    table_mask_img = hdf52png_table(hdf5_path=hdf5_path, ) # white mask for table / desk, without obj
    color_table_mask = table_mask_img[idx_without_obj[0], idx_without_obj[1]]
    pts_table_mask = visualize_points(pts_without_obj.points, color_table_mask/255) # already rotated
    withe_mask_thershold = 0.98
    points = np.asarray(pts_table_mask.points)
    colors = np.asarray(pts_table_mask.colors)
    is_white_mask = np.all(colors > withe_mask_thershold, axis=1)
    white_points = points[is_white_mask]
    white_colors = colors[is_white_mask]

    desk_pcd = o3d.geometry.PointCloud()
    desk_pcd.points = o3d.utility.Vector3dVector(white_points)
    desk_pcd.colors = o3d.utility.Vector3dVector(white_colors)
    desk_bbox = desk_pcd.get_axis_aligned_bounding_box()
    x_min, y_min, z_min = desk_bbox.get_min_bound()
    x_max, y_max, z_max = desk_bbox.get_max_bound()

    # visualize for debugging
    desk_bbox.color = [1, 0, 0]
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    #o3d.visualization.draw_geometries([pts_table_mask, coordinate_frame, desk_bbox])

    sampled_points = kwargs.get("pred_points", None)

    if sampled_points is None:
    # get the red points, which are the sampled points
        points = np.asarray(pts_without_obj.points)
        affordance_pred.points = o3d.utility.Vector3dVector(np.asarray(affordance_pred.points) @ T_plane_without_obj[:3, :3] * 1000)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        #o3d.visualization.draw_geometries([pts_without_obj, affordance_pred, coordinate_frame])
        
        # find the most topk affordance value points
        topk_afffordance = 1
        affordance_value = np.asarray(affordance_value.squeeze())
        topk_idx = np.argsort(affordance_value)[-topk_afffordance:]
        affordance_points = np.asarray(affordance_pred.points)
        topk_points = affordance_points[topk_idx]
        sampled_points = topk_points
    
    #sampled_points = sampled_points @ T_plane_without_obj[:3, :3]
    # check if the points in bbox
    sampled_points = sampled_points @ T_plane_without_obj[:3, :3]
    sampled_sphere = [desk_bbox]
    is_in_bbox = []
    for point in sampled_points:
        x, y, z = point
        desk_height = z_min
        if x_min <= x <= x_max and y_min <= y <= y_max and z_min-10 <= z <= z_min+10:
            is_in_bbox.append(True)
        else:
            is_in_bbox.append(False)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
        sphere.translate(point)
        sampled_sphere.append(sphere)
    
    #o3d.visualization.draw_geometries(sampled_sphere)
    return is_in_bbox

def process_collision_metrics_GeoL(
        affordance_pred, 
        affordance_value,
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
        ):
    obj_mesh_path = kwargs.get("obj_mesh_path", None)
    scene_mesh_path = kwargs.get("scene_mesh_path", None)
    #scene_id = scene_mesh_path.split("/")[-2]
    scene_id = mask_with_obj_path.split("/")[-3]
    scene_json_path = f'/home/stud/zhoy/MasterThesis_zhoy/dataset/scene_gen/scene_mesh_json_aug/{scene_id}.json'
    scene_json = json.load(open(scene_json_path, "r"))
    obj_scale = scene_json[obj_mesh_path.replace("/home/stud/zhoy/MasterThesis_zhoy/", "")][1] # for dense scene
    with open("/home/stud/zhoy/MasterThesis_zhoy/GeoL_net/dataset_gen/obj_size.json", 'r') as f:
        obj_target_size_json = json.load(f)
    obj_category = obj_mesh_path.split("/")[-2].rsplit("_", 2)[0]
    target_size = obj_target_size_json[obj_category]

    obj_rotation_degree = scene_json[obj_mesh_path.replace("/home/stud/zhoy/MasterThesis_zhoy/", "")][2]
    obj_rotation = np.radians(obj_rotation_degree)


    if intrinsics is None:
        intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                    [  0.     , 607.05212/2, 367.35952/2],
                    [  0.     ,   0.     ,   1.     ]])
    
    # get the points_cloud of without object
    depth_with_obj = np.array(cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED))
    depth_without_obj = np.array(cv2.imread(depth_without_obj_path, cv2.IMREAD_UNCHANGED))
    pts_with_obj, idx_with_obj = backproject(depth_with_obj, intrinsics, np.logical_and(depth_with_obj > 0, depth_with_obj < 2000))
    pts_without_obj, idx_without_obj = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj < 2000))
    color_with_obj = cv2.imread(mask_with_obj_path, cv2.COLOR_BGR2RGB)[idx_with_obj[0], idx_with_obj[1]]
    color_without_obj = cv2.imread(mask_without_obj_path, cv2.COLOR_BGR2RGB)[idx_without_obj[0], idx_without_obj[1]]
    pts_with_obj = visualize_points(pts_with_obj, color_with_obj/255)
    pts_without_obj = visualize_points(pts_without_obj, color_without_obj/255)

    # get the obj mesh and scene mesh(completed scene)
    obj_mesh = trimesh.load_mesh(obj_mesh_path)
    current_size = obj_mesh.bounds[1] - obj_mesh.bounds[0]
    obj_scale = [target_size[0]/current_size[0], target_size[1]/current_size[1], target_size[2]/current_size[2]]
    scene_mesh = trimesh.load_mesh(scene_mesh_path)
    obj_mesh.apply_scale(obj_scale)
    obj_points_sampled = obj_mesh.sample(2000)
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points_sampled * 1000) # scale obj
    scene_points_sampled = scene_mesh.sample(10000)
    scene_comp_pcd = o3d.geometry.PointCloud()
    scene_comp_pcd.points = o3d.utility.Vector3dVector(scene_points_sampled * 1000) 
    
    # algin the obj and scene mesh points
    scene_rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    scene_comp_pcd.rotate(scene_rotation_matrix, center=(0,0,0))
    scene_comp_pcd.translate([0, -1000, 1000])
    obj_rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ np.array([[np.cos(obj_rotation), -np.sin(obj_rotation), 0], [np.sin(obj_rotation), np.cos(obj_rotation), 0], [0, 0, 1]])
    obj_pcd.rotate(obj_rotation_matrix, center=(0,0,0)) # rotate obj mesh
    obj_pcd.translate([0, -1000, 1000])
    # o3d.visualization.draw_geometries([scene_comp_pcd])
    # o3d.visualization.draw_geometries([obj_pcd])
    # o3d.visualization.draw_geometries([obj_pcd, scene_comp_pcd])


    # rotate the pts to algin with xy plane
    T_plane_with_obj, plane_model_with_obj = get_tf_for_scene_rotation(np.asarray(pts_with_obj.points), axis="z")
    T_plane_without_obj, plane_model_without_obj = get_tf_for_scene_rotation(np.asarray(pts_without_obj.points), axis="z")
    pts_with_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_with_obj.points) @ T_plane_with_obj[:3, :3])
    pts_without_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_without_obj.points) @ T_plane_without_obj[:3, :3])
    
    # get the ground truth of the receptacle
    # get the mask of the table 

    #o3d.visualization.draw_geometries([pts_with_obj, pts_without_obj, obj_pcd, scene_comp_pcd])

    # get the points with high affordance value
    sampled_points = kwargs.get("pred_points", None)

    if sampled_points is None:
        points = np.asarray(pts_without_obj.points)
        #affordance_pred.points = o3d.utility.Vector3dVector(np.asarray(affordance_pred.points) @ T_plane_without_obj[:3, :3])
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])

        topk_afffordance = 5
        affordance_value = np.asarray(affordance_value.squeeze())
        topk_idx = np.argsort(affordance_value)[-topk_afffordance:]
        affordance_points = np.asarray(affordance_pred.points)
        topk_points = affordance_points[topk_idx]
        sampled_points = topk_points


    # check collision
    non_collision = []
    sampled_sphere = [pts_without_obj, scene_comp_pcd]
    for point in sampled_points:
        # move the obj to the point
        obj_max_bound = obj_pcd.get_max_bound()
        obj_min_bound = obj_pcd.get_min_bound()
        obj_bottom_center = (obj_max_bound + obj_min_bound) / 2
        obj_bottom_center[2] = obj_max_bound[2]  # attention: the z axis is reversed
        
        obj_pcd.translate(point - obj_bottom_center)
        obj_pcd.paint_uniform_color([1, 1, 0])
        sphere = o3d.geometry.TriangleMesh.create_sphere()
        sphere.compute_vertex_normals()
        sphere.scale(50, [0,0,0])

        sphere.translate(point)
        sampled_sphere.append(sphere)
        noncollision = isNonCollision(obj_pcd, scene_comp_pcd, threshold=10)
        non_collision.append(noncollision)
        #print(noncollision)
        #o3d.visualization.draw_geometries([obj_pcd, scene_comp_pcd])  
    #o3d.visualization.draw_geometries([obj_pcd, scene_comp_pcd])  
    #o3d.visualization.draw_geometries(sampled_sphere)
    collision_rate = sum(non_collision) / len(non_collision)
    #print(collision_rate)
    return non_collision

def process_collision_metrics_GeoL_completed(
        affordance_pred, 
        affordance_value,
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
        ):
    obj_mesh_path = kwargs.get("obj_mesh_path", None)
    scene_mesh_path = kwargs.get("scene_mesh_path", None)
    #scene_id = scene_mesh_path.split("/")[-2]
    scene_id = mask_with_obj_path.split("/")[-3]
    scene_json_path = f'/home/stud/zhoy/MasterThesis_zhoy/dataset/scene_gen/scene_mesh_json_aug/{scene_id}.json'
    scene_json = json.load(open(scene_json_path, "r"))
    #obj_scale = scene_json[obj_mesh_path.replace("/home/stud/zhoy/MasterThesis_zhoy/", "")][1]
    with open("/home/stud/zhoy/MasterThesis_zhoy/GeoL_net/dataset_gen/obj_size.json", 'r') as f:
        obj_target_size_json = json.load(f)
    obj_category = obj_mesh_path.split("/")[-2].rsplit("_", 2)[0]
    target_size = obj_target_size_json[obj_category]
    obj_rotation_degree = scene_json[obj_mesh_path.replace("/home/stud/zhoy/MasterThesis_zhoy/", "")][2]
    obj_rotation = np.radians(obj_rotation_degree)


    if intrinsics is None:
        intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
                    [  0.     , 607.05212/2, 367.35952/2],
                    [  0.     ,   0.     ,   1.     ]])
    
    # get the points_cloud of without object
    depth_with_obj = np.array(cv2.imread(depth_with_obj_path, cv2.IMREAD_UNCHANGED)) / 1000
    depth_without_obj = np.array(cv2.imread(depth_without_obj_path, cv2.IMREAD_UNCHANGED)) / 1000
    pts_with_obj, idx_with_obj = backproject(depth_with_obj, intrinsics, np.logical_and(depth_with_obj > 0, depth_with_obj < 2))
    pts_without_obj, idx_without_obj = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj < 2000))
    color_with_obj = cv2.imread(mask_with_obj_path, cv2.COLOR_BGR2RGB)[idx_with_obj[0], idx_with_obj[1]]
    color_without_obj = cv2.imread(mask_without_obj_path, cv2.COLOR_BGR2RGB)[idx_without_obj[0], idx_without_obj[1]]
    pts_with_obj = visualize_points(pts_with_obj, color_with_obj/255)
    pts_without_obj = visualize_points(pts_without_obj, color_without_obj/255)

    # get the obj mesh and scene mesh(completed scene)
    obj_mesh = trimesh.load_mesh(obj_mesh_path)
    current_size = obj_mesh.bounds[1] - obj_mesh.bounds[0]
    obj_scale = [target_size[0]/current_size[0], target_size[1]/current_size[1], target_size[2]/current_size[2]]
    scene_mesh = trimesh.load_mesh(scene_mesh_path)
    obj_mesh.apply_scale(obj_scale)
    obj_points_sampled = obj_mesh.sample(2000)
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points_sampled) # scale obj
    scene_points_sampled = scene_mesh.sample(10000)
    scene_comp_pcd = o3d.geometry.PointCloud()
    scene_comp_pcd.points = o3d.utility.Vector3dVector(scene_points_sampled) 
    
    # algin the obj and scene mesh points
    scene_rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    scene_comp_pcd.rotate(scene_rotation_matrix, center=(0,0,0))
    scene_comp_pcd.translate([0, -1, 1])
    obj_rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ np.array([[np.cos(obj_rotation), -np.sin(obj_rotation), 0], [np.sin(obj_rotation), np.cos(obj_rotation), 0], [0, 0, 1]])
    obj_pcd.rotate(obj_rotation_matrix, center=(0,0,0)) # rotate obj mesh
    obj_pcd.translate([0, -1, 1])
    #o3d.visualization.draw_geometries([scene_comp_pcd])
    #o3d.visualization.draw_geometries([obj_pcd])
    #o3d.visualization.draw_geometries([obj_pcd, scene_comp_pcd])


    # rotate the pts to algin with xy plane
    T_plane_with_obj, plane_model_with_obj = get_tf_for_scene_rotation(np.asarray(pts_with_obj.points), axis="z")
    T_plane_without_obj, plane_model_without_obj = get_tf_for_scene_rotation(np.asarray(pts_without_obj.points), axis="z")
    pts_with_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_with_obj.points) @ T_plane_with_obj[:3, :3])
    pts_without_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_without_obj.points) @ T_plane_without_obj[:3, :3])
    
    # get the ground truth of the receptacle
    # get the mask of the table 

    #o3d.visualization.draw_geometries([pts_with_obj, pts_without_obj, obj_pcd, scene_comp_pcd])

    # get the points with high affordance value
    sampled_points = kwargs.get("pred_points", None)

    if sampled_points is None:
        points = np.asarray(pts_without_obj.points)
        affordance_pred.points = o3d.utility.Vector3dVector(np.asarray(affordance_pred.points) @ T_plane_without_obj[:3, :3] * 1000)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        topk_afffordance = 10
        affordance_value = np.asarray(affordance_value.squeeze())
        topk_idx = np.argsort(affordance_value)[-topk_afffordance:]
        affordance_points = np.asarray(affordance_pred.points)
        topk_points = affordance_points[topk_idx]
        sampled_points = topk_points

    sampled_points = sampled_points @ T_plane_without_obj[:3, :3]
    # check collision
    non_collision = []
    sampled_sphere = [pts_without_obj, scene_comp_pcd]
    for point in sampled_points:
        # move the obj to the point
        obj_max_bound = obj_pcd.get_max_bound()
        obj_min_bound = obj_pcd.get_min_bound()
        obj_bottom_center = (obj_max_bound + obj_min_bound) / 2
        obj_bottom_center[2] = obj_max_bound[2]  # attention: the z axis is reversed
        
        obj_pcd.translate(point - obj_bottom_center)
        obj_pcd.paint_uniform_color([1, 1, 0])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
        sphere.translate(point)
        sampled_sphere.append(sphere)
        noncollision = isNonCollision(obj_pcd, scene_comp_pcd, threshold=0.01)
        if noncollision == False:
            o3d.visualization.draw_geometries([obj_pcd, scene_comp_pcd])
        #print(noncollision)
        non_collision.append(noncollision)
        #o3d.visualization.draw_geometries([obj_pcd, scene_comp_pcd]) 
        #o3d.visualization.draw_geometries([obj_pcd, scene_comp_pcd, pts_with_obj])  
    #o3d.visualization.draw_geometries(sampled_sphere)
    collision_rate = sum(non_collision) / len(non_collision)
    #print(collision_rate)
    return non_collision

def process_success_metrics(
        image_sampled_point: np.ndarray, 
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
):
    is_direction = process_direction_metrics(
        image_sampled_point,
        mask_with_obj_path,
        mask_without_obj_path,
        depth_with_obj_path,
        depth_without_obj_path,
        hdf5_path,
        direction,
        intrinsics,
        **kwargs
    )

    is_in_bbox = process_receptacle_metrics(
        image_sampled_point,
        mask_with_obj_path,
        mask_without_obj_path,
        depth_with_obj_path,
        depth_without_obj_path,
        hdf5_path,
        direction,
        intrinsics,
        **kwargs
    )

    non_collision = process_collision_metrics(
        image_sampled_point,
        mask_with_obj_path,
        mask_without_obj_path,
        depth_with_obj_path,
        depth_without_obj_path,
        hdf5_path,
        direction,
        intrinsics,
        **kwargs
    )

    # calculate the success rate
    is_success = [all(x) for x in zip(is_direction, is_in_bbox, non_collision)]
    return is_success, is_direction, is_in_bbox, non_collision


def rw_process_success_metrics(
        image_sampled_point: np.ndarray, 
        depth_path: str,
        mask_file_path: str,
        pcd_scene_only_obj,
        obj_pc,
):
    
    is_in_mask = rw_process_mask_metrics(
        image_sampled_point,
        depth_path,
        mask_file_path,
    )
    is_non_collision = rw_process_collision_metrics(
        image_sampled_point,
        depth_path,
        obj_pc,
        pcd_scene_only_obj
    )

    is_success = [all(x) for x in zip(is_in_mask, is_non_collision)]
    return is_success, is_in_mask, is_non_collision

def rw_process_collision_metrics(
        image_sampled_point: np.ndarray,
        depth_path: str,
        obj_pc,
        pcd_scene_only_obj,
        intrinsics=None,
    ):
    non_collision = []
    if intrinsics is None:
        intrinsics = np.array([[911.09 ,   0.     , 657.44  ],
                    [  0.     , 910.68, 346.58],
                    [  0.     ,   0.     ,   1.     ]])
    depth = np.array(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED))/1000
    pts, idx = backproject(depth, intrinsics, np.logical_and(depth > 0, depth<2))
    color_sampled_point = image_sampled_point[idx[0], idx[1]]
    pts_sampled_point = visualize_points(pts, color_sampled_point/255)

    colors = np.asarray(pts_sampled_point.colors)
    points = np.asarray(pts_sampled_point.points)

    is_red = (colors[:, 0] >= 0.9) & (colors[:, 1] <= 0.1) & (colors[:, 2] <= 0.1)
    sampled_points = points[is_red]

    obj_pcd = o3d.geometry.PointCloud() 
    obj_pcd.points = o3d.utility.Vector3dVector(obj_pc)
    
    vis = [pcd_scene_only_obj]
    non_collision = []
    for pred_point in sampled_points:
        pred_point = np.asarray(pred_point)
        # get the center of the obj_pc
        obj_max_bound = obj_pcd.get_max_bound()
        obj_min_bound = obj_pcd.get_min_bound()
        obj_bottom_center = (obj_max_bound + obj_min_bound) / 2

        # move the obj to the point
        obj_bottom_center[2] = obj_max_bound[2]  # attention: the z axis is reversed
        # deep copy
        obj_case = copy.deepcopy(obj_pcd)
        obj_case.translate(pred_point - obj_bottom_center)
        noncollision = isNonCollision(obj_case, pcd_scene_only_obj, threshold=0.01)
        vis.append(obj_case)
        non_collision.append(noncollision)
    
    #o3d.visualization.draw_geometries(vis, window_name="collision pred")

    return non_collision


    

def rw_process_mask_metrics(
        image_sampled_point: np.ndarray,
        depth_path,
        mask_with_obj_path: str,

        intrinsics=None,
    ):

    if intrinsics is None:
        intrinsics = np.array([[911.09 ,   0.     , 657.44  ],
                    [  0.     , 910.68, 346.58],
                    [  0.     ,   0.     ,   1.     ]])
    mask_pcd = o3d.io.read_point_cloud(mask_with_obj_path)

    depth = np.array(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED))/1000
    pts, idx = backproject(depth, intrinsics, np.logical_and(depth > 0, depth<2))
    color_sampled_point = image_sampled_point[idx[0], idx[1]]
    pts_sampled_point = visualize_points(pts, color_sampled_point/255)

    colors = np.asarray(pts_sampled_point.colors)
    points = np.asarray(pts_sampled_point.points)
    mask_points = np.asarray(mask_pcd.points)
    mask_colors = np.asarray(mask_pcd.colors)

    is_red = (colors[:, 0] >= 0.9) & (colors[:, 1] <= 0.1) & (colors[:, 2] <= 0.1)
    sampled_points = points[is_red]
    sampled_mask_points = mask_points[is_red]
    sampled_mask_colors = mask_colors[is_red]

    # create sphre on sampled points
    sampled_sphere = [mask_pcd]
    # for sampled_point in sampled_points:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    #     sphere.compute_vertex_normals()
    #     sphere.scale(1, [0,0,0])
    #     sphere.translate(sampled_point)
    #     sampled_sphere.append(sphere)
    # o3d.visualization.draw_geometries(sampled_sphere, window_name="mask and pred")

    is_mask = []
    for sampled_mask_color in sampled_mask_colors:
        if np.all(sampled_mask_color == [1, 0, 0]):
            is_mask.append(True)
        else:
            is_mask.append(False)
    return is_mask

def process_success_metrics_GeoL_completed(
        affordance_pred, 
        affordance_value,
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
):
    print('direction:', direction)
    is_direction = process_direction_metrics_GeoL_completed(
        affordance_pred,
        affordance_value,
        mask_with_obj_path,
        mask_without_obj_path,
        depth_with_obj_path,
        depth_without_obj_path,
        hdf5_path,
        direction,
        intrinsics,
        **kwargs
    )


    non_collision = process_collision_metrics_GeoL_completed(
        affordance_pred,
        affordance_value,
        mask_with_obj_path,
        mask_without_obj_path,
        depth_with_obj_path,
        depth_without_obj_path,
        hdf5_path,
        direction,
        intrinsics,
        **kwargs
    )
    
    # calculate the success rate
    is_success = [all(x) for x in zip(is_direction, non_collision)]
    return is_success, is_direction, non_collision


def rw_process_success_metrics_GeoL_completed(
        pred_points,
        pred_rotations,
        obj_pc,
        mask_file_path,
        pcd_only_obj,
        pcd_scene,
        **kwargs
):

    is_in_mask_result = []
    is_non_collision_result = []

    is_in_mask = rw_process_mask_metrics_GeoL_completed(
        pred_points,
        mask_file_path,
        pcd_scene
    )
    is_in_mask_result = is_in_mask_result + is_in_mask

    in_non_collision = rw_process_collision_metrics_GeoL_completed(
        pred_points,
        pred_rotations,
        obj_pc,
        pcd_only_obj,
        pcd_scene
    )
    is_non_collision_result = is_non_collision_result + in_non_collision

    print(is_in_mask)
    # calculate the success rate
    is_success = [all(x) for x in zip(is_in_mask_result, is_non_collision_result)]
    print("is_success:", is_success)
    print("is_in_mask_result:", is_in_mask_result)
    print("is_non_collision_result:", is_non_collision_result)
    return is_success, is_in_mask_result, is_non_collision_result

def rw_process_mask_metrics_GeoL_completed(
        pred_points,
        mask_file_path,
        pcd_scene
):
    mask_pcd = o3d.io.read_point_cloud(mask_file_path)
    pred_points = np.asarray(pred_points)
    # visualize for debugging
    sampled_sphere = [mask_pcd]
    in_mask = []
    for query_point in pred_points:
        kdtree = o3d.geometry.KDTreeFlann(mask_pcd)
        _, index, _ = kdtree.search_knn_vector_3d(query_point, 1)
        nearest_color = np.asarray(mask_pcd.colors)[index[0]]
        is_red = nearest_color[0] >= 0.9 and nearest_color[1] <= 0.1 and nearest_color[2] <= 0.1
    
        in_mask.append(is_red)

    #     sphere = o3d.geometry.TriangleMesh.create_sphere()
    #     sphere.compute_vertex_normals()
    #     sphere.scale(0.01, [0,0,0])
    #     sphere.translate(query_point)
    #     sampled_sphere.append(sphere)
    # o3d.visualization.draw_geometries(sampled_sphere, window_name="mask and pred")

    return in_mask

def rw_process_collision_metrics_GeoL_completed(
        pred_points,
        pred_rotations,
        obj_pc,
        pcd_only_obj,
        pcd_scene,
        target_size=[0.1, 0.1, 0.1],
    ):
    non_collision = []
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(np.asarray(obj_pc.cpu().numpy()))
    vis = [pcd_only_obj]
    for pred_point in pred_points:
        pred_point = np.asarray(pred_point)
        # get the center of the obj_pc
        obj_max_bound = obj_pcd.get_max_bound()
        obj_min_bound = obj_pcd.get_min_bound()
        obj_bottom_center = (obj_max_bound + obj_min_bound) / 2

        # move the obj to the point
        obj_bottom_center[2] = obj_max_bound[2]  # attention: the z axis is reversed
        # deep copy
        obj_case = copy.deepcopy(obj_pcd)
        obj_case.translate(pred_point - obj_bottom_center)
        noncollision = isNonCollision(obj_case, pcd_only_obj, threshold=0.01)
        vis.append(obj_case)
        non_collision.append(noncollision)
    
    #o3d.visualization.draw_geometries(vis, window_name="collision pred")

    return non_collision


    

def process_success_metrics_GeoL(
        affordance_pred, 
        affordance_value,
        mask_with_obj_path: str,
        mask_without_obj_path: str,
        depth_with_obj_path: str,
        depth_without_obj_path: str,
        hdf5_path:str,
        direction:str,
        intrinsics=None,
        **kwargs
):
    is_direction = process_direction_metrics_GeoL(
        affordance_pred,
        affordance_value,
        mask_with_obj_path,
        mask_without_obj_path,
        depth_with_obj_path,
        depth_without_obj_path,
        hdf5_path,
        direction,
        intrinsics,
        **kwargs
    )

    is_in_bbox = process_receptacle_metrics_GeoL(
        affordance_pred,
        affordance_value,
        mask_with_obj_path,
        mask_without_obj_path,
        depth_with_obj_path,
        depth_without_obj_path,
        hdf5_path,
        direction,
        intrinsics,
        **kwargs
    )

    non_collision = process_collision_metrics_GeoL(
        affordance_pred,
        affordance_value,
        mask_with_obj_path,
        mask_without_obj_path,
        depth_with_obj_path,
        depth_without_obj_path,
        hdf5_path,
        direction,
        intrinsics,
        **kwargs
    )
    
    # calculate the success rate
    print("direction:", is_direction)
    print("in_bbox:", is_in_bbox)
    print("non_collision:", non_collision)
    is_success = [all(x) for x in zip(is_direction, is_in_bbox, non_collision)]
    return is_success, is_direction, is_in_bbox, non_collision
    

if __name__ == "__main__":
    mask_with_obj_path = "dataset/scene_RGBD_mask_v2_kinect_cfg/id1_1/bowl_0002_glass/mask_with_obj.png"
    depth_with_obj_path = "dataset/scene_RGBD_mask_v2_kinect_cfg/id1_1/bowl_0002_glass/with_obj/test_pbr/000000/depth/000000.png"
    mask_without_obj_path = "dataset/scene_RGBD_mask_v2_kinect_cfg/id1_1/bowl_0002_glass/mask_no_obj.png"
    depth_without_obj_path = "dataset/scene_RGBD_mask_v2_kinect_cfg/id1_1/bowl_0002_glass/no_obj/test_pbr/000000/depth/000000.png"

    process_receptacle_metrics(
        sampled_points=None,
        mask_with_obj_path=mask_with_obj_path,
        mask_without_obj_path=mask_without_obj_path,
        depth_with_obj_path=depth_with_obj_path,
        depth_without_obj_path=depth_without_obj_path,
    )
    
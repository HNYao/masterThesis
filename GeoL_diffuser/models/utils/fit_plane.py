import numpy as np
import open3d as o3d






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
    T_plane = np.eye(4)
    if axis == "y":
        T_plane[:3, 1] = -plane_dir
        T_plane[:3, 2] = -np.cross([1, 0, 0], plane_dir)
        T_plane[:3, 0] = np.cross(T_plane[:3, 1], T_plane[:3, 2])
    elif axis == "x":
        T_plane[:3, 0] = -plane_dir  # Set the X-axis to align with the plane normal
        T_plane[:3, 1] = -np.cross([1, 0, 0], plane_dir)  
        T_plane[:3, 2] = np.cross(T_plane[:3, 0], T_plane[:3, 1])
    elif axis == "z":
        T_plane[:3, 2] = -plane_dir  # Set the X-axis to align with the plane normal
        T_plane[:3, 0] = -np.cross([1, 0, 0], plane_dir)  
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

if __name__ == "__main__":
    scene_pcd_path = "dataset/scene_RGBD_mask_v2_kinect_cfg/id2/bottle_0003_cola/mask_Front.ply"
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)
    scene_pcd_points = np.asarray(scene_pcd.points)

    cam_rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0.8, -0.6],
        [0, 0.6, 0.8]
    ])

    camera_scene_pcd_points = scene_pcd_points @ cam_rotation_matrix.T
    camera_scene_pcd = o3d.geometry.PointCloud()
    camera_scene_pcd.points = o3d.utility.Vector3dVector(camera_scene_pcd_points)

    T_plane, plane_model = get_tf_for_scene_rotation(camera_scene_pcd_points, axis="z")
    recovered_scene_points =   camera_scene_pcd_points @ T_plane[:3, :3] + T_plane[:3, 3]
    recovered_scene_pcd = o3d.geometry.PointCloud()
    recovered_scene_pcd.points = o3d.utility.Vector3dVector(recovered_scene_points)
    # black color
    recovered_scene_pcd.colors =  o3d.utility.Vector3dVector(np.ones((recovered_scene_points.shape[0], 3)) * [0, 0, 0])

    # back to original
    back_to_original = recovered_scene_points @ np.linalg.inv(T_plane[:3, :3]) - T_plane[:3, 3]
    back_to_original_pcd = o3d.geometry.PointCloud()
    back_to_original_pcd.points = o3d.utility.Vector3dVector(back_to_original)
    back_to_original_pcd.colors =  o3d.utility.Vector3dVector(np.zeros((back_to_original.shape[0], 3)) * [0, 0, 0])




    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([ camera_scene_pcd, recovered_scene_pcd, coordinate_frame])

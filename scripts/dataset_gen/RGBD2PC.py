"Backproject RGBD to Point Cloud"
import numpy as np
import open3d as o3d
import cv2

def backproject(depth, intrinsics, instance_mask, NOCS_convention=False):
    """
        depth: np.array, [H,W]
        intrinsics: np.array, [3, 3]
        instance_mask: np.array, [H, W]; (np.logical_and(depth>0, depth<2))
    """
    intrinsics_inv = np.linalg.inv(intrinsics)
    # image_shape = depth.shape
    # width = image_shape[1]
    # height = image_shape[0]

    # x = np.arange(width)
    # y = np.arange(height)

    # non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = depth > 0
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
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


if __name__ == "__main__":
    depth_path = "dataset/scene_RGBD_mask/no_obj/test_pbr/000000/depth/000000.png"
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = np.array(depth_image)

    color_path = "dataset/scene_RGBD_mask/no_obj/mask.png"
    color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
    color = np.array(color_image) / 255.0
    
    intr = np.array([[591.0125 ,   0.     , 322.525  ],
                     [  0.     , 590.16775, 244.11084],
                     [  0.     ,   0.     ,   1.     ]])

    points_scene, scene_idx = backproject(
                depth,
                intr,
                np.logical_and(depth > 0, depth > 0),
                NOCS_convention=False,
            )
    
    cam_rotation_matrix = np.array([
        [1, 0, 0],
        [0,0.8,-0.6],
        [0,0.6,0.8]
    ])
 
    points_scene = (cam_rotation_matrix @ points_scene.T).T

    centroid = np.mean(points_scene, axis=0)
    print("centroid:", centroid)
    points_scene = points_scene - centroid

    colors_scene = color[scene_idx[0], scene_idx[1]]
    #colors_scene = color_binary[scene_idx[0], scene_idx[1]]

    #print("points_scene:",points_scene)
    pcd_scene = visualize_points(points_scene, colors_scene)
    o3d.visualization.draw_geometries([pcd_scene])
    #o3d.io.write_point_cloud("dataset/scene_RGBD_mask/mask.ply", pcd_scene)

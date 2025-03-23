import open3d as o3d
import numpy as np

def add_obj_into_pcd(obj_path:str, scene_pcd, target_point:list, rotation_matrix, scaling_factors=[1.0, 1.0, 1.0], is_mesh=True):
    """
    Add obj into pcd scene

    Args:
    obj_path: str
        Path to the obj file
    scene_pcd: open3d.geometry.PointCloud
        Point cloud object
    rotation_matrix: numpy.ndarray
        Rotation matrix
    scalling_matrix: list
        Scalling matrix
    is_mesh: bool
        If the object is a mesh or not
    
    Returns:
        None
    """
    if is_mesh:
        mesh = o3d.io.read_triangle_mesh(obj_path)  
        mesh.compute_vertex_normals()

        if not mesh.has_vertex_colors() and not mesh.has_vertex_normals():
            print("Mesh contains no vertex colors or normals. Using default values.")
            mesh.paint_uniform_color([1.0, 0.0, 0.0])  # default red color

        obj_pcd = mesh.sample_points_uniformly(number_of_points=2048) # or poisson_disk_sampling

        if mesh.has_vertex_colors():
            print("Mesh has vertex colors. Using them.")

        o3d.visualization.draw_geometries([obj_pcd])

    else:
        obj_pcd = o3d.io.read_point_cloud(obj_path)
    
    aabb = obj_pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)

    corner_min = np.array(aabb.get_min_bound())
    corner_max = np.array(aabb.get_max_bound())
    bottom_center_aabb = np.array([
        (corner_min[0] + corner_max[0]) / 2,  
        (corner_min[1] + corner_max[1]) / 2,  
        corner_min[2]  
    ])

    obb = obj_pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)

    obb_points = np.asarray(obb.get_box_points())
    obb_bottom_face = obb_points[np.argsort(obb_points[:, 2])[:4]]  # z 最小的 4 个点
    bottom_center_obb = np.mean(obb_bottom_face, axis=0)
    
    obj_pcd.rotate(rotation_matrix)



    # Create a scaling transformation matrix
    scaling_matrix = np.eye(4)
    scaling_matrix[0, 0] = scaling_factors[0]  # Scale x
    scaling_matrix[1, 1] = scaling_factors[1]  # Scale y
    scaling_matrix[2, 2] = scaling_factors[2]
    scaling_matrix[3,3]

    obj_pcd.transform(scaling_matrix)

    obj_pcd.translate([0,0,0])

    scene_pcd += obj_pcd
    o3d.visualization.draw_geometries([obj_pcd, aabb, obb])

    return None

if __name__ == "__main__":
    scene_pcd = o3d.io.read_point_cloud("dataset/scene_RGBD_mask_v2_kinect_cfg/id1_1/bowl_0002_glass/mask_Front.ply")
    rotation_matrix = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]], dtype=np.float64)
    scaling_factors = [1.5, 2.0, 0.5]

    add_obj_into_pcd(obj_path="dataset/obj/mesh/chessboard/chessboard_0002_grey/mesh.obj", scene_pcd=scene_pcd, target_point=[0,0,0], rotation_matrix=rotation_matrix, scaling_factors=scaling_factors,    is_mesh=True)
    
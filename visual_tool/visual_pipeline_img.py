import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def color_obj_mesh(object_mesh_path, color=[1, 0, 0]):
    """
    Color the object mesh with the given color.
    
    Args:
        object_mesh_path (str): Path to the object mesh file.
        color (list): RGB color values for the object mesh.
        
    Returns:
        o3d.geometry.TriangleMesh: Colored object mesh.
    """

    # sample points from the mesh
    # mesh = o3d.io.read_triangle_mesh(object_mesh_path)
    # mesh.compute_vertex_normals()
    #pcd = mesh.sample_points_uniformly(number_of_points=10000)
    pcd = o3d.io.read_point_cloud(".tmp/pipepine_depth.ply")
    
    points = np.asarray(pcd.points)
    #values = np.sum(points, axis=1)
    values = points[:,2]
    # 

    # x_min = points[:, 0].min()
    # x_max = points[:, 0].max()
    # y_min = points[:, 1].min()
    # y_max = points[:, 1].max()
    # z_min = points[:, 2].min()
    # z_max = points[:, 2].max()

    # # Normalize the values to the range [0, 1]
    # x_norm = (points[:, 0] - x_min) / (x_max - x_min)
    # y_norm = (points[:, 1] - y_min) / (y_max - y_min)
    # z_norm = (points[:, 2] - z_min) / (z_max - z_min)

    # xyz_min = points.min(axis=0)
    # xyz_max = points.max(axis=0)
    # xyz_range = xyz_max - xyz_min + 1e-8
    # normalized_points = (points - xyz_min) / xyz_range

    # pcd.colors = o3d.utility.Vector3dVector(normalized_points)


    normalize_values = (values - np.min(values)) / (np.max(values) - np.min(values))
    colormap = plt.cm.get_cmap('viridis')
    colors = colormap(normalize_values)[:, :3]  # Get RGB values
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(".tmp/pipepine_depth.ply", pcd)
if __name__ == "__main__":
    object_mesh_path = "data_and_weights/mesh_realworld/plate_realworld.obj"
    color_obj_mesh(object_mesh_path, color=[1, 0, 0])
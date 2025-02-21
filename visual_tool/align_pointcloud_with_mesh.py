import trimesh
import open3d as o3d
import numpy as np
import cv2

# read npy as dictionary
npy_file = np.load(
    "selected_scene/align_pointcloud_with_mesh.npy", allow_pickle=True
).item()


intrinsics = np.array(
    [
        [607.09912 / 2, 0.0, 636.85083 / 2],
        [0.0, 607.05212 / 2, 367.35952 / 2],
        [0.0, 0.0, 1.0],
    ]
)
# intrinsics = np.array(npy_file["intrinsics"])

# scene_mesh_path = "selected_scene/scene_mesh.obj"
obj_mesh_path = "selected_scene/mesh.obj"
obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
obj_mesh.compute_vertex_normals()

# rgb_path = "dataset/picked_scene_RGBD_mask/id15_0/keyboard_0009_black/no_obj/test_pbr/000000/rgb/000000.jpg"
# depth_path = "dataset/picked_scene_RGBD_mask/id15_0/keyboard_0009_black/no_obj/test_pbr/000000/depth/000000.png"

# depth_without_obj = np.array(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)) / 1000

# pts_without_obj, idx_without_obj = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj < 2))
# color_without_obj = cv2.imread(rgb_path, cv2.COLOR_BGR2RGB)[idx_without_obj[0], idx_without_obj[1]]
# pts_without_obj = visualize_points(pts_without_obj, color_without_obj/255)

# o3d.visualization.draw_geometries([pts_without_obj])
pts_without_obj = o3d.geometry.PointCloud()
pts_without_obj.points = o3d.utility.Vector3dVector(npy_file["scene_pcd_point"])
pts_without_obj.colors = o3d.utility.Vector3dVector(npy_file["scene_pcd_color"])

# scene_mesh = trimesh.load(scene_mesh_path)
obj_scale = np.array(npy_file["obj_scale"])
obj_mesh.scale(obj_scale[0], center=[0, 0, 0])
obj_pcd = obj_mesh.sample_points_uniformly(number_of_points=10000)


obj_rotation = 0
# obj_rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ np.array([[np.cos(obj_rotation), -np.sin(obj_rotation), 0], [np.sin(obj_rotation), np.cos(obj_rotation), 0], [0, 0, 1]])
obj_rotation_matrix = np.array(npy_file["obj_rotation_matrix"])
obj_mesh.rotate(obj_rotation_matrix, center=[0, 0, 0])  # rotate obj mesh
obj_mesh.translate([0, -1, 1])

obj_pcd.rotate(obj_rotation_matrix, center=[0, 0, 0])  # rotate obj mesh
obj_pcd.translate([0, -1, 1])
obj_pcd_target_point = [-0.2, -0.75, 1]


obj_max_bound = obj_pcd.get_max_bound()
obj_min_bound = obj_pcd.get_min_bound()
obj_bottom_center = (obj_max_bound + obj_min_bound) / 2
obj_bottom_center[2] = obj_max_bound[2]  # attention: the z axis is reversed

obj_pcd.translate(
    obj_pcd_target_point - obj_bottom_center
)  # move obj mesh to target point
obj_mesh.translate(obj_pcd_target_point - obj_bottom_center)  # move obj

# T_plane_without_obj, plane_model_without_obj = get_tf_for_scene_rotation(np.asarray(pts_without_obj.points), axis="z")
T_plane_without_obj = np.array(npy_file["T_plane_without_obj"])
# pts_without_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_without_obj.points) @ T_plane_without_obj[:3, :3])

o3d.visualization.draw([pts_without_obj, obj_pcd, obj_mesh])

# to_save = {
#     "obj_scale": obj_scale,
#     "scene_mesh_point": np.asarray(scene_comp_pcd.points),
#     "scene_pcd_point":np.asarray(pts_without_obj.points),
#     "scene_pcd_color":np.asarray(pts_without_obj.colors),
#     "scene_rotation_matrix": scene_rotation_matrix,
#     "scene_translation": [0, -1, 1],
#     "obj_rotation_matrix": obj_rotation_matrix,
#     "T_plane_without_obj": T_plane_without_obj,
#     "intrinsics": intrinsics,
#     "obj_pcd_translation": [0, -1, 1],
#     "obj_pcd_target_point": obj_pcd_target_point
# }

# np.save("selected_scene/align_pointcloud_with_mesh.npy", to_save)

import trimesh
import open3d as o3d
import numpy as np
import cv2
from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
from GeoL_diffuser.models.utils.fit_plane import fit_plane_from_points, get_tf_for_scene_rotation

intrinsics = np.array([[607.09912/2 ,   0.     , 636.85083/2  ],
            [  0.     , 607.05212/2, 367.35952/2],
            [  0.     ,   0.     ,   1.     ]])
scene_mesh_path = "dataset/picked_scene_RGBD_mask/id15_0/keyboard_0009_black/mesh.obj"
obj_mesh_path = "dataset/obj/mesh/keyboard/keyboard_0009_black/mesh.obj"
rgb_path = "dataset/picked_scene_RGBD_mask/id15_0/keyboard_0009_black/no_obj/test_pbr/000000/rgb/000000.jpg"
depth_path = "dataset/picked_scene_RGBD_mask/id15_0/keyboard_0009_black/no_obj/test_pbr/000000/depth/000000.png"

depth_without_obj = np.array(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)) / 1000

pts_without_obj, idx_without_obj = backproject(depth_without_obj, intrinsics, np.logical_and(depth_without_obj > 0, depth_without_obj < 2))
color_without_obj = cv2.imread(rgb_path, cv2.COLOR_BGR2RGB)[idx_without_obj[0], idx_without_obj[1]]
pts_without_obj = visualize_points(pts_without_obj, color_without_obj/255)

o3d.visualization.draw_geometries([pts_without_obj])

scene_mesh = trimesh.load(scene_mesh_path)
obj_mesh = trimesh.load(obj_mesh_path)
obj_scale = [0.0016315726279011605,0.0016315726279011605,0.0016315726279011605]
obj_mesh.apply_scale(obj_scale) # hardcoded scale
obj_points_sampled = obj_mesh.sample(2000)
scene_points_sampled = scene_mesh.sample(10000)
scene_comp_pcd = o3d.geometry.PointCloud()
scene_comp_pcd.points = o3d.utility.Vector3dVector(scene_points_sampled)

scene_rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
scene_comp_pcd.rotate(scene_rotation_matrix, center=(0,0,0))
scene_comp_pcd.translate([0, -1, 1])

obj_pcd = o3d.geometry.PointCloud()
obj_pcd.points = o3d.utility.Vector3dVector(obj_points_sampled)
obj_rotation = 0
obj_rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ np.array([[np.cos(obj_rotation), -np.sin(obj_rotation), 0], [np.sin(obj_rotation), np.cos(obj_rotation), 0], [0, 0, 1]])
obj_pcd.rotate(obj_rotation_matrix, center=(0,0,0)) # rotate obj mesh
obj_pcd.translate([0, -1, 1])
obj_pcd_target_point = [-0.2, -0.75, 1]


obj_max_bound = obj_pcd.get_max_bound()
obj_min_bound = obj_pcd.get_min_bound()
obj_bottom_center = (obj_max_bound + obj_min_bound) / 2
obj_bottom_center[2] = obj_max_bound[2]  # attention: the z axis is reversed

obj_pcd.translate(obj_pcd_target_point - obj_bottom_center) # move obj mesh to target point



T_plane_without_obj, plane_model_without_obj = get_tf_for_scene_rotation(np.asarray(pts_without_obj.points), axis="z")

pts_without_obj.points = o3d.utility.Vector3dVector(np.asarray(pts_without_obj.points) @ T_plane_without_obj[:3, :3])



o3d.visualization.draw_geometries([scene_comp_pcd, pts_without_obj, obj_pcd])

to_save = {
    "obj_scale": obj_scale,
    "scene_mesh_point": np.asarray(scene_comp_pcd.points),
    "scene_pcd_point":np.asarray(pts_without_obj.points),
    "scene_pcd_color":np.asarray(pts_without_obj.colors),
    "scene_rotation_matrix": scene_rotation_matrix,
    "scene_translation": [0, -1, 1],
    "obj_rotation_matrix": obj_rotation_matrix,
    "T_plane_without_obj": T_plane_without_obj,
    "intrinsics": intrinsics,
}

np.save("selected_scene/align_pointcloud_with_mesh.npy", to_save)




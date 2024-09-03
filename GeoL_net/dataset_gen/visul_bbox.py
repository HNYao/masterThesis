import trimesh
import numpy as np
from utils import get_obj_from_scene, bbox_pos_scale_all_obj, rotate_mesh_around_center, get_obj_from_scene_inslabel, move_trimeshobj_to_position, get_unique_filename

mesh = trimesh.load("dataset/obj/mesh/laptop/laptop_0002_black/mesh.obj")
mesh2 = trimesh.load("dataset/obj/mesh/book/book_0001_blue/mesh.obj")
aabb2_scaled_min = mesh.bounds[0]
aabb2_scaled_max = mesh.bounds[1]
extent2_scaled = aabb2_scaled_max - aabb2_scaled_min
bbox = mesh.bounding_box
#bbox = mesh.bounds
print(mesh.bounds)
print(mesh.centroid)
bbox.visual.face_colors = [0, 255, 0, 50]
aabb_center = mesh.centroid
aabb_bottom_center = np.array(aabb_center)
aabb_bottom_center[2] -= extent2_scaled[2] / 2.0

aabb2_scaled_min = mesh2.bounds[0]
aabb2_scaled_max = mesh2.bounds[1]
extent2_scaled = aabb2_scaled_max - aabb2_scaled_min
aabb_center2 = mesh2.centroid
aabb_bottom_center2 = np.array(aabb_center2)
aabb_bottom_center2[2] -= extent2_scaled[2] / 2.0

mesh1 = move_trimeshobj_to_position(mesh, end_position=[0,0,0], start_position=aabb_bottom_center)
mesh2 = move_trimeshobj_to_position(mesh2, end_position=[0,0,0], start_position=aabb_bottom_center2)

scene = trimesh.Scene([mesh, bbox, mesh2])
scene.show()
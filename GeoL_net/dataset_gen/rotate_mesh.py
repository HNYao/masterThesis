import numpy as np
import trimesh

def rotate_mesh_obj(file_path, axis='x', angle_degrees=90):
    mesh = trimesh.load(file_path)

    angle_radians = np.radians(angle_degrees)

    if axis == 'x':
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, [1, 0, 0])
    elif axis == 'y':
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, [0, 1, 0])
    elif axis == 'z':
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_radians, [0, 0, 1])
    else:
        raise ValueError("Invalid axis. Please choose from 'x', 'y', or 'z'.")


    mesh.apply_transform(rotation_matrix)


    mesh.export(file_path)
file_list = ["dataset/obj/mesh/clock/clock_0001_normal/mesh.obj", "dataset/obj/mesh/clock/clock_0002_wooden/mesh.obj"]
for file_path in file_list:
    rotate_mesh_obj(file_path, axis='x', angle_degrees=90)
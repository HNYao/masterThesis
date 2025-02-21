import numpy as np
import os
import json
import math
import trimesh
import open3d as o3d

def generate_full_mesh_from_json(json_path, export_path=None):
    """
    Generate the full mesh from the json file
    Contain all the objects, including the object to place
    except the table or desk
    
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    with open('GeoL_net/dataset_gen/obj_size.json', 'r') as f:
        obj_size = json.load(f)
    

    
    # load all objects at one time
    scene_mesh_list = []
    obj_amount = len(data)
    exist_obj_amount = 0
    for obj_file_name, obj_pose in data.items():
        exist_obj_amount += 1
        if exist_obj_amount == obj_amount:
            # the last object is the table or desk
            desk_mesh = trimesh.load_mesh(obj_file_name)
            # TODO: maybe need to rotate
            scale_x, scale_y, scale_z = obj_pose[1]
            aabb_min, aabb_max = desk_mesh.bounds
            z_height = aabb_max[2] - aabb_min[2]
            desk_mesh.apply_translation([0, 0, -z_height/2])

            desk_mesh.apply_scale([scale_x, scale_y, scale_z])
            #scene_mesh_list.append(desk_mesh) # NOTE: do not add the desk to the scene, so the collision will only caused by the objects
            break

        # load each object
        obj_mesh = trimesh.load_mesh(obj_file_name)
        rotation_angle = obj_pose[2]
        rotation_radius = np.radians(rotation_angle)
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_radius, [0, 0, 1])
        obj_mesh.apply_transform(rotation_matrix)

        obj_category = obj_file_name.split('/')[3]
        target_size = obj_size[obj_category]
        aabb_min, aabb_max = obj_mesh.bounds
        current_size = aabb_max - aabb_min
        #scale_x, scale_y, scale_z = [target_size[0]/current_size[0], target_size[1]/current_size[1], target_size[2]/current_size[2]]
        scale_x, scale_y, scale_z = data[obj_file_name][1]
        obj_mesh.apply_scale([scale_x, scale_y, scale_z])
        aabb_min, aabb_max = obj_mesh.bounds
        z_height = aabb_max[2] - aabb_min[2]
        obj_mesh.apply_translation([obj_pose[0][0], obj_pose[0][1], z_height/2])

        scene_mesh_list.append(obj_mesh)

    scene_mesh = trimesh.util.concatenate(scene_mesh_list)

   
    scene_mesh.export(export_path)

def generate_incompleted_mesh_from_json(file_path, json_path, export_path=None):
    """
    Generate the incompleted mesh from the json file
    Only contain the object to place, excluding the objec to be placed

    file_path: e.g. "dataset/benchmark_bproc_data/id9/eye_glasses_0002_black"
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    with open('GeoL_net/dataset_gen/obj_size.json', 'r') as f:
        obj_size = json.load(f)
    
    with open(os.path.join(file_path.rsplit("/", 1)[0], 'text_guidance.json'), 'r') as f:
        text_guidance = json.load(f)

    # get the name of the object to place from the file path
    obj_discription = file_path.split('/')[-1].split('_')[-1]
    obj_category = file_path.split('/')[-1].rsplit('_', 2)[0]
    obj_to_place_full_name = f"the {obj_discription} {obj_category}"
    obj_to_place_mesh_path = text_guidance[obj_to_place_full_name][4]

    
    # load all objects at one time
    scene_mesh_list = []
    obj_amount = len(data)
    exist_obj_amount = 0
    for obj_file_name, obj_pose in data.items():
        exist_obj_amount += 1

        if obj_file_name == obj_to_place_mesh_path:
            continue

        if exist_obj_amount == obj_amount:
            # the last object is the table or desk
            desk_mesh = trimesh.load_mesh(obj_file_name)
            # TODO: maybe need to rotate
            scale_x, scale_y, scale_z = obj_pose[1]
            aabb_min, aabb_max = desk_mesh.bounds
            z_height = aabb_max[2] - aabb_min[2]
            desk_mesh.apply_translation([0, 0, -z_height/2])

            desk_mesh.apply_scale([scale_x, scale_y, scale_z])
            #scene_mesh_list.append(desk_mesh) # NOTE: do not add the desk to the scene, so the collision will only caused by the objects
            break

        # load each object
        obj_mesh = trimesh.load_mesh(obj_file_name)
        rotation_angle = obj_pose[2]
        rotation_radius = np.radians(rotation_angle)
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_radius, [0, 0, 1])
        obj_mesh.apply_transform(rotation_matrix)

        obj_category = obj_file_name.split('/')[3]
        target_size = obj_size[obj_category]
        aabb_min, aabb_max = obj_mesh.bounds
        current_size = aabb_max - aabb_min
        #scale_x, scale_y, scale_z = [target_size[0]/current_size[0], target_size[1]/current_size[1], target_size[2]/current_size[2]]
        scale_x, scale_y, scale_z = data[obj_file_name][1]
        obj_mesh.apply_scale([scale_x, scale_y, scale_z])
        aabb_min, aabb_max = obj_mesh.bounds
        z_height = aabb_max[2] - aabb_min[2]
        obj_mesh.apply_translation([obj_pose[0][0], obj_pose[0][1], z_height/2])

        scene_mesh_list.append(obj_mesh)

    scene_mesh = trimesh.util.concatenate(scene_mesh_list)

   
    scene_mesh.export(export_path)


def generate_incompleted_sparse_mesh_from_json(file_path, json_path, export_path=None):
    """
    Generate the incompleted mesh from the json file, sparse scenes
    Only contain the object to place, excluding the objec to be placed

    object size is fron obj_size.json

    file_path: e.g. "dataset/benchmark_bproc_data/id9/eye_glasses_0002_black"
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    with open('GeoL_net/dataset_gen/obj_size.json', 'r') as f:
        obj_size = json.load(f)
    
    with open(os.path.join(file_path.rsplit("/", 1)[0], 'text_guidance.json'), 'r') as f:
        text_guidance = json.load(f)

    # get the name of the object to place from the file path
    obj_discription = file_path.split('/')[-1].split('_')[-1]
    obj_category = file_path.split('/')[-1].rsplit('_', 2)[0]
    obj_to_place_full_name = f"the {obj_discription} {obj_category}"
    obj_to_place_mesh_path = text_guidance[obj_to_place_full_name][4]

    
    # load all objects at one time
    scene_mesh_list = []
    obj_amount = len(data)
    exist_obj_amount = 0
    for obj_file_name, obj_pose in data.items():
        exist_obj_amount += 1

        if obj_file_name == obj_to_place_mesh_path:
            continue

        if exist_obj_amount == obj_amount:
            # the last object is the table or desk
            desk_mesh = trimesh.load_mesh(obj_file_name)
            # TODO: maybe need to rotate
            scale_x, scale_y, scale_z = obj_pose[1]
            aabb_min, aabb_max = desk_mesh.bounds
            z_height = aabb_max[2] - aabb_min[2]
            desk_mesh.apply_translation([0, 0, -z_height/2])

            desk_mesh.apply_scale([scale_x, scale_y, scale_z])
            scene_mesh_list.append(desk_mesh) # NOTE: do not add the desk to the scene, so the collision will only caused by the objects
            break

        # load each object
        obj_mesh = trimesh.load_mesh(obj_file_name)
        rotation_angle = obj_pose[2]
        rotation_radius = np.radians(rotation_angle)
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_radius, [0, 0, 1])
        obj_mesh.apply_transform(rotation_matrix)

        obj_category = obj_file_name.split('/')[3]
        target_size = obj_size[obj_category]
        aabb_min, aabb_max = obj_mesh.bounds
        current_size = aabb_max - aabb_min
        scale_x, scale_y, scale_z = [target_size[0]/current_size[0], target_size[1]/current_size[1], target_size[2]/current_size[2]]
        #scale_x, scale_y, scale_z = data[obj_file_name][1]
        obj_mesh.apply_scale([scale_x, scale_y, scale_z])
        aabb_min, aabb_max = obj_mesh.bounds
        z_height = aabb_max[2] - aabb_min[2]
        obj_mesh.apply_translation([obj_pose[0][0], obj_pose[0][1], z_height/2])

        scene_mesh_list.append(obj_mesh)

    scene_mesh = trimesh.util.concatenate(scene_mesh_list)

   
    scene_mesh.export(export_path)

if __name__ == "__main__":
    ##### generate the full mesh for the benchmark_bproc_data crowded scenes
    # root_dir = "dataset/benchmark_bproc_data"
    # for folder_name in os.listdir(root_dir):
    #     folder_path = os.path.join(root_dir, folder_name)
    #     if os.path.isdir(folder_path):  # 确保是文件夹
    #         print(folder_path)
    #     # get the json file path from the folder path
    #     #folder_path = "dataset/benchmark_bproc_data/id8"
    #     scene_id = folder_path.split('/')[-1]
    #     json_path = f"dataset/scene_gen/scene_mesh_json_kinect/{scene_id}.json"
    #     export_folder = f"dataset/benchmark_bproc_data_mesh/with_obj_to_place/{scene_id}"
    #     # check if export path exists, if not, create the folder
    #     if not os.path.exists(export_folder):
    #         os.makedirs(export_folder)
            
    #     export_path = os.path.join(export_folder, "mesh.obj")
    #     print(export_path)
    #     generate_full_mesh_from_json(json_path, export_path)
    # ######


    ##### generate the incompleted mesh for the benchmark_bproc_data sparse scenes
    # root_dir = "dataset/benchmark_bproc_data"
    # for folder_name in os.listdir(root_dir):
    #     folder_path = os.path.join(root_dir, folder_name)
    #     scene_id = folder_path.split('/')[-1]
    #     json_path = f"dataset/scene_gen/scene_mesh_json_kinect/{scene_id}.json"
    #     if os.path.isdir(folder_path):  # 确保是文件夹
    #         print(folder_path)

    #         for subfolder_name in os.listdir(folder_path):
    #             subfolder_path = os.path.join(folder_path, subfolder_name)
    #             if os.path.isdir(subfolder_path):  # 确保是文件夹
    #                 print(subfolder_name)
    #             else:
    #                 continue
     
    #             export_path = os.path.join(subfolder_path, "mesh.obj")

    #             print(export_path)
    #             generate_incompleted_mesh_from_json(
    #                 file_path=subfolder_path,
    #                 json_path=json_path,
    #                 export_path=export_path)
    ######

    ###### generate the incompleted mesh for the benchmark_bproc_data sparse scenes
    root_dir = "dataset/picked_scene_RGBD_mask"
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        scene_id = folder_path.split('/')[-1]
        json_path = f"dataset/scene_gen/picked_scene_mesh_json/{scene_id}.json"
        if os.path.isdir(folder_path):  # 确保是文件夹
            print(folder_path)

            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path):  # 确保是文件夹
                    print(subfolder_name)
                else:
                    continue
     
                export_path = os.path.join(subfolder_path, "mesh.obj")

                print(export_path)
                generate_incompleted_sparse_mesh_from_json(
                    file_path=subfolder_path,
                    json_path=json_path,
                    export_path=export_path)



    
    ###### debug: check if the mesh and point cloud are aligned
    # scene_mesh = trimesh.load_mesh("dataset/benchmark_bproc_data_mesh/with_obj_to_place/id8/mesh.obj")
    # num_points = 50000

    # mesh_points = scene_mesh.sample(num_points)
    # pc_mesh = o3d.geometry.PointCloud()
    # mesh_points = np.asarray(mesh_points).astype(np.float32)
    # pc_mesh.points = o3d.utility.Vector3dVector(mesh_points * 1000)

    # # align the mesh with the point cloud
    # rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # pc_mesh.rotate(rotation_matrix, center=(0, 0, 0)) # rotate
    # pc_mesh.translate([0, -1000, 1000]) # translate

    # aabb_mesh = pc_mesh.get_axis_aligned_bounding_box()
    # aabb_mesh.color = (1, 0, 0)
    # pc = o3d.io.read_point_cloud("dataset/benchmark_bproc_data/id8/book_0001_black/mask_Left Front.ply")
    # aabb_pc = pc.get_axis_aligned_bounding_box()
    # aabb_pc.color = (0, 1, 0)
    # o3d.visualization.draw_geometries([pc, aabb_pc])
    # o3d.visualization.draw_geometries([pc_mesh, aabb_pc])
    # o3d.visualization.draw_geometries([pc, pc_mesh])
    ######
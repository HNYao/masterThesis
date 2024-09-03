"""
generate a json file, including:
    (obj_mesh_file_path): position, scale, bbox
"""
import time
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
import trimesh
import numpy as np
import os
import random
import json
from GeoL_net.dataset_gen.solver_scene_generate import *
from utils import get_obj_from_scene, bbox_pos_scale_all_obj, rotate_mesh_around_center, get_obj_from_scene_inslabel, move_trimeshobj_to_position, get_unique_filename


def generate_mesh_scene_all_texute_v2(ply_path, npz_path, directory_mesh_save="dataset/scene_gen/mesh", directory_json_save="dataset/scene_gen/scene_mesh_json"):
    """
    Using open3d and trimesh
    OBJ with out texture
    Remove objects with semantic labels greater than 40 from the point cloud scene and replace them with mesh obj.
    1. check if there is desk or table 
        # TODO: only keep the obj and calculate the center and size of desk 
    2. get the pos and scale of the desk or table
    3. write down the obj_file_name, position, z_angle, scale into json file 
    
    
    Parameters:
    pcd_ply_path (str): Path to the input PLY file.
    npz_path (str): Path to the NPZ file containing 'xyz', 'semantic_label', and 'instance_label'.

    
    Returns:
    mesh scene, json file
    """

    pose_dict = {} # obj_file_name: [position, scale, angle, bbox]

    # 1. get the position, z-angle and scale of the desk or table
    # check if table or desk, otherwise None
    desk_pcd= get_obj_from_scene(pcd_ply_path=ply_path, obj_index=14, npz_path=npz_path)
    table_pcd = get_obj_from_scene(pcd_ply_path=ply_path, obj_index=7, npz_path=npz_path)
    if desk_pcd is not None and len(desk_pcd.points) > 200:
        desk_pcd = desk_pcd
        desk_mesh_folder = 'dataset/obj/mesh/desk'
    elif table_pcd is not None and len(table_pcd.points) > 200:
        desk_pcd = table_pcd
        desk_mesh_folder = 'dataset/obj/mesh/table'
    else:
        print('No table or desk!')
        return None
    
    aabb_desk_pcd = desk_pcd.get_axis_aligned_bounding_box()
    min_bound = aabb_desk_pcd.get_min_bound()
    max_bound = aabb_desk_pcd.get_max_bound()
    dimensions_desk_pcd = max_bound - min_bound
    center_desk_pcd = (min_bound + max_bound) / 2.0
    top_center_desk_pcd = np.array(center_desk_pcd)
    top_center_desk_pcd[2] += dimensions_desk_pcd[2] / 2.0

    # get desk or table mesh obj from a random file in desk_mesh_folder
    desk_mesh_subfolders = [os.path.join(desk_mesh_folder, name) for name in os.listdir(desk_mesh_folder)
              if os.path.isdir(os.path.join(desk_mesh_folder, name))]
    chosen_desk_mesh_file = random.choice(desk_mesh_subfolders)
    desk_obj_files = [os.path.join(chosen_desk_mesh_file, name) for name in os.listdir(chosen_desk_mesh_file)
             if name.endswith('.obj')]
    
    desk_mesh =  trimesh.load(desk_obj_files[0], process=False, skip_materials=True)


    # optimize the z
    optimized_z_desk = torch.tensor(0) # hard code
    angle_deg_z_desk = optimized_z_desk.detach().numpy()
    angle_deg_z_desk = np.round(angle_deg_z_desk / 90) * 90 # each 90 degree
    #print("agnle desk:", angle_deg_z_desk)

    # scale desk mesh
    min_bound = desk_mesh.bounds[0]
    max_bound = desk_mesh.bounds[1]
    dimensions_desk_mesh = max_bound - min_bound 

    center_desk_mesh = (min_bound + max_bound) / 2.0
    top_center_desk_mesh = np.array(center_desk_mesh)
    top_center_desk_mesh[2] += dimensions_desk_mesh[2] / 2.0
    
    scaling_factors = dimensions_desk_pcd / dimensions_desk_mesh

    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = scaling_factors[0]
    scale_matrix[1, 1] = scaling_factors[1]
    scale_matrix[2, 2] = scaling_factors[2]
    desk_scale_matrix = scale_matrix
    desk_mesh.apply_transform(desk_scale_matrix)
    
    
    min_bound = desk_mesh.bounds[0]
    max_bound = desk_mesh.bounds[1]
    dimensions_desk_mesh = max_bound - min_bound 

    center_desk_mesh = (min_bound + max_bound) / 2.0
    top_center_desk_mesh = np.array(center_desk_mesh)
    top_center_desk_mesh[2] += dimensions_desk_mesh[2] / 2.0

    mesh_scene = trimesh.Trimesh() # empty scene

    list_bottom_centric = []

    #print("---------")
    #print("Creating mesh scene...")
    dict_bbox_pos_sacle = bbox_pos_scale_all_obj(ply_path, npz_path)


    # 2. read mesh obj in sequence
    sample_mesh_points_list = []
    sample_pcd_points_list = []
    scale_matrix_list = []
    item_bbox_pos_list = []
    keyword_list = []
    mesh_file_path_list = []
    mesh_obj_list = []
    for ins_label, item_instance in dict_bbox_pos_sacle.items():
        obj_exist = 0

        item_semantic_label = item_instance['semantic_label']
        item_bbox_min_bound = item_instance['min_bound']
        item_bbox_max_bound = item_instance['max_bound']
        item_bbox_size = item_instance['size']
        item_bbox_center = [0,0,0] # initialize center
        item_bbox_center[0] = (item_bbox_min_bound[0] + item_bbox_max_bound[0]) / 2 #x
        item_bbox_center[1] = (item_bbox_min_bound[1] + item_bbox_max_bound[1]) / 2 #y
        item_bbox_center[2] = (item_bbox_min_bound[2] + item_bbox_max_bound[2]) / 2 #z
        item_bbox_pos = item_bbox_center
        item_bbox_pos[2] = item_bbox_pos[2] - item_bbox_size[2] / 2 #z bottom
        item_bbox_pos[2] = 0

        
        
        # convert to semantic label to keyword
            # index -- object name
        item_dict = {}
        with open('GeoL_net/dataset_gen/classes.txt', 'r') as file:
            for line in file:
                parts = line.strip().split('	')
                num = int(parts[0])
                item_name = parts[1]
                item_dict[num] = item_name    
        item_keyword = item_dict[item_semantic_label]

        # get the mesh obj 
        keyword = item_keyword # add mapping
        with open('GeoL_net/dataset_gen/obj_dict.json', 'r', encoding='utf-8') as file:
            data=json.load(file)
            keyword = data[keyword]
            
        if keyword == "":
            # print(f"no {item_keyword}")
            continue
        

        folder_path = 'dataset/obj/mesh' # the dataset folder of texture obj
        subfolders = next(os.walk(folder_path))[1]
        if keyword in subfolders:
            obj_exist = 1
            keyword_folder_path = os.path.join(folder_path, keyword)
            subfolders = [f.path for f in os.scandir(keyword_folder_path) if f.is_dir()]
            subfolder = random.choice(subfolders) # select a obj randomly
            mesh_file_path = os.path.join(subfolder, 'mesh.obj')
            if os.path.exists(mesh_file_path):
                #mesh_obj = trimesh.load_mesh(mesh_file_path, process=False, skip_materials=True, force="mesh")
                # print(mesh_file_path)
                mesh_obj = trimesh.load(mesh_file_path, process=False, skip_materials=True, force="mesh")
                #print(f"get obj from {mesh_file_path}")

        if obj_exist == 0: #ModelNet does not have the obj
            #print(f'sorry we dont have {keyword} in texture obj dataset')
            continue

        
        # bbox
        aabb_min_bound = mesh_obj.bounds[0]
        aabb_max_bound = mesh_obj.bounds[1]
        extent1 = item_bbox_size
        extent2 = aabb_max_bound - aabb_min_bound
        scale_factors = extent1 / extent2
        if keyword in ['book', 'pen', 'pencil', 'notebook', "glass_box", "bowl","printer", "keyboard", "laptop"]:
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = scale_factors[0] # align with the target
            scale_matrix[1, 1] = scale_factors[1]
            scale_matrix[2, 2] = scale_factors[2]

        else:
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = scale_factors.mean() # keep the orignal shape
            scale_matrix[1, 1] = scale_factors.mean()
            scale_matrix[2, 2] = scale_factors.mean()
        
        scale_matrix_list.append(scale_matrix)
        keyword_list.append(keyword) # keyword list
        mesh_obj.apply_transform(scale_matrix) # 1. scale

        aabb2_scaled_min = mesh_obj.bounds[0]
        aabb2_scaled_max = mesh_obj.bounds[1]
        #extent2_scaled = aabb2_scaled_max - aabb2_scaled_min
        #aabb_center = mesh_obj.centroid
        #aabb_bottom_center = np.array(aabb_center)
        #aabb_bottom_center[2] -= extent2_scaled[2] / 2.0  
        aabb_bottom_center = (aabb2_scaled_min + aabb2_scaled_max)/2
        aabb_bottom_center[2] = aabb2_scaled_min[2]

        list_bottom_centric.append(item_bbox_pos[2])
        # optimize the z-angle
        pcd = get_obj_from_scene_inslabel(pcd_ply_path=ply_path, ins_index=ins_label, npz_path=npz_path)
        # Sample points randomly from the point cloud
        sampled_indices = np.random.choice(len(pcd.points), 512, replace=True)
        sampled_pcd_points = np.asarray(pcd.points)[sampled_indices]
        sampled_mesh_points = mesh_obj.sample(512) # sample mesh points
        sampled_mesh_points = np.asarray(sampled_mesh_points)

        sample_mesh_points_expanded = np.expand_dims(sampled_mesh_points, axis=0)
        sample_pcd_points_expanded = np.expand_dims(sampled_pcd_points,axis=0)
        sample_mesh_points_list.append(sample_mesh_points_expanded)
        sample_pcd_points_list.append(sample_pcd_points_expanded)

    
        mesh_obj = move_trimeshobj_to_position(mesh_obj, end_position=item_bbox_pos, start_position=aabb_bottom_center) # 4. translate
        mesh_obj_list.append(mesh_obj) # mesh_obj_list
        mesh_file_path_list.append(mesh_file_path)
        item_bbox_pos_list.append(item_bbox_pos)
        
        #print(f"-----Great! {keyword} is done-----")

    sample_mesh_points_list = np.concatenate(sample_mesh_points_list, axis=0)
    sample_pcd_points_list = np.concatenate(sample_pcd_points_list, axis=0)
    start_time = time.time()
    optimized_z_list = optimize_rotation_batch(sample_mesh_points_list, sample_pcd_points_list, initial_angle=0, num_iterations=10, learning_rate=1)
    end_time = time.time()

    assert len(keyword_list) == len(mesh_obj_list)
    for i in range(len(keyword_list)):
        scale_matrix = scale_matrix_list[i]
        scale_matrix.tolist()
        keyword = keyword_list[i]
        mesh_obj = mesh_obj_list[i]
        optimized_z = optimized_z_list[i]
        angle_deg_z = optimized_z.detach().numpy().tolist()
        if keyword in ['laptop','glass_box','camera','clock','monitor','monitor_modelnet','keyboard','printer']:
            angle_deg_z = np.round(angle_deg_z / 90) * 90
        else:
            angle_deg_z = angle_deg_z//10 * 10

        mesh_obj = rotate_mesh_around_center(mesh_obj, angle_deg_z, [0,0,1])
        mesh_bbox = mesh_obj.bounds.tolist()
        #mesh_scene = trimesh.util.concatenate([mesh_scene, mesh_obj]) 为了速度 不生成mesh
        mesh_file_path = mesh_file_path_list[i]
        item_bbox_pos = item_bbox_pos_list[i]
        pose_dict[mesh_file_path] = [item_bbox_pos, [scale_matrix[0,0], scale_matrix[1,1], scale_matrix[2,2]], angle_deg_z, mesh_bbox] # add obj mesh into dict



    # final desk
    avg_bottom_centric = sum(list_bottom_centric) / len(list_bottom_centric)
    top_center_desk_pcd[2] = avg_bottom_centric
    desk_mesh = move_trimeshobj_to_position(desk_mesh, top_center_desk_pcd, top_center_desk_mesh)
    # mesh_scene = trimesh.util.concatenate([mesh_scene, desk_mesh]) 为了速度 不生成mesh


    last_slash_index = ply_path.rfind("/")
    last_dot_index = ply_path.rfind(".")

    filename = ply_path[last_slash_index + 1:last_dot_index]
    #print('filename', filename)
    #unique_file_path = get_unique_filename(directory=directory_mesh_save, filename=filename, extension="_mesh.obj")

    desk_scale_matrix.tolist()
    angle_deg_z_desk = angle_deg_z_desk.tolist()
    pose_dict[desk_obj_files[0]] = [item_bbox_pos, [desk_scale_matrix[0,0], desk_scale_matrix[1,1], desk_scale_matrix[2,2]], angle_deg_z_desk]

    #mesh_scene.export(unique_file_path)
    unique_json_path = get_unique_filename(directory=directory_json_save, filename=filename, extension=".json")
    #print(pose_dict)
    with open(f"{unique_json_path}", "w") as json_file:
        json.dump(pose_dict, json_file, indent=4)  # The indent parameter is optional but makes the JSON more readable
    #print(f"{unique_file_path} is done")
    print(f"{unique_json_path} is done") 
    print("-----------")
    return unique_json_path

if __name__ == "__main__":

    # 打开并读取文件
    
    with open('GeoL_net/dataset_gen/to_scene.txt', 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)
    bad = 0
    good = 0
    start_time = time.time()
    with open('GeoL_net/dataset_gen/to_scene.txt', 'r') as file:
        for line in tqdm(file, total=total_lines):
            line = line.strip()
            print(line)
            
            ply_path = f"dataset/TO_scene_ori/TO-crowd/ply/train/id{line}.ply"
            npz_path = f"dataset/TO_scene_ori/TO-crowd/npz/train/id{line}.npz"
            result = generate_mesh_scene_all_texute_v2(ply_path=ply_path, npz_path=npz_path)
            if result is None:
                bad = bad + 1
            else:
                good = good + 1
    end_time = time.time()
    print(f"运行时间: {end_time - start_time} 秒")
    print("good:", good)
    print("bad:", bad)
'''
    ply_path = f"dataset/TO_scene_ori/TO-crowd/ply/train/id16.ply"
    npz_path = f"dataset/TO_scene_ori/TO-crowd/npz/train/id16.npz"
    generate_mesh_scene_all_texute_v2(ply_path=ply_path, npz_path=npz_path)
    '''
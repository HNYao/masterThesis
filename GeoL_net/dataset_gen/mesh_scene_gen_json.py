"""
generate a json file, including:
    (obj_mesh_file_path): position, scale, bbox
    TODO: aabb bbox存在问题，例如pencil,判断出长宽高
    TODO:旋转显示器
    TODO: 将所有场景normalize到一个x方向上[-1,1]的区间
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
    2. get the pos and scale of the desk or table
    3. write down the obj_file_name, position, z_angle, scale into json file 
    
    
    Parameters:
    pcd_ply_path (str): Path to the input PLY file.
    npz_path (str): Path to the NPZ file containing 'xyz', 'semantic_label', and 'instance_label'.

    
    Returns:
    mesh scene, json file
    """
    # 1. get the global center and size
    pose_dict = {} # obj_file_name: [position, scale, angle, bbox]
    list_bottom_centric = []

    dict_bbox_pos_sacle = bbox_pos_scale_all_obj(ply_path, npz_path) # the sizes of all bboxs

    # get the global center and size
    global_min_bound = [float('inf'), float('inf'), float('inf')]  # initialization inf
    global_max_bound = [-float('inf'), -float('inf'), -float('inf')]  # initialization -inf
    for ins_label, item_instance in dict_bbox_pos_sacle.items():
            min_bound = item_instance['min_bound']  # min_bound of each instance
            max_bound = item_instance['max_bound']  # max_bound of each instance
            for i in range(3):
                global_min_bound[i] = min(global_min_bound[i], min_bound[i])
                global_max_bound[i] = max(global_max_bound[i], max_bound[i])
        
    global_center = [
        (global_min_bound[0] + global_max_bound[0]) / 2,  # x coordinate
        (global_min_bound[1] + global_max_bound[1]) / 2,  # y coordinate
        (global_min_bound[2] + global_max_bound[2]) / 2   # z coordinate, but not used(will be set to 0 in blenderproc)
        ]
    
    global_translation = global_center # move the xy-plane center of tabletop to (0,0)  将整个桌面物体的xy平面中心移到0，0
    global_size = [
        (global_max_bound[0] - global_min_bound[0]) ,  # x length
        (global_max_bound[1] - global_min_bound[1]) ,  # y length
        (global_max_bound[2] - global_min_bound[2])    # z length
        ]
    global_resize = 1.2 / global_size[0] # 以 x为准 （如果有误，改成以y为准） 需要对物体的size和pos进行调整
    # update ylength
    y_len  = global_size[1] * global_resize #x 归一后 y的length
    
    if y_len > 0.8: # if y is too long, resize to 0.8
        global_resize = 0.8 / global_size[1]


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
            
        if keyword == "": # no corresponding obj
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
                mesh_obj = trimesh.load(mesh_file_path, process=False, skip_materials=True, force="mesh")

        if obj_exist == 0: #ModelNet does not have the obj
            #print(f'sorry we dont have {keyword} in texture obj dataset')
            continue
        
        # bbox
        aabb_min_bound = mesh_obj.bounds[0]
        aabb_max_bound = mesh_obj.bounds[1]
        extent1 = item_bbox_size
        extent2 = aabb_max_bound - aabb_min_bound
        scale_factors = extent1 / extent2
        if keyword in []:
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = scale_factors.mean() # keep the orignal shape
            scale_matrix[1, 1] = scale_factors.mean()
            scale_matrix[2, 2] = scale_factors.mean()
        elif keyword in ["cup"]:
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = scale_factors.min() # keep the orignal shape
            scale_matrix[1, 1] = scale_factors.min()
            scale_matrix[2, 2] = scale_factors.min()
        elif keyword in ["pen, pencil"]:
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = scale_factors[2] # the z length is the standard
            scale_matrix[1, 1] = scale_factors[2]
            scale_matrix[2, 2] = scale_factors[2]
 
        scale_matrix_list.append(scale_matrix)
        keyword_list.append(keyword) # keyword list
        mesh_obj.apply_transform(scale_matrix) # 1. scale

        aabb2_scaled_min = mesh_obj.bounds[0]
        aabb2_scaled_max = mesh_obj.bounds[1]
 
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
    optimized_z_list = optimize_rotation_batch(sample_mesh_points_list, sample_pcd_points_list, initial_angle=0, num_iterations=10, learning_rate=1)


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
        # pose_dict[mesh_file_path] = [item_bbox_pos, [scale_matrix[0,0], scale_matrix[1,1], scale_matrix[2,2]], angle_deg_z, mesh_bbox] # add obj mesh into dict
        item_bbox_pos[0] = (item_bbox_pos[0] - global_translation[0]) * global_resize
        item_bbox_pos[1] = (item_bbox_pos[1] - global_translation[1]) * global_resize
        item_bbox_pos[2] = (item_bbox_pos[2] - global_translation[2]) * global_resize
        pose_dict[mesh_file_path] = [item_bbox_pos, 
                                     [scale_matrix[0,0]  * global_resize, scale_matrix[1,1]  * global_resize, scale_matrix[2,2] * global_resize], 
                                     angle_deg_z, 
                                     mesh_bbox # 后续未使用，不做调整
                                     ]


    # final desk
    desk_mesh_folder  = 'dataset/obj/mesh/desk'
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

    #center_desk_mesh = (min_bound + max_bound) / 2.0
    #top_center_desk_mesh = np.array(center_desk_mesh)
    #top_center_desk_mesh[2] += dimensions_desk_mesh[2] / 2.0
    
    scaling_factors_x = 1.4 / dimensions_desk_mesh[0] # desk x轴1.1
    scaling_factors_y = max(y_len-0.4, 1) / dimensions_desk_mesh[1] # 最少y是1

    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = scaling_factors_x
    scale_matrix[1, 1] = scaling_factors_y
    scale_matrix[2, 2] = scaling_factors_x
    desk_scale_matrix = scale_matrix
    
    #top_center_desk_pcd[2] = avg_bottom_centric
    #desk_mesh = move_trimeshobj_to_position(desk_mesh, top_center_desk_pcd, top_center_desk_mesh)
    # mesh_scene = trimesh.util.concatenate([mesh_scene, desk_mesh]) 为了速度 不生成mesh
    

    
    # save the json
    last_slash_index = ply_path.rfind("/")
    last_dot_index = ply_path.rfind(".")
    filename = ply_path[last_slash_index + 1:last_dot_index]
    desk_scale_matrix.tolist()
    angle_deg_z_desk = angle_deg_z_desk.tolist()
    pose_dict[desk_obj_files[0]] = [[0,0,0], [desk_scale_matrix[0,0], desk_scale_matrix[1,1], desk_scale_matrix[2,2]], angle_deg_z_desk]

    unique_json_path = get_unique_filename(directory=directory_json_save, filename=filename, extension=".json")
    with open(f"{unique_json_path}", "w") as json_file:
        json.dump(pose_dict, json_file, indent=4)  # The indent parameter is optional but makes the JSON more readable
    print(f"{unique_json_path} is done") 
    print("-----------")
    return unique_json_path

if __name__ == "__main__":

    # 打开并读取文件
    # generate batch 
    with open('GeoL_net/dataset_gen/to_scene.txt', 'r') as file:
        ungenerated_scene_ids = []
        for line in file:
            try:
                number = int(line.strip())
                ungenerated_scene_ids.append(number)
            except ValueError:
                pass
        total_lines = sum(1 for _ in file)

    #print(ungenerated_scene_ids)
    bad = 0
    good = 0
    start_time = time.time()
    
    for number in tqdm(range(0, 750), total=750-total_lines):
    
        if number not in ungenerated_scene_ids:
            print(number)
            
            ply_path = f"dataset/TO_scene_ori/TO-crowd/ply/train/id{number}.ply"
            npz_path = f"dataset/TO_scene_ori/TO-crowd/npz/train/id{number}.npz"
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
    ply_path = f"dataset/TO_scene_ori/TO-crowd/ply/train/id1.ply"
    npz_path = f"dataset/TO_scene_ori/TO-crowd/npz/train/id1.npz"
    generate_mesh_scene_all_texute_v2(ply_path=ply_path, npz_path=npz_path)
    '''
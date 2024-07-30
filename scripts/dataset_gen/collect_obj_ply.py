"""
1. collect obj ply from scene (except for scene in the to_scene.txt)
2. store the ply in the dataset/obj/ply
3. set a max number of each kind of obj 
"""

import open3d as o3d
from scripts.dataset_gen.utils import * 
import os

# get dict: item_name - num
item_dict = {}
with open('scripts/dataset_gen/classes.txt', 'r') as file:
    for line in file:
        parts = line.strip().split('	')
        num = int(parts[0])
        item_name = parts[1]
        item_dict[num] = item_name    

# get list: inproper scene
inproper_scene_list = []
with open('scripts/dataset_gen/to_scene.txt', 'r') as file:
    for line in file:
        number = int(line.strip())
        inproper_scene_list.append(number)

scene_ply_folder_path = "dataset/TO_scene_ori/TO-crowd/ply/train"
scene_npz_folder_path = "dataset/TO_scene_ori/TO-crowd/npz/train"
for root, dirs, files in os.walk(scene_ply_folder_path):
    len_plyfiles = len(files)

# Traverse all ids (except those in txt)
for id_ply in range(len_plyfiles):
    if id_ply in inproper_scene_list:
        continue
    ply_path = os.path.join(scene_ply_folder_path, f"id{id_ply}.ply")
    npz_path = os.path.join(scene_npz_folder_path, f"id{id_ply}.npz")
    print(ply_path, npz_path)

    for obj_id in item_dict.keys():
        obj_ply = get_obj_from_scene(pcd_ply_path=ply_path, obj_index=obj_id, npz_path=npz_path)
        if obj_ply is None:
            continue
        
        o3d.io.write_point_cloud("example.ply", obj_ply)
        
        # to zero point
        center = obj_ply.get_center()
        obj_ply.translate(-center)

        output_folder = os.path.join("dataset/obj/ply",f"{item_dict[obj_id]}")
        os.makedirs(output_folder, exist_ok=True)
        existing_files = os.listdir(output_folder)
        next_suffix = len(existing_files) + 1
        if len(existing_files) > 20: # max number of one kind of obj
            continue

        output_file = os.path.join(output_folder, f"{item_dict[obj_id]}_{next_suffix}.ply")
        o3d.io.write_point_cloud(output_file, obj_ply)

        print(f"save {output_file}")


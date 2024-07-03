"""
    1. collect obj ply from scene (except for scene in the to_scene.txt)
    2. store the ply in the dataset/obj/ply
"""

import open3d as o3d
import utils
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
        inproper_scene_list.append(line.strip())

scene_folder_path = "dataset/TO_scene_ori/TO-crowd/ply/train"
for root, dirs, files in os.walk(scene_folder_path):
    print(root, dirs, files)
    print(len(files))

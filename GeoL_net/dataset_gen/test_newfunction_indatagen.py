import open3d as o3d
import numpy as np
import json
from tqdm import tqdm
import os
import math
from GeoL_net.text_gen.json2text import key2phrase, determine_direction
import random

def calculate_distance(pos1, pos2):
    """计算两个点之间的欧几里得距离"""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def determine_direction(p1, p2):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    
    angle = math.degrees(math.atan2(y_diff, x_diff))
    if angle < 0:
        angle += 360
    
    if 0 <= angle < 22.5 or 337.5 <= angle < 360:
        return "Left"
    elif 22.5 <= angle < 67.5:
        return "Front Left"
    elif 67.5 <= angle < 112.5:
        return "Front"
    elif 112.5 <= angle < 157.5:
        return "Front Right"
    elif 157.5 <= angle < 202.5:
        return "Right"
    elif 202.5 <= angle < 247.5:
        return "Right Behind"
    elif 247.5 <= angle < 292.5:
        return "Behind"
    elif 292.5 <= angle < 337.5:
        return "Left Behind"



def objs_keep_in_scene(json_file, number_of_objects):
    scene_id = json_file.split("/")[-1].split(".")[0]
    new_json_file = os.path.join("dataset/scene_RGBD_mask_sequence",scene_id, "text_guidance.json")

        

    obj_keep_list = []
    
    # 打开JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 创建用于分类的列表
    high_priority = []
    low_priority = []

    # 遍历JSON文件中的所有物体
    for obj_file_name, obj_pose in data.items():
        if "desk" in obj_file_name or "table" in obj_file_name:
            obj_keep_list.append(obj_file_name)
        if "monitor" in obj_file_name or "laptop" in obj_file_name:
            # 优先保留 monitor 和 laptop
            high_priority.append((obj_file_name, obj_pose[0]))  # 存储物体名和位置信息
        elif any(x in obj_file_name for x in ["pencil", "eraser", "clock"]):
            # 不保留 pencil, eraser, clock
            continue
        else:
            # 其他物体保留到低优先级列表
            low_priority.append((obj_file_name, obj_pose[0]))

    # 循环选择物体
    while len(obj_keep_list) < number_of_objects+1: #算上桌子
        # 优先选择高优先级的物体
        if high_priority:
            random.shuffle(high_priority)  # 随机打乱顺序
            obj_name, obj_pos = high_priority.pop(0)  # 从 high_priority 列表中取出
            obj_keep_list.append(obj_name)
        # 如果高优先级物体处理完，则处理其他物体
        elif low_priority:
            random.shuffle(low_priority)  # 随机打乱顺序
            obj_name, obj_pos = low_priority.pop(0)  # 从 low_priority 列表中取出
            obj_keep_list.append(obj_name)
        else:
            break  # 如果没有更多的物体可供选择，退出循环

    # 保留data中obj_keep_list中的物体
    filtered_data = {k: v for k, v in data.items() if k in obj_keep_list}

    # 选定anchor物体 从filtered_data中选择一个物体作为anchor物体，但anchor物体不能是桌子
    anchor_obj = None
    
    while True:
        obj_name = random.choice(obj_keep_list)
        if "desk" not in obj_name and "table" not in obj_name:
            anchor_obj = obj_name
            break
    anchor_pos = data[anchor_obj][0] # x, y, z
    
    # 选定anchor 后， 遍历所有不在obj_keep_list中的物体，计算其相对于anchor的位置
    direction_list = []
    text_dict = {}
    removed_obj_list = []
    obj_distances = []

    # 计算不在 obj_keep_list 中的物体到锚点的距离，并保存这些物体
    for obj_name, obj_pose in data.items():
        if obj_name in obj_keep_list:
            continue
        distance = calculate_distance(obj_pose[0][:2], anchor_pos[:2])  # 计算物体与锚点的2D距离
        obj_distances.append((obj_name, obj_pose, distance))

    # 按照距离从近到远排序
    obj_distances.sort(key=lambda x: x[2])


    for obj_name, obj_pose, _ in obj_distances:
        if obj_name in obj_keep_list:
            continue
        direction = determine_direction(obj_pose[0][:2], anchor_pos[:2])
        if direction not in direction_list:
            direction_list.append(direction)
            removed_obj_list.append(obj_name)
        else :
            continue
        processed_obj_name = key2phrase(obj_name)
        processed_anchor_obj = key2phrase(anchor_obj)
        short_anchor_obj = processed_anchor_obj.split(" ")[-1]
        text_dict[processed_obj_name] = [processed_anchor_obj,f"{direction} {processed_anchor_obj}",f"{short_anchor_obj}",direction]
    if not os.path.exists(new_json_file):
    # 如果文件不存在，首先确保目录存在
        os.makedirs(os.path.dirname(new_json_file), exist_ok=True)
    with open(new_json_file, "w") as json_file:
        json.dump(text_dict, json_file, indent=4)
    
    
    return obj_keep_list, filtered_data, anchor_obj, removed_obj_list



if __name__ == "__main__":

    obj_keep_list, filtered_data, anchor_obj, removed_obj_list = objs_keep_in_scene("dataset/scene_gen/scene_mesh_json/id728.json", 1)
    print(obj_keep_list)
    print(filtered_data)
    print(anchor_obj)
    print(removed_obj_list)
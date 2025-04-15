import blenderproc as bproc
from blenderproc.python.writer.MyWriterUtility import write_my, write_my_zhoy
from blenderproc.python.writer.CocoWriterUtility import write_coco_annotations
import bpy
import psutil

import numpy as np
import glob
import json
import os
import math


import time

import random
import math



""""
    almost same as the mesh_scene_gen_bproc.py
    the only difference is generating a mask.json containing the bbox parameters of obj.
    json to RGBD(mask)

    updated:
        add the mask of anchor obj (Yellow)
    
        category_id:1 normal object
        category_id:2 desk
        category_id:3 room plane
        category_id:4 removed object
        category_id:5 anchor object

"""
def calculate_distance(pos1, pos2):
    """euclidean distance between two points"""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def key2phrase(key:str):
    key_parts = key.split('/')
    obj_descpt = key_parts[-2]
    parts = obj_descpt.split('_')
    obj_name = '_'.join(key_parts[3:-2]) # pay attention
    obj_descpt = parts[-1]
    phrase = f"the {obj_descpt} {obj_name}"
    return phrase

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


def bproc_gen_mask_with_and_without_obj(
        scene_mesh_json, 
        RGBD_out_dir, 
        removed_obj_path_list=None, 
        anchor_obj_path_list=None,
        view_angle='front'):
    """Render both the scene with and without the removed object.

    Args:
        scene_mesh_json: str, path to the scene mesh json file.
        RGBD_out_dir: str, path to the output directory.
        removed_obj_path_list: list, path to the removed object.
        anchor_obj_path_list: list, path to the anchor object

    Returns:
        None

    """


    # Step 1: load scene mesh json file
    json_file_path = scene_mesh_json
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    with open('GeoL_net/dataset_gen/obj_size.json', 'r') as f:
        obj_size = json.load(f)

    mode = 'train'
    # previous config
    #cfg = dict(
    #    H=480, 
    #    W=640, 
    #    K=[591.0125, 0, 322.525, 0, 590.16775, 244.11084, 0, 0, 1], 
    #    cc_textures_path='resources'
    #)

    # NOTE: kinect config, the real kinect is 720, 1280 
    cfg = dict(
        H=360, 
        W=640, 
        K=[607.09912/2, 0, 636.85083/2, 0, 607.05212/2, 367.35952/2, 0, 0, 1], 
        cc_textures_path='resources'
    )
    K = np.array(cfg['K']).reshape(3, 3)

    # Step 2: camera intrinsics
    bproc.camera.set_intrinsics_from_K_matrix(K, cfg['W'], cfg['H'])

    target_objects = []
    x_min, x_max, y_min, y_max = 100, -100, 100, -100
    exist_obj_amount = 0
    obj_amount = len(data)

    # Step 3: load all objects at one time
    for obj_file_name, obj_pose in data.items():
        exist_obj_amount += 1
        if exist_obj_amount == obj_amount:
            # the last object is the desk
            obj_mesh = bproc.loader.load_obj(obj_file_name)
            desk = obj_mesh[0]
            desk.blender_obj.rotation_euler = (0, 0, 0)
            desk.blender_obj.scale = (obj_pose[1][0], obj_pose[1][1], obj_pose[1][2])
            bbox = desk.get_bound_box()
            z_height = bbox[1][2] - bbox[0][2]
            desk.blender_obj.location = ((x_min + x_max) / 2, (y_min + y_max) / 2, 0 - z_height / 2)

            desk_mat = desk.get_materials()[0]
            desk_mat.set_principled_shader_value("Roughness", np.random.uniform(0.8, 1.0))
            desk_mat.set_principled_shader_value("Specular", np.random.uniform(0.8, 1.0))

            material = desk.get_materials()[0]
            desk.replace_materials(material)

            desk.set_cp("category_id", 2) # category_id: 2 desk
            desk.set_cp("scene_id", 2)
            desk.set_name("desk")

            target_objects.append(desk)
            break

        # load each object
        obj_mesh = bproc.loader.load_obj(obj_file_name)
        z_rotation = math.radians(obj_pose[2])
        obj_category = obj_file_name.split("/")[3]
        #target_size = obj_size[obj_category]
        #current_size = obj_mesh[0].blender_obj.dimensions
        #scale_factor = [target_size[0] / current_size[0], target_size[1] / current_size[1], target_size[2] / current_size[2]]
        scale_factor = obj_pose[1]
        obj_mesh[0].blender_obj.rotation_euler = (0, 0, z_rotation)
        if obj_category in ["mug", "cup", "plant"]:
            obj_mesh[0].blender_obj.scale = (scale_factor[2], scale_factor[2], scale_factor[2])
        else:
            obj_mesh[0].blender_obj.scale = (scale_factor[0], scale_factor[1], scale_factor[2])
        obj_mesh[0].set_name(obj_file_name)
        bbox = obj_mesh[0].get_bound_box()
        z_height = bbox[1][2] - bbox[0][2]
        obj_mesh[0].blender_obj.location = (obj_pose[0][0], obj_pose[0][1], 0 + z_height / 2)

        # set material
        mat = obj_mesh[0].get_materials()[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.8, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0.8, 1.0))


        obj_mesh[0].set_cp("category_id", 1) # category_id: 1  normal object
        obj_mesh[0].set_cp("scene_id", 1)

        target_objects.append(obj_mesh[0])

        # set the bounding box of the scene
        x_min = min(x_min, bbox[0][0])
        x_max = max(x_max, bbox[6][0])
        y_min = min(y_min, bbox[0][1])
        y_max = max(y_max, bbox[6][1])

    # Step 4: room planes
    room_coeff = 10
    room = [bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, 0, obj_pose[0][2] - z_height]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, -room_coeff, room_coeff + obj_pose[0][2] - z_height], rotation=[-1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[0, room_coeff, room_coeff + obj_pose[0][2] - z_height], rotation=[1.570796, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[room_coeff, 0, room_coeff + obj_pose[0][2] - z_height], rotation=[0, -1.570796, 0]),
            bproc.object.create_primitive('PLANE', scale=[room_coeff, room_coeff, 1], location=[-room_coeff, 0, room_coeff+obj_pose[0][2] - z_height], rotation=[0, 1.570796, 0])]

    # sample point light on shell
    light_plane = bproc.object.create_primitive('SPHERE', scale=[1, 1, 1], location=[-1, -1, 2])
    light_plane.set_name('light_point')
    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))     
    light_plane.replace_materials(light_plane_material)

    # sample CC Texture and assign to room planes
    # if cfg['cc_textures_path'] is not None:
    #     cc_textures = bproc.loader.load_ccmaterials(cfg['cc_textures_path'])
    #     for plane in room:
    #         random_cc_texture = np.random.choice(cc_textures)
    #         plane.replace_materials(random_cc_texture)
    
    # # set attributes
    room.append(light_plane)
    for plane in room:
        plane.set_cp('category_id', 3) # category_id: 3 room plane

    # Step 5: camera pose
    i = 0
    radius_min, radius_max = (1.2, 1.5)
    _radius = np.random.uniform(low=radius_min, high=radius_max) 
    num_views= 1

    while i < num_views:

        poi = bproc.object.compute_poi(np.random.choice(target_objects, size=5))

        noise =  np.random.randn() * (_radius / 10) 
        inplane_rot = np.random.uniform(-0.7854, 0.7854) 

        radius = _radius + noise
        # Sample on sphere around ShapeNet object
        location = bproc.sampler.part_sphere(poi, radius=radius, dist_above_center=0.05, mode="SURFACE") # dist_above_center: hight of the ring


        # Compute rotation based on vector going from location towards ShapeNet object
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=inplane_rot)

        # Add homog cam pose based on location an rotation
        #cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        #rotation_matrix = [[1, 0, 0], [0,0.8,-0.6], [0,0.6,0.8]]
        #rotation_matrix = [[0.8660254, -0.3535534,  0.3535534], [0.5000000,  0.6123725, -0.6123725], [-0.0000000,  0.7071068,  0.7071068]] # right view
        
        #cam2world_matrix = bproc.math.build_transformation_mat([0.75568, -1.1737, 1], rotation_matrix) #right view
        if view_angle == "right":
            rotation_matrix = [[0.8660254,  0.3535534, -0.3535534], [-0.5000000,  0.6123725, -0.6123725], [0.0000000,  0.7071068,  0.7071068]] # left view
            cam2world_matrix = bproc.math.build_transformation_mat([-0.75568, -1.1737, 1], rotation_matrix) #right view
        elif view_angle == "left":
            rotation_matrix = [[0.8660254, -0.3535534,  0.3535534], [0.5000000,  0.6123725, -0.6123725], [0.0000000,  0.7071068,  0.7071068]]
            cam2world_matrix = bproc.math.build_transformation_mat([0.75568, -1.1737, 1], rotation_matrix) #left view
        elif view_angle == "front":
            rotation_matrix = [[1, 0, 0], [0,0.8,-0.6], [0,0.6,0.8]]
            cam2world_matrix = bproc.math.build_transformation_mat([0, -1, 1], rotation_matrix) #front view
        # print(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
        i += 1
    
    assert len(anchor_obj_path_list) == len(removed_obj_path_list), "The number of anchor objects and removed objects should be the same."
    for i in range(len(anchor_obj_path_list)):
        # get scene id
        scene_id = json_file_path.split("/")[-1].split(".")[0]
        # get the removed object name
        removed_obj_name = removed_obj_path_list[i].split("/")[-2]

        # 设定 物体的category_id
        for obj in target_objects:
            if removed_obj_path_list[i] in obj.get_name() and obj.get_name() != "desk":
                obj.set_cp("category_id", 4)
            if anchor_obj_path_list[i] in obj.get_name() and obj.get_name() != "desk":
                obj.set_cp("category_id", 5)

        # Step 6: 渲染包含和不包含 removed_obj 的场景
        for remove_flag in [False, True]:
            # 控制 removed_obj 的可见性
            for obj in target_objects:
                print(f"-------{obj.get_name()}-----")
                if removed_obj_path_list[i] in obj.get_name():
                    obj.blender_obj.hide_render = remove_flag  # True: 不渲染该对象, False: 渲染该对象
                    print(f"-------not rendering {obj.get_name()}-----")
                else:
                    obj.blender_obj.hide_render = False  # 其他对象始终渲染
            try:
                bproc.renderer.enable_depth_output(activate_antialiasing=False)
            except RuntimeError:
                pass
            bproc.renderer.set_max_amount_of_samples(1)
            bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id"])
            data = bproc.renderer.render()

            data["depth_kinect"] = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"], 10)
            # Step 7: 保存渲染结果
            out_dir = os.path.join(RGBD_out_dir, scene_id, removed_obj_name)
            if remove_flag:
                out_dir = os.path.join(out_dir, "no_obj")
            else:
                out_dir = os.path.join(out_dir, "with_obj")

            # 保存文件
            write_my(out_dir,
                chunk_name='test_pbr',
                dataset='',
                target_objects=target_objects,
                depths=data["depth"],
                depths_noise=data["depth_kinect"],
                colors=data["colors"],
                instance_masks=data['instance_segmaps'],
                category_masks=data['category_id_segmaps'],
                instance_attribute_maps=data["instance_attribute_maps"],
                color_file_format="JPEG",
                ignore_dist_thres=10,
                frames_per_chunk=1,
                is_shapenet=False
            )

            bproc.writer.write_hdf5(out_dir, data)


        # 重新设定 物体的category_id
        for obj in target_objects:
            if obj.get_name() != "desk": 
                obj.set_cp("category_id", 1) 



    
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

def get_cpu_temperature():
    """从 sensors 命令获取当前 CPU 温度"""
    try:
        result = os.popen("sensors").read()
        lines = result.splitlines()
        for line in lines:
            if "Package id 0" in line:  # 找到 CPU 包温度
                temp_str = line.split(":")[1].split("(")[0]
                return float(temp_str.strip().replace("°C", ""))
    except Exception as e:
        print(f"Fail to get cup temp: {e}")
        return None


# 定义温度阈值
NORMAL_TEMP = 59.0  # 正常工作温度
HIGH_TEMP = 63.0    # 高温警戒
CRIT_TEMP = 100.0   # 临界温度

# 定义休眠时间的范围
MIN_SLEEP = 0       # 最小休眠时间
MAX_SLEEP = 3       # 最大休眠时间（过热时）

def dynamic_sleep():
    """根据 CPU 温度动态调整休眠时间"""

    temp = get_cpu_temperature()
    if temp is not None:
        print(f"Current CPU Temp: {temp}°C")

        if temp < NORMAL_TEMP:
            sleep_time = MIN_SLEEP
        elif NORMAL_TEMP <= temp < HIGH_TEMP:
            sleep_time = (temp - NORMAL_TEMP) / (HIGH_TEMP - NORMAL_TEMP) * MAX_SLEEP
        else:
            sleep_time = MAX_SLEEP

        print(f"Sleep time : {sleep_time:.2f} s")
        time.sleep(1 + sleep_time)
    else:
        print("fail to get CPU temp, sleep 1s")
        time.sleep(1)


if __name__ == "__main__":
    #### generate one case for picked case
    bproc.init()
    json_file = "dataset/scene_gen/picked_scene_mesh_json/id15_0.json"
    parent_dir = "dataset/multi_view"
    scene_id = json_file.split("/")[-1].split(".")[0]
    text_guidance_file = "dataset/picked_scene_RGBD_mask/id15_0/text_guidance.json"

    scene_id_file = os.path.join("dataset/picked_scene_RGBD_mask",scene_id)

    with open(text_guidance_file, 'r') as f:
        text_guidance_data = json.load(f)


    # get the list of anchor object and removed objcet 
    anchor_obj_path_list = []
    removed_obj_path_list = []
    for key in text_guidance_data:
        anchor_obj_name = text_guidance_data[key][0] # e.g. the brown bottle
        removed_obj_name  = key # e.g. the glass bottle
        anchor_obj_path = text_guidance_data[key][5]
        removed_obj_path = text_guidance_data[key][4]

        anchor_obj_path_list.append(anchor_obj_path)
        removed_obj_path_list.append(removed_obj_path)

    #view_angle = "left"
    #view_angle = "right"
    view_angle = "front"
    bproc_gen_mask_with_and_without_obj(
        scene_mesh_json=json_file,
        RGBD_out_dir=parent_dir,
        removed_obj_path_list=removed_obj_path_list,
        anchor_obj_path_list=anchor_obj_path_list,
        view_angle=view_angle
    )

    
    #### generate data in batch
    # bproc.init()

    # json_folder_path = "dataset/scene_gen/scene_mesh_json_aug"
    # json_files = glob.glob(os.path.join(json_folder_path, '*.json'))
    # data_size = 0
    # for json_file_path in json_files:
    #     # clean up the scene
    #     bproc.clean_up()

      
    #     # config
    #     json_file = json_file_path # "dataset/scene_gen/scene_mesh_json/id531_1.json"
    #     parent_dir = "dataset/scene_RGBD_mask_data_aug"
    #     scene_id = json_file.split("/")[-1].split(".")[0]
        
    #     # find the corresponding text guidance file
    #     text_guidance_file = os.path.join("dataset/scene_RGBD_mask_data_aug",scene_id, "text_guidance.json")

    #     # find the scene id file, eg. id531_1
    #     scene_id_file = os.path.join("dataset/scene_RGBD_mask_data_aug",scene_id)

    #     # check if the scene id file contains other directories
    #     if os.path.exists(scene_id_file):
    #         if len(os.listdir(scene_id_file)) > 1: # check if the directory contains other directories
    #             print(f"{scene_id} is already made, so skip")
    #             data_size += 1
    #             print(f"Data size: {data_size}")
    #             start_datasize = data_size
    #             continue

        
    #     # find the anchor object and removed object in the text guidance file
    #     with open(text_guidance_file, 'r') as f:
    #         text_guidance_data = json.load(f)

    #     # get the list of anchor object and removed objcet 
    #     anchor_obj_path_list = []
    #     removed_obj_path_list = []
    #     for key in text_guidance_data:
    #         anchor_obj_name = text_guidance_data[key][0] # e.g. the brown bottle
    #         removed_obj_name  = key # e.g. the glass bottle
    #         anchor_obj_path = text_guidance_data[key][5]
    #         removed_obj_path = text_guidance_data[key][4]

    #         anchor_obj_path_list.append(anchor_obj_path)
    #         removed_obj_path_list.append(removed_obj_path)

    #     # given list of anchor obj and removed obj
    #     bproc_gen_mask_with_and_without_obj(
    #         scene_mesh_json=json_file,
    #         RGBD_out_dir=parent_dir,
    #         removed_obj_path_list=removed_obj_path_list,
    #         anchor_obj_path_list=anchor_obj_path_list
    #     )
    #     data_size += 1

    #     print(f"Data size: {data_size}")

    #     if (data_size - start_datasize) % 30 == 0:
    #         print("RESTART")


    #     bproc.clean_up(True)
        #######

   
    
    ######### test object size hard code
    # bproc.init()
    # json_file = "dataset/scene_gen/scene_mesh_json_aug/id114_id64_0_0.json"
    # parent_dir = "dataset/scene_gen/scene_RGBD_mask_data_aug_test"

    # anchor_obj_path_list = []
    # removed_obj_path_list = []
    # text_guidance_file = os.path.join("dataset/scene_RGBD_mask_data_aug", "id114_id64_0_0", "text_guidance.json")
    # with open(text_guidance_file, 'r') as f:
    #     text_guidance_data = json.load(f)
    # for key in text_guidance_data:
    #     anchor_obj_name = text_guidance_data[key][0] # e.g. the brown bottle
    #     removed_obj_name  = key # e.g. the glass bottle
    #     anchor_obj_path = text_guidance_data[key][5]
    #     removed_obj_path = text_guidance_data[key][4]

    #     anchor_obj_path_list.append(anchor_obj_path)
    #     removed_obj_path_list.append(removed_obj_path)

    # bproc_gen_mask_with_and_without_obj(
    # scene_mesh_json=json_file,
    # RGBD_out_dir=parent_dir,
    # removed_obj_path_list=removed_obj_path_list,
    # anchor_obj_path_list=anchor_obj_path_list
    # )

    # bproc.clean_up(True)

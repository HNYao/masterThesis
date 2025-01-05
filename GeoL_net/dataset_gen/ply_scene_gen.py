"""
given 2 scene ply, create a new scene ply and its npy

1. given 2 scene ply: scene A and scene B
2. check if there is table or desk
3. check if there is laptop or monitor
4. transfer to scene graphs
5. combine 2 graphs, variation, new scene graph
6. use the table or desk in scene, put obj ply in sequence
7. collision detect, if collision, move
"""

import open3d as o3d
import numpy as np
from collections import deque
from GeoL_net.dataset_gen.utils import * 

import os

def check_obj_exist(ply_path, npz_path, obj_id):
    """check if there is a specified obj .

    Args:
        ply_path.
        npz_path.
    
    Returns:
        True or False
    """
    data = np.load(npz_path)
    semantic_labels = data['semantic_label']
    
    if obj_id in semantic_labels:
        return True

    return False

def find_center_object(obj_dict):
    # 筛选出 semantic label 大于 40 的物体
    filtered_objects = {key: value for key, value in obj_dict.items() if value[0][0] > 40}
    # 获取所有物体的位置向量
    positions = [value[2] for value in filtered_objects.values()]
    
    # 计算所有位置的平均值
    mean_position = np.mean(positions, axis=0)
    
    # 初始化最小距离和中心物体
    min_distance = float('inf')
    center_object = None
    
    # 遍历所有物体，找到离平均位置最近的物体
    for key, value in obj_dict.items():
        position = value[2]
        distance = np.linalg.norm(position - mean_position)
        if distance < min_distance:
            min_distance = distance
            center_object = (key, value[1][0], position)  # 返回物体ID、类别和位置
    
    return center_object

def convert_to_graph_v2(data):

    graph = {}

    # 筛选出semantic label大于40的节点
    filtered_data = {instance: info for instance, info in data.items() if info[0][0] > 40}
    

    center_object_key, center_object_semantic, center_object_pos = find_center_object(data)

    # 找到center节点
    for instance, info in filtered_data.items():
        if center_object_key == instance:
            center_node = (instance, info[1][0])
            break

    if center_node is None:
        raise ValueError("No center object found in the filtered data.")
    
    # 将Display节点作为初始节点加入图，并且设置出度为 1
    graph[center_node] = {}
    
    # 记录已经添加到图中的节点
    added_nodes = {center_node}

    # 创建队列用于存放被指向过的节点
    queue = deque()

    # 寻找Display节点到其他节点的距离并排序
    distances = [(instance, np.array(filtered_data[center_node[0]][2]) - np.array(info[2])) for instance, info in filtered_data.items() if instance != center_node[0]]
    distances.sort(key=lambda x: np.linalg.norm(x[1]))

    # 找到距离最近的三个节点
    nearest_nodes = [(instance, distance) for instance, distance in distances[:5]]

    # 将找到的最近节点加入图中，并设置边的权重为欧式距离
    for nearest_node, distance in nearest_nodes:
        x_distance, y_distance = distance[0], distance[1]
        euclidean_distance = np.linalg.norm(distance)
        node_info = (nearest_node, filtered_data[nearest_node][1][0])
        graph[center_node][node_info] = {'x_distance': x_distance, 'y_distance': y_distance, 'euclidean_distance': euclidean_distance}
        # 将被指向过的节点加入队列
        queue.append(node_info)
        # 更新已经处理过的节点集合和已经添加到图中的节点集合
        added_nodes.add(node_info)

    # 处理队列中的节点，指向未被指向过的节点中与其最近的节点
    while queue:
        #print("queue", queue)
        current_node = queue.popleft()
        current_info = filtered_data[current_node[0]]
        #print("current info", current_info)
        current_position = current_info[2]
        nearest_distance = float('inf')
        nearest_node = None

        for instance, info in filtered_data.items():
            if instance != center_node[0] and (instance, info[1][0]) not in added_nodes:
                distance = np.array(current_position) - np.array(info[2])
                if np.linalg.norm(distance) < nearest_distance:
                    nearest_distance = np.linalg.norm(distance)
                    nearest_xy = distance
                    nearest_node = (instance, info[1][0])
        #print('nearest node', nearest_node, info[2])
        if nearest_node is not None:
            x_distance, y_distance = nearest_xy[0], nearest_xy[1]
            euclidean_distance = np.linalg.norm(nearest_xy)
            graph[current_node] = {nearest_node: {'x_distance': x_distance, 'y_distance': y_distance, 'euclidean_distance': euclidean_distance}}
            queue.append(nearest_node)
            added_nodes.add(nearest_node)

    return graph

def find_array_for_first_key(graph, dict):
    # 获取 graph 的第一个 key
    first_key = next(iter(graph))
    instance_id = first_key[0]
    
    # 从 dict 中找到对应的 array
    if instance_id in dict:
        array_value = dict[instance_id][2]
        return array_value
    else:
        return None

def add_objects_to_scene_v2(scene_pcd_path, scene_npz_path, first_node_position, graph, obj_dataset_folder, scale):
    """
    Add objects to the scene point cloud based on the given first node position and graph.
    And make the new npz file parallelly.

    scene_pcd_path: The path to the scene point cloud in .ply format.
    display_position: The position of the display [x, y, z].
    graph: The graph representing the spatial relationships between objects.
    obj_dataset_folder: The folder containing object point clouds in .ply format.
    scale: scale
    
    Returns:
        The augmented scene point cloud.
    """
    # tiem_dict
    item_dict = {}
    with open('GeoL_net/dataset_gen/classes.txt', 'r') as file:
        for line in file:
            parts = line.strip().split('	')
            num = int(parts[0])
            item_name = parts[1]
            item_dict[num] = item_name    

    # scene pcd
    scene_pcd = remove_all_obj_from_scene(scene_pcd_path, scene_npz_path) # remove semantic label > 40
    obj_all_pcd = o3d.geometry.PointCloud() # empty ply

    # filter npz
    npz_data = np.load(scene_npz_path)
    semantic_labels = npz_data['semantic_label']
    indices_to_keep = np.where(semantic_labels < 40)[0]
    updated_npz = {
        'semantic_label': npz_data['semantic_label'][indices_to_keep],
        'instance_label': npz_data['instance_label'][indices_to_keep],
    }

    

    # 将first node节点的位置作为起点
    first_node_position = np.array(first_node_position)

    # 在 graph中找到 first node label
    first_key = next(iter(graph))
    instance_id = first_key[0]
    first_instance_name = first_key[1]
    first_instance_label = instance_id


    # 存储已经添加过的节点的instance label，防止重复添加
    added_nodes = set()

    # 创建队列用于广度优先搜索
    queue = deque()

    # 将display节点添加到队列中
    queue.append((first_instance_label, first_instance_name, first_node_position))

    # 遍历队列中的每个节点
    instance_label = 100
    while queue:
        # 弹出队列中的节点
        current_instance_label, current_name, current_position = queue.popleft()

        # 检查当前节点是否已经处理过
        if current_instance_label in added_nodes:
            continue

        # 将当前节点标记为已添加
        added_nodes.add(current_instance_label)

        # 检索与当前节点直接相连的邻居节点
        neighbors = graph.get((current_instance_label, current_name), {})
        #print('neighbours', neighbors)

        # 遍历当前节点指向的每个邻居节点
        for neighbor_node, neighbor_info in neighbors.items():
            # 获取相对位置信息
            x_distance = neighbor_info['x_distance']
            y_distance = neighbor_info['y_distance']

            # 计算邻居节点的绝对位置
            neighbor_position = current_position - np.array([x_distance , y_distance , 0.0])

            # 获取邻居节点的物体名称
            obj_name = neighbor_node[1]
            if obj_name == 'remote_control':
                obj_name = 'remote control'
            #print('obj name', neighbor_node[0], obj_name)

            # 检查邻居节点是否已经添加过
            if neighbor_node[0] not in added_nodes:
                # 读取物体点云
                #print('enter in obj name', neighbor_node[0], obj_name, current_position, x_distance, y_distance, neighbor_position)
                random_obj_name = get_similar_random_obj(obj_name)
                #random_obj_name = obj_name
                obj_pcd_path = os.path.join(obj_dataset_folder, f"{random_obj_name}")
                all_files = os.listdir(obj_pcd_path)
                ply_files = [file for file in all_files if file.endswith('.ply')]
                random_ply_file = random.choice(ply_files)
                obj_pcd_file_path = os.path.join(obj_pcd_path, random_ply_file)
            
                obj_pcd = o3d.io.read_point_cloud(obj_pcd_file_path)
                obj_position = get_bottom_centric(obj_pcd)

                #根据两个场景的scale，修正最后的位置
                x_obj = neighbor_position[0]
                y_obj = neighbor_position[1]
                x_display = first_node_position[0]
                y_display = first_node_position[1]

                if x_obj >= x_display:
                   x_obj_new = x_display + scale[2] * (x_obj - x_display)
                else:
                    x_obj_new = x_display + scale[0] * (x_obj - x_display)
                
                if y_obj >= y_display:
                    y_obj_new = y_display + scale[1] * (y_obj - y_display)
                else:
                    y_obj_new = y_display + scale[3] * (y_obj - y_display)


                # 将物体点云平移到对应的位置
                obj_pcd.translate([x_obj_new, y_obj_new, neighbor_position[2]] - obj_position)
                
                # 碰撞检测
                if len(obj_all_pcd.points) > 20:
                    collision_percentage = check_collision(obj_all_pcd, obj_pcd)
                    print("cllision percentate:", collision_percentage)
                    
                    collision_threshold = 5  # 碰撞点云的百分比阈值
                    move_step = 0.05  # 每次移动的步长
                    max_moves = 5  # 最大移动次数
                    moves = 0
                    while collision_percentage > collision_threshold and moves < max_moves:
                        print(f"碰撞点云百分比: {collision_percentage:.2f}% 超过 {collision_threshold}% 阈值，移动物体")

                        # 随机选择一个方向进行移动
                        move_direction = np.random.choice(['x', 'y'])
                        if move_direction == 'x':
                            move_object(obj_pcd, [move_step, 0, 0])
                        else:
                            move_object(obj_pcd, [0, move_step, 0])

                        collision_percentage = check_collision(obj_all_pcd, obj_pcd)
                        moves += 1



                if len(obj_all_pcd.points) <=20 or collision_percentage < collision_threshold:
                    # 将物体点云合并到场景点云中
                    obj_all_pcd += obj_pcd

                    # Update semantic and instance labels in npz
                    for key, val in item_dict.items():
                        if val == obj_name:
                            semantic_label = key
                            break
                    
                    updated_npz['semantic_label'] = np.concatenate((updated_npz['semantic_label'], semantic_label * np.ones(len(obj_pcd.points))))
                    
                    updated_npz['instance_label'] = np.concatenate((updated_npz['instance_label'], instance_label * np.ones(len(obj_pcd.points))))
                    instance_label += 1
                #print(f"add {neighbor_node[0], obj_name}")

                

                # 将邻居节点添加到队列中
                queue.append((neighbor_node[0], obj_name, neighbor_position))
                #print("queue:", queue)
    scene_pcd += obj_all_pcd
    return scene_pcd, updated_npz

def check_collision(scene_pcd, new_pcd, distance_threshold=0.05):
    # 使用KDTree建立场景点云的索引
    scene_kd_tree = o3d.geometry.KDTreeFlann(scene_pcd)

    # 统计碰撞点的数量
    collision_count = 0

    for point in new_pcd.points:
        # 查询每个点的最近邻
        [k, idx, _] = scene_kd_tree.search_knn_vector_3d(point, 1)
        if k > 0:
            # 获取最近邻的距离
            closest_distance = np.linalg.norm(np.asarray(scene_pcd.points)[idx[0]] - point)
            if closest_distance < distance_threshold:
                collision_count += 1

    # 计算碰撞点占新点云总点数的百分比
    collision_percentage = (collision_count / len(new_pcd.points)) * 100
    return collision_percentage

def move_object(pcd, translation):
    pcd.translate(translation)

def swap_keys_object_name(dict_a, dict_b):
    # 获取 dict_a 和 dict_b 的 key（假设每个字典只有一个 key）
    key_a = list(dict_a.keys())[0]
    key_b = list(dict_b.keys())[0]

    # 提取 id 和 obj name
    id_a, obj_name_a = key_a
    id_b, obj_name_b = key_b

    # 构建新的 key
    new_key_a = (id_a, obj_name_b)
    new_key_b = (id_b, obj_name_a)

    # 构建新的字典
    new_dict_a = {new_key_a: dict_a[key_a]}
    new_dict_b = {new_key_b: dict_b[key_b]}

    return new_dict_a, new_dict_b


def replace_entries(graph_a, graph_b, obj_name_list):
    '''
    swap key-values in graph_a and graph_b which have same obj name.
    '''
    for obj_name in obj_name_list:
        for key_a in graph_a.keys():
            if obj_name in key_a:
                pointed_node_a = graph_a[key_a]
            else:
                continue
            for key_b in graph_b.keys():
                if obj_name in key_b:
                    pointed_node_b = graph_b[key_b]
            new_dict_a, new_dict_b = swap_keys_object_name(pointed_node_a, pointed_node_b)

            graph_a[key_a] = new_dict_a
            graph_b[key_b] = new_dict_b


def generate_new_scene_from_demos(id_a, id_b):
        # given 2 scenes
    ply_path_b = f'dataset/data_aug/human_demos/ply/{id_b}.ply'
    npz_path_b = f'dataset/data_aug/human_demos/npz/{id_b}.npz'
    ply_path_a = f'dataset/data_aug/human_demos/ply/{id_a}.ply'
    npz_path_a = f'dataset/data_aug/human_demos/npz/{id_a}.npz'
    # convert to dict and graph
    dict_a = convert_dict(ply_path_a, npz_path_a)
    graph_a = convert_to_graph_v2(dict_a)
    dict_b = convert_dict(ply_path_b, npz_path_b)
    graph_b = convert_to_graph_v2(dict_b)

    #print(graph_a)
    #print(graph_b)

    # merge graphs
    a_name_index = {key[1] for key in graph_a.keys()}
    b_name_index = {key[1] for key in graph_b.keys()}
    set_a = set(a_name_index)
    set_b = set(b_name_index)
    common_elements = set_a.intersection(set_b)
    print("a name idex", a_name_index)
    print("b name idex", b_name_index)
    print("common element:", common_elements)
    replace_entries(graph_a, graph_b, common_elements)
    #print(new_graph_a)

    # detemine the pos of first node
    first_node_pos_a = find_array_for_first_key(graph_a, dict_a) # 找到scene a 中的物体的中心点位置
    first_node_pos_b = find_array_for_first_key(graph_b, dict_b) # 找到scene b 中的物体的中心点位置
    #print('first node pos:', first_node_pos_a)
    #print(find_center_object(dict_id15))

    # get the bbox of all objs
    on_desk_bbox_a = get_on_desk_bbox(ply_path_a, npz_path_a)
    on_desk_bbox_b = get_on_desk_bbox(ply_path_b, npz_path_b)
    #print(on_desk_bbox_a[0])
    #print(on_desk_bbox_b[0])

    # relative pos
    relative_postion_bbox_a = get_relative_position_bbox(bbox_position=on_desk_bbox_a[0], obj_postiion=first_node_pos_a) # scene a 中心点相对位置
    relative_postion_bbox_b = get_relative_position_bbox(bbox_position=on_desk_bbox_b[0], obj_postiion=first_node_pos_b) # scene b 中心点相对位置
    scale = [a / b for a, b in zip(relative_postion_bbox_a, relative_postion_bbox_b)]

    # generate new scene ply and npz
    new_scene, new_npz = add_objects_to_scene_v2(scene_pcd_path=ply_path_a, 
                                        scene_npz_path=npz_path_a,
                                        first_node_position=first_node_pos_a, 
                                        graph=graph_b, 
                                        obj_dataset_folder= "dataset/obj/ply",
                                        scale = scale)
    # save ply and npz
    id_a = ply_path_a.split('/')[-1].split('.')[0]
    id_b = ply_path_b.split('/')[-1].split('.')[0]
    output_ply = "dataset/data_aug/generated_data/ply"
    output_npz = "dataset/data_aug/generated_data/npz"
    output_ply_path = os.path.join(output_ply, f"{id_a}_{id_b}.ply")
    output_npz_path = os.path.join(output_npz, f"{id_a}_{id_b}.npz")
    output_ply_path = get_unique_filename(directory=output_ply, filename=f"{id_a}_{id_b}", extension=".ply")
    output_npz_path = get_unique_filename(directory=output_npz, filename=f"{id_a}_{id_b}", extension=".npz")

    o3d.io.write_point_cloud(output_ply_path, new_scene)
    np.savez(output_npz_path, semantic_label=new_npz['semantic_label'], instance_label=new_npz['instance_label'])

    print(f"Point cloud saved to {output_ply_path}")


if __name__ == "__main__":
    # ply folder
    ply_folder = "dataset/data_aug/human_demos/ply"
    id_list = []
    amount_scene = 0
    # Iterate through all .ply files in the ply folder
    for root, dirs, files in os.walk(ply_folder):
        for file in files:
            if file.endswith(".ply"):
                file_path = os.path.join(root, file)
                print(f"Found PLY file: {file_path}")
                id_list.append(file.split('.')[0])
    for id_a in id_list:
        for id_b in id_list:
            if id_a != id_b:
                if os.path.exists(f"dataset/data_aug/generated_data/ply/{id_a}_{id_b}_0.ply"): # no repeat _0
                    continue
                generate_new_scene_from_demos(id_a, id_b)
                print(f"generate new scene from {id_a} and {id_b}")
                amount_scene += 1
                print(f"amount_scene: {amount_scene}")

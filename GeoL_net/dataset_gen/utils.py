import open3d as o3d
import trimesh
import numpy as np
from collections import deque
import os
import random
import json
from GeoL_net.dataset_gen.solver_scene_generate import *
from trimesh.collision import CollisionManager




def get_unique_filename(directory, filename, extension):
    """
    Check if a filename is unique in a specified directory. If the file exists, add a numerical suffix to the filename
    """
    base_name = filename
    counter = 1
    unique_filename = f"{base_name}{extension}"
    unique_file_path = os.path.join(directory, unique_filename)
    
    while os.path.exists(unique_file_path):
        unique_filename = f"{base_name}_{counter}{extension}"
        unique_file_path = os.path.join(directory, unique_filename)
        counter += 1
    
    return unique_file_path

def get_bottom_centric(pcd):
    """get the bottom centric point of a obj pcd

    pcd: point cloud
    return: the bottom centric location of the pcd, [x,y,z]
    """

    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    bottom_centric_point = np.array([(min_bound[0] + max_bound[0]) / 2,
                                     (min_bound[1] + max_bound[1]) / 2,
                                     min_bound[2]])
    #print("min", min_bound)
    #print("max", max_bound)
    #print('bottom centric point', bottom_centric_point)
    return bottom_centric_point

def remove_obj_from_scene(pcd_ply_path, obj_index:int, npz_path):
    """
    remove the obj from the pcd

    pcd: the scene pcd
    obj_index: the index of the obj
    return: the scene pcd without the obj
    """
    data = np.load(npz_path)
    semantic_labels = data['semantic_label']
    instance_labels = data['instance_label']
    point_cloud = o3d.io.read_point_cloud(pcd_ply_path)
    ins_list = []

    for i in range(len(semantic_labels)):
        if semantic_labels[i] == obj_index and instance_labels[i] not in ins_list:
            ins_list.append(instance_labels[i])

    if len(ins_list) == 0:
        print('No such obj in the scene')
        return point_cloud
        
    ins_label = ins_list[0] # reomve the first instance
    label_indices = np.where((semantic_labels == obj_index) & (instance_labels == ins_label))[0]
    filtered_points = np.delete(np.asarray(point_cloud.points), label_indices, axis=0)
    filtered_colors = np.delete(np.asarray(point_cloud.colors), label_indices, axis=0)
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_point_cloud

def remove_all_obj_from_scene(pcd_ply_path, npz_path):
    """
    remove the obj from the pcd

    pcd: the scene pcd
    obj_index: the index of the obj
    return: the scene pcd without the obj
    """
    data = np.load(npz_path)
    points = data['xyz']
    semantic_labels = data['semantic_label']
    instance_labels = data['instance_label']
    point_cloud = o3d.io.read_point_cloud(pcd_ply_path)

    label_indices = np.where((semantic_labels >= 40))[0]
    filtered_points = np.delete(np.asarray(point_cloud.points), label_indices, axis=0)
    filtered_colors = np.delete(np.asarray(point_cloud.colors), label_indices, axis=0)
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_point_cloud


def get_obj_from_scene(pcd_ply_path, obj_index:int, npz_path):
    """get the obj from the scene pcd

    pcd_ply_path: the scene pcd path
    obj_index: the index of the obj
    npz path: npz_path
    return: the obj pcd, if obj does not exist, return None
    """

    data = np.load(npz_path)
    points = data['xyz']
    semantic_labels = data['semantic_label']
    instance_labels = data['instance_label']
    
    ins_list = []
    for i in range(len(semantic_labels)):
        if semantic_labels[i] == obj_index and instance_labels[i] not in ins_list:
            ins_list.append(instance_labels[i])
    if len(ins_list) == 0:
        return None
    ins_label = ins_list[0] # reomve the first instance

    removed_indices = np.where((semantic_labels != obj_index)|(instance_labels != ins_label))[0]

    point_cloud = o3d.io.read_point_cloud(pcd_ply_path)
    removed_points = np.delete(np.asarray(point_cloud.points), removed_indices, axis=0)
    removed_colors = np.delete(np.asarray(point_cloud.colors), removed_indices, axis=0)
    removed_point_cloud = o3d.geometry.PointCloud()
    removed_point_cloud.points = o3d.utility.Vector3dVector(removed_points)
    removed_point_cloud.colors = o3d.utility.Vector3dVector(removed_colors)

    return removed_point_cloud

def get_obj_from_scene_inslabel(pcd_ply_path, ins_index:int, npz_path):
    """get the obj from the scene pcd, through instance label

    pcd_ply_path: the scene pcd path
    ins_index: the index of the instance
    npz path: npz_path
    return: the obj pcd
    """

    
    data = np.load(npz_path)
    points = data['xyz']
    semantic_labels = data['semantic_label']
    instance_labels = data['instance_label']

    removed_indices = np.where((instance_labels != ins_index))[0]

    point_cloud = o3d.io.read_point_cloud(pcd_ply_path)
    removed_points = np.delete(np.asarray(point_cloud.points), removed_indices, axis=0)
    removed_colors = np.delete(np.asarray(point_cloud.colors), removed_indices, axis=0)
    removed_point_cloud = o3d.geometry.PointCloud()
    removed_point_cloud.points = o3d.utility.Vector3dVector(removed_points)
    removed_point_cloud.colors = o3d.utility.Vector3dVector(removed_colors)

    return removed_point_cloud

def move_obj_in_scene(pcd_ply_path, obj_index:int, npz_path, direct="xy",scale=0.5):
    data = np.load(npz_path)
    points = data['xyz']
    semantic_labels = data['semantic_label']
    instance_labels = data['instance_label']
    point_cloud = o3d.io.read_point_cloud(pcd_ply_path)
    ins_list = []
    for i in range(len(semantic_labels)):
        if semantic_labels[i] == obj_index and instance_labels[i] not in ins_list:
            ins_list.append(instance_labels[i])
        #print(ins_list)
        # if the target obejct does not exis in the scene, return 0
    if len(ins_list) == 0:
        print('No such obj in the scene')
        return point_cloud
        
    ins_label = ins_list[0] # reomve the first instance
    label_indices = np.where((semantic_labels == obj_index) & (instance_labels == ins_label))[0]

    # 计算物体的边界框
    obj_points = points[label_indices]
    min_bound = np.min(obj_points, axis=0)
    max_bound = np.max(obj_points, axis=0)
    
    # 添加边界框
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    # 计算移动距离
    
    move_distance = np.abs(max_bound - min_bound) * scale

    if direct == "xy":
        for idx in label_indices:
            points[idx][0] += move_distance[0] # x red
            print('dis x:', move_distance[0])
            points[idx][1] += move_distance[1] # y green
            print('dis y:', move_distance[1])
            #points[idx][2] += move_distance[2] # z blue
            #print('dis z:', move_distance[2])
        
    # 更新点云数据
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud

def rotate_point_cloud(pcd_ply_path, angle_deg, aixs='z'):
    """Rotate the point cloud around Z axis.

    Args:
        pcd: The input point cloud.
        angle_deg: The rotation angle in degrees.
    
    Returns:
        The rotated point cloud.
    """
    # Convert angle from degrees to radians
    pcd = o3d.io.read_point_cloud(pcd_ply_path)
    angle_rad = np.radians(angle_deg)
    
    # Define rotation matrix around Z axis
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                  [np.sin(angle_rad), np.cos(angle_rad), 0],
                  [0, 0, 1]])
    
    # Apply rotation to point cloud
    pcd.rotate(R, center=(0, 0, 0))
    
    return pcd

def exchange_objects(scene1_ply_path, scene2_ply_path, scene1_npz_path, scene2_npz_path, obj1_index, obj2_index):
    # 从第一个场景中移除显示器，并记录其最低点的中心位置
    scene1_pcd = o3d.io.read_point_cloud(scene1_ply_path)
    #monitor_centric_lowest_point = get_bottom_centric(scene1_pcd)
    scene1_without_obj1 = remove_obj_from_scene(scene1_ply_path, obj_index=obj1_index, npz_path=scene1_npz_path)
    obj_1 = get_obj_from_scene(scene1_ply_path, obj1_index, scene1_npz_path)
    obj_1_centric_lowest = get_bottom_centric(obj_1)

    # 从第二个场景中移除笔记本电脑，并记录其最低点的中心位置
    scene2_pcd = o3d.io.read_point_cloud(scene2_ply_path)
    #laptop_centric_lowest_point = get_bottom_centric(scene2_pcd)
    scene2_without_obj2 = remove_obj_from_scene(scene2_ply_path, obj_index=obj2_index, npz_path=scene2_npz_path) 
    obj_2 = get_obj_from_scene(scene2_ply_path, obj2_index, scene2_npz_path)
    obj_2_centric_lowest = get_bottom_centric(obj_2)

    # 将笔记本电脑放置到第一个场景中记录的显示器最低点的中心位置上
    
    obj_2.translate(obj_1_centric_lowest - obj_2_centric_lowest)

    # 将显示器放置到第二个场景中记录的笔记本电脑最低点的中心位置上
    
    obj_1.translate(obj_2_centric_lowest - obj_1_centric_lowest)

    # 保存新的场景
    o3d.io.write_point_cloud("new_scene1_with_laptop.ply", scene1_without_obj1 + obj_2)
    o3d.io.write_point_cloud("new_scene2_with_monitor.ply", scene2_without_obj2 + obj_1)

def convert_dict(pcd_ply_path, npz_path) -> dict:
    """Convert the instance on the surface to a dict

    Args:
        pcd_ply_path: scene pcd path.
        npz_path: npz path.
    
    Returns:
        dist e.g.
        instances = {
            'laptop': {'position': (xl, yl, zl), 'category': 'laptop'},
            'instance1': {'position': (x1, y1, z1), 'category': 'category1'},
            'instance2': {'position': (x2, y2, z2), 'category': 'category2'},
            ...
        }.    
    """
    data = np.load(npz_path)
    semantic_labels = data['semantic_label']
    instance_labels = data['instance_label']

    # 创建字典用于存储实例和语义标签的映射关系
    instance_semantic_dict = {}

    # 遍历每个点的语义标签和实例标签
    for semantic_label, instance_label in zip(semantic_labels, instance_labels):
        # 如果实例标签不为-1，则将其加入字典
        if instance_label != -1:
            # 将语义标签与实例标签关联
            if instance_label not in instance_semantic_dict:
                instance_semantic_dict[instance_label] = set()
            instance_semantic_dict[instance_label].add(semantic_label)
    # 去除字典中重复的键值对
    instance_semantic_dict = {instance: list(semantic_labels) for instance, semantic_labels in instance_semantic_dict.items()}

    # 按照键从小到大排序字典
    instance_semantic_dict = dict(sorted(instance_semantic_dict.items()))

    # index -- object name
    item_dict = {}
    with open('scripts/dataset_gen/classes.txt', 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除行尾的换行符，并将每行内容按空格分割成数字和物品名
            parts = line.strip().split('	')
            # 将第一个部分转换为整数作为数字
            num = int(parts[0])
            # 将第二个部分作为物品名
            item_name = parts[1]
            # 将数字和物品名添加到字典中
            item_dict[num] = item_name    
    

    # 将实例映射到语义标签和物体名称的列表
    instance_label_to_semantic_name_pos = {}

    for instance_label, semantic_labels in instance_semantic_dict.items():
        # 获取每个实例对应的物体名称列表
        names = [item_dict.get(semantic_label) for semantic_label in semantic_labels if semantic_label in item_dict]
        # 如果没有对应的物体名称，则跳过该实例
        if not names:
            continue
        obj_index = semantic_labels[0]
        obj_pcd = get_obj_from_scene_inslabel(pcd_ply_path=pcd_ply_path, ins_index=instance_label, npz_path=npz_path)
        bottom_centric = get_bottom_centric(obj_pcd)
        # 将实例标签映射到语义标签和物体名称的列表
        instance_label_to_semantic_name_pos[instance_label] = [semantic_labels, names, bottom_centric]
    
    return instance_label_to_semantic_name_pos


def calculate_distance(instance1, instance2):
    """
    instance1: position of instance1
    insrance2: position of instance2
    return: x_distance, y_distance, euclidean_distance
    """
    # 提取位置信息
    x1, y1, _ = instance1
    x2, y2, _ = instance2

    # 计算 x 方向和 y 方向的距离
    x_dist = x2 - x1
    y_dist = y2 - y1

    # 计算欧式距离
    euclidean_distance = np.linalg.norm(np.array(instance1) - np.array(instance2))

    return x_dist, y_dist, euclidean_distance



def keep_obj_in_scene(pcd_ply_path, obj_index: int, npz_path):
    """
    Keep only the object with specified index and points with semantic label < 40 in the scene point cloud.

    pcd_ply_path: The path to the scene point cloud in .ply format.
    obj_index: The index of the object to be kept.
    npz_path: The path to the .npz file containing semantic labels.
    
    Returns:
        A tuple containing:
        - A point cloud containing only the specified object and points with semantic label < 40.
        - The bottom centric location of the specified object point cloud.
    
    """
    data = np.load(npz_path)
    points = data['xyz']
    semantic_labels = data['semantic_label']
    point_cloud = o3d.io.read_point_cloud(pcd_ply_path)
    
    obj_indices = np.where((semantic_labels == obj_index) | (semantic_labels < 40))[0]
    
    # If the target object does not exist in the scene, return None
    if len(obj_indices) == 0:
        print('No such obj in the scene')
        return None, None
    
    obj_points = np.asarray(point_cloud.points)[obj_indices]
    obj_colors = np.asarray(point_cloud.colors)[obj_indices]

    kept_point_cloud = o3d.geometry.PointCloud()
    kept_point_cloud.points = o3d.utility.Vector3dVector(obj_points)
    kept_point_cloud.colors = o3d.utility.Vector3dVector(obj_colors)
    
    objpcd = get_obj_from_scene(pcd_ply_path, obj_index, npz_path)
    bottom_centric_point = get_bottom_centric(objpcd)


    return kept_point_cloud, bottom_centric_point

def add_objects_to_scene(scene_pcd_path, display_position, graph, obj_dataset_folder, scale):
    """
    Add objects to the scene point cloud based on the given display position and graph.

    scene_pcd_path: The path to the scene point cloud in .ply format.
    display_position: The position of the display [x, y, z].
    graph: The graph representing the spatial relationships between objects.
    obj_dataset_folder: The folder containing object point clouds in .ply format.
    
    Returns:
        The augmented scene point cloud.
    """
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_path)

    # 将display节点的位置作为起点
    display_position = np.array(display_position)

    # 在 graph中找到 display的instance label
    display_instance_label = None
    for node, neighbors in graph.items():
        #print("node1", node[1])
        if node[1] == 'display':
            display_instance_label = node[0]
            break

    if display_instance_label is None:
        raise ValueError("Display instance label not found in the graph.")

    # 存储已经添加过的节点的instance label，防止重复添加
    added_nodes = set()

    # 创建队列用于广度优先搜索
    queue = deque()

    # 将display节点添加到队列中
    queue.append((display_instance_label, 'display', display_position))

    # 遍历队列中的每个节点
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
            #print('obj name', neighbor_node[0], obj_name)

            # 检查邻居节点是否已经添加过
            if neighbor_node[0] not in added_nodes:
                # 读取物体点云
                #print('enter in obj name', neighbor_node[0], obj_name, current_position, x_distance, y_distance, neighbor_position)
                random_obj_name = get_similar_random_obj(obj_name)
                #random_obj_name = obj_name
                obj_pcd_path = os.path.join(obj_dataset_folder, f"{random_obj_name}.ply")
            
                obj_pcd = o3d.io.read_point_cloud(obj_pcd_path)
                obj_position = get_bottom_centric(obj_pcd)

                #根据两个场景的scale，修正最后的位置
                x_obj = neighbor_position[0]
                y_obj = neighbor_position[1]
                x_display = display_position[0]
                y_display = display_position[1]

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

                # 将物体点云合并到场景点云中
                scene_pcd += obj_pcd
                #print(f"add {neighbor_node[0], obj_name}")

                

                # 将邻居节点添加到队列中
                queue.append((neighbor_node[0], obj_name, neighbor_position))
                #print("queue:", queue)

    return scene_pcd


def convert_to_graph(data):
    graph = {}
    display_node = None

    # 筛选出semantic label大于40的节点
    filtered_data = {instance: info for instance, info in data.items() if info[0][0] > 40}

    # 找到Display节点
    for instance, info in filtered_data.items():
        if 'display' in info[1]:
            display_node = (instance, info[1][0])
            break

    if display_node is None:
        raise ValueError("No Display node found in the filtered data.")

    # 将Display节点作为初始节点加入图，并且设置出度为 1
    graph[display_node] = {}
    
    # 记录已经添加到图中的节点
    added_nodes = {display_node}

    # 创建队列用于存放被指向过的节点
    queue = deque()

    # 寻找Display节点到其他节点的距离并排序
    distances = [(instance, np.array(filtered_data[display_node[0]][2]) - np.array(info[2])) for instance, info in filtered_data.items() if instance != display_node[0]]
    distances.sort(key=lambda x: np.linalg.norm(x[1]))

    # 找到距离最近的三个节点
    nearest_nodes = [(instance, distance) for instance, distance in distances[:5]]

    # 将找到的最近节点加入图中，并设置边的权重为欧式距离
    for nearest_node, distance in nearest_nodes:
        x_distance, y_distance = distance[0], distance[1]
        euclidean_distance = np.linalg.norm(distance)
        node_info = (nearest_node, filtered_data[nearest_node][1][0])
        graph[display_node][node_info] = {'x_distance': x_distance, 'y_distance': y_distance, 'euclidean_distance': euclidean_distance}
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
            if instance != display_node[0] and (instance, info[1][0]) not in added_nodes:
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

def get_on_desk_bbox(pcd_ply_path,  npz_path):
    """get a boundongbox containing all objects on the table and its xy-area

    pcd_ply_path: the scene pcd path
    npz path: npz_path
    return: the bbox position
    """

    data = np.load(npz_path)
    points = data['xyz']
    semantic_labels = data['semantic_label']

    removed_indices = np.where((semantic_labels <= 40))[0]

    point_cloud = o3d.io.read_point_cloud(pcd_ply_path)
    removed_points = np.delete(np.asarray(point_cloud.points), removed_indices, axis=0)
    removed_colors = np.delete(np.asarray(point_cloud.colors), removed_indices, axis=0)
    removed_point_cloud = o3d.geometry.PointCloud()
    removed_point_cloud.points = o3d.utility.Vector3dVector(removed_points)
    removed_point_cloud.colors = o3d.utility.Vector3dVector(removed_colors)
    bbox = removed_point_cloud.get_axis_aligned_bounding_box()

    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    xy_area = abs(min_bound[0] - max_bound[0]) * abs(min_bound[1] - max_bound[1])

    
    return [min_bound, max_bound], xy_area

def get_relative_position_bbox(bbox_position, obj_postiion):
    """Get the obj position relative to bbox position
    """
    bbox_min = bbox_position[0]
    bbox_max = bbox_position[1]
    xmin = bbox_min[0]
    ymin = bbox_min[1]
    xmax = bbox_max[0]
    ymax = bbox_max[1]
    s1 = abs(obj_postiion[0] - xmin)
    s2 = abs(obj_postiion[1] - ymax)
    s3 = abs(obj_postiion[0] - xmax)
    s4 = abs(obj_postiion[1] - ymin)
    return [s1, s2, s3, s4]

def get_similar_random_obj(name):
    """get a similar obj randomly
    name: the name of the obj
    return: pcd of a similar obj
    """
    group_can = ['mug', 'can','bowl','bottle', 'jar']
    group_elec = ['earphone', 'phone', 'calculator', 'charger', 'remote control']
    group_book = ['book','eraser','notebook', 'pencil','ruler']
    group_big = ['lamp', 'plant','vase']
    group_hat = ['hat', 'cap', 'eye_glasses']
    groups = [group_can, group_big, group_hat, group_elec, group_book]
    for group in groups:
        if name in group:
            name = random.choice(group)
            break
    return name


def rotate_mesh(mesh, axis, angle_deg):
    """
    旋转网格

    参数:
        mesh: 要旋转的网格对象 (o3d.geometry.TriangleMesh)
        axis: 旋转轴，一个包含三个值的向量，例如 [1, 0, 0] 表示绕 X 轴旋转
        angle_deg: 旋转角度，单位为度
    返回值:
        旋转后的网格对象 (o3d.geometry.TriangleMesh)
    """
    # 将角度转换为弧度
    angle_rad = np.radians(angle_deg)

    # 创建旋转变换矩阵
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis, angle_rad)

    # 应用旋转变换矩阵到网格顶点
    mesh.rotate(rotation_matrix, center=mesh.get_center())

    return mesh


def bbox_pos_scale_all_obj(pcd_ply_path, npz_path):
    """
    calculate the bbox pos and bbox scale of all obj(semantic label > 40)
    
    Parameters:
    pcd_ply_path (str): Path to the input PLY file.
    npz_path (str): Path to the NPZ file containing 'xyz', 'semantic_label', and 'instance_label'.

    Returns:
    dict_bbox_pos_scale: a dict contain the instance label : [semantic label, bbox pos, bbox scale] 
    """
    # Load NPZ file data
    data = np.load(npz_path)
    #points = data['xyz']
    semantic_labels = data['semantic_label']
    instance_labels = data['instance_label']
    
    # Read point cloud from PLY file
    point_cloud = o3d.io.read_point_cloud(pcd_ply_path)
    # Initialize a dictionary to store bounding box information
    dict_bbox_pos_scale = {}
    
    # Find unique semantic labels greater than 40
    unique_labels = np.unique(semantic_labels[semantic_labels > 40])
    
    for label in unique_labels:
        # Get instance labels for the current semantic label
        instance_mask = (semantic_labels == label)
        instance_ids = np.unique(instance_labels[instance_mask])
        
        for ins_id in instance_ids:
            # Find points corresponding to the current instance
            obj = get_obj_from_scene_inslabel(pcd_ply_path=pcd_ply_path, ins_index=ins_id, npz_path=npz_path)
            aabb = obj.get_axis_aligned_bounding_box()
            # Compute bounding box
            min_bound = aabb.min_bound
            max_bound = aabb.max_bound
            size = max_bound - min_bound
            
            # Store the bounding box information
            dict_bbox_pos_scale[int(ins_id)] = {
                "semantic_label": int(label),
                "min_bound": min_bound.tolist(),
                "max_bound": max_bound.tolist(),
                "size": size.tolist()
            }
    
    return dict_bbox_pos_scale

def bbox_pos_scale_all_scene(pcd_ply_path, npz_path):
    """
    Get position, semnatic label of the whole scene
    
    Parameters:
    pcd_ply_path (str): Path to the input PLY file.
    npz_path (str): Path to the NPZ file containing 'xyz', 'semantic_label', and 'instance_label'.

    
    Returns:
    dict_bbox_pos_scale: a dict contain the instance label : [semantic label, bbox pos, bbox scale] 
    """
    # Load NPZ file data
    data = np.load(npz_path)
    points = data['xyz']
    semantic_labels = data['semantic_label']
    instance_labels = data['instance_label']
    
    # Read point cloud from PLY file
    point_cloud = o3d.io.read_point_cloud(pcd_ply_path)
    # Initialize a dictionary to store bounding box information
    dict_bbox_pos_scale = {}
    
    # Find unique semantic labels greater than 40
    unique_labels = np.unique(semantic_labels[semantic_labels > 1])
    
    for label in unique_labels:
        # Get instance labels for the current semantic label
        instance_mask = (semantic_labels == label)
        instance_ids = np.unique(instance_labels[instance_mask])
        
        for ins_id in instance_ids:
            # Find points corresponding to the current instance
            label_indices = np.where((semantic_labels == label) & (instance_labels == ins_id))[0]
            points_of_instance = points[label_indices]
            
            # Compute bounding box
            min_bound = points_of_instance.min(axis=0)
            max_bound = points_of_instance.max(axis=0)
            size = max_bound - min_bound
            
            # Store the bounding box information
            dict_bbox_pos_scale[int(ins_id)] = {
                "semantic_label": int(label),
                "min_bound": min_bound.tolist(),
                "max_bound": max_bound.tolist(),
                "size": size.tolist()
            }
    
    # Remove points with semantic labels greater than 40
    remove_mask = semantic_labels > 40
    filtered_points = points[~remove_mask]
    
    # Create a filtered point cloud
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    
    return dict_bbox_pos_scale

def generate_mesh_scene(ply_path, npz_path):
    """
    Remove objects with semantic labels greater than 40 from the point cloud scene and replace them with mesh obj from ShapNet.
    
    Parameters:
    pcd_ply_path (str): Path to the input PLY file.
    npz_path (str): Path to the NPZ file containing 'xyz', 'semantic_label', and 'instance_label'.

    
    Returns:
    mesh scene 
    """
    filtered_scene = remove_all_obj_from_scene(ply_path, npz_path)
    print('---------')
    print('Remmove all objs...')
    filtered_scene.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    filtered_scene.orient_normals_consistent_tangent_plane(k=30)
    filtered_scene_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(filtered_scene, depth=9)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    filtered_scene_mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh_scene = filtered_scene_mesh
    print("---------")
    print("Creat mesh scene...")
    dict_bbox_pos_sacle = bbox_pos_scale_all_obj(ply_path, npz_path)

    for ins_label, item_instance in dict_bbox_pos_sacle.items():
        obj_exist = 0
        #print('item instance', item_instance)
        item_semantic_label = item_instance['semantic_label']
        item_bbox_min_bound = item_instance['min_bound']
        item_bbox_max_bound = item_instance['max_bound']
        item_bbox_size = item_instance['size']
        item_bbox_center = [0,0,0] # initialize center
        item_bbox_center[0] = (item_bbox_min_bound[0] + item_bbox_max_bound[0]) / 2
        item_bbox_center[1] = (item_bbox_min_bound[1] + item_bbox_max_bound[1]) / 2
        item_bbox_center[2] = (item_bbox_min_bound[2] + item_bbox_max_bound[2]) / 2
        item_bbox_pos = item_bbox_center
        item_bbox_pos[2] = item_bbox_pos[2] - item_bbox_size[2] / 2
        print('item bbox pos', item_bbox_pos)
        # convert to semantic label to keyword
            # index -- object name
        item_dict = {}
        with open('code/classes.txt', 'r') as file:
            for line in file:
                parts = line.strip().split('	')
                num = int(parts[0])
                item_name = parts[1]
                item_dict[num] = item_name    
        item_keyword = item_dict[item_semantic_label]

        # get the mesh obj 
        json_path = '/home/stud/zhoy/storage/group/dataset_mirrors/ShapeNetV2/ShapeNetCore.v2/taxonomy.json'
        with open(json_path,  'r', encoding='utf-8') as f:
            json_content = json.load(f)
        #print('json content:',json_content[0])
        keyword = item_keyword # maybe add mapping
        for item in json_content:
            if keyword in item['name'] and (keyword=='laptop' or keyword =='lamp' or keyword =='mug'):
                obj_exist = 1
                print('-----------')
                print(f"{item['name']}: {item['synsetId']}")
                directory = os.path.join('/home/stud/zhoy/storage/group/dataset_mirrors/ShapeNetV2/ShapeNetCore.v2', f"{item['synsetId']}")
                try:
                    items_file = os.listdir(directory)
                except FileNotFoundError:
                    print("No such file or directory.")
                    continue
                subdirectories = [item for item in items_file if os.path.isdir(os.path.join(directory, item))]
                first_subdirectory = os.path.join(directory, subdirectories[0])
                print("first subdirectory:", first_subdirectory)
                for _, _, files in os.walk(first_subdirectory):
                    for file in files:
                        print("file:", file)
                        if file.endswith('.obj'):
                            obj_path = os.path.join(first_subdirectory, f"models/{file}")
                            mesh_obj = o3d.io.read_triangle_mesh(obj_path)
                            #o3d.io.write_triangle_mesh(f"{keyword}.obj", mesh_obj)
                break
        if obj_exist == 0:
            print(f'sorry we dont have {keyword}')
            continue
        # after getting the obj mesh, replace the pcd obj
        angle_deg_y = 180
        angle_rad_y = np.radians(angle_deg_y)
        angle_deg_x = 90
        angle_rad_x = np.radians(angle_deg_x)
        rotation_vector = np.array([0, angle_rad_y, 0])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
        mesh_obj.rotate(rotation_matrix, center=mesh_obj.get_center())
        rotation_vector = np.array([angle_rad_x, 0, 0])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
        mesh_obj.rotate(rotation_matrix, center=mesh_obj.get_center())
        
        # bbox
        aabb = mesh_obj.get_axis_aligned_bounding_box()
        extent1 = item_bbox_size
        extent2 = aabb.get_extent()
        scale_factors = extent1 / extent2
        vertices2 = np.asarray(mesh_obj.vertices)
        #vertices2 *= scale_factors
        vertices2 *= scale_factors.mean()
        mesh_obj.vertices = o3d.utility.Vector3dVector(vertices2)
        aabb2_scaled = mesh_obj.get_axis_aligned_bounding_box()
        extent2_scaled = aabb2_scaled.get_extent()
        aabb_center = aabb2_scaled.get_center()
        aabb_bottom_center = np.array(aabb_center)
        aabb_bottom_center[2] -= extent2_scaled[2] / 2.0  # 减去 Z 方向上的一半尺寸  
        translation = item_bbox_pos - aabb_bottom_center
        #mesh_obj.translate(translation)

        # optimize the z-angle
        pcd = get_obj_from_scene_inslabel(pcd_ply_path=ply_path, ins_index=ins_label, npz_path=npz_path)
        # Sample points randomly from the point cloud
        sampled_indices = np.random.choice(len(pcd.points), 512, replace=True)
        sampled_pcd_points = np.asarray(pcd.points)[sampled_indices]
        sampled_mesh_points = mesh_obj.sample_points_uniformly(number_of_points=512)
        sampled_mesh_points = np.asarray(sampled_mesh_points.points)
        optimezed_z = optimize_rotation([sampled_mesh_points], [sampled_pcd_points], initial_angle=-10, num_iterations=1000, learning_rate=0.1)

        angle_deg_z = optimezed_z.detach().numpy()
        angle_rad_z = np.radians(angle_deg_z)
        print("angle deg z", angle_deg_z)
        print("angle rad z", angle_rad_z)
        rotation_vector = np.array([0, 0, angle_rad_z])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
        mesh_obj.rotate(rotation_matrix, center=mesh_obj.get_center())
        mesh_obj.translate(translation)
        mesh_scene = mesh_obj + mesh_scene
        print(f"Great! {keyword} is done")
    o3d.io.write_triangle_mesh("align_test_all_optimized.obj", mesh_scene)


def generate_mesh_scene_all(ply_path, npz_path):
    """
    Remove objects with semantic labels greater than 40 from the point cloud scene and replace them with mesh obj from ShapNet.
    Reomvoe the table and replace it with mesh obj from ModelNet.
    
    Parameters:
    pcd_ply_path (str): Path to the input PLY file.
    npz_path (str): Path to the NPZ file containing 'xyz', 'semantic_label', and 'instance_label'.

    
    Returns:
    mesh scene 
    """
    # color dict
    color_dict = {
        'red': [1, 0, 0],
        'green': [0, 1, 0],
        'blue': [0, 0, 1],
        'yellow': [1, 1, 0]
        }
    # get the size of the desk pcd

    #desk_pcd= get_obj_from_scene(pcd_ply_path=ply_path, obj_index=7, npz_path=npz_path)
    desk_pcd= get_obj_from_scene(pcd_ply_path=ply_path, obj_index=14, npz_path=npz_path)
    aabb_desk_pcd = desk_pcd.get_axis_aligned_bounding_box()
    min_bound = aabb_desk_pcd.get_min_bound()
    max_bound = aabb_desk_pcd.get_max_bound()
    dimensions_desk_pcd = max_bound - min_bound
    center_desk_pcd = (min_bound + max_bound) / 2.0
    top_center_desk_pcd = np.array(center_desk_pcd)
    top_center_desk_pcd[2] += dimensions_desk_pcd[2] / 2.0

    #get the position of the desk pcd
    #get the desk mesh and align the size and bottom centric
    desk_mesh_folder_path = 'mesh_dataset/raplace_mesh/desk'
    off_files = [f for f in os.listdir(desk_mesh_folder_path) if f.endswith('.off')]
    random_file = random.choice(off_files)
    desk_mesh_file_path = os.path.join(desk_mesh_folder_path, random_file)
    desk_mesh =  o3d.io.read_triangle_mesh(desk_mesh_file_path)


    # optimize the z
    sampled_indices_desk = np.random.choice(len(desk_pcd.points), 512, replace=True)
    sampled_pcd_points_desk = np.asarray(desk_pcd.points)[sampled_indices_desk]
    sampled_mesh_points_desk = desk_mesh.sample_points_uniformly(number_of_points=512)
    sampled_mesh_points_desk = np.asarray(sampled_mesh_points_desk.points)
    optimezed_z_desk = optimize_rotation([sampled_mesh_points_desk], [sampled_pcd_points_desk], initial_angle=0, num_iterations=10, learning_rate=50)
    angle_deg_z_desk = optimezed_z_desk.detach().numpy()
    angle_deg_z_desk = np.round(angle_deg_z_desk / 90) * 90
    angle_rad_z_desk = np.radians(angle_deg_z_desk)
    rotation_vector = np.array([0, 0, angle_rad_z_desk])
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
    desk_mesh.rotate(rotation_matrix, center=desk_mesh.get_center())


    # scale
    aabb_desk_mesh = desk_mesh.get_axis_aligned_bounding_box()
    min_bound = aabb_desk_mesh.get_min_bound()
    max_bound = aabb_desk_mesh.get_max_bound()
    dimensions_desk_mesh = max_bound - min_bound 

    center_desk_mesh = (min_bound + max_bound) / 2.0
    top_center_desk_mesh = np.array(center_desk_mesh)
    top_center_desk_mesh[2] += dimensions_desk_mesh[2] / 2.0

    # scale desk mesh
    scaling_factors = dimensions_desk_pcd / dimensions_desk_mesh
    desk_mesh_vertices = np.asarray(desk_mesh.vertices)
    desk_mesh_center = desk_mesh.get_center()
    #scaled_desk_vertices = (desk_mesh_vertices - desk_mesh_center) * scaling_factors + desk_mesh_center
    scaled_desk_vertices = desk_mesh_vertices * scaling_factors
    desk_mesh.vertices = o3d.utility.Vector3dVector(scaled_desk_vertices)
    desk_mesh.compute_vertex_normals()
    aabb_desk_mesh = desk_mesh.get_axis_aligned_bounding_box()
    min_bound = aabb_desk_mesh.get_min_bound()
    max_bound = aabb_desk_mesh.get_max_bound()
    dimensions_desk_mesh = max_bound - min_bound 

    center_desk_mesh = (min_bound + max_bound) / 2.0
    top_center_desk_mesh = np.array(center_desk_mesh)
    top_center_desk_mesh[2] += dimensions_desk_mesh[2] / 2.0



    #translation = top_center_desk_pcd - top_center_desk_mesh
    # create an empty mesh
    #desk_mesh.translate(translation)
    mesh_scene = o3d.geometry.TriangleMesh()
    #mesh_scene = desk_mesh
    list_bottom_centric = []


    print('---------')
    print('Remmove all objs...')

    print("---------")
    print("Creat mesh scene...")
    dict_bbox_pos_sacle = bbox_pos_scale_all_obj(ply_path, npz_path)

    for ins_label, item_instance in dict_bbox_pos_sacle.items():
        obj_exist = 0
        #print('item instance', item_instance)
        item_semantic_label = item_instance['semantic_label']
        item_bbox_min_bound = item_instance['min_bound']
        item_bbox_max_bound = item_instance['max_bound']
        item_bbox_size = item_instance['size']
        item_bbox_center = [0,0,0] # initialize center
        item_bbox_center[0] = (item_bbox_min_bound[0] + item_bbox_max_bound[0]) / 2
        item_bbox_center[1] = (item_bbox_min_bound[1] + item_bbox_max_bound[1]) / 2
        item_bbox_center[2] = (item_bbox_min_bound[2] + item_bbox_max_bound[2]) / 2
        item_bbox_pos = item_bbox_center
        item_bbox_pos[2] = item_bbox_pos[2] - item_bbox_size[2] / 2
        #print('item bbox pos', item_bbox_pos)
        # convert to semantic label to keyword
            # index -- object name
        item_dict = {}
        with open('code/classes.txt', 'r') as file:
            for line in file:
                parts = line.strip().split('	')
                num = int(parts[0])
                item_name = parts[1]
                item_dict[num] = item_name    
        item_keyword = item_dict[item_semantic_label]

        # get the mesh obj 
        json_path = '/home/stud/zhoy/storage/group/dataset_mirrors/ShapeNetV2/ShapeNetCore.v2/taxonomy.json'
        with open(json_path,  'r', encoding='utf-8') as f:
            json_content = json.load(f)
        #print('json content:',json_content[0])
        keyword = item_keyword # maybe add mapping
        if keyword in ['book', 'notebook', 'remote control','pen', 'pencil','phone', 'fruit','eye_glasses','ruler','eraser','charger','calculator','file_box']:# 暂时用glass box代替没有的mesh
            keyword = "glass_box"
        if keyword == 'can':
            keyword = 'cup'
        if keyword == 'tea_pot':
            keyword = 'bowl'
        if keyword in ['globe', 'chessboard']:
            keyword = 'vase'
        if keyword in ['monitor', 'display']:
            keyword = 'monitor'
        if keyword in ['earphone']:
            keyword = 'we dont need that'
        if keyword in ['lamp']:
            keyword = 'lamp_modelnet'

        for item in json_content:
            #if keyword in item['name'] and (keyword=='laptop' or keyword =='lamp' or keyword =='mug'): #
            if keyword in item['name']:
                
                print('-----------')
                #print(f"{item['name']}: {item['synsetId']}")
                directory = os.path.join('/home/stud/zhoy/storage/group/dataset_mirrors/ShapeNetV2/ShapeNetCore.v2', f"{item['synsetId']}")
                try:
                    items_file = os.listdir(directory)
                    subdirectories = [item for item in items_file if os.path.isdir(os.path.join(directory, item))]
                    len_subdirectories = len(subdirectories)
                    random_integer = random.randint(0, len_subdirectories-1)
                    first_subdirectory = os.path.join(directory, subdirectories[random_integer])
                    print("first subdirectory:", first_subdirectory)
                    for _, _, files in os.walk(first_subdirectory):
                        for file in files:
                            print("file:", file)
                            if file.endswith('.obj'):
                                obj_path = os.path.join(first_subdirectory, f"models/{file}")
                                mesh_obj = o3d.io.read_triangle_mesh(obj_path)
                                print(f'we got {keyword}')
                                obj_exist = 1
                            #o3d.io.write_triangle_mesh(f"{keyword}.obj", mesh_obj)
                    break
                except FileNotFoundError:
                    print("No such file or directory.")

                    continue

        if obj_exist == 0: # ShapeNet does not have the obj
            print(f'sorry we dont have {keyword} in ShapeNet')
            root_folder = "mesh_dataset/raplace_mesh"
            for root, dirs, files in os.walk(root_folder):
                for dir_name in dirs:
                    if dir_name == keyword:
                        obj_exist = 1
                        folder_path = os.path.join(root_folder, keyword)
                        #print(folder_path)
                        off_files = [f for f in os.listdir(folder_path) if f.endswith('.off')]
                        random_file = random.choice(off_files)
                        file_path = os.path.join(folder_path, random_file)
                        mesh_obj = o3d.io.read_triangle_mesh(file_path)
                        angle_deg_x = -90
                        angle_rad_x = np.radians(angle_deg_x)
                        rotation_vector = np.array([angle_rad_x, 0, 0])
                        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
                        mesh_obj.rotate(rotation_matrix, center=mesh_obj.get_center())

                




        if obj_exist == 0: #ModelNet does not have the obj
            print(f'sorry we dont have {keyword} in ModelNet')
            continue
        # after getting the obj mesh, replace the pcd obj
        angle_deg_y = 180
        angle_rad_y = np.radians(angle_deg_y)
        angle_deg_x = 90
        angle_rad_x = np.radians(angle_deg_x)
        rotation_vector = np.array([0, angle_rad_y, 0])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
        mesh_obj.rotate(rotation_matrix, center=mesh_obj.get_center())
        rotation_vector = np.array([angle_rad_x, 0, 0])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
        mesh_obj.rotate(rotation_matrix, center=mesh_obj.get_center())
        
        # bbox
        aabb = mesh_obj.get_axis_aligned_bounding_box()
        extent1 = item_bbox_size
        extent2 = aabb.get_extent()
        scale_factors = extent1 / extent2
        vertices2 = np.asarray(mesh_obj.vertices)
        if keyword in ['book', 'pen', 'pencil', 'notebook', "glass_box", "bowl","printer", "keyboard"]:
            vertices2 *= scale_factors
        else:
            vertices2 *= scale_factors.mean()
        mesh_obj.vertices = o3d.utility.Vector3dVector(vertices2)
        aabb2_scaled = mesh_obj.get_axis_aligned_bounding_box()
        extent2_scaled = aabb2_scaled.get_extent()
        aabb_center = aabb2_scaled.get_center()
        aabb_bottom_center = np.array(aabb_center)
        aabb_bottom_center[2] -= extent2_scaled[2] / 2.0  # 减去 Z 方向上的一半尺寸  
        translation = item_bbox_pos - aabb_bottom_center
        #mesh_obj.translate(translation)
        list_bottom_centric.append(item_bbox_pos[2])
        # optimize the z-angle
        pcd = get_obj_from_scene_inslabel(pcd_ply_path=ply_path, ins_index=ins_label, npz_path=npz_path)
        # Sample points randomly from the point cloud
        sampled_indices = np.random.choice(len(pcd.points), 512, replace=True)
        sampled_pcd_points = np.asarray(pcd.points)[sampled_indices]
        sampled_mesh_points = mesh_obj.sample_points_uniformly(number_of_points=512)
        sampled_mesh_points = np.asarray(sampled_mesh_points.points)
        optimezed_z = optimize_rotation([sampled_mesh_points], [sampled_pcd_points], initial_angle=-10, num_iterations=10, learning_rate=10)

        angle_deg_z = optimezed_z.detach().numpy()
        if keyword in ['laptop','glass_box','camera','clock','monitor','monitor_modelnet','keyboard','printer']:
            angle_deg_z = np.round(angle_deg_z / 90) * 90
        if keyword in ['earphone']:
            angle_rad = 90
            rotation_vector = np.array([angle_rad, 0, 0])
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
            mesh_obj.rotate(rotation_matrix, center=mesh_obj.get_center())



        angle_rad_z = np.radians(angle_deg_z)
        #print("angle deg z", angle_deg_z)
        #print("angle rad z", angle_rad_z)
        rotation_vector = np.array([0, 0, angle_rad_z])
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
        mesh_obj.rotate(rotation_matrix, center=mesh_obj.get_center())
        mesh_obj.translate(translation)
        #mesh_obj.vertex_colors = o3d.utility.Vector3dVector(colors)
        chosen_color = random.choice(list(color_dict.values()))

        # 创建颜色数组，给每个顶点赋予相同的颜色
        num_vertices = len(mesh_obj.vertices)
        colors = np.tile(chosen_color, (num_vertices, 1))
        # 将颜色赋予网格的顶点
        mesh_obj.vertex_colors = o3d.utility.Vector3dVector(colors)


        mesh_scene = mesh_obj + mesh_scene
        print(f"Great! {keyword} is done")

    avg_bottom_centric = sum(list_bottom_centric) / len(list_bottom_centric)
    top_center_desk_pcd[2] = avg_bottom_centric
    translation = top_center_desk_pcd - top_center_desk_mesh
    desk_mesh.translate(translation)
    mesh_scene+=desk_mesh

    # add chairs if exist
    '''
    chair_pcd = get_obj_from_scene(pcd_ply_path=ply_path, obj_index=3, npz_path=npz_path)
    if chair_pcd is not None:
        print('adding chair...')
        aabb_chair_pcd = chair_pcd.get_axis_aligned_bounding_box()
        min_bound = aabb_chair_pcd.get_min_bound()
        max_bound = aabb_chair_pcd.get_max_bound()
        dimensions_chair_pcd = max_bound - min_bound
        center_chair_pcd = (min_bound + max_bound) / 2.0
        top_center_chair_pcd = np.array(center_chair_pcd)
        top_center_chair_pcd[2] += dimensions_chair_pcd[2] / 2.0
        chair_mesh_file_path = 'mesh_dataset/raplace_mesh/chair/chair_0001.off'
        chair_mesh =  o3d.io.read_triangle_mesh(chair_mesh_file_path)
        aabb_chair_mesh = chair_mesh.get_axis_aligned_bounding_box()
        min_bound = aabb_chair_mesh.get_min_bound()
        max_bound = aabb_chair_mesh.get_max_bound()
        dimensions_chair_mesh = max_bound - min_bound
        center_chair_mesh = (min_bound + max_bound) / 2.0
        top_center_chair_mesh = np.array(center_chair_mesh)
        top_center_chair_mesh[2] += dimensions_chair_mesh[2] / 2.0
    '''

    last_slash_index = ply_path.rfind("/")
    last_dot_index = ply_path.rfind(".")

    filename = ply_path[last_slash_index + 1:last_dot_index]
    print('filename', filename)
    #output_file_path = os.path.join("myDataset_gen_mesh", f"{filename}_mesh.obj")
    unique_file_path = get_unique_filename(directory="myDataset_gen_mesh", filename=filename, extension="_mesh.obj")
    o3d.io.write_triangle_mesh(unique_file_path, mesh_scene)
    #print(list_bottom_centric)

def rotate_mesh_around_center(mesh, angle_degrees, axis=[0, 0, 1]):
    """
    旋转Trimesh对象绕其中心点指定角度。

    参数：
    mesh (trimesh.Trimesh): 输入的Trimesh对象。
    angle_degrees (float): 旋转角度（以度为单位）。
    axis (list): 旋转轴，默认为绕Z轴旋转。

    返回值：
    trimesh.Trimesh: 旋转后的Trimesh对象。
    """
    # 确保输入是一个Trimesh对象
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input mesh is not a Trimesh object")

    # 计算物体的中心
    center = mesh.centroid

    # 指定旋转角度（以弧度为单位）
    angle_radians = np.deg2rad(angle_degrees)

    # 绕指定轴的旋转矩阵
    rotation_mat = trimesh.transformations.rotation_matrix(angle_radians, axis)

    # 平移到中心的变换矩阵
    translation_to_origin = trimesh.transformations.translation_matrix(-center)

    # 从中心平移回来的变换矩阵
    translation_back = trimesh.transformations.translation_matrix(center)

    # 组合变换矩阵：先平移到中心，然后旋转，最后平移回去
    transform = translation_back @ rotation_mat @ translation_to_origin

    # 应用组合变换
    mesh.apply_transform(transform)

    # 返回旋转后的网格
    return mesh

def move_trimeshobj_to_position(obj_mesh, end_position, start_position):
    # 加载 obj1 和 obj2

    # 计算平移向量
    translation = np.array(end_position) - np.array(start_position)
    
    # 创建平移矩阵
    translation_matrix = np.eye(4)
    translation_matrix[0:3, 3] = translation
    
    # 对 obj2 进行平移
    obj_mesh.apply_transform(translation_matrix)

    return obj_mesh
    

def generate_mesh_scene_all_texute(ply_path, npz_path):
    """
    Using trimesh
    OBJ with texture
    Remove objects with semantic labels greater than 40 from the point cloud scene and replace them with mesh obj.
    
    
    Parameters:
    pcd_ply_path (str): Path to the input PLY file.
    npz_path (str): Path to the NPZ file containing 'xyz', 'semantic_label', and 'instance_label'.

    
    Returns:
    mesh scene 
    """

    # get the size of the desk pcd

    #desk_pcd= get_obj_from_scene(pcd_ply_path=ply_path, obj_index=7, npz_path=npz_path)
    desk_pcd= get_obj_from_scene(pcd_ply_path=ply_path, obj_index=14, npz_path=npz_path)
    aabb_desk_pcd = desk_pcd.get_axis_aligned_bounding_box()
    min_bound = aabb_desk_pcd.get_min_bound()
    max_bound = aabb_desk_pcd.get_max_bound()
    dimensions_desk_pcd = max_bound - min_bound
    center_desk_pcd = (min_bound + max_bound) / 2.0
    top_center_desk_pcd = np.array(center_desk_pcd)
    top_center_desk_pcd[2] += dimensions_desk_pcd[2] / 2.0

    #get the position of the desk pcd
    #get the desk mesh and align the size and bottom centric
    desk_mesh_file_path = 'mesh_dataset/replace_mesh_texture/desk/desk_0202_plastic/mesh.obj'
    desk_mesh =  trimesh.load(desk_mesh_file_path, process=False)


    # optimize the z
    sampled_indices_desk = np.random.choice(len(desk_pcd.points), 512, replace=True)
    sampled_pcd_points_desk = np.asarray(desk_pcd.points)[sampled_indices_desk]
    sampled_mesh_points_desk = desk_mesh.sample(512)
    optimezed_z_desk = optimize_rotation([sampled_mesh_points_desk], [sampled_pcd_points_desk], initial_angle=0, num_iterations=10, learning_rate=50)
    angle_deg_z_desk = optimezed_z_desk.detach().numpy()
    angle_deg_z_desk = np.round(angle_deg_z_desk / 90) * 90
    angle_rad_z_desk = np.radians(angle_deg_z_desk)
    # 绕Z轴旋转的旋转矩阵
    desk_mesh = rotate_mesh_around_center(desk_mesh, angle_deg_z_desk, axis=[0,0,1])


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
    desk_mesh.apply_transform(scale_matrix)
    
    #desk_mesh.compute_vertex_normals()
    min_bound = desk_mesh.bounds[0]
    max_bound = desk_mesh.bounds[1]
    dimensions_desk_mesh = max_bound - min_bound 

    center_desk_mesh = (min_bound + max_bound) / 2.0
    top_center_desk_mesh = np.array(center_desk_mesh)
    top_center_desk_mesh[2] += dimensions_desk_mesh[2] / 2.0



    #translation = top_center_desk_pcd - top_center_desk_mesh
    # create an empty mesh
    #desk_mesh.translate(translation)
    mesh_scene = trimesh.Trimesh()
    #mesh_scene = desk_mesh
    list_bottom_centric = []


    print('---------')
    print('Remmove all objs...')

    print("---------")
    print("Creat mesh scene...")
    dict_bbox_pos_sacle = bbox_pos_scale_all_obj(ply_path, npz_path)

    for ins_label, item_instance in dict_bbox_pos_sacle.items():
        obj_exist = 0
        #print('item instance', item_instance)
        item_semantic_label = item_instance['semantic_label']
        item_bbox_min_bound = item_instance['min_bound']
        item_bbox_max_bound = item_instance['max_bound']
        item_bbox_size = item_instance['size']
        item_bbox_center = [0,0,0] # initialize center
        item_bbox_center[0] = (item_bbox_min_bound[0] + item_bbox_max_bound[0]) / 2
        item_bbox_center[1] = (item_bbox_min_bound[1] + item_bbox_max_bound[1]) / 2
        item_bbox_center[2] = (item_bbox_min_bound[2] + item_bbox_max_bound[2]) / 2
        item_bbox_pos = item_bbox_center
        item_bbox_pos[2] = item_bbox_pos[2] - item_bbox_size[2] / 2
        #print('item bbox pos', item_bbox_pos)
        # convert to semantic label to keyword
            # index -- object name
        item_dict = {}
        with open('code/classes.txt', 'r') as file:
            for line in file:
                parts = line.strip().split('	')
                num = int(parts[0])
                item_name = parts[1]
                item_dict[num] = item_name    
        item_keyword = item_dict[item_semantic_label]

        # get the mesh obj 
        keyword = item_keyword # maybe add mapping
        if keyword in ['book', 'notebook', 'remote control','pen', 'pencil','phone', 'fruit','eye_glasses','ruler','eraser','charger','calculator','file_box']:# 暂时用glass box代替没有的mesh
            keyword = keyword
        if keyword in ['ruler']:
            keyword = 'cup'
        if keyword in ['camera', 'alarm', "glass"]:
            keyword = 'cup'
        if keyword == 'tea_pot':
            keyword = 'bowl'
        if keyword in ['monitor', 'display']:
            keyword = 'monitor'
        if keyword in ['earphone']:
            keyword = 'we dont need that'


        folder_path = 'mesh_dataset/replace_mesh_texture' # the dataset folder of texture obj
        subfolders = next(os.walk(folder_path))[1]
        if keyword in subfolders:
            obj_exist = 1
            keyword_folder_path = os.path.join(folder_path, keyword)
            #os.chdir(keyword_folder_path)
            # 获取所有子文件夹名称
            #print(os.scandir(keyword_folder_path))
            subfolders = [f.path for f in os.scandir(keyword_folder_path) if f.is_dir()]

            # 遍历子文件夹
            for subfolder in subfolders:
                mesh_file_path = os.path.join(subfolder, 'mesh.obj')
                if os.path.exists(mesh_file_path):
                # 读取 mesh.obj 文件
                    mesh_obj = trimesh.load_mesh(mesh_file_path, process=False)


        if obj_exist == 0: #ModelNet does not have the obj
            print(f'sorry we dont have {keyword} in texture obj dataset')
            continue

        
        # after getting the obj mesh, replace the pcd obj
        if keyword in ['bottle', 'cup','lamp']:
            mesh_obj = rotate_mesh_around_center(mesh_obj, -90, [1,0,0])
        angle_deg_y = 180
        mesh_obj = rotate_mesh_around_center(mesh_obj, angle_deg_y, [0,1,0])
        angle_deg_x = 90
        mesh_obj = rotate_mesh_around_center(mesh_obj, angle_deg_x, [1,0,0])
    
        
        # bbox
        aabb_min_bound = mesh_obj.bounds[0]
        aabb_max_bound = mesh_obj.bounds[1]
        extent1 = item_bbox_size
        extent2 = aabb_max_bound - aabb_min_bound
        scale_factors = extent1 / extent2
        obj_mesh_vertices = mesh_obj.vertices
        obj_mesh_center = mesh_obj.centroid
        if keyword in ['book', 'pen', 'pencil', 'notebook', "glass_box", "bowl","printer", "keyboard", "laptop"]:
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = scale_factors[0]
            scale_matrix[1, 1] = scale_factors[1]
            scale_matrix[2, 2] = scale_factors[2]

        else:
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = scale_factors.mean()
            scale_matrix[1, 1] = scale_factors.mean()
            scale_matrix[2, 2] = scale_factors.mean()
            
        mesh_obj.apply_transform(scale_matrix)

        aabb2_scaled_min = mesh_obj.bounds[0]
        aabb2_scaled_max = mesh_obj.bounds[1]
        extent2_scaled = aabb2_scaled_max - aabb2_scaled_min
        aabb_center = mesh_obj.centroid
        aabb_bottom_center = np.array(aabb_center)
        aabb_bottom_center[2] -= extent2_scaled[2] / 2.0  # 减去 Z 方向上的一半尺寸  
        translation = item_bbox_pos - aabb_bottom_center
        #mesh_obj.translate(translation)
        list_bottom_centric.append(item_bbox_pos[2])
        # optimize the z-angle
        pcd = get_obj_from_scene_inslabel(pcd_ply_path=ply_path, ins_index=ins_label, npz_path=npz_path)
        # Sample points randomly from the point cloud
        sampled_indices = np.random.choice(len(pcd.points), 512, replace=True)
        sampled_pcd_points = np.asarray(pcd.points)[sampled_indices]
        sampled_mesh_points = mesh_obj.sample(512)
        sampled_mesh_points = np.asarray(sampled_mesh_points)
        optimezed_z = optimize_rotation([sampled_mesh_points], [sampled_pcd_points], initial_angle=-70, num_iterations=10, learning_rate=10)

        angle_deg_z = optimezed_z.detach().numpy()
        if keyword in ['laptop','glass_box','camera','clock','monitor','monitor_modelnet','keyboard','printer']:
            angle_deg_z = np.round(angle_deg_z / 90) * 90

        mesh_obj = rotate_mesh_around_center(mesh_obj, angle_deg_z, [0,0,1])
        mesh_obj = move_trimeshobj_to_position(mesh_obj, end_position=item_bbox_pos, start_position=aabb_bottom_center)
        #mesh_obj.vertex_colors = o3d.utility.Vector3dVector(colors)

        # drift the object until collision -- waiting to be done
        # hardcode
        if keyword in ['laptop']:
            mesh_obj.apply_translation([0, 0, -0.06])

        mesh_scene = trimesh.util.concatenate([mesh_scene, mesh_obj])
        print(f"Great! {keyword} is done")

    avg_bottom_centric = sum(list_bottom_centric) / len(list_bottom_centric)
    top_center_desk_pcd[2] = avg_bottom_centric
    translation = top_center_desk_pcd - top_center_desk_mesh
    desk_mesh = move_trimeshobj_to_position(desk_mesh, top_center_desk_pcd, top_center_desk_mesh)
    #mesh_scene+=desk_mesh
    mesh_scene = trimesh.util.concatenate([mesh_scene, desk_mesh])


    last_slash_index = ply_path.rfind("/")
    last_dot_index = ply_path.rfind(".")

    filename = ply_path[last_slash_index + 1:last_dot_index]
    print('filename', filename)
    #output_file_path = os.path.join("myDataset_gen_mesh", f"{filename}_mesh.obj")
    unique_file_path = get_unique_filename(directory="myDataset_gen_mesh", filename=filename, extension="_mesh.obj")
    centroid_mesh_scene = mesh_scene.centroid
    

    mesh_scene.export(unique_file_path)
    print(f"{unique_file_path} is done")
    #print(list_bottom_centric)

if __name__ == "__main__":
    '''
        pcd_ply_path = 'test_new_scene_generation/Dataset/id15.ply'
    npz_path = 'test_new_scene_generation/Dataset/id15.npz'
    obj_pcd = get_obj_from_scene(pcd_ply_path=pcd_ply_path, obj_index=48, npz_path=npz_path)
    new_scene_pcd = remove_obj_from_scene(pcd_ply_path=pcd_ply_path, obj_index=49, npz_path=npz_path)
    move_scene_pcd = move_obj_in_scene(pcd_ply_path=pcd_ply_path, obj_index=49,npz_path=npz_path)
    roated_scene_pcd = rotate_point_cloud(pcd_ply_path=pcd_ply_path, angle_deg=90)
    #if obj_pcd is not None:
    #    o3d.io.write_point_cloud("test_new_scene_generation/Dataset/test_obj.ply", obj_pcd)
    #else: print("No such obj in the scene")
    o3d.io.write_point_cloud("test_new_scene_generation/Dataset/test_rotate.ply", roated_scene_pcd)
    '''
    '''    scene1_ply_path = 'test_new_scene_generation/Dataset/id15.ply'
    sceme1_npz_path = 'test_new_scene_generation/Dataset/id15.npz'
    scene2_ply_path = 'test_new_scene_generation/Dataset/id88.ply'
    sceme2_npz_path = 'test_new_scene_generation/Dataset/id88.npz'
    exchange_objects(scene1_ply_path, scene2_ply_path, sceme1_npz_path, sceme2_npz_path, 49, 49)'''

    
    pcd_ply_path = 'test_new_scene_generation/Dataset/id88.ply'
    npz_path = 'test_new_scene_generation/Dataset/id88.npz'
    dict = convert_dict(pcd_ply_path, npz_path)
    #print('dict', dict)
    #print("conver to graph ...")
    graph = convert_to_graph(dict)
    #print("graph", graph)


    pcd_ply_path = 'test_new_scene_generation/Dataset/id15.ply'
    npz_path = 'test_new_scene_generation/Dataset/id15.npz'    
    obj_idx = 49
    newpcd, location = keep_obj_in_scene(pcd_ply_path, obj_index=49, npz_path=npz_path)
    o3d.io.write_point_cloud(f"test_new_scene_generation/Dataset/cleaned_id15.ply", newpcd)
    print(location)

    pcd_ply_path = 'test_new_scene_generation/Dataset/cleaned_id15.ply'

    _, old_location = keep_obj_in_scene(pcd_ply_path='test_new_scene_generation/Dataset/id88.ply', obj_index=49, npz_path = 'test_new_scene_generation/Dataset/id88.npz')
    old_bbox, _ = get_on_desk_bbox(pcd_ply_path='test_new_scene_generation/Dataset/id88.ply',
                                npz_path = 'test_new_scene_generation/Dataset/id88.npz')
    print("old bbox",old_bbox[0], old_bbox[1])
    old_scale = get_relative_position_bbox(old_bbox, old_location)
    print("old scale", old_scale)
    
    new_bbox,_ = get_on_desk_bbox(pcd_ply_path = 'test_new_scene_generation/Dataset/id15.ply',
                                npz_path = 'test_new_scene_generation/Dataset/id15.npz')
    
    new_scale = get_relative_position_bbox(new_bbox, location)
    print("new scale", new_scale)
    scale = [n / o for o, n in zip(old_scale, new_scale)]
    print("scale", scale)

    add_scene = add_objects_to_scene(pcd_ply_path,
                                     display_position=location, 
                                     graph=graph,
                                     obj_dataset_folder='test_new_scene_generation/Dataset/obj_dateset',
                                     scale=scale)
    o3d.io.write_point_cloud(f"test_new_scene_generation/Dataset/new_id15.ply", add_scene)
    '''
    pcd_ply_path = 'test_new_scene_generation/Dataset/id15.ply'
    npz_path = 'test_new_scene_generation/Dataset/id15.npz'    
    obj_idx = 49
    bbox, xy_area= get_on_desk_bbox(pcd_ply_path, npz_path)
    print("bbox", bbox, xy_area)

    pcd = get_obj_from_scene(pcd_ply_path, 49, npz_path)
    bottom_centric = get_bottom_centric(pcd)
    print("bottom centric", bottom_centric)
    '''
    item_dict = {}
    with open('code/classes.txt', 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除行尾的换行符，并将每行内容按空格分割成数字和物品名
            parts = line.strip().split('	')
            # 将第一个部分转换为整数作为数字
            num = int(parts[0])
            # 将第二个部分作为物品名
            item_name = parts[1]
            # 将数字和物品名添加到字典中
            item_dict[num] = item_name  

    for obj_index in range(41,93):
        pcd_ply_path = 'ori_train/ori_train_ply/id171.ply'
        npz_path = 'ori_train/ori_train_npz/id171.npz'
        obj_pcd = get_obj_from_scene(pcd_ply_path, obj_index, npz_path)
        if obj_pcd:
            o3d.io.write_point_cloud(f"test_new_scene_generation/Dataset/obj_dateset/{item_dict[obj_index]}.ply", obj_pcd)

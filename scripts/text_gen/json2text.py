"""
    Generate the text guidance from the json
"""
import json
import math
import random



# 计算欧几里得距离的函数
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def key2phrase(key:str):
    parts = key.split('/')
    obj_descpt = parts[-2]
    parts = obj_descpt.split('_')
    obj_name = parts[0]
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
        return "Back Right"
    elif 247.5 <= angle < 292.5:
        return "Back"
    elif 292.5 <= angle < 337.5:
        return "Back Left"
    
def json2text():
    pass

# 从文件中读取 JSON 数据
with open('dataset/scene_gen/scene_mesh_json/id15_2.json', 'r') as file:
    data = json.load(file)

# 删除最后一个键值对(桌子)
if isinstance(data, dict) and data:
    last_key = list(data.keys())[-1]
    del data[last_key]

# 提取每个物体的位置信息 (x, y)
positions = {key: value[0][:2] for key, value in data.items()}


# 找到每个物体最近的物体
nearest_neighbors = {}
for obj1, pos1 in positions.items():
    nearest_obj = None
    min_distance = float('inf')
    for obj2, pos2 in positions.items():
        if obj1 != obj2:
            distance = euclidean_distance(pos1, pos2)
            if distance < min_distance:
                min_distance = distance
                nearest_obj = obj2
    direction = determine_direction(pos1, positions[nearest_obj])
    nearest_neighbors[obj1] = (nearest_obj, direction)

# text
print(f"There are {len(positions)} objects on the table.")
for obj, (nearest, direction) in nearest_neighbors.items():
    obj1 = key2phrase(obj)
    obj2 = key2phrase(nearest)
    direction = direction
    strings = [
    f'{obj1} is at the {direction} of {obj2}',
    f'{obj1} is at {obj2}\'s {direction}',
    f'{obj1} is on {obj2}\'s {direction} side',
    f'{obj1} is on the {direction} side of {obj2}',
    f'To {direction} of {obj2}, there is a {obj1}'
]
    selected_string = random.choice(strings)

    print(selected_string)
    

'''
# 遍历解析后的数据
for key, value in data.items():
    print(f'Key: {key}')
    parts = key.split('/')
    obj_descpt = parts[-2]
    parts = obj_descpt.split('_')
    obj_name = parts[0]
    obj_descpt = parts[-1]
    print(f"the {obj_descpt} {obj_name}")
    for index, item in enumerate(value):
        print(f'  Item {index + 1}: {item}')
'''
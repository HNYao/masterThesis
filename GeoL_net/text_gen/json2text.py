"""
    Generate the text guidance from the json
    TODO: 体积太小的物体不做参照物，例如plant和耳机
"""
import json
import math
import random
import os


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
    
def json2text(json_path, out_dir=None):

    text_dict = {}

    with open(json_path, 'r') as file:
        data = json.load(file)

    # ignore the desk
    if isinstance(data, dict) and data:
        last_key = list(data.keys())[-1]
        del data[last_key]

    positions = {key: value[0][:2] for key, value in data.items()}

    # nearest
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
        text_dict[obj1] = [obj2, selected_string]
        print(selected_string, "--refrence:", obj2)
        print(text_dict)

    if out_dir is None:
        scene_id = json_path.split("/")[-1].split(".")[0]
        parent_folder = "dataset/scene_RGBD_mask"
        out_dir = os.path.join(parent_folder, scene_id)
        print(out_dir)

    with open(f"{out_dir}/text_guidance.json", "w") as json_file:
        json.dump(text_dict, json_file, indent=4)

if __name__ =="__main__":
    json2text(json_path="dataset/scene_gen/scene_mesh_json/id15_1.json")
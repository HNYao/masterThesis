"""
    Generate the text guidance from the json
    Update: 体积太小的物体不做参照物，例如plant 橡皮 pencil
    Update 11.16 ：过于近的物体不做参照物，距离阈值为0.2


"""
import json
import math
import random
import os
import glob


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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
    
def json2text(json_path, out_dir=None):

    if out_dir is None:
        scene_id = json_path.split("/")[-1].split(".")[0]
        parent_folder = "dataset/scene_RGBD_mask_v2_kinect_cfg"
        out_dir = os.path.join(parent_folder, scene_id)
        # if not exists, create the folder
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print(out_dir, "is making now ")

    if os.path.exists(f"{out_dir}/text_guidance.json"):
        print(f"{out_dir}/text_guidance.json already exists, no need to generate again.")
        return

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
                if "eraser" in obj2 or "pencil" in obj2 or "plant" in obj2:
                    continue
                distance = euclidean_distance(pos1, pos2)
                if distance < min_distance and distance > 0.2: # 0.2 is the threshold
                    min_distance = distance
                    nearest_obj = obj2
        direction = determine_direction(pos1, positions[nearest_obj])
        nearest_neighbors[obj1] = (nearest_obj, direction)

    # text
    #print(f"There are {len(positions)} objects on the table.")
    for obj, (nearest, direction) in nearest_neighbors.items():
        obj1 = key2phrase(obj)
        obj2 = key2phrase(nearest)
        direction = direction
        
        strings = [ # type1
        f'{obj1} is at the {direction} of {obj2}',
        f'{obj1} is at {obj2}\'s {direction}',
        f'{obj1} is on {obj2}\'s {direction} side',
        f'{obj1} is on the {direction} side of {obj2}',
        f'To {direction} of {obj2}, there is a {obj1}'
    ]
        strings = [ #type2
        f'place {obj1} at the {direction} of {obj2}',
        f'place {obj1} at {obj2}\'s {direction}',
        f'place {obj1} on {obj2}\'s {direction} side',
        f'place {obj1} on the {direction} side of {obj2}'
    ]
        obj2_short = obj2.split(" ")[-1]
        strings = [
            f"{direction} {obj2}"
        ]
        selected_string = random.choice(strings)
        text_dict[obj1] = [obj2, selected_string, obj2_short, direction, obj, nearest]
        #print(selected_string, "--refrence:", obj2)
        #print(text_dict)


    with open(f"{out_dir}/text_guidance.json", "w") as json_file:
        json.dump(text_dict, json_file, indent=4)
    
    print(f"Text guidance is saved at {out_dir}/text_guidance.json")

if __name__ =="__main__":

    json_folder_path = "dataset/scene_gen/scene_mesh_json_kinect"
    json_files = glob.glob(os.path.join(json_folder_path, '*.json'))

    amount_dataset = 0
    for json_file_path in json_files:
        json_path = json_file_path
        try:
            json2text(json_path=json_path)
        except FileNotFoundError:
            continue
        amount_dataset += 1

        
    
    print(amount_dataset)
    # json2text(json_path="dataset/scene_gen/scene_mesh_json_kinect/id100.json") generate single json
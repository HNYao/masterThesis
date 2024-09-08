import os


folder_path = 'dataset/scene_gen/scene_mesh_json'


for filename in os.listdir(folder_path):
    if filename.endswith('_1.json'): #_1
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
        print(f"Delete: {file_path}")
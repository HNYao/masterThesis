from pathlib import Path
import os

def count_first_level_subfolders(folder_path):
    # 统计一级子文件夹数量
    path = Path(folder_path)
    return sum(1 for item in path.iterdir() if item.is_dir())


def count_second_level_subfolders(folder_path):
    second_level_subfolder_count = 0
    
    # 遍历一级子文件夹
    for first_level_folder in os.listdir(folder_path):
        first_level_folder_path = os.path.join(folder_path, first_level_folder)
        
        # 如果是一级子文件夹
        if os.path.isdir(first_level_folder_path):
            # 遍历二级子文件夹
            for second_level_folder in os.listdir(first_level_folder_path):
                second_level_folder_path = os.path.join(first_level_folder_path, second_level_folder)
                
                # 如果是二级子文件夹
                if os.path.isdir(second_level_folder_path):
                    second_level_subfolder_count += 1

    return second_level_subfolder_count

# 函数 筛选出二级子文件夹数量小于8的一级子文件夹
def filter_first_level_subfolders(folder_path):
    # 统计一级子文件夹数量
    path = Path(folder_path)
    first_level_subfolders = [item for item in path.iterdir() if item.is_dir()]
    filtered_first_level_subfolders = []
    
    for first_level_subfolder in first_level_subfolders:
        second_level_subfolder_count = sum(1 for item in first_level_subfolder.iterdir() if item.is_dir())
         
        if second_level_subfolder_count <7:
            filtered_first_level_subfolders.append(first_level_subfolder)
    
    return filtered_first_level_subfolders

# 给定文件夹路径
folder_path = 'dataset/scene_RGBD_mask_v2_kinect_cfg'
first_level_subfolder_count = count_first_level_subfolders(folder_path)
second_level_subfolder_count = count_second_level_subfolders(folder_path)
filtered_first_level_subfolders = filter_first_level_subfolders(folder_path)

print(f"Number of first-level subfolders in '{folder_path}': {first_level_subfolder_count}")
print(f"Number of second-level subfolders in '{folder_path}': {second_level_subfolder_count}")
#print(f"First-level subfolders with less than 8 second-level subfolders: {filtered_first_level_subfolders}")


"""
rotate the ply 
save back to the folder
"""
import open3d as o3d
import numpy as np
import math
import json
import os

def rotate_point_cloud(input_ply_path, angle_degrees):
    # 读取点云文件

    pcd = o3d.io.read_point_cloud(input_ply_path)


    angle_radians = math.radians(angle_degrees)  # 转换为弧度   

    # 创建绕z轴旋转90度的旋转矩阵
    R = pcd.get_rotation_matrix_from_xyz((0, 0, angle_radians))  # 90度 = π/2 弧度

    # 应用旋转矩阵到点云
    pcd.rotate(R, center=(0, 0, 0))

    # 保存旋转后的点云到原路径
    o3d.io.write_point_cloud(input_ply_path, pcd)

    print(f"Rotated point cloud saved to {input_ply_path}")


input_ply_path = 'dataset/obj/ply/book/book_20.ply'  # 替换为你的文件路径
angle_degrees = 45
rotate_point_cloud(input_ply_path, angle_degrees)

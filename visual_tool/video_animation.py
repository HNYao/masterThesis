import open3d as o3d
import numpy as np
import cv2
import threading
import time

def rotate_point_cloud_around_x(point_cloud, angle_in_degrees=180):

    angle_in_radians = np.deg2rad(angle_in_degrees)
    

    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                                [0, np.sin(angle_in_radians), np.cos(angle_in_radians)]])
    

    rotated_point_cloud = point_cloud.rotate(rotation_matrix, center=(0, 0, 0))
    
    return rotated_point_cloud

def create_rotation_video(point_cloud, axis, video_name, duration=10, fps=30):
    """
    创建点云绕某一轴旋转的视频。

    参数:
        point_cloud (o3d.geometry.PointCloud): 输入的点云对象。
        axis (str): 旋转轴 ('x', 'y', 'z')。
        video_name (str): 输出视频的文件名。
        duration (int): 视频时长（秒）。
        fps (int): 视频帧率。
    """
    # 创建可视化器
    point_cloud = rotate_point_cloud_around_x(point_cloud)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(point_cloud)

    # 获取视图控制器
    ctr = vis.get_view_control()

    # 定义旋转矩阵
    total_frames = duration * fps
    angle_per_frame = 360 / total_frames  # 每帧旋转的角度（度）
    rotation_matrices = {
        'x': np.array([[1, 0, 0],
                       [0, np.cos(np.deg2rad(angle_per_frame)), -np.sin(np.deg2rad(angle_per_frame))],
                       [0, np.sin(np.deg2rad(angle_per_frame)), np.cos(np.deg2rad(angle_per_frame))]]),
        'y': np.array([[np.cos(np.deg2rad(angle_per_frame)), 0, np.sin(np.deg2rad(angle_per_frame))],
                       [0, 1, 0],
                       [-np.sin(np.deg2rad(angle_per_frame)), 0, np.cos(np.deg2rad(angle_per_frame))]]),
        'z': np.array([[np.cos(np.deg2rad(angle_per_frame)), -np.sin(np.deg2rad(angle_per_frame)), 0],
                       [np.sin(np.deg2rad(angle_per_frame)), np.cos(np.deg2rad(angle_per_frame)), 0],
                       [0, 0, 1]])
    }
    
    # 初始化视频写入器
    width, height = 640, 480
    video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 生成并保存每一帧
    for i in range(total_frames):
        # 旋转点云
        ctr.rotate(10, 0)
        
        # 渲染当前帧
        vis.poll_events()
        vis.update_renderer()

        # 获取帧图像
        image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        image = (255 * image).astype(np.uint8)  # 转换为 8 位图像
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (width, height))  # 确保每帧大小一致
        
        # 写入视频
        video_writer.write(image)
    
    # 释放资源
    video_writer.release()
    vis.destroy_window()

# 读取两个不同的点云文件
point_cloud_1 = o3d.io.read_point_cloud("outputs/point_cloud_heatmap_4cls_whole.ply")  # 替换为你的第一个点云文件路径
point_cloud_2 = o3d.io.read_point_cloud("outputs/point_cloud_4cls.ply")  # 替换为你的第二个点云文件路径

# 创建两个线程，分别生成两个视频
thread_1 = threading.Thread(target=create_rotation_video, args=(point_cloud_1, 'x', 'outputs/video/heatmap.mp4', 10, 30))
thread_2 = threading.Thread(target=create_rotation_video, args=(point_cloud_2, 'x', 'outputs/video/classification.mp4', 10, 30))

# 启动两个线程
thread_1.start()
thread_2.start()

# 等待两个线程结束
thread_1.join()
thread_2.join()

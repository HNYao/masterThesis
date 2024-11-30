import open3d as o3d
import numpy as np
import threading
import time

def rotate_point_cloud_around_x(point_cloud, angle_in_degrees=180):

    angle_in_radians = np.deg2rad(angle_in_degrees)
    

    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                                [0, np.sin(angle_in_radians), np.cos(angle_in_radians)]])
    

    rotated_point_cloud = point_cloud.rotate(rotation_matrix, center=(0, 0, 0))
    
    return rotated_point_cloud


def visualize_point_cloud(point_cloud, window_name):

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)

    vis.add_geometry(point_cloud)

    ctr = vis.get_view_control()

    rotation_angle = 3 
    while True:
        ctr.rotate(rotation_angle, 0)
        
        vis.poll_events()
        vis.update_renderer()
        
        time.sleep(0.01)

    vis.destroy_window()

pc_1 = o3d.io.read_point_cloud("test.ply")
#pc_2 = o3d.io.read_point_cloud("outputs/point_cloud_4cls.ply")
pc_1 = rotate_point_cloud_around_x(pc_1)
#pc_2 = rotate_point_cloud_around_x(pc_2)
thread_1 = threading.Thread(target=visualize_point_cloud, args=(pc_1, "point cloud classification"))
#thread_2 = threading.Thread(target=visualize_point_cloud, args=(pc_2, "point cloud heatmap"))

thread_1.start()
#thread_2.start()
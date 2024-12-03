import open3d as o3d
import numpy as np


def create_bounding_box(min_bound, max_bound):
    points = [
        [min_bound[0], min_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
    ]

    points = np.array(points)

    lines = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


if __name__ == "__main__":

    pcd = o3d.io.read_point_cloud(
        "dataset/scene_RGBD_mask_v2_kinect_cfg/id9/bottle_0001_plastic/mask_Behind.ply"
    )
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    green_mask = colors[:, 1] > 0.3
    filtered_points = points[green_mask]

    min_bound = filtered_points.min(axis=0)
    max_bound = filtered_points.max(axis=0)

    bounding_box = create_bounding_box(min_bound, max_bound)

    o3d.visualization.draw_geometries([bounding_box, pcd])
